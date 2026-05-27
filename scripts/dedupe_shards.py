#!/usr/bin/env python3
"""Post-process WebDataset shards to remove duplicate tar entries.

When the upstream JSON manifests repeat the same record N times to express a
sampling-weight choice (WSRD/REAL_FLARE/ISTD in the UniSER bundle),
pack_shards.py emits a .tar where the same `key.gt.<ext>` / `key.haze_NNN.png`
/ `key.json` appears multiple times. WebDataset's `group_by_keys` then raises:

    ValueError: <name>: duplicate file name in tar file ...

This script streams each shard and renames offending entries so every
(field) is unique per key. Behavior:

  * `gt.<ext>` and `json` — only the first occurrence per key is kept (later
    records have identical bytes anyway, since each duplicate record carries
    the same GT path and the same haze list).
  * `haze_NNN.png` / `haze_NNN.txt` — renumbered to the next sequential slot
    within that key, so all duplicate haze variants are retained but live
    under unique field names (haze_000, haze_001, ..., haze_071 for a 3x
    duplicated WSRD base). The companion `.txt` is paired with its `.png` by
    reusing the same new index.
  * Sources with confirmed unique base_names (HAZESPACE / ITS / OTS /
    HALO_QING per the JSON analysis) are passed through without state
    tracking for speed.

Usage:
    python scripts/dedupe_shards.py \\
        --in-dir  /mnt/localssd/staging/main \\
        --out-dir /mnt/localssd/staging/main_dedup

Result: a parallel directory of shard-NNNNN.tar that webdataset can read.
"""
import argparse
import io
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


# Sources whose base_names are unique by construction (confirmed via JSON
# duplication audit). Entries belonging to these sources are copied through
# without any rename or state tracking.
NO_DUP_SOURCES = {"hazespace", "its", "ots", "halo_qing"}


def split_key_field(member_name: str):
    """`hazespace/abc.haze_000.png` -> ('hazespace/abc', 'haze_000.png')."""
    if "/" in member_name:
        dirpart, fname = member_name.rsplit("/", 1)
    else:
        dirpart, fname = "", member_name
    if "." not in fname:
        return None, None
    stem, _, ext = fname.partition(".")
    key = f"{dirpart}/{stem}" if dirpart else stem
    return key, ext


def source_of(member_name: str):
    """Return the source token (key directory) for a tar member name."""
    if "/" not in member_name:
        return None
    return member_name.split("/", 1)[0].lower()


def dedupe_one_shard(src: Path, dst: Path):
    """Read src tar, write dst tar with duplicate field names renamed.

    Entries belonging to NO_DUP_SOURCES are passed through unchanged. For the
    duplicate-prone sources, we keep the first gt/json and renumber haze
    field indices so all duplicate variants survive under unique names.
    Duplicate haze content (same bytes in multiple records) is preserved
    on disk — only the field name is changed.
    """
    counters = dict(
        members_in=0, members_out=0, passthrough=0,
        dup_gt=0, dup_json=0,
        haze_renamed=0, haze_txt_orphan=0,
        keys=0,
    )

    state: dict = {}
    def init_key():
        counters["keys"] += 1
        return dict(
            gt_done=False,
            json_done=False,
            next_haze_idx=0,
            current_record_orig_to_new={},   # original NNN -> new_idx for
                                             # pairing png with its txt
        )

    now = int(time.time())

    def write_entry(t_out, name: str, data: bytes, mtime):
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = mtime or now
        info.mode = 0o644
        t_out.addfile(info, io.BytesIO(data))
        counters["members_out"] += 1

    with tarfile.open(src, "r") as t_in, tarfile.open(dst, "w") as t_out:
        for m in t_in:
            if not m.isfile():
                continue
            counters["members_in"] += 1

            # Fast path: sources confirmed unique by JSON audit -> pass through.
            src_token = source_of(m.name)
            if src_token in NO_DUP_SOURCES:
                buf = t_in.extractfile(m)
                write_entry(t_out, m.name, buf.read() if buf else b"", m.mtime)
                counters["passthrough"] += 1
                continue

            key, field = split_key_field(m.name)
            if key is None:
                continue

            s = state.setdefault(key, init_key())

            if field.startswith("gt."):
                if s["gt_done"]:
                    # New record boundary -> reset per-record mapping.
                    s["current_record_orig_to_new"] = {}
                    counters["dup_gt"] += 1
                    continue
                s["gt_done"] = True
                s["current_record_orig_to_new"] = {}
                buf = t_in.extractfile(m)
                write_entry(t_out, m.name, buf.read() if buf else b"", m.mtime)
                continue

            if field == "json":
                if s["json_done"]:
                    counters["dup_json"] += 1
                    continue
                s["json_done"] = True
                buf = t_in.extractfile(m)
                write_entry(t_out, m.name, buf.read() if buf else b"", m.mtime)
                continue

            if field.startswith("haze_") and field.endswith(".png"):
                orig = field[len("haze_"):-len(".png")]
                new_idx = s["next_haze_idx"]
                s["next_haze_idx"] += 1
                s["current_record_orig_to_new"][orig] = new_idx
                new_field = f"haze_{new_idx:03d}.png"
                new_name = f"{key}.{new_field}"
                if new_field != field:
                    counters["haze_renamed"] += 1
                buf = t_in.extractfile(m)
                write_entry(t_out, new_name, buf.read() if buf else b"", m.mtime)
                continue

            if field.startswith("haze_") and field.endswith(".txt"):
                orig = field[len("haze_"):-len(".txt")]
                new_idx = s["current_record_orig_to_new"].get(orig)
                if new_idx is None:
                    # .txt arrived without a paired .png in this record (rare).
                    # Fall back to a fresh slot rather than dropping data.
                    new_idx = s["next_haze_idx"]
                    s["next_haze_idx"] += 1
                    counters["haze_txt_orphan"] += 1
                new_field = f"haze_{new_idx:03d}.txt"
                new_name = f"{key}.{new_field}"
                if new_field != field:
                    counters["haze_renamed"] += 1
                buf = t_in.extractfile(m)
                write_entry(t_out, new_name, buf.read() if buf else b"", m.mtime)
                continue

            # Anything else: pass through unchanged.
            buf = t_in.extractfile(m)
            write_entry(t_out, m.name, buf.read() if buf else b"", m.mtime)

    return counters


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True,
                    help="Directory containing shard-*.tar files to dedupe.")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write the deduplicated shards. Must differ "
                         "from --in-dir.")
    ap.add_argument("--pattern", default="shard-*.tar",
                    help="Glob pattern for shards (default: shard-*.tar).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip an output shard that already exists.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if in_dir == out_dir:
        sys.exit("ERROR: --in-dir and --out-dir must differ.")
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(in_dir.glob(args.pattern))
    if not shards:
        sys.exit(f"No shards matching {args.pattern} in {in_dir}")
    print(f"dedupe {len(shards)} shards: {in_dir} -> {out_dir}")

    total = defaultdict(int)
    pbar = tqdm(shards, desc="shards")
    for sh in pbar:
        dst = out_dir / sh.name
        if args.skip_existing and dst.exists():
            pbar.write(f"  skip (exists): {sh.name}")
            continue
        c = dedupe_one_shard(sh, dst)
        for k, v in c.items():
            total[k] += v
        pbar.set_postfix(in_=c["members_in"], out=c["members_out"],
                         dup_gt=c["dup_gt"], renamed=c["haze_renamed"])

    print("\n--- totals ---")
    for k in ("keys", "members_in", "members_out",
              "passthrough", "dup_gt", "dup_json",
              "haze_renamed", "haze_txt_orphan"):
        print(f"  {k:<18} {total[k]:>12}")


if __name__ == "__main__":
    main()
