#!/usr/bin/env python3
"""Render an HTML gallery of random samples from the UniSER-Datasets HF release.

Downloads one or more WebDataset `.tar` shards from the Hugging Face Hub
(default: jdzhang0929/uniser-haze-dataset), pulls a balanced random selection of
samples across sources, and writes a dark-themed `index.html` with the GT
image alongside several different haze types per sample.

Prerequisites:
    pip install -U "huggingface_hub[hf_xet]" pillow
    hf auth login                       # must have accepted the gated terms

Typical usage:
    python scripts/make_haze_gallery.py
    python scripts/make_haze_gallery.py --total 60 --out-dir /tmp/uniser_gallery
"""
import argparse
import io
import json
import math
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image


REPO_ID_DEFAULT = "jdzhang0929/uniser-haze-dataset"
SHARDS_SUBDIR = "shards"
DEFAULT_OUT_DIR = Path("/mnt/localssd/tmp")


# --- WebDataset key parsing -------------------------------------------------
def split_key_field(member_name: str):
    """Return (key, field) given a tar member name.

    Mirrors webdataset's `base_plus_ext`: splits on the FIRST dot after the
    last `/`. e.g. `hazespace/abc.haze_000.png` -> ('hazespace/abc', 'haze_000.png').
    """
    if "/" in member_name:
        dirpart, fname = member_name.rsplit("/", 1)
    else:
        dirpart, fname = "", member_name
    if "." not in fname:
        return None, None
    stem, _, ext = fname.partition(".")
    key = f"{dirpart}/{stem}" if dirpart else stem
    return key, ext


def group_shard_samples(tar_path: Path) -> dict:
    """Read a .tar and return {key -> {field_name -> bytes}}."""
    groups: dict = defaultdict(dict)
    with tarfile.open(tar_path, "r") as tar:
        for m in tar:
            if not m.isfile():
                continue
            key, field = split_key_field(m.name)
            if key is None:
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            groups[key][field] = f.read()
    return groups


# --- Sample decoding --------------------------------------------------------
def haze_type(tag: str) -> str:
    """`out_fog_120` -> `fog`; `out_haze_brownish-dust_v100` -> `haze_brownish-dust`."""
    if tag.startswith("out_"):
        rest = tag[4:]
        parts = rest.split("_")
        if parts and (parts[-1].isdigit()
                      or (parts[-1][:1] in ("v", "h", "d", "b", "k")
                          and parts[-1][1:].replace(".", "").isdigit())):
            parts = parts[:-1]
        return "_".join(parts) if parts else rest
    return tag.split("_")[0]


def decode_sample(key, fields):
    """Build a typed dict from raw bytes; returns None on malformed samples."""
    if "json" not in fields:
        return None
    meta = json.loads(fields["json"])

    gt = None
    for f in fields:
        if f.startswith("gt."):
            gt = Image.open(io.BytesIO(fields[f])).convert("RGB")
            break

    haze_pairs = []
    for f in sorted(fields):
        if f.startswith("haze_") and f.endswith(".png"):
            tag_field = f.replace(".png", ".txt")
            tag = fields.get(tag_field, b"").decode("utf-8", errors="ignore")
            haze_pairs.append((f, tag, fields[f]))
    if not haze_pairs:
        return None

    return {
        "key": key,
        "source": meta.get("source", "UNKNOWN"),
        "base_name": meta.get("base_name", key.split("/")[-1]),
        "gt": gt,
        "haze_pairs": haze_pairs,
        "meta": meta,
    }


def select_diverse_haze(sample, n, rng):
    """Pick n haze variants spanning different type buckets when possible."""
    buckets = defaultdict(list)
    for field, tag, data in sample["haze_pairs"]:
        buckets[haze_type(tag)].append((field, tag, data))
    types = list(buckets)
    rng.shuffle(types)
    picked = []
    for t in types:
        picked.append(rng.choice(buckets[t]))
        if len(picked) == n:
            break
    if len(picked) < n:
        all_v = [v for t in buckets for v in buckets[t]]
        leftover = [x for x in all_v if x not in picked]
        rng.shuffle(leftover)
        picked.extend(leftover[: n - len(picked)])
    return picked


# --- HF shard discovery / download ------------------------------------------
def list_remote_shards(repo_id: str, subdir: str) -> list:
    fs = HfFileSystem()
    base = f"datasets/{repo_id}/{subdir}"
    try:
        entries = fs.ls(base, detail=False)
    except FileNotFoundError:
        sys.exit(
            f"ERROR: {base!r} not found on the Hub. Either the dataset hasn't "
            "been uploaded yet, or you haven't accepted the gated access terms.\n"
            f"Visit https://huggingface.co/datasets/{repo_id} to request access."
        )
    return sorted(Path(p).name for p in entries if p.endswith(".tar"))


def download_shard(repo_id: str, shard_name: str, cache_dir=None) -> Path:
    p = hf_hub_download(
        repo_id=repo_id,
        filename=f"{SHARDS_SUBDIR}/{shard_name}",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return Path(p)


# --- Sampling ---------------------------------------------------------------
def stratified_pick(samples_by_source: dict, total: int, min_per: int, rng):
    """Distribute `total` picks across sources, sqrt-weighted by available pool."""
    pools = {s: len(samples_by_source[s]) for s in samples_by_source}
    if not pools:
        return []
    weights = {s: math.sqrt(n) for s, n in pools.items()}
    alloc = {s: min(min_per, pools[s]) for s in pools}
    remaining = max(0, total - sum(alloc.values()))
    wsum = sum(weights.values())
    if remaining and wsum > 0:
        raw = {s: remaining * weights[s] / wsum for s in pools}
        floors = {s: int(raw[s]) for s in pools}
        leftover = remaining - sum(floors.values())
        frac_order = sorted(pools, key=lambda s: raw[s] - floors[s], reverse=True)
        for s in frac_order[:leftover]:
            floors[s] += 1
        for s in pools:
            alloc[s] = min(alloc[s] + floors[s], pools[s])

    picks = []
    for s, k in alloc.items():
        picks.extend(rng.sample(samples_by_source[s], k))
    rng.shuffle(picks)

    print(f"\nallocation (sqrt-weighted, total={total}, min={min_per}):")
    for s in sorted(pools, key=lambda d: -pools[d]):
        print(f"  {s:<11} pool={pools[s]:>6}  -> samples={alloc[s]}")
    return picks


# --- HTML rendering ---------------------------------------------------------
def save_image_bytes(out_dir: Path, sample_key: str, suffix: str, data: bytes) -> str:
    safe = sample_key.replace("/", "__")
    rel = Path("images") / f"{safe}__{suffix}"
    full = out_dir / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(data)
    return rel.as_posix()


def render_html(samples, out_dir: Path, per_haze: int, repo_id: str, seed: int):
    sections = []
    for s in samples:
        cells = []
        if s["gt"] is not None:
            buf = io.BytesIO()
            s["gt"].save(buf, format="PNG")
            rel = save_image_bytes(out_dir, s["key"], "gt.png", buf.getvalue())
            cells.append(
                f'<div class="cell gt"><img src="{rel}" loading="lazy">'
                f'<div class="cap"><b>GT</b> — {s["source"]}</div></div>'
            )
        else:
            cells.append(
                '<div class="cell"><div class="no-gt">No GT<br><small>'
                '(OTS — fetch via prepare_ots_originals.py)</small></div></div>'
            )

        for field, tag, data in s["chosen_haze"]:
            rel = save_image_bytes(out_dir, s["key"], field, data)
            cells.append(
                f'<div class="cell"><img src="{rel}" loading="lazy">'
                f'<div class="cap"><span class="type">{haze_type(tag)}</span>'
                f'<br><span class="tag">{tag}</span></div></div>'
            )
        sections.append(
            f'<section><h3>{s["source"]} · '
            f'<span class="bn">{s["base_name"]}</span></h3>'
            f'<div class="row">{"".join(cells)}</div></section>'
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UniSER-Datasets — random samples</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif;
          margin: 16px; background: #111; color: #eee; }}
  h1 {{ margin: 6px 0; }}
  .meta {{ color: #888; font-size: 13px; margin-bottom: 18px; }}
  section {{ margin-bottom: 28px; border-bottom: 1px solid #333;
             padding-bottom: 16px; }}
  h3 {{ font-size: 14px; color: #9cf; margin: 6px 0; font-weight: 500; }}
  .bn {{ color: #777; font-weight: 400; font-size: 12px; }}
  .row {{ display: flex; gap: 14px; overflow-x: auto; padding-bottom: 6px; }}
  .cell {{ flex: 0 0 auto; text-align: center; max-width: 280px; }}
  .cell.gt img {{ outline: 2px solid #4c9; }}
  .cell img {{ height: 240px; max-width: 280px; object-fit: contain;
               background: #000; border: 1px solid #444; display: block; }}
  .cap {{ font-size: 12px; color: #ccc; margin-top: 4px;
          word-break: break-all; line-height: 1.3; }}
  .type {{ color: #ffd76b; font-weight: 600; }}
  .tag  {{ color: #888; font-size: 11px; }}
  .no-gt {{ width: 240px; height: 240px; border: 1px dashed #555;
            display: flex; align-items: center; justify-content: center;
            color: #777; font-size: 13px; text-align: center; padding: 8px; }}
</style>
</head>
<body>
<h1>UniSER-Datasets · random samples</h1>
<p class="meta">{len(samples)} samples from
  <a style="color:#9cf" href="https://huggingface.co/datasets/{repo_id}">{repo_id}</a>.
  Each row: ground-truth + {per_haze} different haze variants. seed={seed}.</p>
{"".join(sections)}
</body></html>"""

    out_html = out_dir / "index.html"
    out_html.write_text(html)
    print(f"\nwrote {out_html}")


# --- Main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=REPO_ID_DEFAULT,
                    help=f"HF dataset repo id (default: {REPO_ID_DEFAULT}).")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="How many random shards to download (default: 1). "
                         "Each shard is ~2 GB.")
    ap.add_argument("--total", type=int, default=24,
                    help="Total samples to render (default: 24).")
    ap.add_argument("--min-per-source", type=int, default=1,
                    help="Minimum samples per source (default: 1).")
    ap.add_argument("--per-haze", type=int, default=3,
                    help="Haze variants per sample row (default: 3).")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help=f"Where to write index.html + images/ (default: {DEFAULT_OUT_DIR}).")
    ap.add_argument("--cache-dir", default=None,
                    help="HF cache dir (default: ~/.cache/huggingface).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"listing shards in {args.repo_id}/{SHARDS_SUBDIR}...")
    all_shards = list_remote_shards(args.repo_id, SHARDS_SUBDIR)
    print(f"  found {len(all_shards)} shards on HF.")
    if not all_shards:
        sys.exit("No shards yet — has the upload finished?")

    rng.shuffle(all_shards)
    chosen_shards = all_shards[: args.num_shards]
    print(f"\ndownloading {len(chosen_shards)} shard(s):")
    local_shard_paths = []
    for sh in chosen_shards:
        print(f"  {sh} ...")
        local_shard_paths.append(download_shard(args.repo_id, sh, args.cache_dir))

    print("\nreading samples...")
    samples_by_source: dict = defaultdict(list)
    for sp in local_shard_paths:
        groups = group_shard_samples(sp)
        for key, fields in groups.items():
            sample = decode_sample(key, fields)
            if sample is not None:
                samples_by_source[sample["source"]].append(sample)

    picks = stratified_pick(samples_by_source, args.total, args.min_per_source, rng)
    if not picks:
        sys.exit("No samples to render.")

    for s in picks:
        s["chosen_haze"] = select_diverse_haze(s, args.per_haze, rng)

    render_html(picks, out_dir, args.per_haze, args.repo_id, args.seed)
    print(f"\nopen with: python3 -m http.server --directory {out_dir} 8000")


if __name__ == "__main__":
    main()
