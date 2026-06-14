#!/usr/bin/env python3
"""Render an HTML gallery of random samples from the HALO lens-flare HF release.

Downloads one or more WebDataset `.tar` shards from the Hugging Face Hub
(default: jdzhang0929/halo-flare-dataset), pulls a balanced random selection
of samples across effect types (Streak / Reflective / Glare / Shimmer), and
writes a dark-themed `index.html` showing each sample's clean / flared /
flare-only triplet side-by-side.

Prerequisites:
    pip install -U "huggingface_hub[hf_xet]" pillow
    hf auth login                       # must have accepted the gated terms

Typical usage:
    python scripts/make_halo_gallery.py
    python scripts/make_halo_gallery.py --total 60 --out-dir /tmp/halo_gallery
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


REPO_ID_DEFAULT = "jdzhang0929/halo-flare-dataset"
SHARDS_SUBDIR = "shards"
DEFAULT_OUT_DIR = Path("/tmp/halo_gallery")

EFFECT_ORDER = ["Streak", "Reflective", "Glare", "Shimmer"]


# --- WebDataset key parsing -------------------------------------------------
def split_key_field(member_name: str):
    """Return (key, field) given a tar member name.

    Mirrors webdataset's `base_plus_ext`: splits on the FIRST dot after the
    last `/`. e.g. `halo/Scene003_Glare001_camera01.gt.png` ->
    ('halo/Scene003_Glare001_camera01', 'gt.png').
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
def decode_sample(key, fields):
    """Build a typed dict from raw bytes; returns None on malformed samples."""
    if "json" not in fields:
        return None
    try:
        meta = json.loads(fields["json"])
    except json.JSONDecodeError:
        return None
    needed = ("gt.png", "flare.png", "separate.png")
    if not all(n in fields for n in needed):
        return None
    return {
        "key": key,
        "effect_type": meta.get("effect_type", "Unknown"),
        "effect_id":   meta.get("effect_id",   "Unknown"),
        "scene":       meta.get("scene",       "Unknown"),
        "sample_id":   meta.get("sample_id",   key.split("/")[-1]),
        "bytes": {
            "gt":       fields["gt.png"],
            "flare":    fields["flare.png"],
            "separate": fields["separate.png"],
        },
        "meta": meta,
    }


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
def stratified_pick(samples_by_effect: dict, total: int, min_per: int, rng):
    """Distribute `total` picks across effect types, sqrt-weighted by pool size."""
    pools = {e: len(samples_by_effect[e]) for e in samples_by_effect}
    if not pools:
        return []
    weights = {e: math.sqrt(n) for e, n in pools.items()}
    alloc = {e: min(min_per, pools[e]) for e in pools}
    remaining = max(0, total - sum(alloc.values()))
    wsum = sum(weights.values())
    if remaining and wsum > 0:
        raw = {e: remaining * weights[e] / wsum for e in pools}
        floors = {e: int(raw[e]) for e in pools}
        leftover = remaining - sum(floors.values())
        frac_order = sorted(pools, key=lambda e: raw[e] - floors[e], reverse=True)
        for e in frac_order[:leftover]:
            floors[e] += 1
        for e in pools:
            alloc[e] = min(alloc[e] + floors[e], pools[e])

    picks = []
    for e, k in alloc.items():
        picks.extend(rng.sample(samples_by_effect[e], k))
    rng.shuffle(picks)

    print(f"\nallocation (sqrt-weighted, total={total}, min={min_per}):")
    for e in sorted(pools, key=lambda d: -pools[d]):
        print(f"  {e:<11} pool={pools[e]:>4}  -> samples={alloc[e]}")
    return picks


# --- HTML rendering ---------------------------------------------------------
def save_image_bytes(out_dir: Path, sample_key: str, suffix: str, data: bytes) -> str:
    safe = sample_key.replace("/", "__")
    rel = Path("images") / f"{safe}__{suffix}"
    full = out_dir / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(data)
    return rel.as_posix()


EFFECT_COLOR = {
    "Streak":     "#ffd76b",
    "Reflective": "#9cf",
    "Glare":      "#ff9c6b",
    "Shimmer":    "#b9ff6b",
}


def render_html(samples, out_dir: Path, repo_id: str, seed: int):
    sections = []
    for s in samples:
        cells = []
        for kind in ("gt", "flare", "separate"):
            data = s["bytes"][kind]
            rel = save_image_bytes(out_dir, s["key"], f"{kind}.png", data)
            label = {"gt": "light (clean)",
                     "flare": "flare (with effect)",
                     "separate": "separate (flare only)"}[kind]
            cells.append(
                f'<div class="cell"><img src="{rel}" loading="lazy">'
                f'<div class="cap"><b>{label}</b></div></div>'
            )
        eff = s["effect_type"]
        color = EFFECT_COLOR.get(eff, "#fff")
        sections.append(
            f'<section><h3>'
            f'<span style="color:{color}">{eff}</span> · '
            f'<span class="effid">{s["effect_id"]}</span> · '
            f'<span class="bn">{s["sample_id"]}</span>'
            f'</h3>'
            f'<div class="row">{"".join(cells)}</div></section>'
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HALO Lens-Flare Dataset — random samples</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif;
          margin: 16px; background: #111; color: #eee; }}
  h1 {{ margin: 6px 0; }}
  .meta {{ color: #888; font-size: 13px; margin-bottom: 18px; }}
  section {{ margin-bottom: 28px; border-bottom: 1px solid #333;
             padding-bottom: 16px; }}
  h3 {{ font-size: 14px; margin: 6px 0; font-weight: 500; }}
  .effid {{ color: #aaa; font-weight: 500; font-size: 13px; }}
  .bn {{ color: #777; font-weight: 400; font-size: 12px; }}
  .row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
  .cell {{ text-align: center; }}
  .cell img {{ width: 100%; aspect-ratio: 16/9; object-fit: contain;
               background: #000; border: 1px solid #444; display: block; }}
  .cap {{ font-size: 12px; color: #ccc; margin-top: 4px;
          word-break: break-all; line-height: 1.3; }}
</style>
</head>
<body>
<h1>HALO Lens-Flare Dataset · random samples</h1>
<p class="meta">{len(samples)} samples from
  <a style="color:#9cf" href="https://huggingface.co/datasets/{repo_id}">{repo_id}</a>.
  Each row shows the clean scene, the same scene with flare added, and the
  flare layer alone (transparent background). seed={seed}.</p>
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
    ap.add_argument("--min-per-effect", type=int, default=2,
                    help="Minimum samples per effect type (default: 2).")
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
    samples_by_effect: dict = defaultdict(list)
    for sp in local_shard_paths:
        groups = group_shard_samples(sp)
        for key, fields in groups.items():
            sample = decode_sample(key, fields)
            if sample is not None:
                samples_by_effect[sample["effect_type"]].append(sample)

    picks = stratified_pick(samples_by_effect, args.total, args.min_per_effect, rng)
    if not picks:
        sys.exit("No samples to render.")

    render_html(picks, out_dir, args.repo_id, args.seed)
    print(f"\nopen with: python3 -m http.server --directory {out_dir} 8000")


if __name__ == "__main__":
    main()
