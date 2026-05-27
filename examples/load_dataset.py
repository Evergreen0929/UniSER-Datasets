"""Minimal example: stream the UniSER haze dataset from Hugging Face.

Requires:
    pip install -U "huggingface_hub[hf_xet]" webdataset pillow

Each WebDataset sample is a dict whose keys are file extensions/suffixes for
that sample (WebDataset groups files in a tar by the part of the filename
before the first `.`):

    gt.jpg / gt.png  (bytes)   - clean RGB image (absent when source == "OTS")
    haze_000.png     (bytes)   - synthesized haze variant 0
    haze_000.txt     (bytes)   - descriptive tag for variant 0 (e.g., out_fog_120)
    haze_001.png     (bytes)
    haze_001.txt     (bytes)
    ...
    json             (bytes)   - metadata: source, base_name, upstream info, ...

This script:
  1) lists shard URLs on the Hugging Face Hub
  2) builds a WebDataset pipeline that yields (gt, haze, tag, meta) tuples
  3) prints a few samples and saves a tiny preview grid

Run:
    python examples/load_dataset.py --num 8
"""
import argparse
import io
import json
import random
from pathlib import Path

from huggingface_hub import HfFileSystem
from PIL import Image
import webdataset as wds


REPO_ID = "jdzhang0929/uniser-haze-dataset"
SHARDS_SUBDIR = "shards"


def list_shard_urls(repo_id: str, subdir: str) -> list[str]:
    """Return https URLs for every .tar shard in the dataset repo."""
    fs = HfFileSystem()
    base = f"datasets/{repo_id}/{subdir}"
    paths = [p for p in fs.ls(base, detail=False) if p.endswith(".tar")]
    return [
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/{Path(p).name.split('/')[-1]}"
        if "/" not in str(p) else
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/{subdir}/{Path(p).name}"
        for p in paths
    ]


def decode_sample(s: dict):
    """Decode one raw WebDataset sample dict into typed values.

    Picks one haze variant uniformly at random per sample. Returns None when
    the sample is malformed (e.g., zero haze variants).
    """
    if "json" not in s:
        return None
    meta = json.loads(s["json"])

    haze_imgs = sorted(k for k in s if k.startswith("haze_") and k.endswith(".png"))
    if not haze_imgs:
        return None
    chosen = random.choice(haze_imgs)
    haze = Image.open(io.BytesIO(s[chosen])).convert("RGB")
    tag_key = chosen.replace(".png", ".txt")
    tag = s[tag_key].decode("utf-8") if tag_key in s else ""

    gt_key = next((k for k in s if k.startswith("gt.")), None)
    gt = Image.open(io.BytesIO(s[gt_key])).convert("RGB") if gt_key else None

    return {"gt": gt, "haze": haze, "tag": tag, "meta": meta}


def build_pipeline(shard_urls: list[str], shuffle_buffer: int = 100):
    """Return an iterable WebDataset pipeline yielding decoded dicts."""
    return (
        wds.WebDataset(shard_urls, shardshuffle=True, nodesplitter=wds.split_by_node)
        .shuffle(shuffle_buffer)
        .map(decode_sample)
        .select(lambda x: x is not None)
    )


def save_preview_grid(samples, out_path: Path, thumb_h: int = 256):
    """Write a small contact-sheet PNG for quick visual sanity-check."""
    rows = []
    for s in samples:
        cells = []
        if s["gt"] is not None:
            cells.append(s["gt"])
        cells.append(s["haze"])
        # Resize cells to the same height.
        ws = [int(c.width * thumb_h / c.height) for c in cells]
        row = Image.new("RGB", (sum(ws), thumb_h), "black")
        x = 0
        for c, w in zip(cells, ws):
            row.paste(c.resize((w, thumb_h), Image.LANCZOS), (x, 0))
            x += w
        rows.append(row)

    if not rows:
        print("no samples to preview.")
        return
    width = max(r.width for r in rows)
    grid = Image.new("RGB", (width, thumb_h * len(rows)), "black")
    for i, r in enumerate(rows):
        grid.paste(r, (0, i * thumb_h))
    grid.save(out_path)
    print(f"preview saved to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=8,
                    help="How many samples to fetch for the preview.")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--shards-subdir", default=SHARDS_SUBDIR)
    ap.add_argument("--preview", default="/tmp/uniser_preview.png")
    args = ap.parse_args()

    urls = list_shard_urls(args.repo_id, args.shards_subdir)
    print(f"discovered {len(urls)} shards on HF.")
    if not urls:
        print("Hint: the dataset is gated. Make sure you have accepted the "
              "access terms on the HF dataset page.")
        return

    pipeline = build_pipeline(urls)

    samples = []
    for i, s in enumerate(pipeline):
        meta = s["meta"]
        print(f"#{i:>3}  source={meta['source']:<11} "
              f"base={meta['base_name'][:40]:<40} tag={s['tag']}")
        samples.append(s)
        if len(samples) >= args.num:
            break

    save_preview_grid(samples, Path(args.preview))


if __name__ == "__main__":
    main()
