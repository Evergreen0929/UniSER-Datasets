"""Pack the UniSER haze dataset from S3 into WebDataset .tar shards.

Reads the synthetic-haze manifests (one JSON per upstream source plus
synthetic_haze_all_dataset.json for WSRD / REAL_FLARE / ISTD), streams each
sample's GT image and synthesized haze variants from S3, and writes them
into ~2 GB tar shards under --out-dir.

Sample layout inside a shard (WebDataset convention — files sharing a key
prefix form one sample):

    {source}/{base_name}.gt.<ext>            GT clean image (omitted for OTS)
    {source}/{base_name}.haze_000.png        synthesized haze variant 0
    {source}/{base_name}.haze_000.txt        descriptive tag, e.g. "out_fog_120"
    {source}/{base_name}.haze_001.png
    ...
    {source}/{base_name}.json                per-sample metadata

Six sources are bundled (HALO_QING excluded). OTS GT images are NOT included
because RESIDE-OTS clean photos carry third-party copyrights;
scripts/prepare_ots_originals.py fetches them from the official source.

Sources are mixed across shards (deterministic seed) so each shard is a
balanced mini-batch.
"""
import argparse
import io
import json
import os
import random
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from tqdm import tqdm

# ---------------------------------------------------------------- config
S3_BUCKET_DEFAULT = "adobe-lingzhi-p"

# JSON manifests live outside the repo; override via --json-dir.
JSON_DIR_DEFAULT = (
    "/sensei-fs/users/yizhouw/projects/collaboration/colligo/contrib/Mori"
    "/mori/collections/experimental/foundation_playground/src"
    "/foundation_playground/data/json_paths"
)
JSON_FILES = [
    "synthetic_haze_hazespace.json",
    "synthetic_haze_its.json",
    "synthetic_haze_ots.json",
    "synthetic_haze_all_dataset.json",   # contains WSRD, REAL_FLARE, ISTD (+ HALO_QING, skipped)
]

INCLUDED_SOURCES = {"HAZESPACE", "ITS", "OTS", "WSRD", "REAL_FLARE", "ISTD"}
SKIP_GT_SOURCES = {"OTS"}  # clean photos carry third-party copyrights

UPSTREAM_INFO = {
    "HAZESPACE":  {"upstream": "HazeSpace2M",         "citation": "Islam et al., ACM MM 2024",     "license": "CC-BY-4.0"},
    "ITS":        {"upstream": "RESIDE-ITS",          "citation": "Li et al., TIP 2018",            "license": "No explicit license; academic-use convention"},
    "OTS":        {"upstream": "RESIDE-OTS",          "citation": "Li et al., TIP 2018",            "license": "No explicit license; clean images not redistributed"},
    "WSRD":       {"upstream": "WSRD",                "citation": "Vasluianu et al., CVPRW 2023",  "license": "CC-BY-NC-SA-4.0"},
    "REAL_FLARE": {"upstream": "Flare-R (Flare7K++)", "citation": "Dai et al., TPAMI 2024",         "license": "S-Lab License 1.0"},
    "ISTD":       {"upstream": "ISTD",                "citation": "Wang et al., CVPR 2018",         "license": "Research & non-commercial only"},
}

# ---------------------------------------------------------------- manifest loading
def load_manifest(json_dir: Path):
    """Read all JSONs and return a flat list of (source, record) dicts."""
    records = []
    for fname in JSON_FILES:
        path = json_dir / fname
        print(f"reading {path.name} ({path.stat().st_size / 1e6:.1f} MB)...")
        with open(path) as f:
            data = json.load(f)
        for rec in data:
            src = rec["gt_image"].split("/")[2]
            if src not in INCLUDED_SOURCES:
                continue
            records.append({
                "source": src,
                "gt_image": rec["gt_image"],
                "hazy_images": rec["hazy_images"],
                "base_name": rec.get("base_name", Path(rec["gt_image"]).stem),
            })
        del data
    return records


def sanitize(name: str) -> str:
    """Make a string safe to use as a tar key segment."""
    return name.replace("/", "_").replace(" ", "_")


# ---------------------------------------------------------------- byte fetching
def fetch_blobs(s3, rec):
    """Download every byte we'll need for one sample. Returns (rec, blobs) or
    (rec, None) on error. Blobs is a list of (tar_name, bytes) ready to write.
    """
    src = rec["source"]
    skip_gt = src in SKIP_GT_SOURCES
    key = f"{src.lower()}/{sanitize(rec['base_name'])}"
    bucket = os.environ.get("UNISER_S3_BUCKET", S3_BUCKET_DEFAULT)

    blobs = []
    haze_meta = []

    try:
        # GT image.
        if not skip_gt:
            obj = s3.get_object(Bucket=bucket, Key=rec["gt_image"])
            gt_bytes = obj["Body"].read()
            ext = Path(rec["gt_image"]).suffix.lstrip(".").lower() or "png"
            blobs.append((f"{key}.gt.{ext}", gt_bytes))

        # Haze variants.
        for idx, hz_key in enumerate(rec["hazy_images"]):
            obj = s3.get_object(Bucket=bucket, Key=hz_key)
            hz_bytes = obj["Body"].read()
            tag = Path(hz_key).stem
            blobs.append((f"{key}.haze_{idx:03d}.png", hz_bytes))
            blobs.append((f"{key}.haze_{idx:03d}.txt", tag.encode("utf-8")))
            haze_meta.append({"idx": idx, "filename": f"haze_{idx:03d}.png", "tag": tag})

        info = UPSTREAM_INFO.get(src, {})
        meta = {
            "source": src,
            "base_name": rec["base_name"],
            "gt_included": not skip_gt,
            "haze_variants": haze_meta,
            "upstream_dataset": info.get("upstream", "unknown"),
            "upstream_citation": info.get("citation", ""),
            "upstream_license": info.get("license", ""),
        }
        if skip_gt:
            meta["gt_note"] = (
                "Clean image not redistributed due to upstream third-party copyrights. "
                "Use scripts/prepare_ots_originals.py to fetch it from the official source."
            )
        blobs.append((f"{key}.json", json.dumps(meta, ensure_ascii=False).encode("utf-8")))
        return rec, blobs
    except Exception as e:
        print(f"\n  !! fetch error: {src}/{rec['base_name']}: {e}", file=sys.stderr)
        return rec, None


# ---------------------------------------------------------------- tar writing
def write_blobs(tar, blobs, now):
    written = 0
    for name, data in blobs:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = now
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
        # Tar pads each entry to 512-byte blocks plus a 512-byte header.
        written += 512 + ((len(data) + 511) // 512) * 512
    return written


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True,
                    help="Local directory to write shard-NNNNN.tar files into.")
    ap.add_argument("--json-dir", default=JSON_DIR_DEFAULT,
                    help="Directory containing the synthetic_haze_*.json manifests.")
    ap.add_argument("--shard-bytes", type=int, default=2 * 1024**3,
                    help="Target shard size in bytes (default: 2 GiB).")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="Subset of sources to pack (default: all 6).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total samples (for dry-run).")
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel S3 download workers.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Shuffle seed for cross-source mixing.")
    args = ap.parse_args()

    records = load_manifest(Path(args.json_dir))
    print(f"loaded {len(records)} records before filtering")

    if args.sources:
        wanted = set(args.sources)
        records = [r for r in records if r["source"] in wanted]
        print(f"  filtered to sources {sorted(wanted)}: {len(records)} records")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.limit:
        records = records[: args.limit]
        print(f"  limited to {len(records)} records (dry-run)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distribution stats.
    counts = defaultdict(int)
    for r in records:
        counts[r["source"]] += 1
    print(f"\nsamples per source:")
    for src in sorted(counts, key=lambda s: -counts[s]):
        print(f"  {src:<11} {counts[src]:>8}")

    s3 = boto3.client("s3")
    now = int(time.time())

    shard_idx = 0
    shard_path = out_dir / f"shard-{shard_idx:05d}.tar"
    tar = tarfile.open(shard_path, "w")
    cur_bytes = 0
    samples_in_shard = 0
    samples_done = 0
    samples_failed = 0

    pbar = tqdm(total=len(records), desc="packing", unit="sample")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rec, blobs in ex.map(lambda r: fetch_blobs(s3, r), records):
                if blobs is None:
                    samples_failed += 1
                    pbar.update(1)
                    continue

                cur_bytes += write_blobs(tar, blobs, now)
                samples_in_shard += 1
                samples_done += 1
                pbar.update(1)
                pbar.set_postfix(
                    shard=shard_idx,
                    cur_gb=f"{cur_bytes / 1e9:.2f}",
                    failed=samples_failed,
                )

                if cur_bytes >= args.shard_bytes:
                    tar.close()
                    tqdm.write(
                        f"  -> closed {shard_path.name} "
                        f"({cur_bytes / 1e9:.2f} GB, {samples_in_shard} samples)"
                    )
                    shard_idx += 1
                    shard_path = out_dir / f"shard-{shard_idx:05d}.tar"
                    tar = tarfile.open(shard_path, "w")
                    cur_bytes = 0
                    samples_in_shard = 0
    finally:
        tar.close()
        pbar.close()
        # Drop the last shard if empty (e.g. exact multiple).
        if samples_in_shard == 0 and shard_path.exists():
            shard_path.unlink()
            shard_idx -= 1

    print(f"\ndone: {samples_done} samples ({samples_failed} failed) "
          f"-> {shard_idx + 1} shards in {out_dir}")


if __name__ == "__main__":
    main()
