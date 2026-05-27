"""Fetch clean GT images for the OTS subset and align them to UniSER's layout.

We cannot redistribute the RESIDE-OTS clean images directly because they were
originally sourced from Flickr / other public photo sites and carry
third-party photographer copyrights. This script:

  1. Reads every OTS sample's metadata from the local UniSER bundle.
  2. Asks the user to download RESIDE-β (which contains OTS) from the
     official source — link printed below.
  3. Once a local RESIDE-β folder is provided, matches base_names and
     symlinks (or copies) the clean images into `<out-dir>/OTS/img/`.

After this script runs, the directory layout matches:

    <bundle>/OTS/img/<base_name>.jpg         ← from this script
    <bundle>/OTS/haze_v2/<base_name>/*.png   ← from the HF dataset

So downstream training code can load (gt, haze) pairs without code changes.

Usage:
    python scripts/prepare_ots_originals.py \\
        --bundle-dir /path/to/downloaded/uniser-haze-dataset \\
        --reside-dir /path/to/RESIDE-beta/OTS_BETA  \\
        --out-dir    /path/to/downloaded/uniser-haze-dataset/OTS/img
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

RESIDE_DOWNLOAD_INFO = """
RESIDE-OTS download (official, requires manual download):

  Project page:
      https://sites.google.com/view/reside-dehaze-datasets

  Direct links (RESIDE-beta OTS_BETA, ~25 GB):
      - Google Drive:  https://drive.google.com/file/d/1Vy0dD5IiQ8m9q9DUmpc-nXJBLcXOq3yA/view
      - Baidu Pan:     see project page

  After download, extract the archive; the clean images live in:
      OTS_BETA/clear_images/*.jpg

Cite:  Li et al., "Benchmarking Single-Image Dehazing and Beyond",
       IEEE TIP 2018.
"""


def collect_ots_base_names_from_shards(bundle_dir: Path) -> set[str]:
    """Read .tar shards and collect base_name for every OTS sample."""
    base_names = set()
    shards = sorted(bundle_dir.rglob("shard-*.tar"))
    if not shards:
        sys.exit(
            f"ERROR: no shard-*.tar files found under {bundle_dir}. "
            "Pass the directory containing the downloaded HF shards."
        )
    print(f"scanning {len(shards)} shards for OTS samples...")
    for sh in shards:
        with tarfile.open(sh, "r") as tar:
            for member in tar:
                if not member.name.endswith(".json"):
                    continue
                if not member.name.startswith("ots/"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                meta = json.loads(f.read())
                base_names.add(meta["base_name"])
    return base_names


def index_reside_files(reside_dir: Path) -> dict[str, Path]:
    """Build {base_name (no ext) -> full path} for every image in reside_dir."""
    print(f"indexing RESIDE files under {reside_dir}...")
    idx = {}
    for p in reside_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        idx[p.stem] = p
    print(f"  found {len(idx)} candidate files.")
    return idx


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-dir", required=True,
                    help="Path where the HF UniSER dataset has been downloaded "
                         "(contains shard-*.tar files).")
    ap.add_argument("--reside-dir", default=None,
                    help="Path to an unzipped RESIDE-beta OTS folder. If "
                         "omitted, this script just prints download instructions.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to place matched files (default: "
                         "<bundle-dir>/OTS/img).")
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"],
                    default="symlink",
                    help="How to materialize matched files (default: symlink).")
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    if not bundle_dir.is_dir():
        sys.exit(f"ERROR: --bundle-dir not found: {bundle_dir}")

    ots_base_names = collect_ots_base_names_from_shards(bundle_dir)
    print(f"OTS samples in bundle: {len(ots_base_names)}")

    if not args.reside_dir:
        print(RESIDE_DOWNLOAD_INFO)
        print(f"\nWhen you have RESIDE-beta downloaded, re-run this script "
              f"with --reside-dir <path>.")
        return

    reside_dir = Path(args.reside_dir).resolve()
    if not reside_dir.is_dir():
        sys.exit(f"ERROR: --reside-dir not found: {reside_dir}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else bundle_dir / "OTS" / "img"

    idx = index_reside_files(reside_dir)

    matched = 0
    missing = []
    ext_counter = Counter()
    for bn in sorted(ots_base_names):
        src = idx.get(bn)
        if src is None:
            missing.append(bn)
            continue
        dst = out_dir / f"{bn}{src.suffix.lower()}"
        link_or_copy(src, dst, args.mode)
        matched += 1
        ext_counter[src.suffix.lower()] += 1

    print(f"\nmatched:  {matched} / {len(ots_base_names)}")
    print(f"  extension breakdown: {dict(ext_counter)}")
    print(f"  output dir: {out_dir}  (mode={args.mode})")
    if missing:
        print(f"missing:  {len(missing)} base_names had no match in "
              f"{reside_dir}.  First 10: {missing[:10]}")
        miss_file = out_dir.with_suffix(".missing.txt")
        miss_file.write_text("\n".join(missing))
        print(f"  full missing list written to {miss_file}")


if __name__ == "__main__":
    main()
