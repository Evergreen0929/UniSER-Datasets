#!/usr/bin/env python3
"""Sample, download, QC, and visualize a subset of the HALO lens-flare dataset.

Pipeline:
  1. Read HALO manifest JSON, stratified-sample N records by flare-effect type.
  2. Download each record's images (flare/light/separate) from S3 in parallel.
  3. Generate small JPG thumbnails for fast browser viewing.
  4. Quality-check: missing files, image readability, resolution stats.
  5. Emit a paginated HTML gallery (one section per sample, lazy-loaded images).

Default output layout:
  /mnt/localssd/HALO/
  ├── selection.json     the chosen N records, with effect-type tag
  ├── raw/<s3_path>      full-resolution downloads (kept for later use)
  ├── thumbs/<id>__<field>.jpg
  ├── gallery/
  │   ├── index.html     table of contents
  │   ├── page_001.html  ...page_NNN.html (per_page samples each)
  └── stats.json         per-field success counters
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from PIL import Image
from tqdm import tqdm


# --- Configuration ----------------------------------------------------------
MANIFEST = Path(
    "/sensei-fs-3/users/xzhan/datasets/lens_flares/HALO/"
    "halo_render_dataset_01_reflective_no_lightsource.json"
)
S3_BUCKET = "adobe-lingzhi-p"
OUT_DIR = Path("/mnt/localssd/HALO")
THUMB_HEIGHT = 360  # px; thumbnails resized to this height, JPG quality 80
# Per-record fields to download (drop separate01 since only 60% of records have it
# — keeping a uniform 3-field schema makes downstream packing simpler).
FIELDS_TO_USE = ["flare_image_key", "light_image_key", "separate_image_key"]
FIELD_SHORT = {
    "flare_image_key":    "flare",
    "light_image_key":    "light",
    "separate_image_key": "separate",
}


# --- Sampling ---------------------------------------------------------------
def effect_prefix(path: str) -> str:
    """Extract Streak / Glare / Shimmer / Reflective from a HALO path."""
    for part in path.split("/"):
        m = re.match(r"^(Streak|Glare|Shimmer|Reflective)\d+", part)
        if m:
            return m.group(1)
    return "Unknown"


def category_of(path: str) -> str:
    """Far_Scene or Close_Scene."""
    parts = path.split("/")
    for p in parts:
        if p in ("Far_Scene", "Close_Scene"):
            return p
    return ""


def stratified_sample(records, target, rng):
    """Sample by effect type, preserving original proportions; rounds up to target."""
    by_type = defaultdict(list)
    for rec in records:
        et = effect_prefix(rec["flare_image_key"])
        by_type[et].append(rec)
    total = sum(len(v) for v in by_type.values())

    selected = []
    for et, recs in by_type.items():
        n = round(target * len(recs) / total)
        rng.shuffle(recs)
        picked = recs[:n]
        for r in picked:
            r["_effect"] = et
            r["_category"] = category_of(r["flare_image_key"])
        selected.extend(picked)
    rng.shuffle(selected)
    return selected[:target]


# --- Download + thumbnail ---------------------------------------------------
def sample_id_of(rec) -> str:
    """Stable ID per record: cam-level prefix of the flare path, sanitized."""
    parts = rec["flare_image_key"].split("/")
    # jingdongz-data/3D_Halo_Render/<category>/<scene>/<effect>/<scene_param>/<camera_X.png>
    if len(parts) >= 7:
        scene = parts[3]
        effect = parts[4]
        camera = parts[6].split("_")[0]
        return f"{scene}_{effect}_{camera}"
    return Path(rec["flare_image_key"]).stem


def download_one(s3, bucket, key, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(bucket, key, str(dest))
        return dest.stat().st_size
    except ClientError as e:
        return None
    except Exception:
        return None


def make_thumb(src_path: Path, dst_path: Path, height: int = THUMB_HEIGHT):
    try:
        img = Image.open(src_path).convert("RGB")
        ow, oh = img.size
        w = int(ow * height / oh)
        img_thumb = img.resize((w, height), Image.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img_thumb.save(dst_path, "JPEG", quality=82, optimize=True)
        return (ow, oh)
    except Exception:
        return None


def process_record(s3, bucket, rec, out_dir, idx):
    """Download + thumbnail one record. Returns a dict describing what happened."""
    raw_dir = out_dir / "raw"
    thumb_dir = out_dir / "thumbs"

    sid = sample_id_of(rec)
    result = {
        "idx": idx,
        "sample_id": sid,
        "effect": rec.get("_effect", "Unknown"),
        "category": rec.get("_category", ""),
        "fields": {},
    }
    for fkey in FIELDS_TO_USE:
        if fkey not in rec:
            result["fields"][FIELD_SHORT[fkey]] = {"status": "missing_in_manifest"}
            continue
        s3_key = rec[fkey]
        short = FIELD_SHORT[fkey]

        raw_path = raw_dir / s3_key
        size = download_one(s3, bucket, s3_key, raw_path)
        if size is None:
            result["fields"][short] = {"status": "download_failed", "s3_key": s3_key}
            continue

        thumb_path = thumb_dir / f"{idx:05d}_{sid}__{short}.jpg"
        full_wh = make_thumb(raw_path, thumb_path, THUMB_HEIGHT)
        if full_wh is None:
            result["fields"][short] = {"status": "thumb_failed", "s3_key": s3_key,
                                       "raw_size": size}
            continue

        result["fields"][short] = {
            "status": "ok",
            "raw_size": size,
            "raw_rel": str(raw_path.relative_to(out_dir)),
            "thumb_rel": str(thumb_path.relative_to(out_dir)),
            "orig_wh": full_wh,
        }
    return result


# --- Gallery rendering ------------------------------------------------------
PAGE_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 16px;
       background: #111; color: #eee; }
h1 { margin: 6px 0; }
.nav { position: sticky; top: 0; background: #111; padding: 10px 0;
       border-bottom: 1px solid #333; margin-bottom: 20px; z-index: 10; }
.nav a { color: #9cf; margin-right: 12px; text-decoration: none; }
.nav span { color: #888; }
section { margin-bottom: 24px; border-bottom: 1px solid #333;
          padding-bottom: 16px; }
h3 { font-size: 13px; margin: 6px 0; font-weight: 500; }
.tag { color: #ffd76b; font-weight: 600; }
.cat { color: #8c9; }
.sid { color: #888; font-weight: 400; }
.row { display: flex; gap: 14px; overflow-x: auto; padding-bottom: 6px; }
.cell { flex: 0 0 auto; text-align: center; }
.cell img { height: 360px; background: #000; border: 1px solid #444;
            display: block; }
.cap { font-size: 12px; color: #ccc; margin-top: 4px; }
.cap b { color: #fff; }
.miss { display: inline-block; padding: 10px; border: 1px dashed #555;
        color: #866; font-size: 12px; height: 340px; }
"""


def write_gallery(samples, gallery_dir: Path, per_page: int):
    gallery_dir.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    n_pages = max(1, (n + per_page - 1) // per_page)

    # Effect-type counts for index
    counts = Counter(s["effect"] for s in samples)
    cat_counts = Counter(s["category"] for s in samples)

    # --- index.html ---
    idx_lines = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>HALO Subset Gallery — {n} samples</title>",
        "<style>", PAGE_CSS,
        ".pages { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }",
        ".pages a { padding: 6px 12px; background: #222; border: 1px solid #444; "
        "text-decoration: none; }",
        "table { border-collapse: collapse; margin: 12px 0; }",
        "td { padding: 3px 12px 3px 0; }",
        "</style></head><body>",
        f"<h1>HALO Subset Gallery</h1>",
        f"<p>{n} samples across {n_pages} pages "
        f"(<a style='color:#9cf' href='../stats.json'>stats.json</a>).</p>",
        "<table>",
        "<tr><td><b>Effect</b></td>",
    ]
    for et in sorted(counts):
        idx_lines.append(f"<td>{et}: {counts[et]}</td>")
    idx_lines += [
        "</tr><tr><td><b>Category</b></td>",
    ]
    for ct in sorted(cat_counts):
        idx_lines.append(f"<td>{ct}: {cat_counts[ct]}</td>")
    idx_lines.append("</tr></table>")
    idx_lines.append("<div class='pages'>")
    for i in range(n_pages):
        idx_lines.append(f"<a href='page_{i+1:03d}.html'>Page {i+1}</a>")
    idx_lines.append("</div></body></html>")
    (gallery_dir / "index.html").write_text("\n".join(idx_lines))

    # --- per-page ---
    for pi in range(n_pages):
        sub = samples[pi * per_page: (pi + 1) * per_page]
        nav = ["<div class='nav'><a href='index.html'>← All pages</a>"]
        if pi > 0:
            nav.append(f"<a href='page_{pi:03d}.html'>← Prev</a>")
        if pi < n_pages - 1:
            nav.append(f"<a href='page_{pi+2:03d}.html'>Next →</a>")
        nav.append(f"<span>Page {pi+1} / {n_pages} · "
                   f"samples {pi*per_page+1}–{min((pi+1)*per_page, n)}</span></div>")

        lines = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
                 f"<title>HALO Page {pi+1}/{n_pages}</title>",
                 "<style>", PAGE_CSS, "</style></head><body>"]
        lines.extend(nav)

        for s in sub:
            lines.append(
                f"<section><h3>"
                f"<span class='tag'>{s['effect']}</span> · "
                f"<span class='cat'>{s['category']}</span> · "
                f"<span class='sid'>{s['sample_id']} (#{s['idx']:05d})</span>"
                f"</h3><div class='row'>"
            )
            for short in ("flare", "light", "separate"):
                f = s["fields"].get(short, {})
                if f.get("status") == "ok":
                    rel = "../" + f["thumb_rel"]
                    ow, oh = f["orig_wh"]
                    lines.append(
                        f"<div class='cell'><img src='{rel}' loading='lazy'>"
                        f"<div class='cap'><b>{short}</b> "
                        f"<span style='color:#888'>{ow}×{oh}</span></div></div>"
                    )
                else:
                    lines.append(
                        f"<div class='cell'><div class='miss'><b>{short}</b><br>"
                        f"<small>{f.get('status', 'missing')}</small></div></div>"
                    )
            lines.append("</div></section>")
        lines.append("</body></html>")
        (gallery_dir / f"page_{pi+1:03d}.html").write_text("\n".join(lines))

    return n_pages


# --- Main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total", type=int, default=5000,
                    help="Total samples to draw (default 5000).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total to a small N for testing (overrides --total).")
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel S3 download workers.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--per-page", type=int, default=50,
                    help="Samples per HTML gallery page.")
    ap.add_argument("--manifest", default=str(MANIFEST),
                    help="HALO manifest JSON path.")
    ap.add_argument("--bucket", default=S3_BUCKET, help="S3 bucket name.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read manifest, stratified sample
    print(f"Reading manifest: {args.manifest}")
    with open(args.manifest) as f:
        records = json.load(f)
    print(f"  total records in manifest: {len(records)}")

    rng = random.Random(args.seed)
    n_target = args.limit if args.limit else args.total
    selected = stratified_sample(records, target=n_target, rng=rng)
    print(f"\nStratified sample (target={n_target}, actual={len(selected)}):")
    for et, cnt in Counter(s["_effect"] for s in selected).most_common():
        print(f"  {et:<11} {cnt}")
    for ct, cnt in Counter(s["_category"] for s in selected).most_common():
        print(f"  {ct:<11} {cnt}")

    (out_dir / "selection.json").write_text(json.dumps(selected, indent=2))
    print(f"\nWrote selection.json")

    # 2-3. Parallel download + thumbnail
    s3 = boto3.client("s3")

    def worker(idx_rec):
        idx, rec = idx_rec
        return process_record(s3, args.bucket, rec, out_dir, idx)

    print(f"\nDownloading + thumbnailing ({args.workers} parallel workers)...")
    t0 = time.time()
    results = []
    pbar = tqdm(total=len(selected), unit="rec")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(worker, enumerate(selected)):
            results.append(r)
            pbar.update(1)
            ok = sum(1 for f in r["fields"].values() if f.get("status") == "ok")
            pbar.set_postfix(last_ok=f"{ok}/3")
    pbar.close()
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s ({len(selected) / max(elapsed, 1):.1f} rec/s)")

    # 4. Quality check
    stats = {
        "total_records": len(results),
        "by_effect": dict(Counter(r["effect"] for r in results)),
        "by_category": dict(Counter(r["category"] for r in results)),
        "field_status": defaultdict(Counter),
        "fully_ok": 0,
        "total_raw_bytes": 0,
        "resolution_distribution": Counter(),
    }
    for r in results:
        all_ok = True
        for fname, f in r["fields"].items():
            stats["field_status"][fname][f.get("status", "missing")] += 1
            if f.get("status") != "ok":
                all_ok = False
            else:
                stats["total_raw_bytes"] += f.get("raw_size", 0)
                wh = f.get("orig_wh")
                if wh:
                    stats["resolution_distribution"][f"{wh[0]}x{wh[1]}"] += 1
        if all_ok:
            stats["fully_ok"] += 1
    stats["field_status"] = {k: dict(v) for k, v in stats["field_status"].items()}
    stats["resolution_distribution"] = dict(stats["resolution_distribution"].most_common(8))

    print(f"\n--- Quality Check ---")
    print(f"  total records:    {stats['total_records']}")
    print(f"  fully OK:         {stats['fully_ok']}")
    print(f"  total raw bytes:  {stats['total_raw_bytes']/1e9:.2f} GB")
    print(f"  resolutions (top): {stats['resolution_distribution']}")
    for fname, ss in stats["field_status"].items():
        print(f"  field '{fname}': {ss}")

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    # 5. Gallery
    print(f"\nGenerating HTML gallery ({args.per_page} samples/page)...")
    n_pages = write_gallery(results, out_dir / "gallery", per_page=args.per_page)
    print(f"  wrote {n_pages} pages to {out_dir / 'gallery'}")

    print(f"\n✓ Done. Serve with:")
    print(f"    python3 -m http.server --directory {out_dir} 8000")
    print(f"    Open http://<host>:8000/gallery/index.html")


if __name__ == "__main__":
    main()
