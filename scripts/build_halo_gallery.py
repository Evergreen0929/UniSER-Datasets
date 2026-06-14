#!/usr/bin/env python3
"""Fast HALO gallery with sprite-atlas overview + per-sample detail pages.

Why an atlas: the previous gallery loaded 100 tile images per overview page,
each triggering its own HTTPS request through the VS Code DevTunnel relay.
Combining all tiles into a single JPG atlas drops that to ONE request per
page, removing 99% of the latency.

This script:
  1. Reads selection.json, drops records flagged `_excluded`, and normalizes
     each remaining record's `_effect_type` (Streak / Glare / Shimmer /
     Reflective) and `_effect_id` (e.g. `Streak023`) — even when the
     upstream folder naming is irregular.
  2. Writes valid_selection.json (the 4998-record final usable list).
  3. Builds one sprite atlas per overview page (10×10 tile grid).
  4. Renders overview HTML that references the atlas via CSS
     background-position.
  5. Renders per-sample detail HTML (3 thumbs, click → full 4K PNG),
     with `<link rel="prefetch">` for the next/prev sample's thumbs so
     keyboard navigation feels instant.

Output:
  /mnt/localssd/HALO/
  ├── valid_selection.json
  ├── gallery/
  │   ├── index.html              entry: stats + page nav
  │   ├── atlas_001.jpg ... 050   one sprite per overview page
  │   ├── overview_001.html ...
  │   └── detail/<idx>.html
"""
import argparse
import json
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


HALO = Path("/mnt/localssd/HALO")

TILE_W = 192          # px per tile in the atlas
TILE_H = 108          # 16:9 from 4K
COLS = 10             # grid columns per atlas
PER_PAGE = 100        # COLS * 10 rows
DETAIL_W = 640        # detail-page nominal image display width

# Medium-resolution JPG version of every raw PNG. The detail page serves these
# instead of 360-px thumbs (better fidelity) or 4K PNGs (too slow). Raw PNGs
# are only fetched when the user clicks "open full-res 4K".
MEDIUM_HEIGHT = 720   # 1280x720 keeps each JPG around 150-250 KB
MEDIUM_QUALITY = 82


# --- Normalization ----------------------------------------------------------
EFFECT_RE = re.compile(r"(Streak|Glare|Shimmer|Reflective)(\d*)")


def normalize_effect(path: str):
    """Return (effect_type, effect_id) for a HALO image path.

    Tolerates the two naming irregularities found in QC:
      - Scene009/Reflective/...        (folder has no number suffix)
      - Scene013/01/...Streak023...    (folder name is `01`, effect lives
                                        in the scene-param string)
    """
    parts = path.split("/")
    # 1) Prefer the dedicated effect folder (parts[4]).
    if len(parts) >= 5:
        m = EFFECT_RE.fullmatch(parts[4])
        if m:
            num = m.group(2)
            return m.group(1), (m.group(1) + num) if num else m.group(1)
    # 2) Fall back to scanning the scene-param string for an effect-id.
    if len(parts) >= 6:
        m = EFFECT_RE.search(parts[5])
        if m:
            num = m.group(2)
            return m.group(1), (m.group(1) + num) if num else m.group(1)
    # 3) Last resort: any segment that contains the keyword.
    m = EFFECT_RE.search(path)
    if m:
        num = m.group(2)
        return m.group(1), (m.group(1) + num) if num else m.group(1)
    return "Unknown", "Unknown"


def category_of(path: str) -> str:
    for p in path.split("/"):
        if p in ("Far_Scene", "Close_Scene"):
            return p
    return ""


def sample_id_of(rec) -> str:
    parts = rec["flare_image_key"].split("/")
    if len(parts) >= 7:
        return f"{parts[3]}_{parts[4]}_{parts[6].split('_')[0]}"
    return Path(rec["flare_image_key"]).stem


# --- Build the filtered & normalized list -----------------------------------
def build_valid_selection(halo_dir: Path):
    selection = json.loads((halo_dir / "selection.json").read_text())
    valid = []
    excluded = []
    for orig_idx, rec in enumerate(selection):
        if rec.get("_excluded"):
            excluded.append((orig_idx, rec.get("_excluded")))
            continue
        rec = dict(rec)  # copy so we don't mutate original
        et, eid = normalize_effect(rec["flare_image_key"])
        rec["_orig_idx"] = orig_idx
        rec["_effect_type"] = et
        rec["_effect_id"] = eid
        rec["_category"] = category_of(rec["flare_image_key"])
        rec["_sample_id"] = sample_id_of(rec)
        valid.append(rec)
    print(f"  valid: {len(valid)}, excluded: {len(excluded)}")
    for i, reason in excluded:
        print(f"    excluded #{i:05d}: {reason}")
    out = halo_dir / "valid_selection.json"
    out.write_text(json.dumps(valid, indent=2))
    print(f"  wrote {out}")
    return valid


# --- Medium-resolution JPG generation --------------------------------------
def make_medium(raw_path: Path, dst_path: Path, height: int = MEDIUM_HEIGHT) -> bool:
    """Convert a 4K PNG to a height-px JPG. Returns True on success."""
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return True
    try:
        # Try PIL first.
        img = Image.open(raw_path).convert("RGB")
        w = int(img.width * height / img.height)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.resize((w, height), Image.LANCZOS).save(
            dst_path, "JPEG", quality=MEDIUM_QUALITY, optimize=True)
        return True
    except Exception:
        # Fallback to OpenCV (handles some PNGs PIL chokes on, e.g. CRC errors).
        try:
            import cv2
            arr = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                return False
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            h_in, w_in = arr.shape[:2]
            w_out = int(w_in * height / h_in)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(dst_path),
                cv2.resize(arr, (w_out, height), interpolation=cv2.INTER_LANCZOS4),
                [cv2.IMWRITE_JPEG_QUALITY, MEDIUM_QUALITY],
            )
            return True
        except Exception:
            return False


def generate_mediums(valid, halo_dir: Path, workers: int):
    """For every raw PNG referenced by a valid record, ensure a medium/<key>.jpg exists."""
    raw_dir = halo_dir / "raw"
    med_dir = halo_dir / "medium"
    med_dir.mkdir(exist_ok=True)

    jobs = []
    for rec in valid:
        for fkey in ("flare_image_key", "light_image_key", "separate_image_key"):
            if fkey not in rec:
                continue
            s3_key = rec[fkey]
            raw = raw_dir / s3_key
            # Replace .png suffix with .jpg for the medium copy.
            rel_jpg = Path(s3_key).with_suffix(".jpg")
            dst = med_dir / rel_jpg
            if dst.exists() and dst.stat().st_size > 0:
                continue
            if not raw.exists():
                continue
            jobs.append((raw, dst))
    if not jobs:
        print(f"  medium JPGs: all present, skipping")
        return
    print(f"  generating {len(jobs)} medium JPGs ({MEDIUM_HEIGHT}px height, "
          f"quality {MEDIUM_QUALITY})...")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(lambda j: make_medium(*j), jobs),
                      total=len(jobs), desc="medium"):
            if r: ok += 1
            else: fail += 1
    print(f"  done: {ok} ok, {fail} failed")


# --- Atlas generation -------------------------------------------------------
def build_one_atlas(page_idx, page_records, halo_dir: Path):
    """Compose a 10×10 sprite atlas of tile thumbnails for one overview page."""
    rows = (len(page_records) + COLS - 1) // COLS
    atlas = Image.new("RGB", (COLS * TILE_W, rows * TILE_H), "#0a0a0a")
    for i, rec in enumerate(page_records):
        sid = rec["_sample_id"]
        orig_idx = rec["_orig_idx"]
        tiny = halo_dir / "tiny" / f"{orig_idx:05d}_{sid}__flare.jpg"
        if not tiny.exists():
            continue
        try:
            img = Image.open(tiny).convert("RGB").resize((TILE_W, TILE_H), Image.LANCZOS)
        except Exception:
            continue
        col = i % COLS
        row = i // COLS
        atlas.paste(img, (col * TILE_W, row * TILE_H))
    out = halo_dir / "gallery" / f"atlas_{page_idx:03d}.jpg"
    atlas.save(out, "JPEG", quality=78, optimize=True)
    return out, atlas.size


def build_atlases(valid, halo_dir: Path, workers=8):
    n_pages = max(1, (len(valid) + PER_PAGE - 1) // PER_PAGE)
    pages = [valid[p * PER_PAGE: (p + 1) * PER_PAGE] for p in range(n_pages)]
    print(f"  building {n_pages} atlases ({PER_PAGE} tiles each, "
          f"{COLS * TILE_W}×{10 * TILE_H} px max)...")
    sizes = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for out, size in tqdm(
            ex.map(lambda x: build_one_atlas(x[0] + 1, x[1], halo_dir), enumerate(pages)),
            total=n_pages, desc="atlas",
        ):
            sizes.append(out.stat().st_size)
    total_mb = sum(sizes) / 1e6
    print(f"  done, {total_mb:.1f} MB total ({total_mb / n_pages:.1f} MB/page avg)")
    return n_pages


# --- CSS --------------------------------------------------------------------
BASE_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, sans-serif; margin: 14px;
       background: #111; color: #eee; }
a { color: #9cf; text-decoration: none; }
a:hover { color: #fff; }
h1 { margin: 8px 0 6px; }
.nav { position: sticky; top: 0; background: rgba(17,17,17,.95);
       padding: 10px 0; border-bottom: 1px solid #333;
       margin-bottom: 14px; z-index: 10; backdrop-filter: blur(4px); }
.nav a { margin-right: 12px; }
.nav span { color: #888; }
.tag { color: #ffd76b; font-weight: 600; }
.cat { color: #8c9; }
.muted { color: #888; }
"""

OVERVIEW_CSS_TEMPLATE = BASE_CSS + f"""
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, {TILE_W}px);
        gap: 10px; }}
.tile {{ position: relative; width: {TILE_W}px; padding-bottom: 22px;
        border: 1px solid #333; border-radius: 4px; background: #1a1a1a;
        transition: transform .12s, border-color .12s; }}
.tile:hover {{ transform: scale(1.05); border-color: #8c9; }}
.tile-img {{ display: block; width: {TILE_W}px; height: {TILE_H}px;
            background-repeat: no-repeat; border-radius: 3px 3px 0 0;
            background-color: #000; }}
.tile-cap {{ position: absolute; bottom: 4px; left: 0; right: 0;
            font-size: 11px; color: #ccc; text-align: center;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            padding: 0 4px; }}
.tile.miss {{ height: {TILE_H + 22}px; color: #866; font-size: 12px;
             display: flex; align-items: center; justify-content: center; }}
.pages {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }}
.pages a {{ padding: 6px 12px; background: #222; border: 1px solid #444;
           border-radius: 3px; }}
table.stats td {{ padding: 3px 18px 3px 0; }}
"""

DETAIL_CSS = BASE_CSS + f"""
.meta {{ color: #aaa; margin: 8px 0 18px 0; }}
.row {{ display: flex; gap: 18px; flex-wrap: wrap; }}
.cell {{ flex: 1 1 {DETAIL_W}px; max-width: {DETAIL_W + 80}px; }}
.cell h3 {{ font-size: 14px; margin: 6px 0; color: #fff; }}
.cell a {{ display: block; }}
.cell img {{ width: 100%; height: auto; background: #000;
            border: 1px solid #444; cursor: zoom-in; display: block; }}
.cell .info {{ font-size: 12px; color: #888; margin-top: 4px;
              word-break: break-all; }}
.cell .info a {{ color: #9cf; }}
"""


# --- HTML emission ----------------------------------------------------------
def write_index(valid, gallery_dir: Path, n_pages: int):
    n = len(valid)
    counts = Counter(r["_effect_type"] for r in valid)
    cat_counts = Counter(r["_category"] for r in valid)

    lines = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>HALO Subset Gallery — {n} samples</title>",
        "<style>", OVERVIEW_CSS_TEMPLATE, "</style></head><body>",
        f"<h1>HALO Subset Gallery</h1>",
        f"<p class='muted'>{n} valid samples in {n_pages} pages. "
        f"Each overview page loads 1 sprite atlas image "
        f"({COLS}×{10} tile grid).</p>",
        "<table class='stats'><tr>",
    ]
    for et in sorted(counts):
        lines.append(f"<td><span class='tag'>{et}</span>: {counts[et]}</td>")
    lines.append("</tr><tr>")
    for ct in sorted(cat_counts):
        lines.append(f"<td><span class='cat'>{ct}</span>: {cat_counts[ct]}</td>")
    lines.append("</tr></table>")
    lines.append("<div class='pages'>")
    for i in range(n_pages):
        lines.append(f"<a href='overview_{i+1:03d}.html'>Page {i+1}</a>")
    lines.append("</div></body></html>")
    (gallery_dir / "index.html").write_text("\n".join(lines))


def write_overview_page(page_idx, page_records, n_pages, gallery_dir: Path):
    nav = ["<div class='nav'>",
           "<a href='index.html'>← All pages</a>"]
    if page_idx > 1:
        nav.append(f"<a href='overview_{page_idx-1:03d}.html'>← Prev</a>")
    if page_idx < n_pages:
        nav.append(f"<a href='overview_{page_idx+1:03d}.html'>Next →</a>")
    nav.append(
        f"<span>Page {page_idx} / {n_pages} · "
        f"{len(page_records)} samples</span></div>"
    )

    atlas_url = f"atlas_{page_idx:03d}.jpg"
    prefetch_next = (f"<link rel='prefetch' href='atlas_{page_idx+1:03d}.jpg'>"
                     if page_idx < n_pages else "")
    lines = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>HALO Overview {page_idx}/{n_pages}</title>",
        f"<link rel='preload' as='image' href='{atlas_url}'>",
        prefetch_next,
        "<style>", OVERVIEW_CSS_TEMPLATE, "</style></head><body>",
    ]
    lines.extend(nav)
    lines.append("<div class='grid'>")

    for i, rec in enumerate(page_records):
        col = i % COLS
        row = i // COLS
        bg_pos = f"-{col * TILE_W}px -{row * TILE_H}px"
        detail_url = f"detail/{rec['_orig_idx']:05d}.html"
        lines.append(
            f"<a class='tile' href='{detail_url}'>"
            f"<div class='tile-img' style=\""
            f"background-image:url('{atlas_url}');"
            f"background-position:{bg_pos};\"></div>"
            f"<div class='tile-cap'>"
            f"<span class='tag'>{rec['_effect_type']}</span> · "
            f"<span class='muted'>#{rec['_orig_idx']:05d}</span>"
            f"</div></a>"
        )
    lines.append("</div></body></html>")
    (gallery_dir / f"overview_{page_idx:03d}.html").write_text("\n".join(lines))


def write_detail_pages(valid, gallery_dir: Path):
    detail_dir = gallery_dir / "detail"
    detail_dir.mkdir(exist_ok=True)
    n = len(valid)

    for pos, rec in enumerate(valid):
        prev_rec = valid[pos - 1] if pos > 0 else None
        next_rec = valid[pos + 1] if pos < n - 1 else None
        page_idx = pos // PER_PAGE + 1
        orig_idx = rec["_orig_idx"]
        sid = rec["_sample_id"]

        nav = ["<div class='nav'>",
               f"<a href='../overview_{page_idx:03d}.html'>"
               f"← Grid (page {page_idx})</a>"]
        if prev_rec is not None:
            nav.append(f"<a href='{prev_rec['_orig_idx']:05d}.html'>← Prev</a>")
        if next_rec is not None:
            nav.append(f"<a href='{next_rec['_orig_idx']:05d}.html'>Next →</a>")
        nav.append(f"<span>Sample {pos+1} / {n}</span></div>")

        # Prefetch the next sample's mediums for snappy Next navigation.
        prefetch = []
        if next_rec is not None:
            for fkey in ("flare_image_key", "light_image_key", "separate_image_key"):
                if fkey in next_rec:
                    rel = f"../../medium/{Path(next_rec[fkey]).with_suffix('.jpg')}"
                    prefetch.append(f"<link rel='prefetch' href='{rel}'>")

        lines = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
                 f"<title>HALO #{orig_idx:05d}</title>"]
        lines.extend(prefetch)
        lines.append("<style>")
        lines.append(DETAIL_CSS)
        lines.append("</style></head><body>")
        lines.extend(nav)
        lines.append(
            f"<h1>Sample #{orig_idx:05d}</h1>"
            f"<div class='meta'>"
            f"<span class='tag'>{rec['_effect_type']}</span> "
            f"<span class='muted'>({rec['_effect_id']})</span> · "
            f"<span class='cat'>{rec['_category']}</span> · "
            f"<span class='muted'>{sid}</span>"
            f"</div>"
            "<div class='row'>"
        )
        for fkey, short in [("flare_image_key", "flare"),
                            ("light_image_key", "light"),
                            ("separate_image_key", "separate")]:
            if fkey not in rec:
                lines.append(f"<div class='cell'><h3>{short}</h3>"
                             f"<div class='info muted'>missing in manifest</div></div>")
                continue
            s3_key = rec[fkey]
            medium_rel = f"../../medium/{Path(s3_key).with_suffix('.jpg')}"
            raw_rel = f"../../raw/{s3_key}"
            lines.append(
                f"<div class='cell'><h3>{short}</h3>"
                f"<a href='{raw_rel}' target='_blank' "
                f"title='Open full-res 4K PNG in new tab'>"
                f"<img src='{medium_rel}' loading='eager' alt=''></a>"
                f"<div class='info'>"
                f"<a href='{raw_rel}' target='_blank'>open full-res 4K PNG →</a><br>"
                f"<span class='muted'>{s3_key}</span></div></div>"
            )
        lines.append("</div></body></html>")
        (detail_dir / f"{orig_idx:05d}.html").write_text("\n".join(lines))


# --- Main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--halo-dir", default=str(HALO))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    halo_dir = Path(args.halo_dir)
    gallery_dir = halo_dir / "gallery"

    print(f"Filtering + normalizing selection...")
    valid = build_valid_selection(halo_dir)

    print(f"\nGenerating medium-resolution JPG backups...")
    generate_mediums(valid, halo_dir, workers=args.workers)

    if gallery_dir.exists():
        shutil.rmtree(gallery_dir)
    gallery_dir.mkdir()

    print(f"\nGenerating atlases...")
    n_pages = build_atlases(valid, halo_dir, workers=args.workers)

    print(f"\nWriting HTML pages...")
    pages = [valid[p * PER_PAGE: (p + 1) * PER_PAGE] for p in range(n_pages)]
    for pi, page_records in enumerate(pages, start=1):
        write_overview_page(pi, page_records, n_pages, gallery_dir)
    write_index(valid, gallery_dir, n_pages)
    write_detail_pages(valid, gallery_dir)
    print(f"  {n_pages} overview pages, {len(valid)} detail pages")

    print(f"\n✓ Done. Reload https://7g9p3dbh-9000.usw2.devtunnels.ms/gallery/")


if __name__ == "__main__":
    main()
