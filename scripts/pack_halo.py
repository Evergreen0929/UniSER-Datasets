"""Pack HALO v2 finalist into WebDataset .tar shards for HF upload.

Per-sample layout inside a shard (WebDataset convention):
    halo/{base_name}.gt.png         clean scene (= light)
    halo/{base_name}.flare.png      scene + flare effect
    halo/{base_name}.separate.png   flare-only on transparent background
    halo/{base_name}.json           per-sample metadata

`base_name` is the sanitized `_sample_id` (e.g. "Scene003_Glare001_camera01").

Reads source PNGs from --raw-root. finalist.json's *_image_key fields are already
sanitized (char01..char11, MouseBlack) but on-disk filenames still have the
original tokens — we reverse the sanitization via --rename-map to locate files.

Shards are mixed across scenes (deterministic seed) so each shard is a balanced
mini-batch. Target ~2 GB per shard → ~82 shards total.
"""
import argparse
import io
import json
import os
import random
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path


def build_reverse_replacer(rename_map_path: Path):
    """Build a function that maps a sanitized path string back to its on-disk path."""
    m = json.loads(rename_map_path.read_text())
    new_to_old = m['new_to_old']
    # Sort by length descending — defensive, even though current tokens don't overlap.
    ordered = sorted(new_to_old.items(), key=lambda kv: -len(kv[0]))

    def reverse(path: str) -> str:
        for new, old in ordered:
            if new in path:
                path = path.replace(new, old)
        return path
    return reverse


def fetch_blobs(rec, raw_root: Path, reverse):
    """Read the 3 PNGs from disk + build per-sample metadata. Returns
    (rec, [(tar_name, bytes), ...]) or (rec, None) on error."""
    base = rec['_sample_id']
    key_prefix = f'halo/{base}'

    fields = [
        ('light_image_key', f'{key_prefix}.gt.png'),
        ('flare_image_key', f'{key_prefix}.flare.png'),
        ('separate_image_key', f'{key_prefix}.separate.png'),
    ]
    blobs = []
    try:
        for src_field, tar_name in fields:
            sanitized_path = rec[src_field]
            disk_relpath = reverse(sanitized_path)
            disk_path = raw_root / disk_relpath
            blobs.append((tar_name, disk_path.read_bytes()))
        meta = {
            'scene':       rec['_scene'],
            'effect_type': rec['_effect_type'],
            'effect_id':   rec['_effect_id'],
            'sample_id':   rec['_sample_id'],
            'orig_idx':    rec['_orig_idx'],
            'final_idx':   rec['_final_idx'],
            'light':       f'{base}.gt.png',
            'flare':       f'{base}.flare.png',
            'separate':    f'{base}.separate.png',
        }
        blobs.append((f'{key_prefix}.json',
                      json.dumps(meta, ensure_ascii=False).encode('utf-8')))
        return rec, blobs
    except Exception as e:
        print(f'\n  !! fetch error: {base}: {e}', file=sys.stderr)
        return rec, None


def write_blobs(tar, blobs, now):
    written = 0
    for name, data in blobs:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = now
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
        written += 512 + ((len(data) + 511) // 512) * 512
    return written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--finalist', required=True,
                    help='Path to finalist.json (sanitized record list).')
    ap.add_argument('--rename-map', required=True,
                    help='Path to release_rename_map.json (new_to_old mapping).')
    ap.add_argument('--raw-root', required=True,
                    help='Root of on-disk raw PNGs (e.g. /mnt/localssd/HALO_v2/raw).')
    ap.add_argument('--out-dir', required=True,
                    help='Directory to write shard-NNNNN.tar files into.')
    ap.add_argument('--shard-bytes', type=int, default=2 * 1024**3,
                    help='Target shard size in bytes (default: 2 GiB).')
    ap.add_argument('--workers', type=int, default=16,
                    help='Parallel disk-read workers.')
    ap.add_argument('--seed', type=int, default=42,
                    help='Shuffle seed for cross-scene mixing.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Cap total samples (for dry-run).')
    args = ap.parse_args()

    finalist = json.loads(Path(args.finalist).read_text())
    print(f'finalist: {len(finalist)} samples')

    reverse = build_reverse_replacer(Path(args.rename_map))
    raw_root = Path(args.raw_root)

    rng = random.Random(args.seed)
    records = list(finalist)
    rng.shuffle(records)
    if args.limit:
        records = records[: args.limit]
        print(f'  limited to {len(records)} (dry-run)')

    # Per-scene stats
    counts = Counter(r['_scene'] for r in records)
    print(f'\nscenes in this run: {len(counts)}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = int(time.time())

    shard_idx = 0
    shard_path = out_dir / f'shard-{shard_idx:05d}.tar'
    tar = tarfile.open(shard_path, 'w')
    cur_bytes = 0
    samples_in_shard = 0
    samples_done = 0
    samples_failed = 0
    t0 = time.time()
    last_print = 0
    shard_started = time.time()
    per_shard_log = []

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rec, blobs in ex.map(
                lambda r: fetch_blobs(r, raw_root, reverse), records
            ):
                if blobs is None:
                    samples_failed += 1
                    continue
                cur_bytes += write_blobs(tar, blobs, now)
                samples_in_shard += 1
                samples_done += 1

                now_t = time.time()
                if now_t - last_print > 10 or samples_done == len(records):
                    rate = samples_done / max(now_t - t0, 0.01)
                    eta_min = (len(records) - samples_done) / max(rate, 0.01) / 60
                    print(f'  {samples_done}/{len(records)} '
                          f'shard={shard_idx} cur_gb={cur_bytes/1e9:.2f} '
                          f'rate={rate:.1f}/s eta={eta_min:.1f}min '
                          f'failed={samples_failed}')
                    last_print = now_t

                if cur_bytes >= args.shard_bytes:
                    tar.close()
                    elapsed = time.time() - shard_started
                    print(f'  -> closed {shard_path.name} '
                          f'({cur_bytes/1e9:.2f} GB, {samples_in_shard} samples, '
                          f'{elapsed:.0f}s)')
                    per_shard_log.append({
                        'shard': shard_path.name,
                        'bytes': cur_bytes,
                        'samples': samples_in_shard,
                    })
                    shard_idx += 1
                    shard_path = out_dir / f'shard-{shard_idx:05d}.tar'
                    tar = tarfile.open(shard_path, 'w')
                    cur_bytes = 0
                    samples_in_shard = 0
                    shard_started = time.time()
    finally:
        tar.close()
        # Drop last shard if empty
        if samples_in_shard == 0 and shard_path.exists():
            shard_path.unlink()
            shard_idx -= 1
        elif samples_in_shard > 0:
            per_shard_log.append({
                'shard': shard_path.name,
                'bytes': cur_bytes,
                'samples': samples_in_shard,
            })

    total_bytes = sum(s['bytes'] for s in per_shard_log)
    print(f'\ndone: {samples_done} samples ({samples_failed} failed) '
          f'-> {len(per_shard_log)} shards, {total_bytes/1e9:.1f} GB total')

    (out_dir / 'shard_index.json').write_text(json.dumps({
        'total_samples': samples_done,
        'total_failed':  samples_failed,
        'total_bytes':   total_bytes,
        'shards':        per_shard_log,
    }, indent=2))
    print(f'wrote {out_dir}/shard_index.json')


if __name__ == '__main__':
    main()
