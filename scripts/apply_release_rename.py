"""Minimum-change rename of personal-name + brand tokens in HALO finalist paths.

Goals:
  1. Replace personal-name tokens (3D-character asset slugs that contain real
     romanized names) with generic `char01..char11` codes.
  2. Replace brand token `MouseThermaltakeBlack` with brand-free `MouseBlack`.
  3. Apply ONLY token-level substring substitution to the 4 *_image_key fields
     and the `_sample_id` field — minimum change.
  4. Assert no path collisions after substitution.

The mapping is recorded in `release_rename_map.json` so it can be reapplied
later to rename files when uploading to Hugging Face, and so users can audit
the substitution after the fact.

Usage:
    python scripts/apply_release_rename.py \\
        --finalist /path/to/finalist.json \\
        --map-out  /path/to/release_rename_map.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path


NAME_MAP = {
    # Personal-name tokens. Ordering matters: replace the LONGEST keys first
    # (handled below via SORTED_KEYS) so e.g. "HuangXinYin" replaces cleanly
    # before any shorter prefix could match.
    'HongChengHong':  'char01',
    'HuangKangHua':   'char02',
    'HuangXinYin':    'char03',
    'LiuLiSha':       'char04',
    'LiuXiaoXiao':    'char05',
    'HuangD':         'char06',
    'LiuCJ':          'char07',
    'Jennt':          'char08',
    'Lilya':          'char09',
    'HuMS':           'char10',
    'HuSQ':           'char11',
    # Brand token
    'MouseThermaltakeBlack': 'MouseBlack',
}
SORTED_KEYS = sorted(NAME_MAP, key=len, reverse=True)

FIELDS = ['light_image_key', 'flare_image_key', 'separate_image_key',
          'separate01_image_key', '_sample_id']


def rename(path: str) -> str:
    """Apply token-level substitutions to a path string."""
    for old in SORTED_KEYS:
        if old in path:
            path = path.replace(old, NAME_MAP[old])
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--finalist', required=True, type=Path,
                    help='Path to finalist.json (will be modified in place).')
    ap.add_argument('--map-out', required=True, type=Path,
                    help='Where to write release_rename_map.json.')
    args = ap.parse_args()

    sel = json.loads(args.finalist.read_text())
    print(f'records: {len(sel)}')

    changed_records = 0
    field_change_counts = Counter()
    for r in sel:
        rec_changed = False
        for f in FIELDS:
            old = r.get(f)
            if not old:
                continue
            new = rename(old)
            if new != old:
                r[f] = new
                field_change_counts[f] += 1
                rec_changed = True
        if rec_changed:
            changed_records += 1
    print(f'changed records: {changed_records}')
    print('per-field changes:')
    for f, n in field_change_counts.most_common():
        print(f'  {f}: {n}')

    # Collision check: every light_image_key must remain unique across records.
    seen = {}
    dup = 0
    for r in sel:
        k = r['light_image_key']
        if k in seen:
            print(f'DUP: idx={r["_orig_idx"]} vs {seen[k]} path={k}')
            dup += 1
        seen[k] = r['_orig_idx']
    if dup == 0:
        print('OK: all light_image_key paths still unique')
    else:
        raise SystemExit(f'ABORT: {dup} duplicate paths after rename')

    # Also verify no record still contains any of the banned tokens.
    leftover = 0
    for r in sel:
        for f in FIELDS:
            v = r.get(f) or ''
            for tok in NAME_MAP:
                if tok in v:
                    leftover += 1
                    print(f'LEFTOVER: idx={r["_orig_idx"]} field={f} token={tok}')
    if leftover == 0:
        print('OK: no banned token remains in finalist paths')
    else:
        raise SystemExit(f'ABORT: {leftover} leftover banned tokens')

    args.finalist.write_text(json.dumps(sel, indent=2))
    print(f'\nwrote {args.finalist}')

    args.map_out.write_text(json.dumps({
        'description': 'token-level substring substitution applied to finalist paths',
        'old_to_new': NAME_MAP,
        'new_to_old': {v: k for k, v in NAME_MAP.items()},
        'sorted_apply_order_longest_first': SORTED_KEYS,
    }, indent=2))
    print(f'wrote {args.map_out}')


if __name__ == '__main__':
    main()
