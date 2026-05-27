# Dataset Structure

## Bundle layout

```
shards/
├── shard-00000.tar      ~2 GB
├── shard-00001.tar
├── ...
└── shard-NNNNN.tar
```

Each `.tar` is a [WebDataset](https://github.com/webdataset/webdataset)-format shard. Samples are recovered by grouping entries that share the same prefix before the first `.` in the filename's basename.

## Per-sample fields

```
<source>/<base_name>.gt.<ext>            clean RGB image (absent for OTS)
<source>/<base_name>.haze_NNN.png        synthesized haze variant
<source>/<base_name>.haze_NNN.txt        descriptive tag (e.g. out_fog_120)
<source>/<base_name>.json                metadata: source, base_name, upstream info
```

`<source>` is one of `hazespace`, `its`, `ots`, `wsrd`, `real_flare`, `istd`.

## Number of haze variants per sample

| Source | Unique GT images | Haze variants per GT |
|---|---:|---:|
| HAZESPACE | 66,133 | 24 |
| ITS | 11,000 | 19 |
| OTS | 2,061 (no GT shipped) | 37 |
| WSRD | 1,000 | 24 |
| REAL_FLARE | 600 | 24 |
| ISTD | 135 | 24 |

## Duplicate samples in WSRD / REAL_FLARE / ISTD

For these three subsets, the upstream JSON manifests intentionally repeat each base image multiple times to express a per-source sampling-weight choice carried over from the original training pipeline:

| Source | Duplication factor | Effective sample multiplicity |
|---|---:|:---:|
| WSRD | 3× | 1,000 unique GTs presented as 3,000 records |
| REAL_FLARE | 5× | 600 unique GTs presented as 3,000 records |
| ISTD | 10× | 135 unique GTs presented as 1,350 records |

When packed into WebDataset shards, the duplicate records collapse into a single tar key per base image with all haze copies retained under sequential field indices (e.g. for a 3× WSRD base, `haze_000.png` through `haze_071.png` — three copies of the same 24 haze variants under different field names).

**Implication for downstream users**

- Iterating the dataset will yield **one sample per unique base image**, with a haze-variant count proportional to the upstream duplication factor (e.g., 72 for WSRD, 120 for REAL_FLARE, 240 for ISTD).
- Each duplicated `haze_NNN.png` is **byte-identical** to its original 24-variant counterpart — the same haze image is simply named under multiple field indices.
- To reproduce the original training-time mixing ratio, draw a single random haze field per sample (`random.choice(haze_fields)`); the inflated per-sample haze count restores the upstream weighting naturally.
- To get exactly 24 unique haze variants per WSRD/REAL_FLARE/ISTD sample, dedupe by content hash at the loader level, or read only the first 24 `haze_NNN.png` fields.

## Total bundle size

| Source | On-disk size (approx.) |
|---|---:|
| HAZESPACE | ~2.3 TB |
| ITS | ~103 GB |
| REAL_FLARE | ~76 GB |
| WSRD | ~50 GB |
| OTS | ~33 GB (haze only; no GT) |
| ISTD | ~1 GB |
| **Total** | **~2.55 TB** across ~1,300 shards |

OTS clean images are not shipped; see [`scripts/prepare_ots_originals.py`](../scripts/prepare_ots_originals.py) to fetch them from the official RESIDE source.
