# HALO Lens-Flare Dataset Structure

## Bundle layout

```
shards/
├── shard-00000.tar      ~2.15 GB
├── shard-00001.tar
├── ...
└── shard-00070.tar
shard_index.json         per-shard size + sample-count manifest
finalist.json            full record table (4945 samples)
release_rename_map.json  token-level sanitization mapping (transparency)
README.md                Hugging Face dataset card
CITATION.bib             citation file
```

71 WebDataset shards, ~2.15 GB each, **153 GB total**. Samples are recovered by grouping tar entries that share the same prefix before the first `.` in the basename.

## Per-sample fields

Each sample carries a flare-free / flared / flare-only triplet plus per-sample metadata:

```
halo/<base_name>.gt.png          clean scene without flare      (RGBA, 3840×2160)
halo/<base_name>.flare.png       same scene with flare added    (RGBA, 3840×2160)
halo/<base_name>.separate.png    flare-only on transparent bg   (RGBA, 3840×2160)
halo/<base_name>.json            per-sample metadata
```

`<base_name>` encodes scene + effect + camera, e.g. `Scene003_Glare001_camera01`.

### Per-sample metadata schema

```json
{
  "scene":       "Scene003",
  "effect_type": "Glare",
  "effect_id":   "Glare001",
  "sample_id":   "Scene003_Glare001_camera01",
  "orig_idx":    7313,
  "final_idx":   0,
  "light":       "Scene003_Glare001_camera01.gt.png",
  "flare":       "Scene003_Glare001_camera01.flare.png",
  "separate":    "Scene003_Glare001_camera01.separate.png"
}
```

`orig_idx` is the index in the pre-filter render set; `final_idx` is the 0-based index in the released 4945-sample finalist (look up further details via `finalist.json`).

## The triplet — how to use it

The three PNGs support three training formulations:

| Task | Input | Target |
|---|---|---|
| **Flare removal** | `flare.png` | `gt.png` |
| **Flare synthesis** | `gt.png` | `flare.png` |
| **Decomposition** | `flare.png` | (`gt.png`, `separate.png`) where `flare ≈ gt + separate` |

`separate.png` retains its alpha channel — the flare is rendered against a transparent background so it can be re-composited over arbitrary clean images at training time for data augmentation.

## Effect-type / scene distribution

4945 samples across **32 scenes** × **4 flare types**:

| Effect type | Count | Share |
|---|---:|---:|
| Streak | 1,656 | 33.5% |
| Reflective | 1,655 | 33.5% |
| Glare | 817 | 16.5% |
| Shimmer | 817 | 16.5% |

Per-scene counts:

| Scene | Total | Streak | Reflective | Glare | Shimmer |
|---|---:|---:|---:|---:|---:|
| Scene011 | 302 | 101 | 101 | 50 | 50 |
| Scene075 | 254 | 84 | 86 | 42 | 42 |
| Scene039 | 253 | 85 | 84 | 42 | 42 |
| Scene022 | 252 | 84 | 84 | 42 | 42 |
| Scene013 | 226 | 77 | 74 | 37 | 38 |
| Scene025 | 204 | 68 | 69 | 34 | 33 |
| Scene041 | 204 | 68 | 69 | 34 | 33 |
| Scene017 | 203 | 68 | 69 | 33 | 33 |
| Scene045 | 203 | 69 | 68 | 33 | 33 |
| Scene021 | 202 | 68 | 68 | 33 | 33 |
| Scene029 | 202 | 68 | 68 | 33 | 33 |
| Scene033 | 202 | 69 | 67 | 33 | 33 |
| Scene049 | 202 | 68 | 68 | 33 | 33 |
| Scene051 | 202 | 68 | 68 | 33 | 33 |
| Scene072 | 201 | 68 | 67 | 33 | 33 |
| Scene005 | 152 | 50 | 52 | 25 | 25 |
| Scene003 | 150 | 50 | 50 | 25 | 25 |
| Scene007 | 150 | 50 | 50 | 25 | 25 |
| Scene023 | 150 | 50 | 50 | 25 | 25 |
| Scene037 | 150 | 50 | 50 | 25 | 25 |
| Scene053 | 150 | 50 | 50 | 25 | 25 |
| Scene048 | 118 | 39 | 39 | 20 | 20 |
| Scene054 | 115 | 38 | 37 | 20 | 20 |
| Scene071 | 101 | 33 | 34 | 17 | 17 |
| Scene015 | 100 | 33 | 33 | 17 | 17 |
| Scene064 | 100 | 33 | 33 | 17 | 17 |
| Scene065 | 49 | 16 | 17 | 8 | 8 |
| Scene077 | 48 | 17 | 16 | 7 | 8 |
| Scene057 | 34 | 11 | 11 | 6 | 6 |
| Scene069 | 34 | 11 | 11 | 6 | 6 |
| Scene035 | 16 | 6 | 6 | 2 | 2 |
| Scene061 | 16 | 6 | 6 | 2 | 2 |

## `finalist.json` schema

`finalist.json` is the master record table (3.8 MB, 4945 entries). Each entry preserves the full provenance of one sample:

```json
{
  "flare_image_key":     "jingdongz-data/3D_Halo_Render/Far_Scene/Scene003/Glare001/Scene003_03_3_3_Glare001_versveldpas_4K/camera01_flare.png",
  "light_image_key":     "jingdongz-data/3D_Halo_Render/.../camera01_light01.png",
  "separate_image_key":  "jingdongz-data/3D_Halo_Render/.../camera01_separate.png",
  "separate01_image_key":"jingdongz-data/3D_Halo_Render/.../camera01_separate01.png",
  "_orig_idx":  7313,
  "_scene":     "Scene003",
  "_effect_type":"Glare",
  "_effect_id": "Glare001",
  "_sample_id": "Scene003_Glare001_camera01",
  "_final_idx": 0
}
```

The `_4K/` parent directory in each `*_image_key` encodes the source HDRI slug (e.g. `versveldpas`, `kiara-4-mid-morning`) — these match canonical Poly Haven / freepoly.org asset names.

## Sanitization mapping

A small number of 3D-asset names contained real-person romanizations or commercial-brand tokens. These were swapped for generic codes via [`scripts/apply_release_rename.py`](../scripts/apply_release_rename.py); the mapping ships alongside the dataset as `release_rename_map.json` for transparency.

| Sanitized token | Replaces | Count of affected finalist records |
|---|---|---:|
| `char01` | `HongChengHong` | 6 |
| `char02` | `HuangKangHua` | 10 |
| `char03` | `HuangXinYin` | 8 |
| `char04` | `LiuLiSha` | 3 |
| `char05` | `LiuXiaoXiao` | 2 |
| `char06` | `HuangD` | 2 |
| `char07` | `LiuCJ` | 2 |
| `char08` | `Jennt` | 5 |
| `char09` | `Lilya` | 4 |
| `char10` | `HuMS` | 4 |
| `char11` | `HuSQ` | 11 |
| `MouseBlack` | `MouseThermaltakeBlack` | 5 |

The substitution is a strict string replacement — every other path component (Scene id, effect, camera, HDRI slug) is untouched. Resulting paths remain unique across the 4945-record finalist.

## License

HALO is released under **CC-BY-NC-SA-4.0**. Upstream HDRI environments retain their original licenses (Poly Haven CC0; freepoly.org terms) — both equal to or more permissive than the bundled license.
