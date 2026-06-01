---
license: cc-by-nc-sa-4.0
language:
- en
pretty_name: UniSER Synthetic Haze Dataset
size_categories:
- 1M<n<10M
task_categories:
- image-to-image
tags:
- image-restoration
- dehazing
- haze-removal
- soft-effects-removal
- webdataset
- computer-vision
- cvpr2026

extra_gated_heading: "Non-commercial academic use only"
extra_gated_description: >
  This dataset is released for non-commercial academic research only.
  By requesting access you agree to: (1) use it solely for non-commercial
  research; (2) cite our CVPR 2026 paper and every upstream subset you use
  (HazeSpace2M, RESIDE-ITS/OTS, WSRD, Flare7K++, ISTD); (3) comply with each
  constituent subset's upstream license — these may impose additional
  restrictions beyond CC-BY-NC-SA-4.0; (4) not redistribute under terms more
  permissive than CC-BY-NC-SA-4.0. Clean images from RESIDE-OTS are NOT
  included due to third-party photographer copyrights; a helper script
  (scripts/prepare_ots_originals.py in the companion GitHub repo) fetches
  them from the official RESIDE source.
extra_gated_prompt: "I agree to the above terms."
extra_gated_button_content: "I agree, request access"
extra_gated_fields:
  Full Name: text
  Affiliation: text
  Intended use: text
  I confirm non-commercial research use: checkbox
---

# UniSER Synthetic Haze Dataset 🌫️

[![arXiv](https://img.shields.io/badge/arXiv-2511.14183-b31b1b.svg)](https://arxiv.org/abs/2511.14183)
[![GitHub](https://img.shields.io/badge/GitHub-UniSER--Datasets-181717.svg?logo=github)](https://github.com/Evergreen0929/UniSER-Datasets)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA--4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Synthetic haze dataset released with our CVPR 2026 paper, *UniSER: A Foundation Model for Unified Soft Effects Removal*. The dataset bundles ~80k unique clean images with ~2 million physically-motivated haze / fog / smoke renderings — covering homogeneous, non-homogeneous, indoor, outdoor, daytime, and dense atmospheric conditions — for training and benchmarking single-image dehazing.

📦 **Size**: ~2.5 TB across 1,327 WebDataset shards
📄 **Paper**: [arXiv 2511.14183](https://arxiv.org/abs/2511.14183)
🐙 **Code**: [github.com/Evergreen0929/UniSER-Datasets](https://github.com/Evergreen0929/UniSER-Datasets)

## 📦 Dataset Composition

Six upstream sources, augmented with a physically-motivated atmospheric rendering on Marigold-predicted depth.

| Source | Unique GTs | Haze variants / image | GT shipped? |
|---|---:|---:|:---:|
| HAZESPACE2M | 66,133 | 24 | ✅ |
| RESIDE-ITS | 11,000 | 19 | ✅ |
| RESIDE-OTS | 2,061 | 37 | ❌ (fetch script in GitHub repo) |
| WSRD | 1,000 | 24 | ✅ |
| Flare-R | 600 | 24 | ✅ |
| ISTD | 135 | 24 | ✅ |
| **Total** | **~80.9k** | varied | — |

## 🗂️ Format

The dataset is sharded in [WebDataset](https://github.com/webdataset/webdataset) format. Each `shards/shard-NNNNN.tar` contains samples whose tar member names follow:

```
<source>/<base_name>.gt.<ext>           clean RGB image (absent for OTS)
<source>/<base_name>.haze_NNN.png       synthesized haze variant
<source>/<base_name>.haze_NNN.txt       descriptive tag (e.g. out_fog_120)
<source>/<base_name>.json               per-sample metadata
```

WebDataset groups entries sharing the same prefix-before-the-first-dot into one sample. Standard loaders (`webdataset`, `torchdata`, `litdata`) consume the shards directly via HTTP without any prior download.

> ⚠️ **Why is the HF Dataset Viewer "not available"?**
> HF's auto-viewer requires every WebDataset sample to share an identical set of field names. Our samples have **variable haze counts per source** (HAZESPACE has 24 variants, ITS has 19, OTS has 37, ISTD has up to 240; OTS samples also have no `gt`). This mirrors the upstream design and is intentional. **Use the loader below** — the dataset is fully functional via the `webdataset` library.

## 🚀 Quick Start

```bash
pip install -U "huggingface_hub[hf_xet]" webdataset pillow
hf auth login
```

```python
import io, json, random
from huggingface_hub import HfFileSystem
import webdataset as wds
from PIL import Image

REPO = "jdzhang0929/uniser-haze-dataset"
urls = [
    f"https://huggingface.co/datasets/{REPO}/resolve/main/{p[len(f'datasets/{REPO}/'):]}"
    for p in HfFileSystem().ls(f"datasets/{REPO}/shards", detail=False)
    if p.endswith(".tar")
]

def decode(s):
    if "json" not in s:
        return None
    meta = json.loads(s["json"])
    haze_keys = sorted(k for k in s if k.startswith("haze_") and k.endswith(".png"))
    if not haze_keys:
        return None
    chosen = random.choice(haze_keys)
    gt_key = next((k for k in s if k.startswith("gt.")), None)
    return {
        "source":    meta["source"],
        "base_name": meta["base_name"],
        "gt":        Image.open(io.BytesIO(s[gt_key])).convert("RGB") if gt_key else None,
        "haze":      Image.open(io.BytesIO(s[chosen])).convert("RGB"),
        "tag":       s[chosen.replace(".png", ".txt")].decode(),
    }

pipeline = (wds.WebDataset(urls, shardshuffle=True)
              .shuffle(1000)
              .map(decode)
              .select(lambda x: x is not None))

for sample in pipeline:
    print(sample["source"], sample["base_name"], sample["tag"])
    break
```

A full example with a preview-grid renderer, plus helpers to fetch the missing RESIDE-OTS clean images, are in the [companion GitHub repo](https://github.com/Evergreen0929/UniSER-Datasets) under `examples/` and `scripts/`.

## ⚠️ Caveats

**RESIDE-OTS clean images are not bundled** because the originals carry third-party photographer copyrights. After downloading, run `scripts/prepare_ots_originals.py` from the [GitHub repo](https://github.com/Evergreen0929/UniSER-Datasets) to fetch them from the official RESIDE source and align them locally by `base_name`.

## 📜 License

This bundled release is distributed under **CC-BY-NC-SA-4.0** — the strictest applicable clause inherited from upstream WSRD. Each subset retains its own upstream license; downstream redistribution and use must comply with each. See the GitHub repo's `docs/upstream_licenses.md` for per-source details.

## 📚 Citation

```bibtex
@article{zhang2025uniser,
  title={UniSER: A Foundation Model for Unified Soft Effects Removal},
  author={Zhang, Jingdong and Zhang, Lingzhi and Liu, Qing and Chiu, Mang Tik and Barnes, Connelly and Wang, Yizhou and You, Haoran and Liu, Xiaoyang and Zhou, Yuqian and Lin, Zhe and others},
  journal={arXiv preprint arXiv:2511.14183},
  year={2025}
}
```

Please also cite every upstream subset you use — full BibTeX in the GitHub repo's `CITATION.bib`.

## 📮 Contact

Please contact [Jingdong Zhang](https://evergreen0929.github.io/) with any questions.
