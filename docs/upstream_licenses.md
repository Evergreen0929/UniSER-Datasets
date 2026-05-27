# Upstream Dataset Licenses

The UniSER-Datasets release builds atmospheric-effect augmentations on top of six existing public datasets. Each subset retains its upstream license; users are responsible for complying with each.

## Summary

| Subset | Upstream | License | Originals redistributed? |
|---|---|---|---|
| HAZESPACE | HazeSpace2M (Islam et al., ACM MM 2024) | CC-BY-4.0 | Yes |
| ITS | RESIDE-ITS (Li et al., TIP 2018) | No explicit license — academic-use convention | Yes |
| OTS | RESIDE-OTS (Li et al., TIP 2018) | No explicit license — Flickr-sourced clean images | **No** — see [`scripts/prepare_ots_originals.sh`](../scripts/prepare_ots_originals.sh) |
| WSRD | WSRD (Vasluianu et al., CVPRW 2023) | CC-BY-NC-SA-4.0 | Yes |
| REAL_FLARE | Flare-R from Flare7K++ (Dai et al., TPAMI 2024) | S-Lab License 1.0 | Yes |
| ISTD | ISTD (Wang et al., CVPR 2018) | Research & non-commercial only | Yes |

## Composite license for the bundled release

Because WSRD carries CC-BY-NC-SA-4.0 (the strictest applicable redistribution clause) and ISTD restricts use to non-commercial research, **the bundled release is distributed under CC-BY-NC-SA-4.0**.

The synthesized haze derivatives (`haze_v2/`) and the code in this repository are released under MIT (see [`LICENSE`](../LICENSE)) — but bundled with images they inherit the composite license above.

## Per-subset citations

If you use this dataset, please cite UniSER-Datasets **and** every upstream source whose subset you use:

```bibtex
@inproceedings{hazespace2m_2024,
  title     = {HazeSpace2M: A Dataset for Haze Aware Single Image Dehazing},
  author    = {Islam, Md Tanvir and others},
  booktitle = {ACM Multimedia},
  year      = {2024},
}

@article{reside_2018,
  title   = {Benchmarking Single-Image Dehazing and Beyond},
  author  = {Li, Boyi and Ren, Wenqi and Fu, Dengpan and Tao, Dacheng and Feng, Dan and Zeng, Wenjun and Wang, Zhangyang},
  journal = {IEEE Transactions on Image Processing},
  year    = {2018},
}

@inproceedings{wsrd_2023,
  title     = {WSRD: A Novel Benchmark for High Resolution Image Shadow Removal},
  author    = {Vasluianu, Florin-Alexandru and others},
  booktitle = {CVPRW (NTIRE)},
  year      = {2023},
}

@article{flare7kpp_2024,
  title   = {Flare7K++: Mixing Synthetic and Real Datasets for Nighttime Flare Removal and Beyond},
  author  = {Dai, Yuekun and others},
  journal = {IEEE TPAMI},
  year    = {2024},
}

@inproceedings{istd_2018,
  title     = {Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal},
  author    = {Wang, Jifeng and Li, Xiang and Yang, Jian},
  booktitle = {CVPR},
  year      = {2018},
}
```

## Note on RESIDE-OTS

The clean outdoor images in RESIDE-OTS were scraped from public photo-sharing sites and carry third-party photographer copyrights. We do **not** redistribute the originals; only our synthesized `haze_v2/` derivatives ship with the dataset. To pair them with the original clean images, run [`scripts/prepare_ots_originals.sh`](../scripts/prepare_ots_originals.sh), which fetches the originals from the official RESIDE page and aligns them to our `base_name` convention.

## Reporting issues

If you are an upstream dataset author and would like a subset removed or its license treatment changed, please open an issue on the GitHub repository.
