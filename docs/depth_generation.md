# Depth Map Generation with Marigold

The synthesizer ([`scripts/synthesize_haze.py`](../scripts/synthesize_haze.py)) needs a depth map per clean RGB image so that the atmospheric scattering grows with scene distance. We use [Marigold v1.1](https://github.com/prs-eth/Marigold) to predict relative monocular depth.

## Setup

Clone Marigold into `third_party/` (see [`third_party/README.md`](../third_party/README.md)):

```bash
git clone https://github.com/prs-eth/Marigold third_party/Marigold
cd third_party/Marigold && git checkout b1dffaa
pip install -r requirements.txt
```

## Generating depth maps

[`scripts/run_marigold_depth.py`](../scripts/run_marigold_depth.py) runs Marigold in parallel across all available GPUs. It accepts either a local directory or an S3 prefix as input/output and writes:

```
<output>/img/<rel_path>                  copy of the input image
<output>/depth/<rel_stem>_depth.png      Marigold predicted depth (8-bit)
```

The relative path under `--input` is preserved under both `img/` and `depth/`, so subdirectories carry through.

### Local example

```bash
python scripts/run_marigold_depth.py \
    --input  /path/to/raw_images \
    --output /path/to/staging \
    --fp16
```

### S3 example

```bash
python scripts/run_marigold_depth.py \
    --input  s3://my-bucket/raw/ITS \
    --output s3://my-bucket/processed/ITS \
    --fp16
```

### Common flags

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | local dir or `s3://...` containing raw RGB images |
| `--output` | required | local dir or `s3://...`; populates `img/` + `depth/` |
| `--fp16` | off | half-precision inference (recommended) |
| `--ensemble_size N` | 1 | number of predictions to average; raise to 5+ for higher quality |
| `--processing_res N` | None | inference resolution; `0` = original |
| `--io_batch_size N` | 8 | I/O batch per GPU worker |

## Notes

- Depth maps are normalized and inverted so that **near = 255 (bright)**, **far = 0 (dark)**. The synthesizer assumes this convention; if you change it, update `scripts/synthesize_haze.py` accordingly.
- Image extensions recognized: `.jpg .jpeg .png .bmp .webp`.
- For best quality use `--ensemble_size 5+`; the default `1` is faster but noisier.
