# Third-party Dependencies

This directory holds external code that we depend on but do not vendor. Clone the upstream repositories here before running scripts that reference them.

## Marigold (depth estimation backbone)

[`scripts/run_marigold_depth.py`](../scripts/run_marigold_depth.py) uses Marigold to generate depth maps from clean RGB images.

We pin Marigold to commit `b1dffaa` (the v1.1 release on the upstream `main` branch). Newer commits may also work — the script only relies on the public `MarigoldDepthPipeline` API.

```bash
git clone https://github.com/prs-eth/Marigold third_party/Marigold
cd third_party/Marigold
git checkout b1dffaa
pip install -r requirements.txt
```

License: Apache 2.0. The pretrained checkpoint (`prs-eth/marigold-depth-v1-1` on Hugging Face) carries an additional model license; see `third_party/Marigold/LICENSE-MODEL.txt` after cloning.
