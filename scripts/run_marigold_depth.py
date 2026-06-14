# Parallel depth estimation with Marigold (local or S3, multi-GPU).
#
# Reads RGB images from --input (local dir or s3://...) and writes:
#   <output>/img/<rel_path>           a copy of the input image
#   <output>/depth/<rel_stem>_depth.png  Marigold predicted depth (8-bit)
#
# Strictly mirrors upstream Marigold inference defaults from prs-eth/Marigold's
# run.py; the only additions are I/O orchestration and multi-GPU distribution.
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import boto3
import cv2
import numpy as np
import torch
from PIL import Image
from botocore.exceptions import ClientError
import torch.multiprocessing as mp
from tqdm import tqdm

# Import path: expects upstream Marigold cloned into <repo>/third_party/Marigold.
# See third_party/README.md for setup instructions.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "Marigold")))

from marigold import MarigoldDepthPipeline


# --- S3 client (initialized per worker process when needed) -------------
s3_client = None

def init_s3_client():
    """Initialize a thread-safe S3 client for a worker process."""
    global s3_client
    if s3_client is None:
        s3_client = boto3.client("s3")


# ==============================================================================
# ## I/O abstraction over local paths and s3:// URIs
# ==============================================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    assert is_s3_uri(uri)
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    return bucket, key.rstrip("/")


def list_input_images(input_uri: str) -> List[Tuple[str, str]]:
    """Return [(image_uri, rel_path_str), ...] for every image under input.

    rel_path_str is the path of each image relative to the input root, used to
    place outputs at the same relative location under <output>/img and
    <output>/depth.
    """
    if is_s3_uri(input_uri):
        bucket, prefix = parse_s3_uri(input_uri)
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        out: List[Tuple[str, str]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                ext = Path(key).suffix.lower()
                if ext not in IMG_EXTS:
                    continue
                rel = key[len(prefix):].lstrip("/") if prefix else key
                out.append((f"s3://{bucket}/{key}", rel))
        return out

    root = Path(input_uri).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--input is not a directory: {root}")
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        rel = p.relative_to(root).as_posix()
        out.append((str(p), rel))
    return out


def download_to_temp(uri: str, temp_dir_base: str) -> str:
    """Localize an image URI to a path under temp_dir_base. Returns the local path."""
    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(uri)
        local = os.path.join(temp_dir_base, key.replace("/", "_"))
        s3_client.download_file(bucket, key, local)
        return local
    return uri


def write_image_bytes(output_uri: str, rel_path: str, data: bytes):
    """Write `data` at <output_uri>/<rel_path> (local or S3)."""
    if is_s3_uri(output_uri):
        bucket, prefix = parse_s3_uri(output_uri)
        key = f"{prefix}/{rel_path}".lstrip("/") if prefix else rel_path
        s3_client.put_object(Bucket=bucket, Key=key, Body=data)
    else:
        out = Path(output_uri) / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)


def copy_image(src_uri: str, output_uri: str, rel_path: str, local_src: str):
    """Copy the source image to <output>/img/<rel_path>."""
    rel = f"img/{rel_path}"
    with open(local_src, "rb") as f:
        data = f.read()
    write_image_bytes(output_uri, rel, data)


def write_depth(output_uri: str, rel_path: str, depth_8bit: np.ndarray):
    """Write the depth PNG to <output>/depth/<rel_stem>_depth.png."""
    rel = Path(rel_path)
    out_rel = f"depth/{rel.with_name(rel.stem + '_depth.png').as_posix()}"
    ok, buf = cv2.imencode(".png", depth_8bit)
    if not ok:
        raise IOError(f"Failed to encode depth PNG for {rel_path}")
    write_image_bytes(output_uri, out_rel, buf.tobytes())


# ==============================================================================
# ## Model loading
# ==============================================================================
def load_marigold_model(device, half_precision):
    """Load the pretrained Marigold depth pipeline onto the given device."""
    print(f"[{os.getpid()}] Loading Marigold model onto {device}...")
    checkpoint_path = "prs-eth/marigold-depth-v1-1"

    if half_precision:
        dtype = torch.float16
        print(f"[{os.getpid()}] Using half precision (float16).")
    else:
        dtype = torch.float32
        print(f"[{os.getpid()}] Using full precision (float32).")

    pipe = MarigoldDepthPipeline.from_pretrained(checkpoint_path, torch_dtype=dtype)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass

    pipe = pipe.to(device)
    print(f"[{os.getpid()}] Model loaded successfully on {device}.")
    return pipe


# ==============================================================================
# ## Core batched processing
# ==============================================================================
def process_batch(batch_jobs, model, output_uri, temp_dir_base, model_args):
    """Load a batch of images, estimate depth with Marigold, and write results."""
    batch_images = []
    batch_metadata = []

    for img_uri, rel_path in batch_jobs:
        try:
            local_image_path = download_to_temp(img_uri, temp_dir_base)
            if not os.path.exists(local_image_path):
                print(f"  WARN: File does not exist, skipping: {local_image_path}")
                continue
            image = Image.open(local_image_path).convert("RGB")
            batch_images.append(image)
            batch_metadata.append({
                "img_uri": img_uri,
                "rel_path": rel_path,
                "local_image_path": local_image_path,
            })
        except Exception as e:
            print(f"  ERROR pre-processing {img_uri}, skipping. Details: {e}")

    if not batch_images:
        return

    # Inference (parameters mirror upstream run.py).
    depth_predictions = []
    with torch.no_grad():
        for image in batch_images:
            generator = None
            if model_args["seed"] is not None:
                generator = torch.Generator(device=model.device)
                generator.manual_seed(model_args["seed"])

            pipeline_output = model(
                image,
                denoising_steps=model_args["denoise_steps"],
                ensemble_size=model_args["ensemble_size"],
                processing_res=model_args["processing_res"],
                match_input_res=not model_args["output_processing_res"],
                batch_size=model_args["batch_size"],
                resample_method=model_args["resample_method"],
                generator=generator,
                show_progress_bar=False,
            )
            depth_predictions.append(pipeline_output.depth_np)

    # Save outputs.
    for i, item in enumerate(batch_metadata):
        try:
            relative_depth_map = depth_predictions[i]
            min_val, max_val = relative_depth_map.min(), relative_depth_map.max()
            if max_val > min_val:
                normalized_depth = (max_val - relative_depth_map) / (max_val - min_val)
            else:
                normalized_depth = np.zeros_like(relative_depth_map)
            depth_8bit = (normalized_depth * 255).astype(np.uint8)

            copy_image(item["img_uri"], output_uri, item["rel_path"], item["local_image_path"])
            write_depth(output_uri, item["rel_path"], depth_8bit)

            # Clean up the staged local file (only if we downloaded it from S3).
            if item["local_image_path"].startswith(temp_dir_base):
                try:
                    os.remove(item["local_image_path"])
                except OSError:
                    pass
        except Exception as e:
            print(f"  ERROR during post-processing for {item['rel_path']}, skipping. Details: {e}")


# ==============================================================================
# ## Worker process (one per GPU)
# ==============================================================================
def process_worker(rank, world_size, all_jobs, output_uri, io_batch_size, model_args):
    """Worker process running on a single GPU."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    init_s3_client()
    model = load_marigold_model(device, model_args["half_precision"])
    jobs_for_this_worker = all_jobs[rank::world_size]
    num_jobs = len(jobs_for_this_worker)
    print(f"[GPU:{rank}] Assigned {num_jobs} images.")

    with tempfile.TemporaryDirectory() as temp_dir:
        worker_temp_dir = os.path.join(temp_dir, f"worker_{rank}")
        os.makedirs(worker_temp_dir, exist_ok=True)

        progress_bar = tqdm(total=-(-num_jobs // io_batch_size), desc=f"GPU:{rank}", position=rank)
        for i in range(0, num_jobs, io_batch_size):
            batch_jobs = jobs_for_this_worker[i:i + io_batch_size]
            try:
                process_batch(batch_jobs, model, output_uri, worker_temp_dir, model_args)
            except Exception as e:
                print(f"FATAL ERROR processing batch on GPU {rank}: {e}")
            progress_bar.update(1)
        progress_bar.close()


# ==============================================================================
# ## Main
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel depth estimation with Marigold (local or S3, multi-GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- I/O ---
    parser.add_argument("--input", required=True,
                        help="Input source: local directory or s3://bucket/prefix "
                             "containing clean RGB images (any depth of subdirs).")
    parser.add_argument("--output", required=True,
                        help="Output destination: local directory or s3://bucket/prefix. "
                             "Writes <output>/img/<rel_path> and <output>/depth/<rel_stem>_depth.png.")
    parser.add_argument("--io_batch_size", type=int, default=8,
                        help="Number of image paths to process per I/O batch.")

    # --- Model arguments (mirror upstream run.py defaults) ---
    parser.add_argument("--denoise_steps", type=int, default=None, help="Diffusion denoising steps.")
    parser.add_argument("--processing_res", type=int, default=None, help="Resolution for estimation. 0 = original resolution.")
    parser.add_argument("--ensemble_size", type=int, default=1, help="Number of predictions to ensemble.")
    parser.add_argument("--batch_size", type=int, default=0, help="Inference batch size (passed to model). 0 = auto.")
    parser.add_argument("--half_precision", "--fp16", action="store_true", help="Run with half-precision (float16).")
    parser.add_argument("--output_processing_res", action="store_true", help="Output result at processing resolution.")
    parser.add_argument("--resample_method", choices=["bilinear", "bicubic", "nearest"], default="bilinear", help="Resampling method.")
    parser.add_argument("--seed", type=int, default=None, help="Reproducibility seed.")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA is not available.")
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Starting multi-process inference...")

    model_args = {
        "denoise_steps": args.denoise_steps,
        "processing_res": args.processing_res,
        "ensemble_size": args.ensemble_size,
        "batch_size": args.batch_size,
        "half_precision": args.half_precision,
        "output_processing_res": args.output_processing_res,
        "resample_method": args.resample_method,
        "seed": args.seed,
    }

    print(f"\n--- Listing images under {args.input} ---")
    init_s3_client()
    jobs = list_input_images(args.input)
    print(f"Found {len(jobs)} images.")
    if not jobs:
        sys.exit("Nothing to do.")

    mp.set_start_method("spawn", force=True)
    spawn_args = (world_size, jobs, args.output, args.io_batch_size, model_args)
    mp.spawn(process_worker, args=spawn_args, nprocs=world_size, join=True)
    print("\n--- Done ---")
