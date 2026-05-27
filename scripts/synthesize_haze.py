#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atmospheric Effects Batch Synthesizer (Multi-Process, Multi-GPU)
================================================================

Reads paired clean RGB images + depth maps (Marigold-style 8-bit) from either
a local directory or an S3 prefix, renders a fixed palette of haze / fog /
smoke variants via a physically-motivated atmospheric model on the GPU, and
writes each variant to <output>/haze_v2/<base_name>/<tag>.png.

Expected input layout (either local or S3):

    <input>/img/<base_name>.<ext>          clean RGB image
    <input>/depth/<base_name>_depth.png    8-bit depth (255 = near, 0 = far)

Optional non-uniform haze:
    Enable with --use-smoke-texture. The renderer picks a random texture from
    <smoke-texture-dir>, takes a random crop of the image's size, and uses it
    as a per-pixel multiplier on the optical thickness so the haze is denser
    in some regions and thinner in others. Off by default.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union

import boto3
import cv2
import numpy as np
import multiprocessing as mp
import random

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm


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
def is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """`s3://bucket/key` -> ('bucket', 'key')."""
    assert is_s3_uri(uri)
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    return bucket, key.rstrip("/")


def read_image_uri(uri: str, flag=cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image (BGR for color, grayscale otherwise) from a local path or s3://."""
    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(uri)
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        buf = np.frombuffer(obj["Body"].read(), dtype=np.uint8)
        return cv2.imdecode(buf, flag)
    return cv2.imread(uri, flag)


def write_image_uri(uri: str, img: np.ndarray):
    """Write an image to a local path or s3://. Extension determines encoder."""
    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(uri)
        ext = "." + key.rsplit(".", 1)[-1]
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            raise IOError(f"cv2.imencode failed for {uri}")
        s3_client.put_object(Bucket=bucket, Key=key, Body=buf.tobytes())
    else:
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(uri, img)


def list_input_pairs(input_uri: str) -> List[Tuple[str, str, str]]:
    """Return a list of (image_uri, depth_uri, base_name) tuples.

    Matches files in <input>/img/ to <input>/depth/<base>_depth.png.
    """
    if is_s3_uri(input_uri):
        bucket, prefix = parse_s3_uri(input_uri)
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        img_prefix = f"{prefix}/img/" if prefix else "img/"
        depth_prefix = f"{prefix}/depth/" if prefix else "depth/"

        depth_basenames = set()
        for page in paginator.paginate(Bucket=bucket, Prefix=depth_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("/"):
                    continue
                depth_basenames.add(os.path.basename(obj["Key"]))

        pairs = []
        for page in paginator.paginate(Bucket=bucket, Prefix=img_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("/"):
                    continue
                img_name = os.path.basename(obj["Key"])
                base, _ = os.path.splitext(img_name)
                expected = f"{base}_depth.png"
                if expected in depth_basenames:
                    pairs.append((
                        f"s3://{bucket}/{obj['Key']}",
                        f"s3://{bucket}/{depth_prefix}{expected}",
                        base,
                    ))
        return pairs

    root = Path(input_uri)
    img_dir = root / "img"
    depth_dir = root / "depth"
    if not img_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{img_dir}' and '{depth_dir}' to exist (img/ + depth/ layout)."
        )
    depth_names = {p.name for p in depth_dir.iterdir() if p.is_file()}
    pairs = []
    for img in sorted(img_dir.iterdir()):
        if not img.is_file():
            continue
        base = img.stem
        expected = f"{base}_depth.png"
        if expected in depth_names:
            pairs.append((str(img), str(depth_dir / expected), base))
    return pairs


def haze_output_uri(output_uri: str, base_name: str, tag: str) -> str:
    """Compose <output>/haze_v2/<base_name>/<tag> as either a local path or s3 URI."""
    if is_s3_uri(output_uri):
        bucket, prefix = parse_s3_uri(output_uri)
        key = f"{prefix}/haze_v2/{base_name}/{tag}" if prefix else f"haze_v2/{base_name}/{tag}"
        return f"s3://{bucket}/{key}"
    return str(Path(output_uri) / "haze_v2" / base_name / tag)


# ==============================================================================
# ## 1. CORE RENDERING ENGINE (GPU-accelerated)
# ==============================================================================
def srgb_to_linear_gpu(img: torch.Tensor) -> torch.Tensor:
    a = 0.055
    threshold = 0.04045
    low_mask = img <= threshold
    high_mask = ~low_mask
    out = torch.empty_like(img, dtype=torch.float32)
    out[low_mask] = img[low_mask] / 12.92
    out[high_mask] = torch.pow(((img[high_mask] + a) / (1 + a)), 2.4)
    return out


def linear_to_srgb_gpu(img_lin: torch.Tensor) -> torch.Tensor:
    a = 0.055
    threshold = 0.0031308
    low_mask = img_lin <= threshold
    high_mask = ~low_mask
    out = torch.empty_like(img_lin, dtype=torch.float32)
    out[low_mask] = 12.92 * img_lin[low_mask]
    img_lin_clamped = torch.clamp(img_lin[high_mask], 0, 1)
    out[high_mask] = (1 + a) * torch.pow(img_lin_clamped, 1 / 2.4) - a
    return out


def parse_color_gpu(vals: Tuple[float, float, float], device: torch.device) -> torch.Tensor:
    return torch.clamp(torch.tensor(vals, dtype=torch.float32, device=device) / 255.0, 0.0, 1.0)


def compute_height_proxy_gpu(hmax: float, H: int, W: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, H, dtype=torch.float32, device=device).view(H, 1)
    h = hmax * (1.0 - y)
    return h.expand(-1, W)


PRESETS = {
    "fog":   {"omega0": (1.0, 1.0, 1.0), "g": 0.9, "eta": 0.7, "H": 100.0,  "airlight": (1.0, 1.0, 1.0),  "spectral": (1.0, 1.0, 1.0)},
    "haze":  {"omega0": (1.0, 1.0, 1.0), "g": 0.8, "eta": 1.0, "H": 1200.0, "airlight": (0.6, 0.75, 1.0), "spectral": (1.1, 1.0, 0.9)},
    "smoke": {"omega0": (0.8, 0.8, 0.8), "g": 0.7, "eta": 0.9, "H": 50.0,   "airlight": (0.75, 0.65, 0.55), "spectral": (0.9, 1.0, 1.1)},
}


def compose_atmosphere_gpu(
    img_enc_tensor: torch.Tensor,
    depth_norm_tensor: torch.Tensor,
    baseline_tau: float,
    apply_dithering: bool,
    smoke_mask: Optional[torch.Tensor] = None,
    smoke_strength: float = 0.5,
    **kwargs: Any,
) -> np.ndarray:
    """GPU-accelerated atmospheric rendering.

    Args:
        img_enc_tensor: input sRGB image as a PyTorch tensor (H, W, 3) in [0, 1].
        depth_norm_tensor: normalized depth (H, W) in [0, 1].
        baseline_tau: random offset added to the optical thickness for baseline fog.
        smoke_mask: optional (H, W) tensor in [0, 1] used as a per-pixel
            multiplier on optical thickness. When provided, density varies as
            tau * ((1 - smoke_strength) + smoke_strength * smoke_mask), creating
            non-uniform haze.
        smoke_strength: blend strength when smoke_mask is set. 0 disables, 1 lets
            the texture fully modulate density.
        **kwargs: rendering parameters.
    """
    device = img_enc_tensor.device
    mode = kwargs["mode"]
    P = PRESETS[mode]
    dmax = float(kwargs.get("dmax", 50.0))
    zgamma = float(kwargs.get("zgamma", 1.0))
    hmax = float(kwargs.get("hmax", P["H"]))
    visibility = float(kwargs["visibility"])
    spectral = P["spectral"]
    omega0 = kwargs.get("omega0", P["omega0"])
    eta = float(kwargs.get("eta", P["eta"]))
    H_scale = float(kwargs.get("H", P["H"]))
    airlight_rgb255 = kwargs.get("airlight", None)

    airlight_enc = (parse_color_gpu(airlight_rgb255, device)
                    if airlight_rgb255 else
                    torch.tensor(P["airlight"], dtype=torch.float32, device=device))

    H_img, W_img, _ = img_enc_tensor.shape

    img_lin = srgb_to_linear_gpu(img_enc_tensor)
    A_enc = airlight_enc.view(1, 1, 3)
    A_lin = srgb_to_linear_gpu(A_enc)

    d = (torch.pow(torch.clamp(depth_norm_tensor, 0, 1), zgamma)) * dmax
    h = compute_height_proxy_gpu(hmax, H_img, W_img, device)

    beta0 = 3.912 / visibility
    s = torch.tensor(spectral, dtype=torch.float32, device=device).view(1, 1, 3)
    beta_t0 = beta0 * s

    phi = torch.exp(-torch.clamp(h, min=0) / max(1e-6, H_scale))
    phi = phi.view(H_img, W_img, 1)

    tau = beta_t0 * phi * d.view(H_img, W_img, 1)

    # Baseline fog: random offset added to optical thickness.
    tau = tau + baseline_tau

    # Non-uniform haze: modulate tau by a smoke-texture crop.
    if smoke_mask is not None:
        modulator = (1.0 - smoke_strength) + smoke_strength * smoke_mask.unsqueeze(-1)
        tau = tau * modulator

    T = torch.exp(-tau)

    kappa_val = 1.0
    kappa3 = torch.tensor([kappa_val, kappa_val, kappa_val], dtype=torch.float32, device=device).view(1, 1, 3)
    omega0v = torch.tensor(omega0, dtype=torch.float32, device=device).view(1, 1, 3)

    T_boost = torch.pow(T, eta)
    air_term = A_lin * (omega0v * kappa3) * (1.0 - T_boost)

    out_lin = img_lin * T + air_term
    out_enc = torch.clamp(linear_to_srgb_gpu(torch.clamp(out_lin, 0.0, 1.0)), 0.0, 1.0)

    if apply_dithering:
        noise = (torch.rand_like(out_enc) - 0.5) * (10.0 / 255.0)
        out_enc = torch.clamp(out_enc + noise, 0.0, 1.0)

    out_cpu_numpy = out_enc.cpu().numpy()
    out_u8 = (out_cpu_numpy * 255.0 + 0.5).astype(np.uint8)

    return cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)


# ==============================================================================
# ## 2. BATCH PROCESSING & CONFIGURATION
# ==============================================================================
def desaturate_color(rgb_color: tuple, factor: float) -> tuple:
    color_bgr_pixel = np.uint8([[list(reversed(rgb_color))]])
    color_hsv_pixel = cv2.cvtColor(color_bgr_pixel, cv2.COLOR_BGR2HSV)
    saturation = color_hsv_pixel[0][0][1]
    new_saturation = np.clip(saturation * factor, 0, 255)
    color_hsv_pixel[0][0][1] = new_saturation
    new_color_bgr_pixel = cv2.cvtColor(color_hsv_pixel, cv2.COLOR_HSV2BGR)
    b, g, r = new_color_bgr_pixel[0][0]
    return (int(r), int(g), int(b))


def get_render_configs() -> List[Dict[str, Any]]:
    configs = []
    haze_params = {
        "sky-blue":      (153, 174, 215),
        "warm-urban":    (200, 180, 140),
        "pale-gray":     (210, 210, 220),
        "brownish-dust": (190, 170, 150),
    }
    haze_params["sky-blue"] = desaturate_color(haze_params["sky-blue"], 0.7)

    for name, airlight in haze_params.items():
        for vis in [100, 200, 300, 500, 1000]:
            configs.append({"mode": "haze", "dmax": 120, "zgamma": 1.0,
                            "airlight": airlight, "visibility": vis,
                            "filename": f"out_haze_{name}_v{vis}.png"})

    fog_base = {"mode": "fog", "dmax": 60, "zgamma": 1.0}
    configs.extend([
        {**fog_base, "visibility": 1000, "eta": 1.0,  "filename": "out_fog_1000.png"},
        {**fog_base, "visibility": 500,  "eta": 0.95, "filename": "out_fog_500.png"},
        {**fog_base, "visibility": 300,  "eta": 1.0,  "filename": "out_fog_300.png"},
        {**fog_base, "visibility": 200,  "eta": 0.95, "filename": "out_fog_200.png"},
        {**fog_base, "visibility": 120,  "eta": 0.9,  "filename": "out_fog_120.png"},
        {**fog_base, "visibility": 80,   "eta": 0.85, "filename": "out_fog_80.png"},
        {**fog_base, "visibility": 40,   "eta": 0.5,  "filename": "out_fog_40.png"},
        {**fog_base, "visibility": 30,   "H": 30, "hmax": 120, "filename": "out_fog_valley_h30.png"},
        {**fog_base, "visibility": 50,   "H": 45, "hmax": 120, "filename": "out_fog_valley_h45.png"},
        {**fog_base, "visibility": 70,   "H": 60, "hmax": 120, "filename": "out_fog_valley_h60.png"},
    ])

    smoke_base = {"mode": "smoke", "dmax": 100, "zgamma": 1.0}
    configs.extend([
        {**smoke_base, "visibility": 250,  "airlight": (200, 180, 160), "eta": 0.95, "omega0": (0.85, 0.85, 0.85), "filename": "out_smoke_light.png"},
        {**smoke_base, "visibility": 150,  "airlight": (180, 150, 120), "eta": 0.9,  "omega0": (0.8,  0.8,  0.8),  "filename": "out_smoke_medium.png"},
        {**smoke_base, "visibility": 80,   "airlight": (160, 120, 90),  "H": 40, "hmax": 60, "eta": 0.85, "omega0": (0.75, 0.75, 0.75), "filename": "out_smoke_heavy.png"},
        {**smoke_base, "visibility": 120,  "airlight": (180, 180, 190), "eta": 0.92, "omega0": (0.85, 0.85, 0.85), "filename": "out_smoke_coolgray_120.png"},
        {**smoke_base, "visibility": 200,  "airlight": (180, 180, 190), "eta": 0.92, "omega0": (0.85, 0.85, 0.85), "filename": "out_smoke_coolgray_200.png"},
        {**smoke_base, "visibility": 500,  "airlight": (180, 180, 190), "eta": 0.92, "omega0": (0.85, 0.85, 0.85), "filename": "out_smoke_coolgray_500.png"},
        {**smoke_base, "visibility": 1000, "airlight": (180, 180, 190), "eta": 0.92, "omega0": (0.85, 0.85, 0.85), "filename": "out_smoke_coolgray_1000.png"},
    ])
    return configs


# ==============================================================================
# ## 3. SMOKE-TEXTURE BLEND HELPERS
# ==============================================================================
def discover_smoke_textures(smoke_dir: str) -> List[str]:
    p = Path(smoke_dir)
    if not p.is_dir():
        return []
    return sorted(str(x) for x in p.iterdir()
                  if x.is_file() and x.suffix.lower() in {".png", ".jpg", ".jpeg"})


def sample_smoke_mask(texture_paths: List[str], H: int, W: int,
                      device: torch.device) -> Optional[torch.Tensor]:
    """Pick a random texture, take a random HxW crop, return [H, W] tensor in [0, 1]."""
    if not texture_paths:
        return None
    tex_path = random.choice(texture_paths)
    tex = cv2.imread(tex_path, cv2.IMREAD_GRAYSCALE)
    if tex is None:
        return None
    th, tw = tex.shape
    if th < H or tw < W:
        nh, nw = max(H, th), max(W, tw)
        tex = cv2.resize(tex, (nw, nh), interpolation=cv2.INTER_LINEAR)
        th, tw = tex.shape
    y0 = random.randint(0, th - H)
    x0 = random.randint(0, tw - W)
    crop = tex[y0:y0 + H, x0:x0 + W]
    return torch.from_numpy(crop.astype(np.float32) / 255.0).to(device)


# ==============================================================================
# ## 4. MULTIPROCESSING WORKER TASK
# ==============================================================================
def worker_task(args: tuple):
    (img_uri, depth_uri, base_name, output_uri, gpu_id,
     apply_dithering, smoke_texture_paths, smoke_strength) = args

    device = (torch.device(f"cuda:{gpu_id}")
              if gpu_id != -1 and torch.cuda.is_available()
              else torch.device("cpu"))

    render_configs = get_render_configs()
    use_smoke = bool(smoke_texture_paths)

    try:
        original_image_bgr = read_image_uri(img_uri)
        if original_image_bgr is None:
            raise IOError(f"Failed to read image: {img_uri}")

        original_image_rgb_np = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        original_image_tensor = torch.from_numpy(original_image_rgb_np).to(device)

        depth_map_raw = read_image_uri(depth_uri, flag=cv2.IMREAD_GRAYSCALE)
        if depth_map_raw is None:
            raise IOError(f"Failed to read depth map: {depth_uri}")

        sharp_depth_map_inverted_np = 255 - depth_map_raw
        sharp_depth_map_norm_np = sharp_depth_map_inverted_np.astype(np.float32) / 255.0
        sharp_depth_tensor = torch.from_numpy(sharp_depth_map_norm_np).to(device)

        H_img, W_img = sharp_depth_map_inverted_np.shape

        for config in render_configs:
            current_depth_tensor = sharp_depth_tensor
            render_params = config.copy()

            if "airlight" in render_params:
                r, g, b = render_params["airlight"]
                r += random.randint(-5, 5)
                g += random.randint(-5, 5)
                b += random.randint(-5, 5)
                render_params["airlight"] = tuple(np.clip([r, g, b], 0, 255))

            if render_params["mode"] == "fog":
                vis = render_params["visibility"]
                apply_blur, kernel_size = False, 0

                # Depth-related Gaussian blur for extra-heavy fog regimes.
                # Kernel size is tuned for ~512 px; adjust for higher resolutions.
                if vis <= 100:
                    apply_blur, kernel_size = True, 3
                    if 30 <= vis < 50:
                        kernel_size = 7
                    elif 50 <= vis < 70:
                        kernel_size = 11
                elif random.random() < 0.5:
                    apply_blur, kernel_size = True, 3

                if apply_blur:
                    depth_to_blur = torch.from_numpy(sharp_depth_map_inverted_np).to(device).float().unsqueeze(0)
                    blurred_depth = TF.gaussian_blur(depth_to_blur, kernel_size=[kernel_size, kernel_size])
                    current_depth_tensor = blurred_depth.squeeze(0) / 255.0

            # Random baseline optical thickness for variety.
            baseline_tau_value = random.uniform(0.0, 0.2)

            # Optional non-uniform haze via smoke texture.
            smoke_mask = None
            if use_smoke:
                smoke_mask = sample_smoke_mask(smoke_texture_paths, H_img, W_img, device)

            final_image_bgr = compose_atmosphere_gpu(
                img_enc_tensor=original_image_tensor,
                depth_norm_tensor=current_depth_tensor,
                baseline_tau=baseline_tau_value,
                apply_dithering=apply_dithering,
                smoke_mask=smoke_mask,
                smoke_strength=smoke_strength,
                **render_params,
            )

            output_filename = render_params["filename"]
            out_uri = haze_output_uri(output_uri, base_name, output_filename)
            write_image_uri(out_uri, final_image_bgr)

        return True
    except Exception as e:
        print(f"\n!! ERROR processing {base_name} on device {device}: {e}")
        return False
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# ## 5. MAIN
# ==============================================================================
DEFAULT_SMOKE_DIR = str(Path(__file__).resolve().parent.parent / "smoke_texture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch synthesize atmospheric effects with multi-GPU support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Input source: local directory or s3://bucket/prefix "
                             "containing img/ and depth/ subdirs.")
    parser.add_argument("--output", required=True,
                        help="Output destination: local directory or s3://bucket/prefix. "
                             "Variants are written under <output>/haze_v2/<base>/.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Worker processes per GPU (default: 2).")
    parser.add_argument("--apply-dithering", action="store_true",
                        help="Add ±1 LSB uniform noise to reduce 8-bit color banding.")

    # Smoke-texture blend (off by default).
    parser.add_argument("--use-smoke-texture", action="store_true",
                        help="Modulate haze density by a random crop of a smoke texture "
                             "for non-uniform haze. Disabled by default.")
    parser.add_argument("--smoke-texture-dir", default=DEFAULT_SMOKE_DIR,
                        help=f"Directory of smoke texture PNGs (default: {DEFAULT_SMOKE_DIR}).")
    parser.add_argument("--smoke-strength", type=float, default=0.5,
                        help="Blend strength in [0, 1]; 0 = disabled, 1 = full texture "
                             "modulation. Default 0.5.")
    args = parser.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing context already started. Assuming 'spawn' method.")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} available GPU(s).")
    else:
        num_gpus = 0
        print("No CUDA-enabled GPU found, falling back to CPU.")

    total_workers = args.num_workers * num_gpus if num_gpus > 0 else args.num_workers
    total_workers = min(total_workers, os.cpu_count() or 1)
    print(f"Using {total_workers} CPU processes across {num_gpus} GPU(s).")
    print("Color adjustment: 'sky-blue' is desaturated for all haze effects.")

    # Resolve smoke textures up-front so workers all get the same list.
    smoke_texture_paths: List[str] = []
    if args.use_smoke_texture:
        smoke_texture_paths = discover_smoke_textures(args.smoke_texture_dir)
        if not smoke_texture_paths:
            print(f"WARN: --use-smoke-texture set but no textures found in "
                  f"{args.smoke_texture_dir}. Falling back to uniform haze.")
        else:
            print(f"Smoke-texture blend enabled with {len(smoke_texture_paths)} "
                  f"textures from {args.smoke_texture_dir} (strength={args.smoke_strength}).")

    print(f"\n--- Matching images and depth maps in '{args.input}' ---")
    init_s3_client()  # in case input/output is S3 in main process
    pairs = list_input_pairs(args.input)
    print(f"Found {len(pairs)} (img, depth) pairs.")
    if not pairs:
        sys.exit("Nothing to do.")

    tasks = []
    for i, (img_uri, depth_uri, base_name) in enumerate(pairs):
        gpu_id = i % num_gpus if num_gpus > 0 else -1
        tasks.append((
            img_uri, depth_uri, base_name, args.output, gpu_id,
            args.apply_dithering, smoke_texture_paths, args.smoke_strength,
        ))

    with mp.Pool(processes=total_workers, initializer=init_s3_client) as pool:
        with tqdm(total=len(tasks), desc="Synthesizing") as pbar:
            for _ in pool.imap_unordered(worker_task, tasks):
                pbar.update(1)
    print("\n--- Done ---")
