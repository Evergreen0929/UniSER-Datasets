import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import noise
from numba import jit


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "smoke_texture"


# --- 1. Base noise generation ---
def generate_raw_noise(width, height, scale, octaves, seed):
    array = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            array[y, x] = noise.pnoise2(x / scale, y / scale,
                                        octaves=octaves,
                                        persistence=0.5,
                                        lacunarity=2.0,
                                        repeatx=width,
                                        repeaty=height,
                                        base=seed)
    return array


# --- 2. Vector field and path-blur kernel ---
def generate_vector_field(width, height, scale, octaves, seed):
    x_components = generate_raw_noise(width, height, scale, octaves, seed)
    y_components = generate_raw_noise(width, height, scale, octaves, seed + 1)

    magnitude = np.sqrt(x_components**2 + y_components**2)
    magnitude[magnitude == 0] = np.finfo(float).eps

    dx = x_components / magnitude
    dy = y_components / magnitude

    return dx, dy


# Edge handling fix: clamp coordinates so bilinear interpolation never reads
# past the image bounds.
@jit(nopython=True)
def apply_path_blur_kernel(image_array, dx_array, dy_array, steps, step_length):
    height, width = image_array.shape
    blurred_array = image_array.astype(np.float32)

    # Pull the bounds in by a tiny epsilon so x_floor+1 / y_floor+1 are always
    # safe to index during bilinear interpolation.
    width_limit = width - 1.000001
    height_limit = height - 1.000001

    for _ in range(steps):
        prev_step_array = blurred_array.copy()
        for y in range(height):
            for x in range(width):
                dx = dx_array[y, x] * step_length
                dy = dy_array[y, x] * step_length
                x2, y2 = x + dx, y + dy

                # Clamp-to-edge boundary handling.
                if x2 < 0: x2 = 0
                if y2 < 0: y2 = 0
                if x2 > width_limit: x2 = width_limit
                if y2 > height_limit: y2 = height_limit

                # Bilinear interpolation (clamp above makes extra bounds check unnecessary).
                x_floor, y_floor = int(x2), int(y2)
                x_frac, y_frac = x2 - x_floor, y2 - y_floor

                p00 = prev_step_array[y_floor, x_floor]
                p10 = prev_step_array[y_floor, x_floor + 1]
                p01 = prev_step_array[y_floor + 1, x_floor]
                p11 = prev_step_array[y_floor + 1, x_floor + 1]

                inter_val_1 = p00 * (1 - x_frac) + p10 * x_frac
                inter_val_2 = p01 * (1 - x_frac) + p11 * x_frac
                neighbor_val = inter_val_1 * (1 - y_frac) + inter_val_2 * y_frac

                blurred_array[y, x] = (prev_step_array[y, x] * 0.5) + (neighbor_val * 0.5)

    return np.clip(blurred_array, 0, 255).astype(np.uint8)


# --- 3. Batch generation entry point ---
def batch_generate(num_images, width, height, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Images will be saved to: {os.path.abspath(output_folder)}")

    PARAM_RANGES = {
        'base_scale': (180.0, 500.0),
        'base_octaves': (5, 7),
        'angle_scale': (350.0, 550.0),
        'angle_octaves': (3, 5),
        'blur_steps': (15, 35),
        'blur_length': (1.0, 2.0),
    }

    tasks = [True] * (num_images // 2) + [False] * (num_images - num_images // 2)
    random.shuffle(tasks)

    total_start_time = time.time()

    for i in range(num_images):
        img_start_time = time.time()
        apply_blur = tasks[i]

        large_seed = int(time.time() * 1000)
        safe_seed = large_seed % 128

        print(f"\n--- Generating image {i + 1}/{num_images} (safe seed: {safe_seed}) ---")

        base_scale = random.uniform(*PARAM_RANGES['base_scale'])
        base_octaves = random.randint(*PARAM_RANGES['base_octaves'])

        print("Step 1/X: Generating base smoke noise...")
        base_noise = generate_raw_noise(width, height, base_scale, base_octaves, safe_seed)
        base_noise_normalized = (base_noise - np.min(base_noise)) / (np.max(base_noise) - np.min(base_noise))
        final_array = (base_noise_normalized * 255).astype(np.uint8)

        filename_suffix = "base"

        if apply_blur:
            base_image_for_blend = final_array.copy()

            print("Step 2/3: Applying path blur...")
            filename_suffix = "pathblur"

            angle_scale = random.uniform(*PARAM_RANGES['angle_scale'])
            angle_octaves = random.randint(*PARAM_RANGES['angle_octaves'])
            blur_steps = random.randint(*PARAM_RANGES['blur_steps'])
            blur_length = random.uniform(*PARAM_RANGES['blur_length'])

            dx_array, dy_array = generate_vector_field(width, height, angle_scale, angle_octaves, safe_seed + 1)
            blurred_array = apply_path_blur_kernel(final_array, dx_array, dy_array, blur_steps, blur_length)

            print("Step 3/3: Blending with the base image...")
            blend_ratio = random.uniform(0.3, 0.7)
            print(f"Blend ratio: {blend_ratio:.2f}")

            blurred_float = blurred_array.astype(np.float32)
            base_float = base_image_for_blend.astype(np.float32)

            blended_float = (blurred_float * blend_ratio) + (base_float * (1.0 - blend_ratio))

            final_array = np.clip(blended_float, 0, 255).astype(np.uint8)

        else:
            print("Step 2/2: Skipping path blur / blend.")

        img_to_save = Image.fromarray(final_array, 'L')
        filename = f"smoke_{i+1:04d}_{filename_suffix}.png"
        filepath = os.path.join(output_folder, filename)
        img_to_save.save(filepath)

        img_end_time = time.time()
        print(f"Image {i + 1} saved to {filepath} (elapsed: {img_end_time - img_start_time:.2f}s)")

    total_end_time = time.time()
    print(f"\n--- All done ---")
    print(f"Generated {num_images} images in total. Total time: {(total_end_time - total_start_time) / 60:.2f} min.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Generate Perlin-noise smoke textures.")
    ap.add_argument("--num-images", type=int, default=2000,
                    help="How many textures to generate (default: 2000).")
    ap.add_argument("--resolution", type=int, default=2048,
                    help="Output resolution (square, default: 2048).")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help=f"Where to write textures (default: {DEFAULT_OUTPUT_DIR}).")
    args = ap.parse_args()

    batch_generate(
        num_images=args.num_images,
        width=args.resolution,
        height=args.resolution,
        output_folder=args.output_dir,
    )
