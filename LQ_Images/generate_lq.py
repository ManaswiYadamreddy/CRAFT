"""
Generate Low-Quality (LQ) face images from High-Quality (HQ) images
using the dual-stage degradation model from WaveFace (Eq. 15), as used by OSDFace.

Degradation pipeline (applied twice for dual-stage):
    x = { [(y ⊗ k_σ) ↓_s + n_δ]_JPEG_q } ↑_s

Parameters sampled uniformly:
    σ ∈ [0.1, 10]   - Gaussian blur kernel std
    s ∈ [0.8, 16]    - downsampling scale
    δ ∈ [0, 15]      - Gaussian noise std
    q ∈ [40, 95]      - JPEG quality factor
"""

import os
import io
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import multiprocessing as mp
from functools import partial


def gaussian_blur(img_np, sigma):
    """Apply Gaussian blur using PIL."""
    img = Image.fromarray(img_np)
    # Kernel size should be odd and large enough for the sigma
    kernel_size = int(np.ceil(sigma * 6)) | 1  # ensure odd
    kernel_size = max(kernel_size, 3)
    img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(img)


def downsample(img_np, scale):
    """Downsample image by scale factor."""
    img = Image.fromarray(img_np)
    h, w = img_np.shape[:2]
    new_h = max(int(h / scale), 1)
    new_w = max(int(w / scale), 1)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    return np.array(img), (w, h)


def upsample(img_np, target_size):
    """Upsample image back to target size (w, h)."""
    img = Image.fromarray(img_np)
    img = img.resize(target_size, Image.BICUBIC)
    return np.array(img)


def add_gaussian_noise(img_np, delta):
    """Add Gaussian noise with std delta."""
    if delta == 0:
        return img_np
    noise = np.random.randn(*img_np.shape) * delta
    noisy = np.clip(img_np.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy


def jpeg_compress(img_np, quality):
    """Apply JPEG compression."""
    img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    img = Image.open(buffer)
    return np.array(img)


def single_stage_degradation(img_np, target_size=(512, 512)):
    """
    Apply one stage of degradation:
        x = { [(y ⊗ k_σ) ↓_s + n_δ]_JPEG_q } ↑_s

    Parameters sampled uniformly per the WaveFace paper.
    """
    sigma = np.random.uniform(0.1, 10)
    scale = np.random.uniform(0.8, 16)
    delta = np.random.uniform(0, 15)
    quality = np.random.uniform(40, 95)

    # 1. Gaussian blur
    img = gaussian_blur(img_np, sigma)

    # 2. Downsample
    img, original_size = downsample(img, scale)

    # 3. Add Gaussian noise
    img = add_gaussian_noise(img, delta)

    # 4. JPEG compression
    img = jpeg_compress(img, quality)

    # 5. Upsample back to target size
    img = upsample(img, target_size)

    return img


def dual_stage_degradation(img_np, target_size=(512, 512)):
    """
    Apply dual-stage degradation (RealESRGAN-style), where the
    single-stage degradation pipeline is applied twice.
    """
    img = single_stage_degradation(img_np, target_size)
    img = single_stage_degradation(img, target_size)
    return img


def process_image(filename, input_dir, output_dir):
    """Process a single image: load, degrade, save."""
    input_path = os.path.join(input_dir, filename)

    # Build output filename with _LQ suffix
    stem = Path(filename).stem
    ext = Path(filename).suffix
    out_filename = f"{stem}_LQ{ext}"
    output_path = os.path.join(output_dir, out_filename)

    # Skip if already exists
    if os.path.exists(output_path):
        return f"SKIP {filename}"

    try:
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img)

        # Apply single-stage degradation (WaveFace Eq. 15)
        lq_np = single_stage_degradation(img_np, target_size=(512, 512))

        lq_img = Image.fromarray(lq_np)
        lq_img.save(output_path, quality=95)

        return f"OK   {filename}"
    except Exception as e:
        return f"FAIL {filename}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate LQ face images using WaveFace dual-stage degradation"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/projectnb/cs585/projects/craft/data/train/images512x512",
        help="Path to HQ images directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/projectnb/cs585/projects/craft/data/train/LQ_images_512x512",
        help="Path to save LQ images",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect image files
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    filenames = sorted(
        f
        for f in os.listdir(args.input_dir)
        if Path(f).suffix.lower() in valid_exts
    )

    print(f"Found {len(filenames)} images in {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using {args.num_workers} workers")

    # Process images in parallel
    func = partial(
        process_image,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    done = 0
    failed = 0
    with mp.Pool(args.num_workers) as pool:
        for result in pool.imap_unordered(func, filenames, chunksize=32):
            done += 1
            if result.startswith("FAIL"):
                failed += 1
                print(result)
            if done % 1000 == 0:
                print(f"Progress: {done}/{len(filenames)}  (failed: {failed})")

    print(f"\nDone. Processed {done} images, {failed} failures.")


if __name__ == "__main__":
    main()