"""
precompute_masks.py — Pre-compute face parsing masks for CRAFT training.

Runs the BiSeNet face parser on all images once and saves 16x16 region index
maps to disk. This avoids running the face parser during every training step
in Phases B and C.

Output structure:
    data_root/
    └── masks_16x16/
        ├── 00000_hq.pt       # (16, 16) int64 region indices for HQ image
        ├── 00000_lq.pt       # (16, 16) int64 region indices for LQ image
        └── ...

Usage:
    python precompute_masks.py --data_root /path/to/data/train \
        --parser_ckpt pretrained/79999_iter.pth

    # Specify GPU and batch size:
    python precompute_masks.py --data_root data/train \
        --parser_ckpt pretrained/79999_iter.pth \
        --batch_size 64 --device cuda
"""

import argparse
import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models.face_parser import FaceParser


class _ImageDataset(Dataset):
    """Simple dataset that loads images and returns (tensor, stem, suffix)."""

    def __init__(self, image_paths, suffix):
        self.image_paths = image_paths
        self.suffix = suffix
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(
            (512, 512),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        stem = os.path.splitext(os.path.basename(path))[0]
        # Remove _LQ suffix for consistent naming
        if stem.endswith("_LQ"):
            stem = stem[:-3]
        img = Image.open(path).convert("RGB")
        tensor = self.resize(self.to_tensor(img))  # (3, 512, 512) in [0, 1]
        return tensor, stem, self.suffix


def main():
    parser = argparse.ArgumentParser(description="Pre-compute face parsing masks")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--parser_ckpt", type=str, required=True)
    parser.add_argument("--hq_folder", type=str, default="images512x512")
    parser.add_argument("--lq_folder", type=str, default="LQ_images_512x512")
    parser.add_argument("--output_folder", type=str, default="masks_16x16")
    parser.add_argument("--target_h", type=int, default=16)
    parser.add_argument("--target_w", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Output directory
    out_dir = os.path.join(args.data_root, args.output_folder)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load face parser
    print(f"Loading face parser from {args.parser_ckpt}")
    face_parser = FaceParser(checkpoint_path=args.parser_ckpt).to(device)

    # Collect image paths
    hq_dir = os.path.join(args.data_root, args.hq_folder)
    lq_dir = os.path.join(args.data_root, args.lq_folder)

    hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.png")))
    lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))
    print(f"Found {len(hq_paths)} HQ images, {len(lq_paths)} LQ images")

    # Process both HQ and LQ
    datasets = []
    if hq_paths:
        datasets.append(_ImageDataset(hq_paths, "hq"))
    if lq_paths:
        datasets.append(_ImageDataset(lq_paths, "lq"))

    total_saved = 0
    for ds in datasets:
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        suffix = ds.suffix
        print(f"\nProcessing {suffix.upper()} images ({len(ds)} total)...")

        for batch_idx, (images, stems, _) in enumerate(loader):
            images = images.to(device)

            with torch.no_grad():
                region_indices = face_parser.get_region_indices(
                    images,
                    target_h=args.target_h,
                    target_w=args.target_w,
                )  # (B, target_h, target_w) int64

            # Save each mask
            for i, stem in enumerate(stems):
                out_path = os.path.join(out_dir, f"{stem}_{suffix}.pt")
                torch.save(region_indices[i].cpu(), out_path)
                total_saved += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)} done "
                      f"({total_saved} masks saved)")

    print(f"\nDone! Saved {total_saved} mask files to {out_dir}")


if __name__ == "__main__":
    main()
