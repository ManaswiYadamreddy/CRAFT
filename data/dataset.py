"""
dataset.py — Dataset for CRAFT Stage 1 VQVAE training.

Loads paired HQ/LQ face images from the FFHQ dataset structure:

    train/
    ├── images512x512/          # HQ images (FFHQ originals)
    │   ├── 00000.png
    │   ├── 00001.png
    │   └── ...
    └── LQ_images_512x512/      # LQ images (synthetically degraded)
        ├── 00000_LQ.png
        ├── 00001_LQ.png
        └── ...

Images are loaded as tensors in [0, 1] range (for face parser compatibility)
or optionally normalized to [-1, 1] (for encoder/decoder training).
"""

import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FFHQPairedDataset(Dataset):
    """
    Paired HQ/LQ dataset for CRAFT Stage 1 training.
    
    Each sample returns a dict with:
        'hq':       (3, 512, 512) HQ image tensor in [-1, 1]
        'lq':       (3, 512, 512) LQ image tensor in [-1, 1]
        'hq_01':    (3, 512, 512) HQ image tensor in [0, 1] (for face parser)
        'lq_01':    (3, 512, 512) LQ image tensor in [0, 1] (for face parser)
        'filename': str, e.g. '00000'
    
    Args:
        data_root:   Path to 'train/' directory containing the image folders.
        hq_folder:   Name of HQ image subfolder (default: 'images512x512').
        lq_folder:   Name of LQ image subfolder (default: 'LQ_images_512x512').
        resolution:  Target resolution (default: 512).
        hq_only:     If True, only load HQ images (for HQ VQVAE pre-training).
                     'lq' and 'lq_01' will be None.
    """

    def __init__(
        self,
        data_root,
        hq_folder="images512x512",
        lq_folder="LQ_images_512x512",
        resolution=512,
        hq_only=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.hq_dir = os.path.join(data_root, hq_folder)
        self.lq_dir = os.path.join(data_root, lq_folder)
        self.resolution = resolution
        self.hq_only = hq_only

        # Discover HQ images
        hq_paths = sorted(glob.glob(os.path.join(self.hq_dir, "*.png")))
        if not hq_paths:
            raise FileNotFoundError(
                f"No .png files found in {self.hq_dir}. "
                f"Check that data_root='{data_root}' is correct."
            )

        # Build list of (hq_path, lq_path, stem) tuples
        self.samples = []
        for hq_path in hq_paths:
            stem = os.path.splitext(os.path.basename(hq_path))[0]  # e.g. '00000'
            lq_path = os.path.join(self.lq_dir, f"{stem}_LQ.png")

            if not hq_only and not os.path.exists(lq_path):
                continue  # skip if LQ counterpart missing

            self.samples.append((hq_path, lq_path, stem))

        if not self.samples:
            raise FileNotFoundError(
                f"No valid HQ/LQ pairs found. "
                f"HQ dir: {self.hq_dir}, LQ dir: {self.lq_dir}"
            )

        # Transforms
        self.to_tensor = transforms.ToTensor()  # PIL → [0, 1] tensor
        self.resize = transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hq_path, lq_path, stem = self.samples[idx]

        # Load HQ
        hq_img = Image.open(hq_path).convert("RGB")
        hq_01 = self.resize(self.to_tensor(hq_img))  # (3, 512, 512) in [0, 1]
        hq = hq_01 * 2.0 - 1.0  # [-1, 1]

        output = {
            "hq": hq,
            "hq_01": hq_01,
            "filename": stem,
        }

        # Load LQ
        if not self.hq_only:
            lq_img = Image.open(lq_path).convert("RGB")
            lq_01 = self.resize(self.to_tensor(lq_img))
            lq = lq_01 * 2.0 - 1.0
            output["lq"] = lq
            output["lq_01"] = lq_01

        return output