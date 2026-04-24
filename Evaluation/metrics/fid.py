"""
fid.py — Frechet Inception Distance via pyiqa.

OSDFace paper uses two FID variants on CelebA-Test (Table 1):
    FID(FFHQ): restored images vs FFHQ HQ set  — recovery of "real" face distr.
    FID(HQ):   restored images vs CelebA-Test HQ set — recovery of this dataset

and one FID on real-world datasets (Table 2):
    FID(FFHQ): vs FFHQ.

`pyiqa.create_metric('fid')` takes two folder paths and returns a scalar FID,
using Inception-V3 features (same as FFHQ-based papers). This is a dataset-
level metric — it cannot be computed per-image.
"""
from __future__ import annotations

import os
from typing import Optional

import torch


class FIDScorer:
    def __init__(self, device: torch.device | str = "cuda"):
        import pyiqa

        self.device = torch.device(device)
        self.metric = pyiqa.create_metric("fid", device=self.device)

    @torch.no_grad()
    def compute(
        self,
        restored_dir: str,
        reference_dir: Optional[str],
        label: str = "fid",
    ) -> dict[str, float]:
        """
        Args:
            restored_dir:  folder with restored PNG/JPG images
            reference_dir: folder with reference images (FFHQ or HQ GT).
                           If None or missing, returns NaN.
            label:         metric name in output dict (e.g. 'fid_ffhq', 'fid_hq')

        Returns:
            {label: float} — lower is better.
        """
        if reference_dir is None or not os.path.isdir(reference_dir):
            print(f"  [WARN] FID reference dir missing: {reference_dir!r}  "
                  f"→ {label}=NaN")
            return {label: float("nan")}
        if not os.path.isdir(restored_dir):
            print(f"  [WARN] FID restored dir missing: {restored_dir!r}")
            return {label: float("nan")}

        try:
            score = self.metric(restored_dir, reference_dir)
            score = float(score.item() if torch.is_tensor(score) else score)
        except Exception as e:
            print(f"  [WARN] FID failed ({label}): {e}")
            score = float("nan")
        return {label: score}


__all__ = ["FIDScorer"]
