"""
iqa.py — Image Quality Assessment metrics via pyiqa.

Covers the per-image metrics used in the OSDFace paper (Tables 1 & 2):

    Full-reference (needs HQ ground truth):
        LPIPS ↓, DISTS ↓, PSNR ↑, SSIM ↑

    No-reference (restored image only):
        MUSIQ ↑, NIQE ↓, CLIPIQA ↑, MANIQA ↑

All metrics accept two tensors in [0, 1] of shape (B, 3, H, W). Full-reference
metrics require the second tensor (HQ GT); no-reference metrics ignore it.

Public API:
    IQAMetrics(device, full_ref=True, no_ref=True)  — wraps the pyiqa handles
    .compute(restored_01, hq_01=None)               — returns {metric: float}
"""
from __future__ import annotations

from typing import Optional

import torch


# Metric names → pyiqa identifiers
_FULL_REF = {
    "lpips":  "lpips",      # VGG-based perceptual
    "dists":  "dists",      # Deep image structure/texture similarity
    "psnr":   "psnr",
    "ssim":   "ssim",
}
_NO_REF = {
    "musiq":   "musiq",
    "niqe":    "niqe",
    "clipiqa": "clipiqa",
    "maniqa":  "maniqa",
}


class IQAMetrics:
    def __init__(
        self,
        device: torch.device | str = "cuda",
        full_ref: bool = True,
        no_ref: bool = True,
        metrics: Optional[list[str]] = None,
    ):
        """
        Args:
            device:   torch device
            full_ref: load full-reference metrics (LPIPS/DISTS/PSNR/SSIM)
            no_ref:   load no-reference metrics (MUSIQ/NIQE/CLIPIQA/MANIQA)
            metrics:  optional subset of metric names; overrides full_ref/no_ref
                      when given (e.g. ['lpips', 'dists', 'niqe']).
        """
        import pyiqa

        self.device = torch.device(device)
        self.metrics: dict[str, object] = {}

        if metrics is None:
            wanted: list[str] = []
            if full_ref: wanted += list(_FULL_REF.keys())
            if no_ref:   wanted += list(_NO_REF.keys())
        else:
            wanted = list(metrics)

        for name in wanted:
            pyiqa_id = _FULL_REF.get(name) or _NO_REF.get(name)
            if pyiqa_id is None:
                raise ValueError(f"Unknown IQA metric: {name}")
            try:
                m = pyiqa.create_metric(pyiqa_id, device=self.device, as_loss=False)
                m.eval()
                self.metrics[name] = m
            except Exception as e:
                print(f"  [WARN] Failed to load {name!r}: {e}")

        self.full_ref_names = [n for n in self.metrics if n in _FULL_REF]
        self.no_ref_names   = [n for n in self.metrics if n in _NO_REF]

    @torch.no_grad()
    def compute(
        self,
        restored_01: torch.Tensor,
        hq_01: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """
        Compute all loaded metrics on a single batch.

        Args:
            restored_01: (B, 3, H, W) in [0, 1], restored images
            hq_01:       (B, 3, H, W) in [0, 1] HQ GT (required for full-ref)

        Returns:
            dict of metric_name → scalar value (mean over the batch).
        """
        restored_01 = restored_01.to(self.device).clamp(0, 1).float()
        if hq_01 is not None:
            hq_01 = hq_01.to(self.device).clamp(0, 1).float()

        out: dict[str, float] = {}
        for name, metric in self.metrics.items():
            try:
                if name in _FULL_REF:
                    if hq_01 is None:
                        continue
                    val = metric(restored_01, hq_01)
                else:
                    val = metric(restored_01)
                val = val.mean().item() if torch.is_tensor(val) else float(val)
                out[name] = float(val)
            except Exception as e:
                print(f"  [WARN] {name} failed: {e}")
                out[name] = float("nan")
        return out


__all__ = ["IQAMetrics"]
