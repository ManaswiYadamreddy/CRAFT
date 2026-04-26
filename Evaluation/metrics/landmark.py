"""
landmark.py — Landmark distance (LMD) for the OSDFace paper Table 1.

LMD is the mean L2 distance (in pixels at 512×512) between 68 facial landmarks
of the restored face and those of the HQ ground truth, averaged over all
68 points, then averaged across images.

Uses `face-alignment` (Bulat & Tzimiropoulos, 2017) — FAN 2D 68-point detector.
Weights auto-download on first use.

Failure modes: if either restored or HQ face cannot be located / landmarked,
that image is skipped. `compute()` returns the mean over successful images plus
a `n_valid` count for reporting.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class LandmarkDistance:
    def __init__(self, device: torch.device | str = "cuda"):
        import face_alignment

        self.device = torch.device(device)
        # FAN 2D 68-point predictor
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=str(self.device),
            flip_input=False,
        )

    def _landmarks(self, img_uint8_hwc: np.ndarray) -> Optional[np.ndarray]:
        """Return (68, 2) float numpy, or None if detection failed."""
        lms = self.fa.get_landmarks_from_image(img_uint8_hwc)
        if lms is None or len(lms) == 0:
            return None
        # Take the largest face if multiple are detected.
        # NumPy 2.0 removed `arr.ptp()`; use the free function np.ptp(arr).
        if len(lms) > 1:
            areas = [(np.ptp(lm[:, 0]) * np.ptp(lm[:, 1])) for lm in lms]
            lms = [lms[int(np.argmax(areas))]]
        return lms[0].astype(np.float32)

    @staticmethod
    def _to_uint8_hwc(img_01: torch.Tensor) -> np.ndarray:
        arr = img_01.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        return (arr * 255.0).astype(np.uint8)

    @torch.no_grad()
    def compute(
        self,
        restored_01: torch.Tensor,
        hq_01: torch.Tensor,
    ) -> dict[str, float]:
        """
        Args:
            restored_01: (B, 3, H, W) in [0, 1]
            hq_01:       (B, 3, H, W) in [0, 1]

        Returns:
            {'lmd': mean pixel distance ↓, 'n_valid': count, 'n_failed': count}
            NaN if every detection failed.
        """
        B = restored_01.shape[0]
        per_image = []
        n_failed = 0
        for i in range(B):
            lm_r = self._landmarks(self._to_uint8_hwc(restored_01[i]))
            lm_h = self._landmarks(self._to_uint8_hwc(hq_01[i]))
            if lm_r is None or lm_h is None:
                n_failed += 1
                continue
            # Mean per-landmark Euclidean distance
            d = np.linalg.norm(lm_r - lm_h, axis=1).mean()
            per_image.append(float(d))
        if not per_image:
            return {"lmd": float("nan"), "n_valid": 0, "n_failed": n_failed}
        return {
            "lmd":      float(np.mean(per_image)),
            "n_valid":  len(per_image),
            "n_failed": n_failed,
        }


__all__ = ["LandmarkDistance"]
