"""
identity.py — Face identity distance ("Deg.") via ArcFace.

Implements the "Deg." column of Table 1 in the OSDFace paper: the angular
distance (in degrees) between ArcFace embeddings of the restored face and the
HQ ground-truth face, averaged over all images.

    cos_sim = <f(restored), f(hq)> / (||f(restored)|| * ||f(hq)||)
    deg     = arccos(cos_sim) * 180 / pi   (in degrees)

Smaller "Deg." means closer to the ground-truth identity.

Uses facexlib's ArcFace (ResNet50, pretrained on MS-Celeb-1M) — auto-downloaded
on first use into ~/.cache/torch/hub/.

The 512×512 inputs are resized to 112×112 and shifted to the ArcFace input
convention (RGB, range [-1, 1]). We assume the test faces are already aligned
(CelebA-Test / LFW cropped_faces both are), so we skip re-detection /
re-alignment to avoid introducing extra failure modes on restored images.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class IdentityDistance:
    """Wraps facexlib's ArcFace recognition model for OSDFace's `Deg.` metric."""

    def __init__(self, device: torch.device | str = "cuda"):
        from facexlib.recognition import init_recognition_model

        self.device = torch.device(device)
        # Downloads arcface_resnet18.pth on first call into ~/.cache.
        self.net = init_recognition_model("arcface", device=self.device)
        self.net.eval()

    @torch.no_grad()
    def _embed(self, imgs_01: torch.Tensor) -> torch.Tensor:
        """
        imgs_01: (B, 3, H, W) RGB in [0, 1].
        Returns: (B, 512) L2-normalised identity embeddings.
        """
        x = imgs_01.to(self.device).float().clamp(0, 1)
        if x.shape[-2:] != (112, 112):
            x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        # ArcFace expects RGB in [-1, 1]
        x = x * 2.0 - 1.0
        feats = self.net(x)
        return F.normalize(feats, p=2, dim=1)

    @torch.no_grad()
    def compute(
        self,
        restored_01: torch.Tensor,
        hq_01: torch.Tensor,
    ) -> dict[str, float]:
        """
        Args:
            restored_01: (B, 3, H, W) restored faces in [0, 1]
            hq_01:       (B, 3, H, W) HQ GT faces in [0, 1]

        Returns:
            dict with:
                'deg'     — mean angular distance in degrees (paper's "Deg." ↓)
                'cos_sim' — mean cosine similarity (↑, convenience)
        """
        f_r = self._embed(restored_01)   # (B, 512)
        f_h = self._embed(hq_01)
        cos = (f_r * f_h).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)   # (B,)
        deg = torch.acos(cos) * (180.0 / math.pi)                     # (B,)
        return {
            "deg":     float(deg.mean().item()),
            "cos_sim": float(cos.mean().item()),
        }


__all__ = ["IdentityDistance"]
