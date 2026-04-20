"""
arcface.py — Facial-identity loss for CRAFT Stage 2 (OSDFace Eq. 13).

Wraps a pretrained ArcFace IR-50 (from facexlib) and computes:

    L_ID(I_H, Î_H) = 1 − cos( F(I_H), F(Î_H) )

Inputs are (B, 3, 512, 512) in [0, 1] — we resize to 112×112 and normalize
before running ArcFace. FFHQ faces are already centered and cropped, so we
skip 5-landmark alignment and just resize; the same transform applies to
both inputs so the cosine similarity remains a consistent training signal.

Usage:
    arc = ArcFaceID().to(device)      # loads pretrained weights lazily
    loss = arc(pred_01, target_01)    # scalar
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceID(nn.Module):
    """
    ArcFace identity loss. Model weights are loaded lazily on first forward
    to keep `__init__` cheap (so it's OK to construct before moving to GPU).

    Args:
        input_size: target size for the ArcFace network (default: 112).
        half_precision: whether to load the ArcFace weights in half precision.
    """

    _ARCFACE_MEAN = 127.5 / 255.0
    _ARCFACE_STD = 128.0 / 255.0

    def __init__(
        self,
        input_size: int = 112,
        half_precision: bool = False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.half_precision = bool(half_precision)
        self._arcface: nn.Module | None = None

    # ------------------------------------------------------------------
    # Lazy builder
    # ------------------------------------------------------------------

    def _build_arcface(self, device, dtype):
        """Lazy: defer heavy import + download to first real call."""
        from facexlib.recognition import init_recognition_model
        arcface = init_recognition_model(
            "arcface",
            half=self.half_precision,
            device=device,
        )
        for p in arcface.parameters():
            p.requires_grad_(False)
        arcface.eval()
        self._arcface = arcface

    def train(self, mode: bool = True):
        super().train(mode)
        if self._arcface is not None:
            self._arcface.eval()
        return self

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, img_01: torch.Tensor) -> torch.Tensor:
        """
        (B, 3, H, W) in [0, 1] → (B, 3, 112, 112) normalized for ArcFace.

        ArcFace training normalization: (pixel − 127.5) / 128 with pixel in
        [0, 255]. In our [0, 1] input: (x − 0.5) / (128/255) = (x − 0.5) / 0.5020.
        """
        x = F.interpolate(
            img_01, size=(self.input_size, self.input_size),
            mode="bicubic", align_corners=False, antialias=True,
        )
        x = (x - self._ARCFACE_MEAN) / self._ARCFACE_STD
        return x

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, img_01: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [0, 1] → (B, 512) L2-normalized ArcFace embedding."""
        if self._arcface is None:
            self._build_arcface(img_01.device, img_01.dtype)
        x = self._preprocess(img_01)
        # Run ArcFace in fp32 for numerical stability regardless of AMP context
        with torch.autocast(device_type=x.device.type, enabled=False):
            emb = self._arcface(x.float())
        emb = F.normalize(emb, dim=-1)
        return emb

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def forward(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_01, target_01: (B, 3, H, W) in [0, 1].
        Returns:
            scalar = mean(1 − cos(F(pred), F(target))).
        """
        emb_pred = self.embed(pred_01)
        emb_target = self.embed(target_01)
        cos_sim = (emb_pred * emb_target).sum(dim=-1)  # (B,)
        return (1.0 - cos_sim).mean()
