"""
dists.py — DISTS + Edge-Aware DISTS loss for CRAFT Stage 2.

Wraps the DISTS implementation from `pyiqa` (Deep Image Structure and Texture
Similarity, Ding et al. 2022) and adds the edge-aware variant from OSDFace:

    L_EA-DISTS(Î, I) = DISTS(Î, I) + DISTS(Sobel(Î), Sobel(I))          (Eq. 14)

pyiqa ships the original paper's learned α/β weights, so the metric is
training-quality out of the box. Gradients flow through to the inputs.

Inputs are expected in [0, 1] range, (B, 3, H, W).

Usage:
    ea = EADists().to(device)
    loss = ea(pred_01, target_01)       # scalar
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Sobel edge extractor
# ----------------------------------------------------------------------

class Sobel(nn.Module):
    """
    3x3 Sobel edge magnitude per RGB channel.

    Output has the same shape as input, (B, 3, H, W), values in [0, 1] after
    min-max scaling per image (to keep the downstream DISTS call in range).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2).contiguous()
        # (2, 1, 3, 3) — one kernel for x, one for y
        kernel = torch.cat([kx, ky], dim=0)
        self.register_buffer("kernel", kernel, persistent=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) in [0, 1].

        Returns:
            (B, 3, H, W) edge magnitude, scaled to [0, 1] per-image.
        """
        B, C, H, W = x.shape
        # Apply sobel per channel (group convolution)
        # reshape (B,C,H,W) → (B*C,1,H,W), conv, then back
        flat = x.reshape(B * C, 1, H, W)
        gx_gy = F.conv2d(flat, self.kernel, padding=1)         # (B*C, 2, H, W)
        mag = torch.sqrt(gx_gy.pow(2).sum(dim=1, keepdim=True) + self.eps)
        mag = mag.reshape(B, C, H, W)
        # Per-image min-max scale to [0, 1] to keep DISTS in a sane range
        mag_flat = mag.reshape(B, -1)
        mn = mag_flat.min(dim=1, keepdim=True).values
        mx = mag_flat.max(dim=1, keepdim=True).values
        denom = (mx - mn).clamp_min(self.eps)
        mag = (mag_flat - mn) / denom
        mag = mag.reshape(B, C, H, W)
        return mag


# ----------------------------------------------------------------------
# DISTS + EA-DISTS
# ----------------------------------------------------------------------

class EADists(nn.Module):
    """
    Edge-Aware DISTS = DISTS(Î, I) + DISTS(Sobel(Î), Sobel(I)).

    Inputs in [0, 1], shape (B, 3, H, W). Returns a scalar (batch-mean).

    Internally uses `pyiqa.create_metric('dists', as_loss=True)` which loads
    the original paper's learned α/β and keeps the pretrained VGG16 frozen.
    """

    def __init__(self):
        super().__init__()
        # Lazy import so this file is importable even if pyiqa is missing
        import pyiqa
        self.dists = pyiqa.create_metric(
            "dists", as_loss=True, device="cpu",
        )
        # Ensure frozen (metric network is just a VGG feature extractor)
        for p in self.dists.parameters():
            p.requires_grad_(False)
        self.dists.eval()
        self.sobel = Sobel()

    def train(self, mode: bool = True):
        # Keep the internal DISTS VGG in eval mode regardless of outer mode
        super().train(mode)
        self.dists.eval()
        return self

    def forward(
        self,
        pred_01: torch.Tensor,
        target_01: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_01, target_01: (B, 3, H, W) in [0, 1].
        Returns:
            scalar loss = mean(DISTS(pred, target) + DISTS(Sobel(pred), Sobel(target))).
        """
        # pyiqa's dists(as_loss=True) returns scalar (1-dim) per call.
        # We mean it to be defensive.
        l_img = self.dists(pred_01, target_01).mean()
        l_edge = self.dists(self.sobel(pred_01), self.sobel(target_01)).mean()
        return l_img + l_edge
