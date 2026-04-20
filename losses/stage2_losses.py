"""
stage2_losses.py — CRAFT Stage 2 generator + discriminator loss composition.

Implements OSDFace Eqs. 12, 14, 15, 16. All losses are operated on images in
[0, 1] range for EA-DISTS and ArcFace, and on scaled VAE latents for the
adversarial terms. Call sites pass:

    generator_loss(I_H, I_hat_H,  z_H, z_hat_H,  discriminator)
    discriminator_loss(             z_H, z_hat_H,  discriminator)

The adversarial term operates on forward-diffused latents F(z, t) where
t ~ U{1, T} — the discriminator sees (noisy latent, t) at a random time
shared by real and fake within each step.

Loss weights (OSDFace defaults + common face-restoration ID weight):
    λ_MSE  = 1.0
    λ_per  = 1.0
    λ_ID   = 0.1
    λ_dis  = 0.8
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dists import EADists
from .arcface import ArcFaceID


# ----------------------------------------------------------------------
# Forward diffusion helper (for D input)
# ----------------------------------------------------------------------

def forward_diffuse(
    z: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    F(z, t) = √(ᾱ_t) · z + √(1 − ᾱ_t) · ε,  ε ∼ 𝒩(0, I).

    Args:
        z:              (B, C, H, W) clean latent.
        t:              (B,) long, indices into alphas_cumprod.
        alphas_cumprod: (T,) cumulative alpha products.

    Returns:
        z_noisy:        same shape as z.
    """
    ab = alphas_cumprod[t].to(z.dtype)                # (B,)
    sqrt_ab = ab.sqrt().view(-1, 1, 1, 1)
    sqrt_1m_ab = (1.0 - ab).sqrt().view(-1, 1, 1, 1)
    eps = torch.randn_like(z)
    return sqrt_ab * z + sqrt_1m_ab * eps


# ----------------------------------------------------------------------
# Stage-2 generator + discriminator loss composition
# ----------------------------------------------------------------------

class Stage2Loss(nn.Module):
    """
    Args:
        alphas_cumprod: (T,) tensor of cumulative alpha products from the SD
                        scheduler. Stored as a buffer (non-persistent).
        lambda_mse, lambda_per, lambda_id, lambda_dis: loss weights.
        enable_id:      If False, skips ArcFace loading and contributes 0
                        to the ID term. Useful for quick smoke tests.
    """

    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        lambda_mse: float = 1.0,
        lambda_per: float = 1.0,
        lambda_id:  float = 0.1,
        lambda_dis: float = 0.8,
        enable_id: bool = True,
    ):
        super().__init__()
        self.lambda_mse = float(lambda_mse)
        self.lambda_per = float(lambda_per)
        self.lambda_id  = float(lambda_id)
        self.lambda_dis = float(lambda_dis)
        self.enable_id = bool(enable_id)

        self.register_buffer(
            "alphas_cumprod", alphas_cumprod.clone().float(), persistent=False,
        )
        self.num_train_timesteps = int(alphas_cumprod.shape[0])

        self.ea_dists = EADists()
        self.arcface = ArcFaceID() if enable_id else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_01(self, x_11: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → [0, 1], clamped for EA-DISTS/ArcFace sanity."""
        return ((x_11 + 1.0) * 0.5).clamp(0.0, 1.0)

    def _sample_t(self, batch_size: int, device) -> torch.Tensor:
        """t ∼ U{1, ..., T-1}. Index 0 corresponds to ᾱ₀≈1 (clean), skip it."""
        return torch.randint(
            low=1, high=self.num_train_timesteps,
            size=(batch_size,), device=device, dtype=torch.long,
        )

    # ------------------------------------------------------------------
    # Generator loss
    # ------------------------------------------------------------------

    def generator_loss(
        self,
        I_H_11: torch.Tensor,
        I_hat_H_11: torch.Tensor,
        z_hat_H: torch.Tensor,
        discriminator: nn.Module,
    ) -> Tuple[torch.Tensor, dict]:
        """
        OSDFace Eq. 12, with L_G from Eq. 15.

        L_gen = λ_MSE · MSE(I_H, Î_H)
              + λ_per · L_EA-DISTS(I_H, Î_H)
              + λ_ID  · L_ID(I_H, Î_H)
              + λ_dis · L_G
        L_G = −E_t [ log D( F(ẑ_H, t) ) ]
        """
        logs: dict = {}

        # -- pixel-space MSE --
        mse = F.mse_loss(I_hat_H_11, I_H_11)
        logs["mse"] = mse.detach()

        # -- perceptual EA-DISTS (inputs in [0, 1]) --
        I_H_01 = self._to_01(I_H_11)
        I_hat_H_01 = self._to_01(I_hat_H_11)
        ea = self.ea_dists(I_hat_H_01, I_H_01)
        logs["ea_dists"] = ea.detach()

        # -- identity loss --
        if self.enable_id:
            id_loss = self.arcface(I_hat_H_01, I_H_01)
            logs["id"] = id_loss.detach()
        else:
            id_loss = torch.zeros((), device=mse.device, dtype=mse.dtype)
            logs["id"] = id_loss

        # -- adversarial G loss on latent --
        B = z_hat_H.shape[0]
        t = self._sample_t(B, z_hat_H.device)
        z_noisy_fake = forward_diffuse(z_hat_H, t, self.alphas_cumprod)
        d_logit_fake = discriminator(z_noisy_fake, t)
        # Non-saturating formulation: maximize log D(fake) = minimize softplus(−logit)
        gan_g = F.softplus(-d_logit_fake).mean()
        logs["gan_g"] = gan_g.detach()

        total = (
            self.lambda_mse * mse
            + self.lambda_per * ea
            + self.lambda_id * id_loss
            + self.lambda_dis * gan_g
        )
        logs["total_gen"] = total.detach()
        return total, logs

    # ------------------------------------------------------------------
    # Discriminator loss
    # ------------------------------------------------------------------

    def discriminator_loss(
        self,
        z_H: torch.Tensor,
        z_hat_H: torch.Tensor,
        discriminator: nn.Module,
    ) -> Tuple[torch.Tensor, dict]:
        """
        OSDFace Eq. 16 (non-saturating binary cross-entropy form):

            L_D = softplus(−D(F(z_H, t))) + softplus(D(F(ẑ_H.detach(), t)))

        Real and fake share the same sampled t within a step.
        """
        logs: dict = {}

        B = z_H.shape[0]
        t = self._sample_t(B, z_H.device)

        z_noisy_real = forward_diffuse(z_H.detach(), t, self.alphas_cumprod)
        z_noisy_fake = forward_diffuse(z_hat_H.detach(), t, self.alphas_cumprod)

        d_real = discriminator(z_noisy_real, t)
        d_fake = discriminator(z_noisy_fake, t)

        loss_real = F.softplus(-d_real).mean()   # want D(real) → large
        loss_fake = F.softplus(d_fake).mean()    # want D(fake) → small
        total = loss_real + loss_fake

        logs["d_real"] = loss_real.detach()
        logs["d_fake"] = loss_fake.detach()
        logs["d_real_logit"] = d_real.detach().mean()
        logs["d_fake_logit"] = d_fake.detach().mean()
        logs["total_dis"] = total.detach()
        return total, logs
