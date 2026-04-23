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
        lambda_region_l1: float = 0.0,
        enable_id: bool = True,
        t_max_dis: int | None = None,
    ):
        super().__init__()
        self.lambda_mse = float(lambda_mse)
        self.lambda_per = float(lambda_per)
        self.lambda_id  = float(lambda_id)
        self.lambda_dis = float(lambda_dis)
        self.lambda_region_l1 = float(lambda_region_l1)
        self.enable_id = bool(enable_id)

        self.register_buffer(
            "alphas_cumprod", alphas_cumprod.clone().float(), persistent=False,
        )
        self.num_train_timesteps = int(alphas_cumprod.shape[0])
        # Cap the adversarial-t upper bound. None/>=T falls back to T-1.
        self.t_max_dis = (
            self.num_train_timesteps if t_max_dis is None
            else min(int(t_max_dis), self.num_train_timesteps)
        )

        self.ea_dists = EADists()
        self.arcface = ArcFaceID() if enable_id else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_01(self, x_11: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → [0, 1], clamped for EA-DISTS/ArcFace sanity."""
        return ((x_11 + 1.0) * 0.5).clamp(0.0, 1.0)

    def _sample_t(self, batch_size: int, device) -> torch.Tensor:
        """
        t ∼ U{1, ..., t_max_dis - 1}. Index 0 corresponds to ᾱ₀≈1 (clean), skip it.
        Capped by t_max_dis (default T) — keeping D in the informative-noise
        regime prevents near-pure-noise samples at t≈T from polluting gradients.
        """
        return torch.randint(
            low=1, high=self.t_max_dis,
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
        region_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        OSDFace Eq. 12, with L_G from Eq. 15, plus optional region-L1 term.

        L_gen = λ_MSE · MSE(I_H, Î_H)
              + λ_per · L_EA-DISTS(I_H, Î_H)
              + λ_ID  · L_ID(I_H, Î_H)
              + λ_dis · L_G
              + λ_region · L1_region(I_H, Î_H)  (if region_mask and λ_region > 0)
        L_G = −E_t [ log D( F(ẑ_H, t) ) ]

        Args:
            region_mask: optional (B, 1, H, W) float mask in [0, 1] selecting
                         pixels to weight in the region-L1 term. When provided
                         and lambda_region_l1 > 0, adds a masked-mean L1 loss.
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

        # -- region-weighted L1 (optional) --
        if self.lambda_region_l1 > 0 and region_mask is not None:
            abs_err = (I_hat_H_11 - I_H_11).abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
            region_mask = region_mask.to(abs_err.dtype)
            denom = region_mask.sum().clamp_min(1.0)
            region_l1 = (abs_err * region_mask).sum() / denom
            logs["region_l1"] = region_l1.detach()
        else:
            region_l1 = torch.zeros((), device=mse.device, dtype=mse.dtype)
            logs["region_l1"] = region_l1

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
            + self.lambda_region_l1 * region_l1
            + self.lambda_dis * gan_g
        )
        logs["total_gen"] = total.detach()
        return total, logs

    # ------------------------------------------------------------------
    # R1 gradient penalty on discriminator (Mescheder 2018)
    # ------------------------------------------------------------------

    def r1_penalty(
        self,
        z_H: torch.Tensor,
        discriminator: nn.Module,
        gamma: float = 10.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        R1 = γ/2 · E [ ‖∇_z D(F(z_real, t))‖² ]

        Computed in fp32 outside any autocast context for numerical stability
        of the second-derivative path (create_graph=True). Call this every
        `r1_every` D updates to amortize cost — the penalty is ~2× more
        expensive than a vanilla D step.

        Args:
            z_H:  real HQ latent (no grad required from caller).
            discriminator: the D network (already spectral-normalized).
            gamma: R1 coefficient. 0 disables (returns 0 loss).

        Returns:
            (penalty_tensor, logs). Caller should `.backward()` this directly
            on the D optimizer — D params will receive gradients; z_H will not.
        """
        if gamma <= 0:
            zero = torch.zeros((), device=z_H.device, dtype=torch.float32)
            return zero, {"r1": zero.detach()}

        B = z_H.shape[0]
        t = self._sample_t(B, z_H.device)
        with torch.autocast(device_type=z_H.device.type, enabled=False):
            z_real = z_H.detach().float().requires_grad_(True)
            eps = torch.randn_like(z_real)
            ab = self.alphas_cumprod[t].to(z_real.dtype).view(-1, 1, 1, 1)
            z_noisy = ab.sqrt() * z_real + (1.0 - ab).sqrt() * eps
            d_logit = discriminator(z_noisy, t).float()
            # retain_graph defaults to create_graph=True, so the forward-pass
            # saved tensors stay alive for the subsequent r1.backward() which
            # traverses the higher-order graph created here.
            (grads,) = torch.autograd.grad(
                outputs=d_logit.sum(), inputs=z_real,
                create_graph=True,
            )
            penalty = grads.pow(2).flatten(1).sum(dim=1).mean()
            r1 = 0.5 * gamma * penalty
        return r1, {"r1": r1.detach(), "r1_raw": penalty.detach()}

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
