"""
stage2_discriminator.py — D³SR-style latent-space discriminator for CRAFT Stage 2.

Follows the D³SR / OSDFace formulation: a CNN that operates on forward-diffused
VAE latents F(z, t) = √(ᾱ_t)·z + √(1−ᾱ_t)·ε with timestep t injected via FiLM
after every stage. This lets the discriminator tell real vs. fake at any noise
level sampled uniformly over {1, …, T−1}, which matches the adversarial term
used by `losses/stage2_losses.py`.

Architecture (CRAFT Stage 2 plan §4):

    Input:   z_noisy (B, 4, 64, 64),  t (B,) long in [0, T)
    Output:  logit   (B,)

    t_emb  = sinusoidal(t, 256) → Linear(256→512) → SiLU → Linear(512→512)

    stem   = SN-Conv2d(4 → 64,  k=3, s=1) → SiLU                   # (64, 64, 64)
    block1 = DownBlock(64  → 128, stride=2 + FiLM)                 # (128, 32, 32)
    block2 = DownBlock(128 → 256, stride=2 + FiLM)                 # (256, 16, 16)
    block3 = DownBlock(256 → 512, stride=2 + FiLM)                 # (512, 8, 8)
    block4 = DownBlock(512 → 512, stride=1 + FiLM)                 # (512, 8, 8)

    head   = AdaptiveAvgPool2d(1) → Flatten → SN-Linear(512 → 1)   # (B,)

Each DownBlock = [SN-Conv(keep-ch, s=1) → GN → SiLU → FiLM(t_emb)
                  → SN-Conv(→out-ch, s or 1) → GN → SiLU].

Spectral norm (torch >= 2.0 parametrization) is wrapped around every Conv2d
and the final Linear for training stability (D³SR uses SN; it is the single
most impactful regularizer for GAN convergence in this setup).

~10 M parameters. Inputs are assumed to be raw scaled latents (VAE encoder
output × 0.18215) with the diffusion noise already mixed in upstream.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as _sn


# ----------------------------------------------------------------------
# Timestep embedding
# ----------------------------------------------------------------------

def sinusoidal_timestep_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal embedding (same as Transformer / DDPM).

    Args:
        t:   (B,) long or float timesteps.
        dim: embedding dimension. If odd, result is right-padded with 0.

    Returns:
        (B, dim) float tensor on t.device.
    """
    half = dim // 2
    device = t.device
    # exp(-log(10000) * k / half), k = 0..half-1
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, device=device, dtype=torch.float32)
        / max(half, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)           # (B, half)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)            # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ----------------------------------------------------------------------
# FiLM modulation
# ----------------------------------------------------------------------

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation conditioning.

    Given a (B, cond_dim) vector, produce per-channel (scale, shift) and apply
    `x * (1 + scale) + shift`. The "(1 + scale)" form initializes to identity
    when the projection weights are small, which helps early-training stability.
    """

    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * num_features)
        # Init scale/shift near 0 so the block starts close to identity on t
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + scale) + shift


# ----------------------------------------------------------------------
# Downsampling residual-ish block with timestep FiLM
# ----------------------------------------------------------------------

class DownBlock(nn.Module):
    """
    [SN-Conv(in→in, s=1) → GN → SiLU] → FiLM(t_emb) →
    [SN-Conv(in→out, s={1,2}) → GN → SiLU]

    GroupNorm uses 32 groups; the channel counts we use (64, 128, 256, 512)
    are all divisible by 32.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_dim: int,
        downsample: bool = True,
        num_groups: int = 32,
    ):
        super().__init__()
        assert in_ch % num_groups == 0, (in_ch, num_groups)
        assert out_ch % num_groups == 0, (out_ch, num_groups)
        self.conv1 = _sn(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1))
        self.gn1 = nn.GroupNorm(num_groups, in_ch)
        self.film = FiLM(t_dim, in_ch)
        stride = 2 if downsample else 1
        self.conv2 = _sn(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1))
        self.gn2 = nn.GroupNorm(num_groups, out_ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.film(h, cond)
        h = F.silu(self.gn2(self.conv2(h)))
        return h


# ----------------------------------------------------------------------
# D³SR-style latent discriminator
# ----------------------------------------------------------------------

class LatentDiscriminator(nn.Module):
    """
    Discriminator over noisy SD-2.1-base latents with timestep conditioning.

    Args:
        in_channels: input latent channels (SD-2.1-base → 4).
        base_ch:     width of the stem conv (default 64).
        t_embed_dim: sinusoidal timestep dim before the MLP (default 256).
        t_mlp_dim:   hidden/output dim of the timestep MLP (default 512). This
                     is the `cond_dim` fed to every FiLM layer.
        num_groups:  GroupNorm groups.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_ch: int = 64,
        t_embed_dim: int = 256,
        t_mlp_dim: int = 512,
        num_groups: int = 32,
    ):
        super().__init__()
        self.t_embed_dim = int(t_embed_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, t_mlp_dim),
            nn.SiLU(),
            nn.Linear(t_mlp_dim, t_mlp_dim),
        )

        # Stem: 4 → 64, preserves 64×64 spatial.
        self.stem = _sn(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=1, padding=1)
        )

        # Down cascade: 64 → 128 → 256 → 512, spatial 64 → 32 → 16 → 8.
        self.block1 = DownBlock(base_ch,      base_ch * 2,  t_mlp_dim, downsample=True,  num_groups=num_groups)
        self.block2 = DownBlock(base_ch * 2,  base_ch * 4,  t_mlp_dim, downsample=True,  num_groups=num_groups)
        self.block3 = DownBlock(base_ch * 4,  base_ch * 8,  t_mlp_dim, downsample=True,  num_groups=num_groups)
        # Final refinement at (512, 8, 8), no further downsample.
        self.block4 = DownBlock(base_ch * 8,  base_ch * 8,  t_mlp_dim, downsample=False, num_groups=num_groups)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = _sn(nn.Linear(base_ch * 8, 1))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_noisy: (B, 4, 64, 64). Forward-diffused VAE latent.
            t:       (B,) long or float timesteps.

        Returns:
            (B,) real-valued logits (pre-sigmoid). The loss module applies
            softplus(±logit) for the non-saturating BCE form.
        """
        emb = sinusoidal_timestep_embed(t, self.t_embed_dim)
        cond = self.t_mlp(emb)                                   # (B, t_mlp_dim)

        h = F.silu(self.stem(z_noisy))                           # (B, 64, 64, 64)
        h = self.block1(h, cond)                                 # (B, 128, 32, 32)
        h = self.block2(h, cond)                                 # (B, 256, 16, 16)
        h = self.block3(h, cond)                                 # (B, 512,  8,  8)
        h = self.block4(h, cond)                                 # (B, 512,  8,  8)

        h = self.pool(h).flatten(1)                              # (B, 512)
        return self.head(h).squeeze(-1)                          # (B,)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------
# Standalone sanity test
# ----------------------------------------------------------------------

def _sanity_test():
    """Run: `python -m models.stage2_discriminator`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    D = LatentDiscriminator().to(device)
    print(f"LatentDiscriminator: {D.num_parameters() / 1e6:.2f} M parameters")

    B, T = 4, 1000
    z_noisy = torch.randn(B, 4, 64, 64, device=device)
    t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

    # Forward
    logits = D(z_noisy, t)
    assert logits.shape == (B,), logits.shape
    print(f"forward OK: logits shape={tuple(logits.shape)}, "
          f"mean={logits.mean().item():+.4f}, std={logits.std().item():.4f}")

    # Backward
    loss = F.softplus(-logits).mean() + F.softplus(logits).mean()
    loss.backward()
    grad_norm = sum(
        p.grad.detach().pow(2).sum().item()
        for p in D.parameters() if p.grad is not None
    ) ** 0.5
    print(f"backward OK: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

    # Timestep sensitivity: two different t values should give different logits.
    with torch.no_grad():
        t_low = torch.zeros(B, device=device, dtype=torch.long)
        t_hi  = torch.full((B,), T - 1, device=device, dtype=torch.long)
        diff = (D(z_noisy, t_low) - D(z_noisy, t_hi)).abs().mean().item()
    # FiLM is zero-init, so initially diff should be ~0. That's fine; this test
    # just confirms the path runs and is differentiable.
    print(f"t-sensitivity (expect ~0 at init): {diff:.6f}")


if __name__ == "__main__":
    _sanity_test()
