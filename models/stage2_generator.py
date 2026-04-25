"""
stage2_generator.py — CRAFT Stage 2 one-step diffusion generator.

Architecture (see CRAFT Stage 2 plan §1):

    I_L (B,3,512,512) in [-1,1]
      ├─► [FROZEN] Stage-1 LQ VQVAE (phase_d/final.pt)
      │       encoder + RegionAwareVQ.forward → z_q (B, 512, 16, 16)
      │       to_prompt()                      → p_L  (B, 256, 512)
      │       Linear 512 → 1024 [TRAINABLE]    → p_L' (B, 256, 1024)
      │       LayerNorm(1024)    [TRAINABLE]   → context
      │                                              │
      │                                              ▼ K, V
      └─► [FROZEN] SD-2.1-base VAE encoder   ──► z_L (B, 4, 64, 64)  (·0.18215)
                                                  │
                                                  ▼ (t = T_L = 999, context)
                      SD-2.1-base UNet ε_θ  [LoRA rank 16 TRAINABLE]
                                                  │
                                                  ▼  ε̂
               ẑ_H = (z_L − √(1−ᾱ_{T_L}) · ε̂) / √(ᾱ_{T_L})    (OSDFace Eq. 2)
                                                  │
                                                  ▼
                   [FROZEN] SD-2.1-base VAE decoder  → Î_H (B, 3, 512, 512) in [-1,1]

Only trainable modules returned in .trainable_parameters():
    - prompt_proj (Linear 512 → 1024 + LayerNorm)
    - the LoRA adapters injected into the SD UNet

Everything else is frozen. The VAE decoder is frozen but runs WITH autograd
(no `torch.no_grad()`) so pixel-space loss gradients can reach the UNet/LoRA
through the decoder. The VAE encoder and Stage-1 model both run under
`torch.no_grad()` since nothing upstream of the UNet needs gradients.

Usage:
    gen = Stage2Generator(
        stage1_ckpt_path="checkpoints/phase_d/final.pt",
        parser_ckpt="pretrained/79999_iter.pth",
    ).to(device)
    I_hat, z_hat, z_true, p_L = gen(I_L_11, I_L_01, I_H_11)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model

from .vqvae import build_lq_vqvae
from .region_aware_vq import RegionAwareVQ


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _freeze(module: nn.Module) -> None:
    """Set all params to requires_grad=False and put in eval() mode."""
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


# ----------------------------------------------------------------------
# Stage-1 wrapper (loads phase_d/final.pt, exposes only the VRE path)
# ----------------------------------------------------------------------

class _Stage1VRE(nn.Module):
    """
    Thin frozen wrapper around the Stage-1 LQ VQVAE.

    Only the encoder + RegionAwareVQ path is exercised (the decoder is not
    used in Stage 2). Auto-detects magnitude_head in the checkpoint so the
    same class handles both Phase-C and Phase-D checkpoints.
    """

    def __init__(
        self,
        ckpt_path: str,
        parser_ckpt: str,
        embed_dim: int = 512,
        rq_levels: int = 3,
    ):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        key = "model" if "model" in ckpt else "lq_model"
        state = ckpt[key]

        has_mag_head = any("magnitude_head" in k for k in state.keys())

        ravq = RegionAwareVQ(
            e_dim=embed_dim,
            n_levels=rq_levels,
            parser_ckpt=parser_ckpt,
            use_magnitude_head=has_mag_head,
        )
        self.vqvae = build_lq_vqvae(ravq, embed_dim=embed_dim)
        self.vqvae.load_state_dict(state)

        self.has_mag_head = has_mag_head
        _freeze(self)

    @torch.no_grad()
    def forward(
        self,
        I_L_11: torch.Tensor,
        I_L_01: torch.Tensor,
        masks: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Args:
            I_L_11: (B, 3, 512, 512) LQ image in [-1, 1].
            I_L_01: (B, 3, 512, 512) LQ image in [0, 1] (for face parser).
            masks:  Optional pre-computed region masks dict.

        Returns:
            p_L: (B, 256, 512) visual prompt.
        """
        z = self.vqvae.encode(I_L_11)  # (B, 512, 16, 16)
        z_q, _, _ = self.vqvae.quantizer(z, images=I_L_01, masks=masks)
        # (B, 512, 16, 16) → (B, 16*16, 512) = (B, 256, 512)
        B, C, H, W = z_q.shape
        p_L = z_q.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return p_L


# ----------------------------------------------------------------------
# Stage-2 generator
# ----------------------------------------------------------------------

class Stage2Generator(nn.Module):
    """
    One-step diffusion generator for CRAFT Stage 2.

    Args:
        stage1_ckpt_path: Path to Phase-D (or Phase-C) checkpoint.
        parser_ckpt:      Path to BiSeNet face-parsing checkpoint.
        sd_model_name:    HuggingFace repo ID for the SD backbone.
        t_fixed:          Pre-defined one-step timestep T_L (0..T).
        lora_rank:        LoRA rank for UNet attention layers.
        lora_alpha:       LoRA alpha (scale = alpha / rank).
        prompt_dim:       Stage-1 visual-prompt channel count (= RegionAwareVQ
                          embed_dim). Default 512.
        context_dim:      UNet cross-attention context dim. Must match the
                          SD backbone (1024 for SD-2.1-base).
    """

    # Which attention linears to LoRA-adapt
    _LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

    def __init__(
        self,
        stage1_ckpt_path: str,
        parser_ckpt: str,
        sd_model_name: str = "stabilityai/stable-diffusion-2-1-base",
        t_fixed: int = 999,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        prompt_dim: int = 512,
        context_dim: int = 1024,
        sd_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.t_fixed = int(t_fixed)
        self.prompt_dim = prompt_dim
        self.context_dim = context_dim

        # --- Stage-1 VRE (frozen) -----------------------------------
        self.stage1 = _Stage1VRE(
            ckpt_path=stage1_ckpt_path,
            parser_ckpt=parser_ckpt,
            embed_dim=prompt_dim,
        )

        # --- SD VAE (frozen, grads FLOW through decoder at training) -
        self.vae = AutoencoderKL.from_pretrained(
            sd_model_name, subfolder="vae", cache_dir=sd_cache_dir,
        )
        _freeze(self.vae)
        self.vae_scale = float(self.vae.config.scaling_factor)  # 0.18215

        # --- SD UNet (frozen base + LoRA adapters) -------------------
        unet = UNet2DConditionModel.from_pretrained(
            sd_model_name, subfolder="unet", cache_dir=sd_cache_dir,
        )
        assert unet.config.cross_attention_dim == context_dim, (
            f"SD backbone '{sd_model_name}' has cross_attention_dim="
            f"{unet.config.cross_attention_dim}, plan expects {context_dim}."
        )
        # Freeze everything first
        for p in unet.parameters():
            p.requires_grad_(False)
        # Wrap with LoRA
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=self._LORA_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
        )
        self.unet = get_peft_model(unet, lora_cfg)
        # peft sets requires_grad only on LoRA params — we do NOT want the
        # base UNet weights trainable, and LoRA params are. Verify below.

        # --- Trainable projection p_L 512 → 1024 --------------------
        self.prompt_proj = nn.Linear(prompt_dim, context_dim)
        self.prompt_ln = nn.LayerNorm(context_dim)
        # Scaled init so projected prompt starts near unit scale
        nn.init.xavier_uniform_(self.prompt_proj.weight, gain=1.0)
        nn.init.zeros_(self.prompt_proj.bias)

        # --- Diffusion schedule (for ᾱ_{T_L} and for forward-diffusion
        #     in discriminator sampling) --------------------------------
        scheduler = DDPMScheduler.from_pretrained(
            sd_model_name, subfolder="scheduler", cache_dir=sd_cache_dir,
        )
        # (T,) cumulative alpha products
        self.register_buffer(
            "alphas_cumprod",
            scheduler.alphas_cumprod.clone().float(),
            persistent=False,
        )
        self.num_train_timesteps = int(scheduler.config.num_train_timesteps)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """All parameters that should receive gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def frozen_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def describe(self) -> str:
        return (
            f"Stage2Generator | trainable={self.trainable_param_count():,}"
            f" | frozen={self.frozen_param_count():,}"
            f" | T_L={self.t_fixed}"
            f" | vae_scale={self.vae_scale:.5f}"
            f" | mag_head={self.stage1.has_mag_head}"
        )

    # ------------------------------------------------------------------
    # Encoding helpers (run under no_grad — all inputs to UNet are frozen)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(
        self,
        I_L_11: torch.Tensor,
        I_L_01: torch.Tensor,
        masks: Optional[dict] = None,
    ) -> torch.Tensor:
        """Stage-1 visual prompt, (B, 256, 512)."""
        return self.stage1(I_L_11, I_L_01, masks=masks)

    @torch.no_grad()
    def encode_vae(self, I_11: torch.Tensor) -> torch.Tensor:
        """
        SD VAE encode to scaled latent.

        Args:
            I_11: (B, 3, 512, 512) in [-1, 1].
        Returns:
            z:    (B, 4, 64, 64) scaled latent (·scaling_factor).
        """
        posterior = self.vae.encode(I_11).latent_dist
        z = posterior.mean * self.vae_scale
        return z

    def decode_vae(self, z: torch.Tensor) -> torch.Tensor:
        """
        SD VAE decode with gradients flowing (frozen params).

        Args:
            z: (B, 4, 64, 64) scaled latent.
        Returns:
            img: (B, 3, 512, 512) in (~[-1, 1]).
        """
        return self.vae.decode(z / self.vae_scale).sample

    # ------------------------------------------------------------------
    # Project p_L → context for cross-attention
    # ------------------------------------------------------------------

    def project_prompt(self, p_L: torch.Tensor) -> torch.Tensor:
        """(B, 256, 512) → (B, 256, 1024) with LayerNorm."""
        return self.prompt_ln(self.prompt_proj(p_L))

    # ------------------------------------------------------------------
    # One-step denoise (OSDFace Eq. 2)
    # ------------------------------------------------------------------

    def one_step_denoise(
        self, z_L: torch.Tensor, eps_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        ẑ_H = (z_L − √(1−ᾱ_{T_L}) · ε̂) / √(ᾱ_{T_L})
        """
        ab = self.alphas_cumprod[self.t_fixed].to(z_L.dtype)
        sqrt_ab = ab.sqrt()
        sqrt_1m_ab = (1.0 - ab).sqrt()
        return (z_L - sqrt_1m_ab * eps_pred) / sqrt_ab

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def forward(
        self,
        I_L_11: torch.Tensor,
        I_L_01: torch.Tensor,
        I_H_11: Optional[torch.Tensor] = None,
        masks: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Full Stage-2 forward pass.

        Args:
            I_L_11: (B, 3, 512, 512) LQ image in [-1, 1].
            I_L_01: (B, 3, 512, 512) LQ image in [0, 1] (for face parser).
            I_H_11: Optional (B, 3, 512, 512) HQ image in [-1, 1]. When
                    provided, also returns z_H = VAE_enc(I_H) for the
                    latent-space discriminator.
            masks:  Optional pre-computed region masks dict.

        Returns:
            I_hat_H_11: (B, 3, 512, 512) in (~[-1, 1]).
            z_hat_H:    (B, 4, 64, 64) predicted HQ latent.
            z_H:        (B, 4, 64, 64) true HQ latent, or None if I_H_11 was
                        not supplied.
            p_L_proj:   (B, 256, 1024) projected visual prompt (for logging).
        """
        B = I_L_11.shape[0]
        device = I_L_11.device

        # --- Stage-1 visual prompt (frozen, no_grad) --------------------
        p_L = self.encode_prompt(I_L_11, I_L_01, masks=masks)  # (B,256,512)
        p_L_proj = self.project_prompt(p_L)                    # (B,256,1024)

        # --- LQ latent (frozen VAE, no_grad) ----------------------------
        z_L = self.encode_vae(I_L_11)                          # (B,4,64,64)

        # --- UNet ε prediction (gradients flow: LoRA + prompt_proj) -----
        # timestep broadcasts to batch
        t = torch.full(
            (B,), self.t_fixed, device=device, dtype=torch.long,
        )
        eps_pred = self.unet(
            sample=z_L,
            timestep=t,
            encoder_hidden_states=p_L_proj,
        ).sample                                               # (B,4,64,64)

        # --- One-step denoise (OSDFace Eq. 2) ---------------------------
        z_hat_H = self.one_step_denoise(z_L, eps_pred)         # (B,4,64,64)

        # --- VAE decode (frozen params, grads flow) ---------------------
        I_hat_H_11 = self.decode_vae(z_hat_H)                  # (B,3,512,512)

        # --- Optional HQ latent for GAN ---------------------------------
        z_H = self.encode_vae(I_H_11) if I_H_11 is not None else None

        return I_hat_H_11, z_hat_H, z_H, p_L_proj


# ----------------------------------------------------------------------
# Sanity test
# ----------------------------------------------------------------------

def _sanity_test() -> None:
    """
    Builds the generator on CPU with tiny random inputs (2 images) and
    runs one forward pass. Verifies shapes and that only the expected
    parameter set has requires_grad=True.

    Usage:
        python -m models.stage2_generator \
            --stage1_ckpt checkpoints/phase_d/final.pt \
            --parser_ckpt pretrained/79999_iter.pth
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--parser_ckpt", required=True)
    ap.add_argument(
        "--sd_model_name",
        default="stabilityai/stable-diffusion-2-1-base",
    )
    ap.add_argument("--sd_cache_dir", default=None)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32

    print(f"[sanity] building Stage2Generator on {device} ...")
    gen = Stage2Generator(
        stage1_ckpt_path=args.stage1_ckpt,
        parser_ckpt=args.parser_ckpt,
        sd_model_name=args.sd_model_name,
        sd_cache_dir=args.sd_cache_dir,
    ).to(device=device, dtype=dtype)
    print(f"[sanity] {gen.describe()}")

    # Identify which modules have trainable params
    train_owners = set()
    for name, p in gen.named_parameters():
        if p.requires_grad:
            # e.g. "unet.base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora_A.default.weight"
            top = name.split(".", 1)[0]
            train_owners.add(top)
    print(f"[sanity] trainable top-level submodules: {sorted(train_owners)}")
    assert "stage1" not in train_owners, "Stage-1 must be frozen"
    assert "vae" not in train_owners, "VAE must be frozen"

    # Make dummy inputs
    B = args.batch
    I_L_11 = torch.rand(B, 3, 512, 512, device=device) * 2 - 1
    I_L_01 = (I_L_11 + 1) / 2
    I_H_11 = torch.rand(B, 3, 512, 512, device=device) * 2 - 1

    print("[sanity] running forward ...")
    with torch.no_grad():
        I_hat, z_hat, z_H, p_L_proj = gen(I_L_11, I_L_01, I_H_11=I_H_11)

    print(f"[sanity]   I_hat   : {tuple(I_hat.shape)}")
    print(f"[sanity]   z_hat   : {tuple(z_hat.shape)}")
    print(f"[sanity]   z_H     : {tuple(z_H.shape)}")
    print(f"[sanity]   p_L_proj: {tuple(p_L_proj.shape)}")

    assert I_hat.shape == (B, 3, 512, 512)
    assert z_hat.shape == (B, 4, 64, 64)
    assert z_H.shape == (B, 4, 64, 64)
    assert p_L_proj.shape == (B, 256, 1024)

    # Gradient flow test — require_grad on prompt_proj should deliver grads
    gen.train()
    I_hat, z_hat, z_H, _ = gen(I_L_11, I_L_01, I_H_11=I_H_11)
    loss = F.mse_loss(I_hat, I_H_11)
    loss.backward()
    proj_grad = gen.prompt_proj.weight.grad
    assert proj_grad is not None and proj_grad.abs().sum().item() > 0, (
        "prompt_proj did not receive gradients"
    )
    # LoRA params should have grads
    lora_with_grad = 0
    for name, p in gen.named_parameters():
        if "lora" in name.lower() and p.grad is not None and p.grad.abs().sum().item() > 0:
            lora_with_grad += 1
    assert lora_with_grad > 0, "No LoRA params received gradients"
    print(f"[sanity] gradient flow ✓ (loss={loss.item():.4f}, "
          f"{lora_with_grad} lora params with grad)")
    print("[sanity] ALL CHECKS PASSED")


if __name__ == "__main__":
    _sanity_test()
