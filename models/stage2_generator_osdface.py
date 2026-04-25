"""
stage2_generator_osdface.py — OSDFace baseline Stage 2 one-step diffusion
generator. Parallel to stage2_generator.py; intentionally does NOT import
the region-aware VQ or the face parser.

Architecture (identical to CRAFT's Stage 2 except the Stage-1 prompt encoder):

    I_L (B,3,512,512) in [-1,1]
      ├─► [FROZEN] OSDFace Stage-1 LQ VQVAE (GlobalVQ, flat codebook)
      │       encoder + GlobalVQ.forward → z_q (B, 512, 16, 16)
      │       reshape                     → p_L  (B, 256, 512)
      │       Linear 512 → ctx_dim        → p_L' (B, 256, ctx_dim)      [TRAINABLE]
      │       LayerNorm(ctx_dim)          → context                     [TRAINABLE]
      │
      └─► [FROZEN] SD VAE encoder  → z_L (B, 4, 64, 64)  (·0.18215)
                                        │
                                        ▼ (t = T_L = 999, context)
                  SD UNet ε_θ  [LoRA rank 16 TRAINABLE]
                                        │
                                        ▼  ε̂
             ẑ_H = (z_L − √(1−ᾱ_{T_L}) · ε̂) / √(ᾱ_{T_L})    (OSDFace Eq. 2)
                                        │
                                        ▼
                  [FROZEN] SD VAE decoder  → Î_H (B, 3, 512, 512) in [-1,1]

Trainable modules:
    - prompt_proj (Linear 512 → ctx_dim) + LayerNorm
    - LoRA adapters inside the SD UNet

Differences vs. `Stage2Generator` (CRAFT):
    - Stage-1 wrapper uses `build_hq_vqvae` (GlobalVQ) instead of RegionAwareVQ.
    - No face parser. The `masks` argument is accepted but ignored for API
      symmetry with the CRAFT path.
    - Checkpoint key is auto-detected: `lq_model` (OSDFace format) is tried
      first, falling back to `model` (CRAFT Phase-A format).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model

from .vqvae import build_hq_vqvae


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _freeze(module: nn.Module) -> None:
    """Set all params to requires_grad=False and put in eval() mode."""
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


# ----------------------------------------------------------------------
# OSDFace Stage-1 wrapper (GlobalVQ, no parser, no masks)
# ----------------------------------------------------------------------

class _Stage1VREOSDFace(nn.Module):
    """
    Thin frozen wrapper around the OSDFace Stage-1 LQ VQ-VAE.

    The LQ branch was trained by `train_osdface_stage1.py` using the same
    factory as the HQ branch (`build_hq_vqvae` → GlobalVQ, n_codes=1024,
    embed_dim=512). Only the encoder + quantizer path is exercised; the
    decoder is never called in Stage 2.

    Args:
        ckpt_path: path to OSDFace Stage-1 checkpoint (Phase B or C).
        n_codes:   codebook size (must match training; default 1024).
        embed_dim: codebook embedding dim (must match training; default 512).
    """

    def __init__(
        self,
        ckpt_path: str,
        n_codes: int = 1024,
        embed_dim: int = 512,
    ):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # OSDFace trainer saves under "lq_model"; HQ ckpts use "model".
        # Accept either so we can also reuse a frozen HQ ckpt as the prompt
        # source for sanity experiments.
        if "lq_model" in ckpt:
            state = ckpt["lq_model"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            raise KeyError(
                f"OSDFace Stage-1 ckpt at {ckpt_path} has neither 'lq_model' "
                f"nor 'model' key. Found: {list(ckpt.keys())}"
            )

        self.vqvae = build_hq_vqvae(n_codes=n_codes, embed_dim=embed_dim)
        self.vqvae.load_state_dict(state)
        _freeze(self)

    @torch.no_grad()
    def forward(
        self,
        I_L_11: torch.Tensor,
        I_L_01: Optional[torch.Tensor] = None,
        masks: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Args:
            I_L_11: (B, 3, 512, 512) LQ image in [-1, 1].
            I_L_01: accepted for API symmetry, ignored.
            masks:  accepted for API symmetry, ignored.

        Returns:
            p_L: (B, 256, 512) visual prompt.
        """
        z = self.vqvae.encode(I_L_11)                           # (B, 512, 16, 16)
        # GlobalVQ returns (z_q, vq_losses, vq_info); only z_q is needed.
        z_q, _, _ = self.vqvae.quantizer(z, images=None, masks=None)
        B, C, H, W = z_q.shape
        p_L = z_q.permute(0, 2, 3, 1).reshape(B, H * W, C)      # (B, 256, 512)
        return p_L


# ----------------------------------------------------------------------
# Stage-2 generator (OSDFace baseline)
# ----------------------------------------------------------------------

class Stage2GeneratorOSDFace(nn.Module):
    """
    One-step diffusion generator — OSDFace baseline (no region-aware prompt).

    Args:
        stage1_ckpt_path: Path to OSDFace Stage-1 LQ checkpoint
                          (trained by train_osdface_stage1.py, Phase B or C).
        sd_model_name:    HuggingFace repo ID for the SD backbone.
        t_fixed:          One-step timestep T_L in [0, T).
        lora_rank:        LoRA rank for UNet attention layers.
        lora_alpha:       LoRA alpha (scale = alpha / rank).
        prompt_dim:       Stage-1 visual-prompt channel count (= 512).
        context_dim:      UNet cross-attention context dim (MUST match the SD
                          backbone: 768 for SD-1.5, 1024 for SD-2.x).
        stage1_n_codes:   OSDFace Stage-1 codebook size (default 1024).
        sd_cache_dir:     Optional HF cache override.
    """

    _LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

    def __init__(
        self,
        stage1_ckpt_path: str,
        sd_model_name: str = "sd-legacy/stable-diffusion-v1-5",
        t_fixed: int = 999,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        prompt_dim: int = 512,
        context_dim: int = 768,
        stage1_n_codes: int = 1024,
        sd_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.t_fixed = int(t_fixed)
        self.prompt_dim = int(prompt_dim)
        self.context_dim = int(context_dim)

        # --- OSDFace Stage-1 (frozen) -------------------------------
        self.stage1 = _Stage1VREOSDFace(
            ckpt_path=stage1_ckpt_path,
            n_codes=stage1_n_codes,
            embed_dim=prompt_dim,
        )

        # --- SD VAE (frozen, grads FLOW through decoder at training) -
        self.vae = AutoencoderKL.from_pretrained(
            sd_model_name, subfolder="vae", cache_dir=sd_cache_dir,
        )
        _freeze(self.vae)
        self.vae_scale = float(self.vae.config.scaling_factor)

        # --- SD UNet (frozen base + LoRA adapters) -------------------
        unet = UNet2DConditionModel.from_pretrained(
            sd_model_name, subfolder="unet", cache_dir=sd_cache_dir,
        )
        assert unet.config.cross_attention_dim == context_dim, (
            f"SD backbone '{sd_model_name}' has cross_attention_dim="
            f"{unet.config.cross_attention_dim}, plan expects {context_dim}."
        )
        for p in unet.parameters():
            p.requires_grad_(False)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=self._LORA_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
        )
        self.unet = get_peft_model(unet, lora_cfg)

        # --- Trainable prompt projection ----------------------------
        self.prompt_proj = nn.Linear(prompt_dim, context_dim)
        self.prompt_ln = nn.LayerNorm(context_dim)
        nn.init.xavier_uniform_(self.prompt_proj.weight, gain=1.0)
        nn.init.zeros_(self.prompt_proj.bias)

        # --- Diffusion schedule --------------------------------------
        scheduler = DDPMScheduler.from_pretrained(
            sd_model_name, subfolder="scheduler", cache_dir=sd_cache_dir,
        )
        self.register_buffer(
            "alphas_cumprod",
            scheduler.alphas_cumprod.clone().float(),
            persistent=False,
        )
        self.num_train_timesteps = int(scheduler.config.num_train_timesteps)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def frozen_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def describe(self) -> str:
        return (
            f"Stage2GeneratorOSDFace | trainable={self.trainable_param_count():,}"
            f" | frozen={self.frozen_param_count():,}"
            f" | T_L={self.t_fixed}"
            f" | vae_scale={self.vae_scale:.5f}"
        )

    # ------------------------------------------------------------------
    # Encoders (frozen; under no_grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(
        self,
        I_L_11: torch.Tensor,
        I_L_01: Optional[torch.Tensor] = None,
        masks: Optional[dict] = None,
    ) -> torch.Tensor:
        """OSDFace Stage-1 visual prompt, (B, 256, 512)."""
        return self.stage1(I_L_11, I_L_01, masks=masks)

    @torch.no_grad()
    def encode_vae(self, I_11: torch.Tensor) -> torch.Tensor:
        """SD VAE encode → scaled latent (B, 4, 64, 64)."""
        posterior = self.vae.encode(I_11).latent_dist
        return posterior.mean * self.vae_scale

    def decode_vae(self, z: torch.Tensor) -> torch.Tensor:
        """SD VAE decode (frozen params, grads flow)."""
        return self.vae.decode(z / self.vae_scale).sample

    # ------------------------------------------------------------------
    # Prompt projection
    # ------------------------------------------------------------------

    def project_prompt(self, p_L: torch.Tensor) -> torch.Tensor:
        return self.prompt_ln(self.prompt_proj(p_L))

    # ------------------------------------------------------------------
    # One-step denoise (OSDFace Eq. 2)
    # ------------------------------------------------------------------

    def one_step_denoise(
        self, z_L: torch.Tensor, eps_pred: torch.Tensor
    ) -> torch.Tensor:
        ab = self.alphas_cumprod[self.t_fixed].to(z_L.dtype)
        sqrt_ab = ab.sqrt()
        sqrt_1m_ab = (1.0 - ab).sqrt()
        return (z_L - sqrt_1m_ab * eps_pred) / sqrt_ab

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        I_L_11: torch.Tensor,
        I_L_01: Optional[torch.Tensor] = None,
        I_H_11: Optional[torch.Tensor] = None,
        masks: Optional[dict] = None,          # ignored; kept for API parity
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
            I_hat_H_11: (B, 3, 512, 512) predicted HQ in (~[-1, 1]).
            z_hat_H:    (B, 4, 64, 64) predicted HQ latent.
            z_H:        (B, 4, 64, 64) true HQ latent or None.
            p_L_proj:   (B, 256, context_dim) projected visual prompt.
        """
        B = I_L_11.shape[0]
        device = I_L_11.device

        # Stage-1 prompt (frozen, no_grad)
        p_L = self.encode_prompt(I_L_11, I_L_01)                # (B,256,512)
        p_L_proj = self.project_prompt(p_L)                     # (B,256,ctx)

        # LQ latent (frozen VAE, no_grad)
        z_L = self.encode_vae(I_L_11)                           # (B,4,64,64)

        # UNet ε prediction (gradients flow: LoRA + prompt_proj)
        t = torch.full((B,), self.t_fixed, device=device, dtype=torch.long)
        eps_pred = self.unet(
            sample=z_L,
            timestep=t,
            encoder_hidden_states=p_L_proj,
        ).sample                                                 # (B,4,64,64)

        z_hat_H = self.one_step_denoise(z_L, eps_pred)           # (B,4,64,64)
        I_hat_H_11 = self.decode_vae(z_hat_H)                    # (B,3,512,512)

        z_H = self.encode_vae(I_H_11) if I_H_11 is not None else None

        return I_hat_H_11, z_hat_H, z_H, p_L_proj


# ----------------------------------------------------------------------
# Sanity test
# ----------------------------------------------------------------------

def _sanity_test():
    """Run: python -m models.stage2_generator_osdface"""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m models.stage2_generator_osdface <osdface_stage1_ckpt>")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    gen = Stage2GeneratorOSDFace(
        stage1_ckpt_path=sys.argv[1],
        sd_model_name="sd-legacy/stable-diffusion-v1-5",
        context_dim=768,
    ).to(device)
    print(gen.describe())

    B = 2
    I_L = torch.randn(B, 3, 512, 512, device=device).clamp(-1, 1)
    I_H = torch.randn(B, 3, 512, 512, device=device).clamp(-1, 1)
    I_hat, z_hat, z_H, p_L_proj = gen(I_L, I_H_11=I_H)
    print(f"[sanity] I_hat={tuple(I_hat.shape)}  z_hat={tuple(z_hat.shape)}  "
          f"z_H={tuple(z_H.shape)}  p_L_proj={tuple(p_L_proj.shape)}")


if __name__ == "__main__":
    _sanity_test()
