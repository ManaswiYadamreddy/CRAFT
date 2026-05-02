"""
benchmark_original_osdface.py — Step Time / Param / MACs for the *original*
pretrained OSDFace one-step diffusion pipeline (the published checkpoint,
not the Stage 1 VRE comparison from `benchmark_efficiency.py`).

Pipeline matches `infer_concat.py`'s OSDFace_Concat.forward:

    img_encoder (vqvae_encoder, root vqvae.py)         : LQ → (B, 77, 512)
    embedding_change (TwoLayerConv1x1)                 : (B, 77, 512) → (B, 77, 1024)
    [empty CLIP text embedding, cached out of the loop : (B, 77, 1024)]
    cat                                                : (B, 154, 1024)
    SD-2.1 VAE encode                                  : LQ → (B, 4, 64, 64)
    SD-2.1 UNet (LoRA-merged, single timestep t=399)   : ε
    x_0 from ε                                          : latent
    SD-2.1 VAE decode                                  : (B, 3, 512, 512)

Usage
-----
    python benchmark_original_osdface.py \\
        --img_encoder_weight ~/Downloads/pretrained/associate_2.ckpt \\
        --ckpt_path          ~/Downloads/pretrained \\
        --pretrained_model_name_or_path /path/to/stable-diffusion-2-1-base \\
        --merge_lora \\
        --dtype fp16

`--ckpt_path` is the directory containing `embedding_change_weights.pth` and
`pytorch_lora_weights.safetensors` (same convention as `infer_concat.py`).
"""

import argparse
import os
import time
import warnings
from types import SimpleNamespace

import torch
import torch.nn as nn

# Imports from this repo
from lq_embed import vqvae_encoder, TwoLayerConv1x1


# ──────────────────────────────────────────────────────────────────────────────
# Original OSDFace pipeline (mirrors OSDFace_Concat in infer_concat.py)
# ──────────────────────────────────────────────────────────────────────────────

class OriginalOSDFacePipeline(nn.Module):
    """One-step OSDFace inference with the original published VRE + LoRA."""

    def __init__(
        self,
        img_encoder_weight: str,
        ckpt_path: str,
        sd_model: str,
        device,
        dtype,
        merge_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        timestep: int = 399,
    ):
        super().__init__()
        from diffusers import (
            DDIMScheduler, AutoencoderKL, UNet2DConditionModel,
        )
        from transformers import CLIPTokenizer, CLIPTextModel

        self.dtype = dtype
        self.timestep = timestep

        # ── Args namespace expected by vqvae_encoder ──────────────────────
        # All flags default to False — matches infer_concat.py with no extras.
        enc_args = SimpleNamespace(
            img_encoder_weight=img_encoder_weight,
            cat_prompt_embedding=False,
            use_pos_embedding=False,
            use_att_pool=False,
            learnable_pos_emb=False,
        )
        self.img_encoder = vqvae_encoder(enc_args).to(device, dtype=dtype).eval()

        # ── Visual-token projection (loaded from checkpoint) ──────────────
        self.embedding_change = TwoLayerConv1x1(512, 1024)
        ec_path = os.path.join(ckpt_path, "embedding_change_weights.pth")
        if os.path.exists(ec_path):
            self.embedding_change.load_state_dict(
                torch.load(ec_path, map_location="cpu", weights_only=False)
            )
        else:
            warnings.warn(f"{ec_path} not found — embedding_change is random.")
        self.embedding_change.to(device, dtype=dtype).eval()

        # ── SD 2.1 components ─────────────────────────────────────────────
        scheduler = DDIMScheduler.from_pretrained(sd_model, subfolder="scheduler")
        alpha_t = scheduler.alphas_cumprod[timestep]
        self.register_buffer("alpha_t", alpha_t.to(device, dtype=dtype))

        self.vae = AutoencoderKL.from_pretrained(
            sd_model, subfolder="vae"
        ).to(device, dtype=dtype).eval()

        unet = UNet2DConditionModel.from_pretrained(sd_model, subfolder="unet")
        if merge_lora:
            unet = self._merge_lora(unet, ckpt_path, lora_rank, lora_alpha)
        self.unet = unet.to(device, dtype=dtype).eval()

        # ── CLIP empty-prompt embedding (cached, NOT inside timed loop) ──
        tokenizer = CLIPTokenizer.from_pretrained(sd_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            sd_model, subfolder="text_encoder"
        ).to(device, dtype=dtype).eval()
        with torch.no_grad():
            tok = tokenizer(
                [""], padding="max_length",
                max_length=tokenizer.model_max_length, return_tensors="pt",
            ).input_ids.to(device)
            text_emb = text_encoder(tok).last_hidden_state.to(dtype)
        self.register_buffer("text_emb", text_emb)
        # text_encoder dropped — same convention as benchmark_efficiency.py.

    @staticmethod
    def _merge_lora(unet, ckpt_path, rank, alpha):
        """LoRA merge identical to infer_concat.py:merge_unet."""
        from safetensors import safe_open
        lora_path = os.path.join(ckpt_path, "pytorch_lora_weights.safetensors")
        if not os.path.exists(lora_path):
            warnings.warn(f"{lora_path} not found — using base UNet only.")
            return unet
        scale = float(alpha / rank)
        with safe_open(lora_path, framework="pt") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
        sd_unet = unet.state_dict()
        for key in sd:
            if "lora_A" in key:
                kb = key.replace("lora_A", "lora_B")
                uk = key.replace(".lora_A.weight", ".weight").replace("unet.", "")
                sd_unet[uk] = sd_unet[uk] + scale * torch.mm(sd[kb], sd[key])
        unet.load_state_dict(sd_unet)
        return unet

    @torch.no_grad()
    def forward(self, x):
        # `x` = LQ image in [-1, 1], (B, 3, 512, 512). Named `x` for ptflops.
        B = x.shape[0]
        # VRE → (B, 77, 512), then split-reshape and project to (B, 77, 1024)
        feat = self.img_encoder(x).reshape(B, 77, -1)
        visual_embeds = self.embedding_change(feat)
        # Concat with cached empty text → (B, 154, 1024)
        text_embeds = self.text_emb.expand(B, -1, -1)
        prompt_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        # SD VAE encode
        lq_lat = (
            self.vae.encode(x.to(self.dtype)).latent_dist.sample()
            * self.vae.config.scaling_factor
        )
        # UNet single ε-prediction
        eps = self.unet(
            lq_lat, self.timestep, encoder_hidden_states=prompt_embeds,
        ).sample
        # x_0 = (x_t − √(1−ᾱ) · ε) / √(ᾱ)
        a = self.alpha_t
        x0 = (lq_lat - (1 - a).sqrt() * eps) / a.sqrt()
        return self.vae.decode(x0 / self.vae.config.scaling_factor).sample


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, in_shape, in_dtype, device):
    from ptflops import get_model_complexity_info

    def constructor(shape):
        return {"x": torch.randn(1, *shape, device=device, dtype=in_dtype)}

    with torch.no_grad():
        macs, _ = get_model_complexity_info(
            model, in_shape,
            input_constructor=constructor,
            as_strings=False, print_per_layer_stat=False,
            verbose=False, backend="pytorch",
        )
    return int(macs) if macs is not None else -1


@torch.no_grad()
def time_step(model, x, n_warmup, n_iter, device):
    is_cuda = str(device).startswith("cuda")
    for _ in range(n_warmup):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    return (time.time() - t0) / n_iter


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_encoder_weight", required=True,
                    help="Path to the published VQVAE encoder ckpt "
                         "(e.g. ~/Downloads/pretrained/associate_2.ckpt).")
    ap.add_argument("--ckpt_path", required=True,
                    help="Directory containing pytorch_lora_weights.safetensors "
                         "and embedding_change_weights.pth.")
    ap.add_argument("--pretrained_model_name_or_path", "--sd_model",
                    dest="sd_model",
                    default="stabilityai/stable-diffusion-2-1-base",
                    help="HF id or local path of SD 2.1 base.")
    ap.add_argument("--merge_lora", action="store_true",
                    help="Merge OSDFace's LoRA into the UNet (recommended).")
    ap.add_argument("--lora_rank",  type=int,   default=16)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--timestep",   type=int,   default=399)
    ap.add_argument("--dtype",  choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--res",      type=int, default=512)
    ap.add_argument("--n_warmup", type=int, default=10)
    ap.add_argument("--n_iter",   type=int, default=50)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"Original OSDFace benchmark — Device: {device}    "
          f"Res: {args.res}×{args.res}    dtype: {args.dtype}    "
          f"merge_lora: {args.merge_lora}")

    # Build pipeline
    print("\nBuilding pipeline (this loads SD 2.1 + the VRE)...")
    model = OriginalOSDFacePipeline(
        img_encoder_weight=args.img_encoder_weight,
        ckpt_path=args.ckpt_path,
        sd_model=args.sd_model,
        device=device,
        dtype=dtype,
        merge_lora=args.merge_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        timestep=args.timestep,
    ).eval()

    # Input
    x = torch.randn(1, 3, args.res, args.res, device=device, dtype=dtype)

    # Time
    print(f"\nTiming forward passes ({args.n_warmup} warmup + {args.n_iter} timed)...")
    t = time_step(model, x, args.n_warmup, args.n_iter, device)

    # Params
    p = count_params(model)
    p_vre  = count_params(model.img_encoder)
    p_proj = count_params(model.embedding_change)
    p_vae  = count_params(model.vae)
    p_unet = count_params(model.unet)

    # MACs
    print("Counting MACs (ptflops)...")
    in_shape = (3, args.res, args.res)
    try:
        macs = count_macs(model, in_shape, in_dtype=dtype, device=device)
    except Exception as e:
        warnings.warn(f"ptflops failed: {e}. MACs reported as -1.")
        macs = -1

    # Report
    print()
    print("=" * 60)
    print(f"Original OSDFace — full one-step pipeline "
          f"(batch=1, {args.res}×{args.res}, {args.dtype})")
    print("-" * 60)
    print(f"{'Step Time (s)':<16}: {t:.4f}")
    print(f"{'Param (M)':<16}: {p/1e6:.2f}")
    if macs > 0:
        print(f"{'MACs (G)':<16}: {macs/1e9:.2f}")
    else:
        print(f"{'MACs (G)':<16}: n/a (ptflops failed)")
    print("-" * 60)
    print(f"  VRE (img_encoder): {p_vre/1e6:7.2f} M")
    print(f"  embedding_change : {p_proj/1e6:7.2f} M")
    print(f"  SD VAE           : {p_vae/1e6:7.2f} M")
    print(f"  SD UNet (+LoRA)  : {p_unet/1e6:7.2f} M")
    print("=" * 60)


if __name__ == "__main__":
    main()
