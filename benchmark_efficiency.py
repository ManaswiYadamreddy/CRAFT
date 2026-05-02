"""
benchmark_efficiency.py — Step Time / Params / MACs for the CRAFT and OSDFace
final models.

Two modes:

    --mode full   (default)  Time the FULL one-step diffusion pipeline:
                  VRE (encode + quantize) → adapter → SD-2.1 VAE encode
                  → SD-2.1 UNet (one timestep) → VAE decode.
                  This is what the OSDFace paper's "Step Time" column reports.

    --mode vre    Time only the Stage 1 VRE (encode → quantize → decode).
                  Useful as an ablation comparing CRAFT's region-aware VQ to
                  OSDFace's flat VQ in isolation. NOT the paper's number.

Reported columns:

    Step Time (s)  : average wall time of one forward pass (batch 1, 512×512).
    Param (M)      : trainable + frozen parameter count, in millions.
    MACs (G)       : multiply-accumulate ops per forward pass, in giga.
                     (1 MAC ≈ 2 FLOPs.)

Usage
-----
Run from inside the CRAFT repo so `models.*` is importable.

  # Full pipeline (paper's number) — needs SD 2.1 weights:
  python benchmark_efficiency.py \\
      --mode full \\
      --craft_ckpt   /path/to/checkpoints/phase_c/final.pt \\
      --osdface_ckpt /path/to/checkpoints_osdface/phase_c/final.pt \\
      --parser_ckpt  /path/to/79999_iter.pth \\
      --sd_model     stabilityai/stable-diffusion-2-1-base \\
      --dtype fp16

  # VRE only (Stage 1 ablation, no SD download required):
  python benchmark_efficiency.py --mode vre \\
      --craft_ckpt   /path/to/checkpoints/phase_c/final.pt \\
      --osdface_ckpt /path/to/checkpoints_osdface/phase_c/final.pt

Notes
-----
* The SD components (UNet + VAE + CLIP) and the (B,256,512)→(B,77,1024)
  adapter are identical for the CRAFT and OSDFace runs — only the VRE
  swaps. So the difference between the two rows is exactly what's
  attributable to the VRE choice.
* The adapter is a small untrained Conv1d + Linear. Output quality is
  meaningless without a trained Stage 2; we are measuring time / MACs /
  params, not reconstruction fidelity.
* MACs are counted with `ptflops`, which hooks standard nn modules
  (Conv2d, Linear, GroupNorm, …) and does NOT count raw torch.matmul
  inside attention blocks. Same convention every diffusion paper uses.
* `--lora_ckpt` (optional) merges OSDFace's pretrained LoRA into the UNet
  the same way `infer_concat.py` does. Step Time / Param / MACs don't
  depend on the merged weights — pass it only if you want the same exact
  UNet that's used at inference time.
"""

import argparse
import time
import warnings

import torch
import torch.nn as nn

from models.vqvae import build_hq_vqvae, build_lq_vqvae
from models.region_aware_vq import RegionAwareVQ


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 model loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_craft(ckpt_path, parser_ckpt, device, embed_dim=512, rq_levels=3):
    """CRAFT Stage 1 LQ branch: VQVAE + RegionAwareVQ (Phase C / Phase D)."""
    state, has_mag_head = None, False
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        key = "model" if "model" in ckpt else "lq_model"
        state = ckpt[key]
        has_mag_head = any("magnitude_head" in k for k in state.keys())
        if has_mag_head:
            print("  CRAFT: detected magnitude_head → Phase D checkpoint")

    ravq = RegionAwareVQ(
        e_dim=embed_dim,
        n_levels=rq_levels,
        parser_ckpt=parser_ckpt,
        use_magnitude_head=has_mag_head,
    )
    model = build_lq_vqvae(ravq, embed_dim=embed_dim)
    if state is not None:
        model.load_state_dict(state)
    return model.to(device).eval()


def load_osdface(ckpt_path, device, embed_dim=512, lq_n_codes=1024):
    """OSDFace Stage 1 LQ branch: VQVAE + flat GlobalVQ (Phase C)."""
    model = build_hq_vqvae(n_codes=lq_n_codes, embed_dim=embed_dim)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        key = "model" if "model" in ckpt else "lq_model"
        model.load_state_dict(ckpt[key])
    return model.to(device).eval()


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers
# ──────────────────────────────────────────────────────────────────────────────

class VREOnlyWrapper(nn.Module):
    """Stage 1 forward: encode → quantize → decode."""
    def __init__(self, model, images_01=None, masks=None):
        super().__init__()
        self.model = model
        self.images_01 = images_01
        self.masks = masks

    def forward(self, x):
        x_rec, *_ = self.model(x, images_01=self.images_01, masks=self.masks)
        return x_rec


class VRETokens(nn.Module):
    """Run VRE as a prompt extractor: returns flattened tokens (B, 256, 512).

    No decoder pass — Stage 2 only needs the quantized features as a prompt.
    """
    def __init__(self, vqvae, images_01=None, masks=None):
        super().__init__()
        self.vqvae = vqvae
        self.images_01 = images_01
        self.masks = masks

    def forward(self, x):
        z = self.vqvae.encode(x)  # (B, 512, 16, 16)
        z_q, _, _ = self.vqvae.quantizer(z, images=self.images_01, masks=self.masks)
        B, C, H, W = z_q.shape
        return z_q.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 256, 512)


class FullPipeline(nn.Module):
    """One complete forward pass of OSDFace-style one-step diffusion.

    VRE → adapter (256→77 tokens, 512→1024 ch) → cat with empty CLIP text →
    SD VAE encode → SD UNet (single timestep) → SD VAE decode.
    """
    def __init__(self, vre_tokens: VRETokens, sd_model: str, device, dtype,
                 lora_ckpt: str = None, timestep: int = 399):
        super().__init__()
        from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPTokenizer, CLIPTextModel

        self.dtype = dtype
        self.timestep = timestep

        # ── SD 2.1 components ──────────────────────────────────────────────
        scheduler = DDIMScheduler.from_pretrained(sd_model, subfolder="scheduler")
        # alphas_cumprod indexed by self.timestep
        alpha_t = scheduler.alphas_cumprod[timestep]
        self.register_buffer("alpha_t", alpha_t.to(device, dtype=dtype))

        self.vae = AutoencoderKL.from_pretrained(
            sd_model, subfolder="vae"
        ).to(device, dtype=dtype).eval()

        unet = UNet2DConditionModel.from_pretrained(sd_model, subfolder="unet")
        if lora_ckpt:
            from peft import PeftModel
            unet = PeftModel.from_pretrained(unet, lora_ckpt).merge_and_unload()
        self.unet = unet.to(device, dtype=dtype).eval()

        tokenizer = CLIPTokenizer.from_pretrained(sd_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            sd_model, subfolder="text_encoder"
        ).to(device, dtype=dtype).eval()
        # Cache empty-prompt embedding so we don't tokenize in the inner loop.
        with torch.no_grad():
            tok = tokenizer(
                [""], padding="max_length",
                max_length=tokenizer.model_max_length, return_tensors="pt",
            ).input_ids.to(device)
            text_emb = text_encoder(tok).last_hidden_state.to(dtype)
        self.register_buffer("text_emb", text_emb)
        # text_encoder is dropped after caching — it's not part of the timed path.

        # ── VRE + adapter (only piece that differs between runs) ───────────
        self.vre_tokens = vre_tokens.to(device).eval()
        self.token_reduce = nn.Conv1d(256, 77, kernel_size=1, bias=False).to(device, dtype=dtype)
        self.embed_proj   = nn.Linear(512, 1024).to(device, dtype=dtype)

    @torch.no_grad()
    def forward(self, x):
        # `x` is the LQ image in [-1, 1], shape (B, 3, 512, 512). Named `x`
        # (not `lq`) so ptflops' input_constructor kwargs match.
        B = x.shape[0]
        # VRE → (B, 256, 512)
        tokens = self.vre_tokens(x).to(self.dtype)
        # (B, 256, 512) → (B, 77, 512) → (B, 77, 1024)
        tokens = self.token_reduce(tokens)
        visual_embeds = self.embed_proj(tokens)
        # Concat with cached text embedding → (B, 154, 1024)
        text_embeds = self.text_emb.expand(B, -1, -1)
        prompt_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        # SD VAE encode
        lq_lat = (
            self.vae.encode(x.to(self.dtype)).latent_dist.sample()
            * self.vae.config.scaling_factor
        )
        # UNet single ε-prediction step
        eps = self.unet(
            lq_lat, self.timestep, encoder_hidden_states=prompt_embeds,
        ).sample
        # x_0 from noise: x0 = (x_t − √(1−ᾱ) · ε) / √(ᾱ)
        a = self.alpha_t
        x0 = (lq_lat - (1 - a).sqrt() * eps) / a.sqrt()
        # SD VAE decode
        return self.vae.decode(x0 / self.vae.config.scaling_factor).sample


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, input_shape, input_dtype=torch.float32, device="cpu"):
    """Returns MACs as a Python int. Uses ptflops."""
    from ptflops import get_model_complexity_info

    def constructor(shape):
        # ptflops calls model(x) with x of shape (1, *shape) — match our dtype.
        return {"x": torch.randn(1, *shape, device=device, dtype=input_dtype)}

    with torch.no_grad():
        macs, _ = get_model_complexity_info(
            model, input_shape,
            input_constructor=constructor,
            as_strings=False, print_per_layer_stat=False,
            verbose=False, backend="pytorch",
        )
    return int(macs)


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

def benchmark_one(name, wrapped, x, n_warmup, n_iter, device, in_shape,
                  in_dtype, extra_param_count=0, sub_param_to_exclude=0):
    """Run all three measurements and return a dict."""
    print(f"\n[{name}] timing...")
    t = time_step(wrapped, x, n_warmup, n_iter, device)
    p = count_params(wrapped) - sub_param_to_exclude + extra_param_count
    print(f"[{name}] counting MACs (ptflops)...")
    try:
        m = count_macs(wrapped, in_shape, input_dtype=in_dtype, device=device)
    except Exception as e:
        warnings.warn(f"[{name}] ptflops failed: {e}. MACs reported as -1.")
        m = -1
    return {"step_s": t, "param": p, "macs": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vre", "full"], default="full",
                    help="vre = Stage 1 VQVAE only; "
                         "full = one-step diffusion pipeline (paper).")
    ap.add_argument("--craft_ckpt",   default=None)
    ap.add_argument("--osdface_ckpt", default=None)
    ap.add_argument("--parser_ckpt",  default=None)
    # full-mode args
    ap.add_argument("--sd_model",
                    default="stabilityai/stable-diffusion-2-1-base",
                    help="HF repo id or local path of SD 2.1 base.")
    ap.add_argument("--lora_ckpt", default=None,
                    help="Optional dir with pytorch_lora_weights.safetensors "
                         "to merge into the UNet (matches infer_concat.py).")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--timestep", type=int, default=399)
    # shared
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--n_warmup", type=int, default=10)
    ap.add_argument("--n_iter",   type=int, default=50)
    ap.add_argument("--include_parser", action="store_true",
                    help="Include the BiSeNet face parser in CRAFT timing/MACs.")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"Mode: {args.mode}    Device: {device}    "
          f"Res: {args.res}×{args.res}    dtype: {args.dtype}")

    # ── Load Stage 1 VREs ──────────────────────────────────────────────────
    print("\nLoading CRAFT VRE...")
    craft = load_craft(args.craft_ckpt, args.parser_ckpt, device)
    print("Loading OSDFace VRE...")
    osdface = load_osdface(args.osdface_ckpt, device)

    # ── Inputs + (optional) precomputed face-parser masks ──────────────────
    x_11_f32 = torch.randn(1, 3, args.res, args.res, device=device)
    x_01     = (x_11_f32.clamp(-1, 1) + 1) / 2

    if args.include_parser:
        masks = None
    else:
        with torch.no_grad():
            masks = craft.quantizer.face_parser.get_region_masks(
                x_01, target_h=args.res // 32, target_w=args.res // 32,
            )
    p_parser = count_params(craft.quantizer.face_parser)

    # ── Build the wrappers we'll benchmark ─────────────────────────────────
    if args.mode == "vre":
        in_dtype = torch.float32
        x = x_11_f32
        craft_w   = VREOnlyWrapper(craft, images_01=x_01, masks=masks)
        osdface_w = VREOnlyWrapper(osdface)
    else:  # full
        in_dtype = dtype
        x = x_11_f32.to(dtype)
        # Cast VREs to the run dtype. Their internal float32 promotion in
        # the quantizer still works because tensors are upcast there.
        craft_t   = VRETokens(craft.to(dtype),   images_01=x_01.to(dtype), masks=masks)
        osdface_t = VRETokens(osdface.to(dtype))
        print(f"\nLoading SD 2.1 from {args.sd_model} (this may download / take a moment)...")
        craft_w = FullPipeline(
            craft_t, args.sd_model, device, dtype,
            lora_ckpt=args.lora_ckpt, timestep=args.timestep,
        )
        osdface_w = FullPipeline(
            osdface_t, args.sd_model, device, dtype,
            lora_ckpt=args.lora_ckpt, timestep=args.timestep,
        )

    in_shape = (3, args.res, args.res)

    # ── Run benchmarks ─────────────────────────────────────────────────────
    # Exclude face parser from CRAFT's reported param count when masks are
    # precomputed (paper convention: parser is upstream, not part of the model).
    craft_excl = 0 if args.include_parser else p_parser
    craft_res   = benchmark_one(
        "CRAFT", craft_w, x, args.n_warmup, args.n_iter, device,
        in_shape, in_dtype, sub_param_to_exclude=craft_excl,
    )
    osdface_res = benchmark_one(
        "OSDFace", osdface_w, x, args.n_warmup, args.n_iter, device,
        in_shape, in_dtype,
    )

    # ── Report ─────────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    title = "Full one-step pipeline" if args.mode == "full" else "Stage 1 VRE only"
    print(f"{title}  —  batch=1, {args.res}×{args.res}, dtype={args.dtype}")
    print(f"{'Model':<10} {'Step Time (s)':>16} {'Param (M)':>14} {'MACs (G)':>14}")
    print("-" * 64)
    for name, r in [("CRAFT", craft_res), ("OSDFace", osdface_res)]:
        macs_str = f"{r['macs']/1e9:>14.2f}" if r["macs"] > 0 else f"{'n/a':>14}"
        print(f"{name:<10} {r['step_s']:>16.4f} "
              f"{r['param']/1e6:>14.2f} {macs_str}")
    print("=" * 64)
    if not args.include_parser:
        print(f"(CRAFT excludes BiSeNet face parser: "
              f"{p_parser/1e6:.2f} M params, masks precomputed.)")
    if args.mode == "full":
        print("(Full pipeline = VRE + adapter + SD-2.1 VAE encode + UNet "
              "+ VAE decode.\n SD components and adapter are identical "
              "between the two rows; only the VRE swaps.)")


if __name__ == "__main__":
    main()
