"""
infer_osdface.py — Run OSDFace on a directory of LQ face images.

Supports both stages:

    --stage 1   OSDFace Stage-1 VQVAE (flat GlobalVQ; no region parser)
                Checkpoint from train_osdface_stage1.py.

    --stage 2   OSDFace Stage-2 one-step diffusion (SD 1.5 + LoRA + Stage-1).
                Loads `final.pt` from train_stage2_osdface.py — a dict with
                {iter, generator, discriminator, optim_g, optim_d, args}. We
                rebuild Stage2GeneratorOSDFace and load the `generator` state
                dict (LoRA + prompt_proj + prompt_ln) onto the frozen SD 1.5
                backbone + OSDFace Stage-1 encoder.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Put repo root on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


# ══════════════════════════════════════════════════════════════════════════
# Stage 1 — OSDFace VQVAE (flat GlobalVQ) direct reconstruction
# ══════════════════════════════════════════════════════════════════════════

def _load_osdface_stage1(
    ckpt_path: str, device: torch.device,
    embed_dim: int = 512, lq_n_codes: int = 1024,
):
    """OSDFace Stage-1 uses CRAFT's build_hq_vqvae (flat GlobalVQ, cosine)."""
    from models.vqvae import build_hq_vqvae

    model = build_hq_vqvae(n_codes=lq_n_codes, embed_dim=embed_dim)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt["lq_model"]
    model.load_state_dict(state)
    return model.to(device).eval()


def _list_images(input_dir: str) -> list[str]:
    paths: list[str] = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(input_dir, ext))
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(input_dir, "*", ext))
    return sorted(set(paths))


@torch.no_grad()
def _run_stage1(args, device: torch.device):
    print("Loading OSDFace Stage-1 model...")
    model = _load_osdface_stage1(
        ckpt_path=args.stage1_ckpt, device=device,
        embed_dim=args.embed_dim, lq_n_codes=args.lq_n_codes,
    )
    print(f"  loaded from {args.stage1_ckpt}")

    os.makedirs(args.output_dir, exist_ok=True)
    paths = _list_images(args.input_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {args.input_dir}")
    print(f"Found {len(paths)} images in {args.input_dir}")

    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    for path in tqdm(paths, desc="OSDFace-S1"):
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output_dir, f"{name}.png")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        img = Image.open(path).convert("RGB")
        x01 = to_tensor(img).unsqueeze(0).to(device)
        if x01.shape[-2:] != (args.resolution, args.resolution):
            x01 = F.interpolate(x01, size=(args.resolution, args.resolution),
                                mode="bilinear", align_corners=False)
        x11 = x01 * 2.0 - 1.0

        # OSDFace Stage-1 VQVAE's forward: model(x) returns (x_rec, ...)
        # (the flat VQVAE does not require images_01 since there's no parser)
        out = model(x11)
        x_rec = out[0] if isinstance(out, (list, tuple)) else out
        x_rec01 = ((x_rec.clamp(-1, 1) + 1) / 2).float()
        to_pil(x_rec01[0].cpu()).save(out_path)


# ══════════════════════════════════════════════════════════════════════════
# Stage 2 — SD 1.5 + LoRA + OSDFace Stage-1 VRE (one-step diffusion)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _run_stage2(args, device: torch.device):
    from models.stage2_generator_osdface import Stage2GeneratorOSDFace

    if not os.path.isfile(args.stage2_ckpt):
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {args.stage2_ckpt}")

    print("Loading OSDFace Stage-2 generator...")
    gen = Stage2GeneratorOSDFace(
        stage1_ckpt_path=args.stage1_ckpt,
        sd_model_name=args.pretrained_model,
        t_fixed=args.t_fixed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        prompt_dim=args.embed_dim,
        context_dim=args.context_dim,
        stage1_n_codes=args.lq_n_codes,
    )
    print(f"  {gen.describe()}")

    ckpt = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
    gen_state = ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt
    missing, unexpected = gen.load_state_dict(gen_state, strict=False)
    trainable_missing = [k for k in missing if "lora" in k.lower() or k.startswith(("prompt_proj", "prompt_ln"))]
    if trainable_missing:
        raise RuntimeError(
            f"Stage-2 ckpt is missing {len(trainable_missing)} trainable keys, "
            f"e.g. {trainable_missing[:3]}"
        )
    if unexpected:
        print(f"  [stage2] WARNING: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    print(f"  loaded {len(gen_state)} tensors from {args.stage2_ckpt} (iter={ckpt.get('iter', '?')})")

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    gen.to(device=device, dtype=weight_dtype)
    gen.eval()

    paths = _list_images(args.input_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(paths)} images in {args.input_dir}")

    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    for path in tqdm(paths, desc="OSDFace-S2"):
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output_dir, f"{name}.png")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        img = Image.open(path).convert("RGB")
        x01 = to_tensor(img).unsqueeze(0).to(device, dtype=weight_dtype)
        if x01.shape[-2:] != (args.resolution, args.resolution):
            x01 = F.interpolate(x01, size=(args.resolution, args.resolution),
                                mode="bilinear", align_corners=False)
        x11 = x01 * 2.0 - 1.0

        I_hat, *_ = gen(x11)
        I_hat01 = ((I_hat.clamp(-1, 1) + 1) / 2).float()
        to_pil(I_hat01[0].cpu()).save(out_path)


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)

    ap.add_argument("--input_dir",  required=True)
    ap.add_argument("--output_dir", required=True)

    # Stage-1 (required; also used as VRE for stage 2)
    ap.add_argument("--stage1_ckpt", required=True,
                    help="OSDFace Stage-1 VQVAE checkpoint "
                         "(checkpoints_osdface/phase_c/final.pt)")
    ap.add_argument("--embed_dim",  type=int, default=512)
    ap.add_argument("--lq_n_codes", type=int, default=1024)
    ap.add_argument("--resolution", type=int, default=512)

    # Stage-2
    ap.add_argument("--stage2_ckpt", default="",
                    help="Path to train_stage2_osdface.py final.pt (the bundle dict)")
    ap.add_argument("--pretrained_model",
                    default="sd-legacy/stable-diffusion-v1-5",
                    help="Local SD 1.5 directory (or HF id if reachable)")
    ap.add_argument("--mixed_precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--lora_rank",  type=int,   default=16)
    ap.add_argument("--lora_alpha", type=float, default=16)
    ap.add_argument("--t_fixed",    type=int,   default=999)
    ap.add_argument("--context_dim", type=int,  default=768,
                    help="UNet cross-attention dim (768 for SD 1.5)")
    ap.add_argument("--seed",  type=int, default=42)
    # Accepted-but-ignored (kept for run_eval.sh backward compat)
    ap.add_argument("--merge_lora", action="store_true",
                    help="(no-op) — LoRA stays attached via PEFT for stage 2")
    ap.add_argument("--prompts_json", default=None,
                    help="(no-op) — stage 2 generator does not use text prompts")
    ap.add_argument("--gpu_ids", nargs="+", type=int, default=[0],
                    help="(no-op) — single-GPU inference")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   stage: {args.stage}")

    if args.stage == 1:
        _run_stage1(args, device)
    else:
        if not args.stage2_ckpt:
            raise ValueError("--stage2_ckpt is required for stage 2")
        _run_stage2(args, device)

    print(f"\nSaved restorations → {args.output_dir}")


if __name__ == "__main__":
    main()
