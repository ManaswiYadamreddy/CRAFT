"""
infer_craft.py — Run CRAFT on a directory of LQ face images.

Supports both stages:

    --stage 1   CRAFT Stage-1 VQVAE (region-aware, checkpoints/phase_d/final.pt)
                Direct LQ → HQ reconstruction through the learned codebook.

    --stage 2   CRAFT Stage-2 diffusion (SD 2.1 + LoRA + Stage-1 VRE)
                Delegates to infer_concat.py with CRAFT's Stage-1 encoder as the
                visual representation embedder. The Stage-2 checkpoint directory
                must contain `pytorch_lora_weights.safetensors` and
                `embedding_change_weights.pth`. If --stage2_ckpt points to a
                file (e.g. final.pt), we use its parent directory.

Usage:
    # Stage 1
    python -m Evaluation.inference.infer_craft \
        --stage 1 \
        --stage1_ckpt /projectnb/.../checkpoints/phase_d/final.pt \
        --parser_ckpt /projectnb/.../pretrained/79999_iter.pth \
        --input_dir   /projectnb/.../LQ \
        --output_dir  /projectnb/.../results/craft_s1

    # Stage 2
    python -m Evaluation.inference.infer_craft \
        --stage 2 \
        --stage1_ckpt /projectnb/.../checkpoints/phase_d/final.pt \
        --stage2_ckpt /projectnb/.../checkpoints_stage2/final.pt \
        --parser_ckpt /projectnb/.../pretrained/79999_iter.pth \
        --input_dir   /projectnb/.../LQ \
        --output_dir  /projectnb/.../results/craft_s2 \
        --merge_lora --mixed_precision fp16
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

# Repo root (parent of Evaluation/) on sys.path so `models.*` imports resolve.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


# ══════════════════════════════════════════════════════════════════════════
# Stage 1 — CRAFT VQVAE (region-aware) direct reconstruction
# ══════════════════════════════════════════════════════════════════════════

def _load_craft_stage1(
    ckpt_path: str, device: torch.device, parser_ckpt: str,
    embed_dim: int = 512, rq_levels: int = 3,
):
    from models.region_aware_vq import RegionAwareVQ
    from models.vqvae import build_lq_vqvae

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt["lq_model"]

    has_mag_head = any("magnitude_head" in k for k in state.keys())
    if has_mag_head:
        print("  Detected magnitude_head → Phase D checkpoint")

    ravq = RegionAwareVQ(
        e_dim=embed_dim, n_levels=rq_levels,
        parser_ckpt=parser_ckpt, use_magnitude_head=has_mag_head,
    )
    model = build_lq_vqvae(ravq, embed_dim=embed_dim)
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
    print("Loading CRAFT Stage-1 model...")
    model = _load_craft_stage1(
        ckpt_path=args.stage1_ckpt, device=device,
        parser_ckpt=args.parser_ckpt,
        embed_dim=args.embed_dim, rq_levels=args.rq_levels,
    )
    print(f"  loaded from {args.stage1_ckpt}")

    os.makedirs(args.output_dir, exist_ok=True)
    paths = _list_images(args.input_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {args.input_dir}")
    print(f"Found {len(paths)} images in {args.input_dir}")

    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    for path in tqdm(paths, desc="CRAFT-S1"):
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

        x_rec, *_ = model(x11, images_01=x01)
        x_rec01 = ((x_rec.clamp(-1, 1) + 1) / 2).float()
        to_pil(x_rec01[0].cpu()).save(out_path)


# ══════════════════════════════════════════════════════════════════════════
# Stage 2 — SD 2.1 + LoRA + Stage-1 VRE (delegates to infer_concat.py)
# ══════════════════════════════════════════════════════════════════════════

def _normalize_stage2_ckpt_dir(p: str) -> str:
    """
    infer_concat.py expects `ckpt_path` to be a DIRECTORY containing
    pytorch_lora_weights.safetensors + embedding_change_weights.pth.
    If the user passes `/…/final.pt` (file), we fall back to its parent dir.
    """
    if os.path.isdir(p):
        return p
    if os.path.isfile(p) or p.endswith(".pt"):
        return os.path.dirname(p)
    return p


def _run_stage2(args, device: torch.device):
    # Lazy import to avoid pulling diffusers/peft when running stage 1
    import random
    import numpy as np
    import torch.multiprocessing as mp
    from infer_concat import merge_unet, run_inference

    ckpt_dir = _normalize_stage2_ckpt_dir(args.stage2_ckpt)
    if ckpt_dir != args.stage2_ckpt:
        print(f"  [stage2] Using ckpt directory: {ckpt_dir}")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Stage-2 checkpoint dir not found: {ckpt_dir}")

    # Build an argparse.Namespace compatible with infer_concat.run_inference
    s2_args = argparse.Namespace(
        input_image     = args.input_dir,
        output_dir      = args.output_dir,
        ckpt_path       = ckpt_dir,
        img_encoder_weight = args.stage1_ckpt,   # CRAFT's Stage-1 acts as VRE
        prompts_json    = args.prompts_json or None,
        pretrained_model_name_or_path = args.pretrained_model,
        mixed_precision = args.mixed_precision,
        gpu_ids         = args.gpu_ids,
        merge_lora      = args.merge_lora,
        lora_rank       = args.lora_rank,
        lora_alpha      = args.lora_alpha,
        seed            = args.seed,
        cat_prompt_embedding = False,
        use_pos_embedding    = False,
        use_att_pool         = False,
        learnable_pos_emb    = False,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    os.makedirs(args.output_dir, exist_ok=True)
    unet_merged = merge_unet(s2_args) if args.merge_lora else None
    run_inference(s2_args, unet_merged)


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)

    # Data
    ap.add_argument("--input_dir",  required=True)
    ap.add_argument("--output_dir", required=True)

    # Stage-1 checkpoints (required for stage 1, and as VRE for stage 2)
    ap.add_argument("--stage1_ckpt", required=True,
                    help="CRAFT Stage-1 VQVAE checkpoint (phase_d/final.pt)")
    ap.add_argument("--parser_ckpt", required=True,
                    help="BiSeNet face parser (79999_iter.pth)")
    ap.add_argument("--embed_dim",   type=int, default=512)
    ap.add_argument("--rq_levels",   type=int, default=3)
    ap.add_argument("--resolution",  type=int, default=512)

    # Stage-2 checkpoints + SD settings
    ap.add_argument("--stage2_ckpt", default="",
                    help="CRAFT Stage-2 LoRA checkpoint dir (or any file inside it)")
    ap.add_argument("--pretrained_model", default="stabilityai/stable-diffusion-2-1-base")
    ap.add_argument("--mixed_precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--lora_rank",  type=int,   default=16)
    ap.add_argument("--lora_alpha", type=float, default=16)
    ap.add_argument("--prompts_json", default=None)
    ap.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    ap.add_argument("--seed",  type=int, default=42)

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
