"""
experiment_b_pca.py — Revelio-style PCA feature visualization for OSDFace

Reproduces the analysis from Revelio Fig. 10, 11, 12:
    "up_ft1 captures very clear localized semantic information"

We compare three conditions on the same N test faces:
    (A) Baseline OSDFace — no FiLM active
    (B) OSDFace + FiLM   — your trained text conditioner active

For each condition we extract features from:
    - bottleneck  (up_blocks has no bottleneck per se — we use mid_block)
    - up_ft0      (up_blocks.0)
    - up_ft1      (up_blocks.1)  ← Revelio's key layer
    - up_ft2      (up_blocks.2)

We then run PCA across the batch (same as Revelio) and visualize the
first 3 principal components as RGB, producing a grid like Revelio Fig. 11.

Usage:
    python experiment_b_pca.py \
        --lq_dir            /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2 \
        --prompts_json      /projectnb/cs585/projects/craft/data/test/CelebA/celeba_prompts_output.json \
        --pretrained_model   /projectnb/cs585/projects/craft/osdface/pretrained/sd21\
        --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
        --ckpt_path         /projectnb/cs585/projects/craft/osdface/pretrained \
        --conditioner_ckpt  checkpoints/textcond_v5/checkpoint-50000/text_conditioner.pth \
        --output_dir        results/experiment_b \
        --n_images          8 \
        --mixed_precision   fp16

Output:
    results/experiment_b/
        pca_grid_bottleneck.png   — 2 rows (baseline / film), N cols
        pca_grid_up_ft0.png
        pca_grid_up_ft1.png       ← most important
        pca_grid_up_ft2.png
        norm_ratios.json          — per-layer FiLM norm perturbation
        layer_names.txt           — all hooked block names (for verification)
"""

import os
import json
import glob
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from safetensors import safe_open

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from transformers import CLIPTextModel, CLIPTokenizer

from lq_embed import vqvae_encoder, TwoLayerConv1x1
from text_conditioner import TextConditioner
from utils.others import get_x0_from_noise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_preamble(text: str) -> str:
    for marker in ["in the description of ", "not in the description of "]:
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
    return text.strip()


def merge_lora_into_unet(args, device="cpu"):
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet"
    )
    alpha = float(args.pretrained_lora_alpha / args.pretrained_lora_rank)
    processed_keys = set()

    lora_path = os.path.join(args.ckpt_path, "pytorch_lora_weights.safetensors")
    with safe_open(lora_path, framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    sd_unet = unet.state_dict()
    for key in state_dict:
        if "lora_A" in key:
            lora_b_key = key.replace("lora_A", "lora_B")
            unet_key   = key.replace(".lora_A.weight", ".weight").replace("unet.", "")
            W_A = state_dict[key]
            W_B = state_dict[lora_b_key]
            orig = sd_unet[unet_key]
            processed_keys.update([key, lora_b_key])
            if orig.ndim == 4:
                rank = W_A.shape[0]
                out_ch, in_ch, kH, kW = orig.shape
                delta = torch.matmul(
                    W_B.view(out_ch, rank),
                    W_A.view(rank, -1),
                ).view(out_ch, in_ch, kH, kW)
            else:
                delta = torch.mm(W_B, W_A)
            sd_unet[unet_key] = orig + alpha * delta
        elif "lora.up.weight" in key:
            lora_down_key = key.replace("lora.up.weight", "lora.down.weight")
            unet_key      = key.replace(".lora.up.weight", ".weight").replace("unet.", "")
            W_up   = state_dict[key]
            W_down = state_dict[lora_down_key]
            orig   = sd_unet[unet_key]
            processed_keys.update([key, lora_down_key])
            if orig.ndim == 2:
                sd_unet[unet_key] = orig + alpha * torch.mm(W_up, W_down)

    unet.load_state_dict(sd_unet)
    print("OSDFace LoRA merged.")
    return unet


@torch.no_grad()
def encode_text(prompts, tokenizer, text_encoder, device, dtype):
    ids = tokenizer(
        prompts, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    hidden = text_encoder(ids).last_hidden_state  # (B, 77, 1024)
    pooled = hidden.mean(dim=1).unsqueeze(1).expand(-1, 77, -1)
    return pooled.to(dtype)


# ---------------------------------------------------------------------------
# Layer → up_block mapping
# Revelio naming: bottleneck = mid_block, up_ft0/1/2 = up_blocks.0/1/2
# ---------------------------------------------------------------------------

LAYER_GROUPS = {
    "bottleneck": "mid_block",
    "up_ft0":     "up_blocks.0",
    "up_ft1":     "up_blocks.1",
    "up_ft2":     "up_blocks.2",
}


def register_feature_hooks(unet):
    """
    Registers forward hooks on the LAST BasicTransformerBlock in each
    layer group. Returns:
        captured: dict  name → list of activation tensors (filled during forward)
        hooks:    list  of hook handles (call h.remove() to clean up)

    We capture the last transformer block per group because:
    - It sees the most processed representation within that spatial scale
    - Matches Revelio's spatially-pooled activation approach
    """
    captured = {k: [] for k in LAYER_GROUPS}
    hooks    = []

    # Find the last BasicTransformerBlock in each group
    group_blocks = {k: [] for k in LAYER_GROUPS}
    for name, module in unet.named_modules():
        if isinstance(module, BasicTransformerBlock):
            for group_name, prefix in LAYER_GROUPS.items():
                if name.startswith(prefix):
                    group_blocks[group_name].append((name, module))

    # Save all found block names for verification
    all_block_names = {
        k: [n for n, _ in v] for k, v in group_blocks.items()
    }

    for group_name, blocks in group_blocks.items():
        if not blocks:
            print(f"WARNING: No BasicTransformerBlock found for {group_name} ({LAYER_GROUPS[group_name]})")
            continue

        # Use the last block in the group
        last_name, last_module = blocks[-1]
        print(f"Hooking {group_name} → {last_name}")

        def make_hook(gname):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                # h shape: (B, seq_len, channels) for transformer blocks
                # Convert to spatial: assume seq_len = H*W for the spatial scale
                captured[gname].append(h.detach().float().cpu())
            return hook

        handle = last_module.register_forward_hook(make_hook(group_name))
        hooks.append(handle)

    return captured, hooks, all_block_names


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# PCA visualization (Revelio Fig. 10/11 style)
# ---------------------------------------------------------------------------

def pca_to_rgb(features: torch.Tensor, n_components: int = 3) -> np.ndarray:
    """
    Run PCA across the batch on spatial features and map first 3 PCs → RGB.

    Args:
        features: (B, H, W, C) — spatial feature maps
    Returns:
        rgb_imgs: (B, H, W, 3) uint8 numpy array
    """
    B, H, W, C = features.shape
    X = features.reshape(B * H * W, C)  # (N, C)

    # Center
    X_mean = X.mean(dim=0)
    X_centered = X - X_mean

    # SVD-based PCA (same as Revelio — no sklearn dependency needed)
    try:
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    except Exception:
        # Fallback for older torch
        U, S, Vt = torch.svd(X_centered)
        Vt = Vt.T

    # Project onto first n_components
    components = Vt[:n_components]              # (3, C)
    projected  = X_centered @ components.T      # (N, 3)
    projected  = projected.reshape(B, H, W, 3)  # (B, H, W, 3)

    # Normalize per-component across the batch to [0, 1]
    # (same normalization Revelio uses so colors are comparable across images)
    for c in range(3):
        ch = projected[..., c]
        lo, hi = ch.min(), ch.max()
        projected[..., c] = (ch - lo) / (hi - lo + 1e-8)

    return (projected.numpy() * 255).astype(np.uint8)


def reshape_transformer_features(h: torch.Tensor, target_h: int, target_w: int):
    """
    Transformer block output is (B, seq_len, C).
    Reshape to (B, H, W, C) using the known spatial dimensions.
    """
    B, seq_len, C = h.shape
    H = target_h
    W = target_w
    assert H * W == seq_len, (
        f"seq_len={seq_len} doesn't match H={H} W={W}. "
        f"Check that target_h/w are correct for this layer."
    )
    return h.reshape(B, H, W, C)


# Revelio Fig 2: spatial resolution per layer for SD 1.5/2.1 at 512x512 input
# mid_block: 8x8, up_blocks.0: 8x8, up_blocks.1: 16x16, up_blocks.2: 32x32
LAYER_SPATIAL = {
    "bottleneck": (8,  8),
    "up_ft0":     (8,  8),
    "up_ft1":     (16, 16),
    "up_ft2":     (32, 32),
}


# ---------------------------------------------------------------------------
# Norm ratio: quantify how much FiLM perturbs each layer
# ---------------------------------------------------------------------------

def compute_norm_ratios(features_baseline, features_film):
    """
    For each layer, compute mean L2 norm ratio: ||film|| / ||baseline||
    > 1.0 means FiLM is amplifying activations
    < 1.0 means FiLM is suppressing activations
    = 1.0 means FiLM is identity (good for restoration layers)
    """
    ratios = {}
    for layer_name in LAYER_GROUPS:
        if not features_baseline[layer_name] or not features_film[layer_name]:
            continue
        base = torch.cat(features_baseline[layer_name], dim=0)  # (B, seq, C)
        film = torch.cat(features_film[layer_name],     dim=0)

        norm_base = base.norm(dim=-1).mean().item()
        norm_film = film.norm(dim=-1).mean().item()
        ratio     = norm_film / (norm_base + 1e-8)
        ratios[layer_name] = {
            "baseline_norm": round(norm_base, 4),
            "film_norm":     round(norm_film, 4),
            "ratio":         round(ratio, 4),
        }
        print(f"  {layer_name}: baseline={norm_base:.3f}  film={norm_film:.3f}  ratio={ratio:.3f}")
    return ratios


# ---------------------------------------------------------------------------
# Plot grid (Revelio Fig 11 style)
# ---------------------------------------------------------------------------

def save_pca_grid(
    pca_baseline: np.ndarray,   # (B, H, W, 3)
    pca_film:     np.ndarray,   # (B, H, W, 3)
    lq_images:    list,         # list of PIL images
    layer_name:   str,
    output_path:  str,
    img_size:     int = 128,
):
    """
    Saves a grid with 3 rows:
        Row 0: LQ input images
        Row 1: PCA maps — baseline OSDFace (no FiLM)
        Row 2: PCA maps — OSDFace + FiLM
    """
    B = pca_baseline.shape[0]
    fig, axes = plt.subplots(3, B, figsize=(B * 2.5, 3 * 2.5))
    if B == 1:
        axes = axes.reshape(3, 1)

    row_labels = ["LQ Input", "Baseline\n(no FiLM)", "FiLM\nconditioned"]

    for i in range(B):
        # Row 0: LQ input
        lq_resized = lq_images[i].resize((img_size, img_size))
        axes[0, i].imshow(lq_resized)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel(row_labels[0], fontsize=10, rotation=0,
                                   labelpad=60, va="center")

        # Row 1: Baseline PCA
        pca_b = Image.fromarray(pca_baseline[i]).resize((img_size, img_size),
                                                         Image.NEAREST)
        axes[1, i].imshow(pca_b)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel(row_labels[1], fontsize=10, rotation=0,
                                   labelpad=60, va="center")

        # Row 2: FiLM PCA
        pca_f = Image.fromarray(pca_film[i]).resize((img_size, img_size),
                                                     Image.NEAREST)
        axes[2, i].imshow(pca_f)
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel(row_labels[2], fontsize=10, rotation=0,
                                   labelpad=60, va="center")

    fig.suptitle(
        f"PCA Feature Maps — {layer_name}\n"
        f"(Revelio Fig. 11 style: spatially coherent = good restoration signal)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    # ── Load models ───────────────────────────────────────────────────────
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae"
    ).to(device, dtype=dtype)
    vae.requires_grad_(False).eval()

    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    print("Loading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder"
    ).to(device, dtype=dtype)
    text_encoder.requires_grad_(False).eval()

    print("Merging OSDFace LoRA into UNet...")
    unet = merge_lora_into_unet(args)
    unet.requires_grad_(False)
    unet.to(device, dtype=dtype)
    unet.eval()

    print("Loading VQ-VAE encoder...")
    img_encoder = vqvae_encoder(args).to(device, dtype=dtype)
    img_encoder.requires_grad_(False).eval()

    embedding_change = TwoLayerConv1x1(512, 1024)
    embedding_change.load_state_dict(
        torch.load(
            os.path.join(args.ckpt_path, "embedding_change_weights.pth"),
            weights_only=False,
        )
    )
    embedding_change.to(device, dtype=dtype)
    embedding_change.requires_grad_(False).eval()

    print("Loading TextConditioner...")
    conditioner = TextConditioner(unet, text_dim=1024)
    conditioner.load(args.conditioner_ckpt, map_location=device)
    conditioner.register_hooks(unet)
    conditioner.to(device)
    conditioner.eval()

    # ── Save all hooked block names for verification ───────────────────────
    _, tmp_hooks, all_block_names = register_feature_hooks(unet)
    remove_hooks(tmp_hooks)
    with open(os.path.join(args.output_dir, "layer_names.txt"), "w") as f:
        for group, names in all_block_names.items():
            f.write(f"\n=== {group} ({LAYER_GROUPS[group]}) ===\n")
            for n in names:
                f.write(f"  {n}\n")
    print(f"Layer names saved to {args.output_dir}/layer_names.txt")

    # ── Load test images ──────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    with open(args.prompts_json) as f:
        prompts_data = json.load(f)

    # Build filename → prompt lookup
    prompt_lookup = {}
    for item in prompts_data:
        stem, ext = os.path.splitext(item["image"])
        for key in [item["image"], f"{stem}_LQ{ext}", f"{stem}_lq{ext}"]:
            prompt_lookup[key] = item

    all_lq = sorted(glob.glob(os.path.join(args.lq_dir, "*")))
    random.seed(args.seed)
    random.shuffle(all_lq)
    selected = all_lq[:args.n_images]

    lq_tensors  = []
    lq_pil      = []
    pos_prompts = []
    na_prompts  = []

    for lq_path in selected:
        fname  = os.path.basename(lq_path)
        entry  = prompt_lookup.get(fname, {"pos": "", "na": ""})
        img    = Image.open(lq_path).convert("RGB")
        lq_pil.append(img)
        t = transform(img) * 2.0 - 1.0
        lq_tensors.append(t)
        pos_prompts.append(strip_preamble(entry.get("pos", "")))
        na_prompts.append(strip_preamble(entry.get("na",  "")))

    lq_batch = torch.stack(lq_tensors).to(device, dtype=dtype)  # (B, 3, 512, 512)
    B = lq_batch.shape[0]
    timestep = 399

    print(f"\nRunning on {B} images at timestep {timestep}...")

    # ── Compute shared inputs (same for both conditions) ──────────────────
    with torch.no_grad():
        visual_embeds = embedding_change(
            img_encoder(lq_batch).reshape(B, 77, -1)
        )  # (B, 77, 1024)

        pos_embeds = encode_text(pos_prompts, tokenizer, text_encoder, device, dtype)
        na_embeds  = encode_text(na_prompts,  tokenizer, text_encoder, device, dtype)

        lq_latent = vae.encode(lq_batch).latent_dist.sample() * vae.config.scaling_factor

    # ── Condition A: Baseline — no FiLM ──────────────────────────────────
    print("\n[A] Extracting baseline features (no FiLM)...")
    conditioner.clear_text_embedding()
    captured_baseline, hooks_baseline, _ = register_feature_hooks(unet)

    with torch.no_grad():
        _ = unet(lq_latent, timestep, encoder_hidden_states=visual_embeds).sample

    remove_hooks(hooks_baseline)

    # ── Condition B: FiLM active ──────────────────────────────────────────
    print("[B] Extracting FiLM-conditioned features...")
    conditioner.set_text_embedding(pos_embeds, na_embeds, neg_weight=args.film_neg_weight)
    captured_film, hooks_film, _ = register_feature_hooks(unet)

    with torch.no_grad():
        _ = unet(lq_latent, timestep, encoder_hidden_states=visual_embeds).sample

    conditioner.clear_text_embedding()
    remove_hooks(hooks_film)

    # ── Norm ratios ────────────────────────────────────────────────────────
    print("\nNorm ratios (film/baseline) per layer:")
    norm_ratios = compute_norm_ratios(captured_baseline, captured_film)
    with open(os.path.join(args.output_dir, "norm_ratios.json"), "w") as f:
        json.dump(norm_ratios, f, indent=2)
    print(f"Norm ratios saved → {args.output_dir}/norm_ratios.json")

    # ── PCA visualization per layer ────────────────────────────────────────
    print("\nGenerating PCA visualizations...")
    for layer_name, (H, W) in LAYER_SPATIAL.items():
        feats_b = captured_baseline[layer_name]
        feats_f = captured_film[layer_name]

        if not feats_b or not feats_f:
            print(f"  Skipping {layer_name} — no features captured")
            continue

        # Each captured item is (B, seq_len, C) — take the first (and only) batch
        hb = feats_b[0]  # (B, seq_len, C)
        hf = feats_f[0]

        # Verify spatial dimensions match expectation
        if hb.shape[1] != H * W:
            print(
                f"  WARNING: {layer_name} expected seq_len={H*W} "
                f"but got {hb.shape[1]}. Trying to infer H/W..."
            )
            seq_len = hb.shape[1]
            H = W = int(seq_len ** 0.5)
            if H * W != seq_len:
                print(f"  Cannot reshape — skipping {layer_name}")
                continue

        hb_spatial = reshape_transformer_features(hb, H, W)  # (B, H, W, C)
        hf_spatial = reshape_transformer_features(hf, H, W)

        # Stack all images for PCA (Revelio does PCA across the batch)
        pca_b = pca_to_rgb(hb_spatial)  # (B, H, W, 3)
        pca_f = pca_to_rgb(hf_spatial)

        out_path = os.path.join(args.output_dir, f"pca_grid_{layer_name}.png")
        save_pca_grid(pca_b, pca_f, lq_pil, layer_name, out_path)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT B COMPLETE")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print("\nWhat to look for (Revelio Fig 10/11 reference):")
    print("  up_ft1 baseline: spatially coherent color regions per face region")
    print("  up_ft1 FiLM:     if colors are blended/incoherent → layer corrupted")
    print("  up_ft2 baseline: fine-grained texture patterns (high freq)")
    print("  up_ft2 FiLM:     high norm ratio here = source of MUSIQ/NIQE collapse")
    print("\nNorm ratio interpretation:")
    print("  ratio ≈ 1.0 → FiLM is near-identity (safe)")
    print("  ratio >> 1.0 → FiLM is amplifying (destructive for restoration)")
    print("  ratio << 1.0 → FiLM is suppressing (may lose face prior)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lq_dir",           required=True)
    p.add_argument("--prompts_json",     required=True)
    p.add_argument("--pretrained_model", default="pretrained/sd21")
    p.add_argument("--img_encoder_weight", default="pretrained/associate_2.ckpt")
    p.add_argument("--ckpt_path",        required=True)
    p.add_argument("--conditioner_ckpt", required=True,
                   help="Path to text_conditioner.pth from your training run")
    p.add_argument("--output_dir",       default="results/experiment_b")
    p.add_argument("--n_images",         type=int,   default=8)
    p.add_argument("--film_neg_weight",  type=float, default=0.5)
    p.add_argument("--mixed_precision",  choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--pretrained_lora_rank",  type=int,   default=16)
    p.add_argument("--pretrained_lora_alpha", type=float, default=16)
    # VRE passthrough
    p.add_argument("--cat_prompt_embedding", action="store_true")
    p.add_argument("--use_pos_embedding",    action="store_true")
    p.add_argument("--use_att_pool",         action="store_true")
    p.add_argument("--learnable_pos_emb",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    run(args)
