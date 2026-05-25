"""
experiment_full.py — Full Revelio-style interpretability analysis for OSDFace

Runs four analyses on the NEW trained checkpoint:

  1. PCA feature maps (Revelio Fig 11 style)
     Baseline vs correct-prompt vs swapped-prompt, per layer.
     4 rows x N cols per layer.

  2. Three-condition norm ratio table (Revelio Sec 4.3 style)
     For each layer: ||correct|| / ||baseline|| and ||swapped|| / ||baseline||
     Asymmetry between correct and swapped = FiLM is prompt-sensitive.

  3. Top-activating image grid (Revelio Fig 1/3 style)
     Ranks test images by feature delta between correct and swapped prompt.
     Top images = where FiLM has the strongest effect.

  4. Label purity per layer (Revelio Table 2 style)
     If --attributes_json provided: measures how well features cluster
     by binary attribute (e.g. glasses/no glasses) per layer and condition.

Usage:
    python experiment_full.py \
        --lq_dir              /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2 \
        --prompts_json        /projectnb/cs585/projects/craft/data/test/CelebA/celeba_prompts_output.json \
        --pretrained_model    /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \
        --img_encoder_weight  /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
        --ckpt_path           /projectnb/cs585/projects/craft/osdface/pretrained \
        --conditioner_ckpt    checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
        --output_dir          results/experiment_full \
        --n_images            8 \
        --attributes_json     data/test/attributes.json \
        --probe_attribute     eyeglasses \
        --mixed_precision     fp16


    python experiment_full.py \
        --lq_dir              /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2 \
        --prompts_json        /projectnb/cs585/projects/craft/data/test/CelebA/celeba_prompts_output.json \
        --pretrained_model    /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \
        --img_encoder_weight  /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
        --ckpt_path           /projectnb/cs585/projects/craft/osdface/pretrained \
        --conditioner_ckpt    checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
        --output_dir          results/experiment_full \
        --film_neg_weight 0.1\
        --n_images           8 \
        --n_top_scan         200 \
        --n_top              10 \
        --mixed_precision    fp16

Output:
    results/experiment_full/
        pca_grid_up_ft1.png          ← main figure (4 rows: LQ/baseline/correct/swapped)
        pca_grid_bottleneck.png
        pca_grid_up_ft2.png
        norm_ratios.json             ← 3-condition table
        norm_ratios_table.png        ← publication-ready bar chart
        top_activating_grid.png      ← Revelio Fig 1 style
        label_purity.json            ← optional, needs --attributes_json
        layer_names.txt
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
                    W_B.view(out_ch, rank), W_A.view(rank, -1)
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
def encode_text_mean_pool(prompts, tokenizer, text_encoder, device, dtype):
    ids = tokenizer(
        prompts, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    hidden = text_encoder(ids).last_hidden_state        # (B, 77, 1024)
    pooled = hidden.mean(dim=1).unsqueeze(1).expand(-1, tokenizer.model_max_length, -1)
    return pooled.to(dtype)
 
 
@torch.no_grad()
def encode_text_eos(prompts, tokenizer, text_encoder, device, dtype):
    ids = tokenizer(
        prompts, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    hidden = text_encoder(ids).last_hidden_state        # (B, 77, 1024)
    eos_positions = (ids == tokenizer.eos_token_id).float().argmax(dim=1)
    eos_emb = hidden[torch.arange(hidden.shape[0], device=ids.device), eos_positions]
    eos_emb = eos_emb.unsqueeze(1).expand(-1, tokenizer.model_max_length, -1)
    return eos_emb.to(dtype)
 
 
# ── Swap this alias to change encoding mode across the entire script ──────
# encode_text_eos       → trained with --text_embed_mode eos  (current)
# encode_text_mean_pool → trained with --text_embed_mode mean_pool
encode_text_mean_pool = encode_text_eos
 
 
# ---------------------------------------------------------------------------
# Layer config  (Revelio naming → diffusers module prefix)
# ---------------------------------------------------------------------------
 
LAYER_GROUPS = {
    "bottleneck": "mid_block",
    "up_ft1":     "up_blocks.1",
    "up_ft2":     "up_blocks.2",
}
 
LAYER_SPATIAL = {
    "bottleneck": (8,  8),
    "up_ft1":     (16, 16),
    "up_ft2":     (32, 32),
}
 
 
# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
 
def get_last_transformer_block(unet, prefix):
    """Return the last BasicTransformerBlock whose name starts with prefix."""
    found = None
    for name, module in unet.named_modules():
        if isinstance(module, BasicTransformerBlock) and name.startswith(prefix):
            found = (name, module)
    return found
 
 
def extract_features_single_pass(unet, lq_latent, timestep, visual_embeds,
                                  conditioner, pos_emb, na_emb,
                                  neg_weight, layer_groups):
    """
    Run one UNet forward pass under a given conditioning state.
    Returns dict: layer_name → tensor (B, seq_len, C)
    """
    captured = {k: None for k in layer_groups}
    hooks    = []
 
    for layer_name, prefix in layer_groups.items():
        result = get_last_transformer_block(unet, prefix)
        if result is None:
            print(f"  WARNING: no transformer block found for {layer_name} ({prefix})")
            continue
        block_name, block = result
 
        def make_hook(lname):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[lname] = h.detach().float().cpu()
            return hook
 
        hooks.append(block.register_forward_hook(make_hook(layer_name)))
 
    # Set conditioning state
    if pos_emb is not None:
        conditioner.set_text_embedding(pos_emb, na_emb, neg_weight=neg_weight)
    else:
        conditioner.clear_text_embedding()
 
    with torch.no_grad():
        _ = unet(lq_latent, timestep, encoder_hidden_states=visual_embeds)
 
    conditioner.clear_text_embedding()
    for h in hooks:
        h.remove()
 
    return captured
 
 
# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
 
def pca_to_rgb(features: torch.Tensor) -> np.ndarray:
    """
    features: (B, H, W, C)
    Returns:  (B, H, W, 3) uint8
    """
    B, H, W, C = features.shape
    X = features.reshape(B * H * W, C)
    X = X - X.mean(dim=0)
    try:
        _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    except Exception:
        _, _, Vt = torch.svd(X); Vt = Vt.T
    projected = (X @ Vt[:3].T).reshape(B, H, W, 3)
    for c in range(3):
        ch = projected[..., c]
        projected[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    return (projected.numpy() * 255).astype(np.uint8)
 
 
def reshape_to_spatial(h: torch.Tensor, H: int, W: int):
    B, seq_len, C = h.shape
    if seq_len != H * W:
        side = int(seq_len ** 0.5)
        H = W = side
    return h.reshape(B, H, W, C), H, W
 
 
# ---------------------------------------------------------------------------
# Analysis 1: PCA grid (4 rows: LQ / baseline / correct / swapped)
# ---------------------------------------------------------------------------
 
def save_pca_grid_4row(pca_base, pca_correct, pca_swapped,
                        lq_pil, layer_name, output_path, img_size=128):
    B = pca_base.shape[0]
    fig, axes = plt.subplots(4, B, figsize=(B * 2.2, 4 * 2.2))
    if B == 1:
        axes = axes.reshape(4, 1)
 
    row_labels = ["LQ Input", "Baseline\n(no FiLM)", "Correct\nPrompt", "Swapped\nPrompt"]
 
    for i in range(B):
        # Row 0: LQ
        axes[0, i].imshow(lq_pil[i].resize((img_size, img_size)))
        axes[0, i].axis("off")
 
        for row_idx, pca_data in enumerate([pca_base, pca_correct, pca_swapped], start=1):
            img = Image.fromarray(pca_data[i]).resize((img_size, img_size), Image.NEAREST)
            axes[row_idx, i].imshow(img)
            axes[row_idx, i].axis("off")
 
        if i == 0:
            for row_idx, label in enumerate(row_labels):
                axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=0,
                                             labelpad=65, va="center")
 
    fig.suptitle(
        f"PCA Feature Maps — {layer_name}\n"
        f"Revelio Fig. 11 style — color shift between rows = FiLM effect",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path}")
 
 
# ---------------------------------------------------------------------------
# Analysis 2: Three-condition norm ratio table + bar chart
# ---------------------------------------------------------------------------
 
def compute_three_condition_norms(feats_base, feats_correct, feats_swapped):
    """
    Returns dict: layer_name → {baseline_norm, correct_norm, swapped_norm,
                                  correct_ratio, swapped_ratio, asymmetry}
    asymmetry = |correct_ratio - swapped_ratio|
    High asymmetry → FiLM responds differently to prompt content (good!)
    """
    results = {}
    for layer_name in LAYER_GROUPS:
        b = feats_base[layer_name]
        c = feats_correct[layer_name]
        s = feats_swapped[layer_name]
        if b is None or c is None or s is None:
            continue
 
        nb = b.norm(dim=-1).mean().item()
        nc = c.norm(dim=-1).mean().item()
        ns = s.norm(dim=-1).mean().item()
 
        correct_ratio  = nc / (nb + 1e-8)
        swapped_ratio  = ns / (nb + 1e-8)
        asymmetry      = abs(correct_ratio - swapped_ratio)
 
        results[layer_name] = {
            "baseline_norm":  round(nb, 4),
            "correct_norm":   round(nc, 4),
            "swapped_norm":   round(ns, 4),
            "correct_ratio":  round(correct_ratio, 4),
            "swapped_ratio":  round(swapped_ratio, 4),
            "asymmetry":      round(asymmetry, 4),
        }
        print(f"  {layer_name:12s}: "
              f"base={nb:.3f}  "
              f"correct={nc:.3f} (×{correct_ratio:.3f})  "
              f"swapped={ns:.3f} (×{swapped_ratio:.3f})  "
              f"asymmetry={asymmetry:.4f}")
    return results
 
 
def save_norm_ratio_chart(norm_results, output_path):
    """
    Bar chart: correct_ratio and swapped_ratio side by side per layer.
    Asymmetry between bars = FiLM is prompt-sensitive at that layer.
    """
    layers  = list(norm_results.keys())
    correct = [norm_results[l]["correct_ratio"] for l in layers]
    swapped = [norm_results[l]["swapped_ratio"] for l in layers]
 
    x   = np.arange(len(layers))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
 
    bars_c = ax.bar(x - w/2, correct, w, label="Correct prompt",
                    color="#4C72B0", alpha=0.85)
    bars_s = ax.bar(x + w/2, swapped, w, label="Swapped prompt",
                    color="#DD8452", alpha=0.85)
 
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
               label="Baseline (ratio=1.0)")
 
    # Annotate asymmetry
    for i, layer in enumerate(layers):
        asym = norm_results[layer]["asymmetry"]
        ymax = max(correct[i], swapped[i])
        ax.text(i, ymax + 0.002, f"Δ={asym:.3f}",
                ha="center", va="bottom", fontsize=8, color="black")
 
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=10)
    ax.set_ylabel("Norm ratio (FiLM / baseline)", fontsize=10)
    ax.set_title(
        "Three-Condition Norm Ratios per Layer\n"
        "Asymmetry (Δ) = FiLM responds to prompt content",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0.95, max(max(correct), max(swapped)) + 0.02)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path}")
 
 
# ---------------------------------------------------------------------------
# Analysis 3: Top-activating image grid (Revelio Fig 1/3 style)
# ---------------------------------------------------------------------------
 
def compute_delta_norms_full_set(unet, vae, img_encoder, embedding_change,
                                  conditioner, tokenizer, text_encoder,
                                  all_image_paths, prompt_lookup,
                                  device, dtype, timestep,
                                  neg_weight, n_top=10):
    """
    For every image in all_image_paths:
        delta = ||feat_correct - feat_swapped||  at up_ft1
    Returns top n_top (delta, path) pairs — highest delta = FiLM most active.
    """
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
 
    # Hook up_ft1
    target_block = None
    for name, module in unet.named_modules():
        if isinstance(module, BasicTransformerBlock) and name.startswith("up_blocks.1"):
            target_block = module
 
    assert target_block is not None
 
    deltas = []
 
    for img_path in all_image_paths:
        fname = os.path.basename(img_path)
        entry = prompt_lookup.get(fname, None)
        if entry is None:
            continue
 
        img = Image.open(img_path).convert("RGB")
        lq  = transform(img) * 2.0 - 1.0
        lq  = lq.unsqueeze(0).to(device, dtype=dtype)
 
        pos_text = strip_preamble(entry.get("pos", ""))
        na_text  = strip_preamble(entry.get("na",  ""))
 
        with torch.no_grad():
            visual_embeds = embedding_change(
                img_encoder(lq).reshape(1, 77, -1)
            )
            lq_latent = vae.encode(lq).latent_dist.sample() * vae.config.scaling_factor
 
        pos_emb = encode_text_mean_pool([pos_text], tokenizer, text_encoder, device, dtype)
        na_emb  = encode_text_mean_pool([na_text],  tokenizer, text_encoder, device, dtype)
 
        # Swapped
        pos_emb_sw = encode_text_mean_pool([na_text],  tokenizer, text_encoder, device, dtype)
        na_emb_sw  = encode_text_mean_pool([pos_text], tokenizer, text_encoder, device, dtype)
 
        feat_store = []
 
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            feat_store.append(h.detach().float().cpu())
 
        handle = target_block.register_forward_hook(hook)
 
        conditioner.set_text_embedding(pos_emb, na_emb, neg_weight=neg_weight)
        with torch.no_grad():
            _ = unet(lq_latent, timestep, encoder_hidden_states=visual_embeds)
        conditioner.clear_text_embedding()
        feat_correct = feat_store[-1]  # (1, seq, C)
 
        feat_store.clear()
        conditioner.set_text_embedding(pos_emb_sw, na_emb_sw, neg_weight=neg_weight)
        with torch.no_grad():
            _ = unet(lq_latent, timestep, encoder_hidden_states=visual_embeds)
        conditioner.clear_text_embedding()
        feat_swapped = feat_store[-1]
 
        handle.remove()
 
        delta = (feat_correct - feat_swapped).norm().item()
        deltas.append((delta, img_path, img))
 
    deltas.sort(key=lambda x: x[0], reverse=True)
    return deltas[:n_top]
 
 
def save_top_activating_grid(top_items, output_path, n_top=10):
    """
    Grid: 2 rows (delta score label / image thumbnail).
    Revelio Fig 1 style — images sorted by FiLM sensitivity.
    """
    n = min(len(top_items), n_top)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.0, 2.8))
    if n == 1:
        axes = [axes]
 
    for i, (delta, img_path, img) in enumerate(top_items[:n]):
        thumb = img.resize((128, 128))
        axes[i].imshow(thumb)
        axes[i].axis("off")
        axes[i].set_title(f"Δ={delta:.2f}", fontsize=8, pad=3)
 
    fig.suptitle(
        "Top Images by FiLM Sensitivity at up_ft1\n"
        "(||feat_correct − feat_swapped||  —  Revelio Fig. 1/3 style)",
        fontsize=9, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path}")
 
 
# ---------------------------------------------------------------------------
# Analysis 4: Label purity (Revelio Table 2)
# ---------------------------------------------------------------------------
 
def compute_label_purity(feats_base, feats_correct, feats_swapped,
                          labels, layer_groups):
    """
    Revelio's σ_label: lower = features cluster more purely by class.
    We compute: mean intra-class std / inter-class distance (lower is better).
    Also compute linear separability as accuracy of a 1-NN classifier.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
 
    results = {}
 
    for layer_name in layer_groups:
        results[layer_name] = {}
        for cond_name, feats in [("baseline", feats_base),
                                   ("correct",  feats_correct),
                                   ("swapped",  feats_swapped)]:
            h = feats[layer_name]
            if h is None:
                continue
 
            # Spatially pool: (B, seq, C) → (B, C)
            pooled = h.mean(dim=1).numpy()
 
            # 1-NN cross-val accuracy
            clf = KNeighborsClassifier(n_neighbors=1)
            try:
                scores = cross_val_score(clf, pooled, labels, cv=5, scoring="accuracy")
                acc = scores.mean()
            except Exception:
                acc = float("nan")
 
            # Intra-class std (lower = purer clusters)
            sigma = np.concatenate([
                pooled[labels == c].std(axis=0)
                for c in np.unique(labels)
            ]).mean()
 
            results[layer_name][cond_name] = {
                "1nn_accuracy": round(float(acc), 4),
                "intra_class_std": round(float(sigma), 4),
            }
            print(f"  {layer_name:12s} [{cond_name:8s}]: "
                  f"1-NN acc={acc:.3f}  σ={sigma:.3f}")
 
    return results
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    timestep = 399
 
    # ── Load models ───────────────────────────────────────────────────────
    print("Loading models...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae"
    ).to(device, dtype=dtype)
    vae.requires_grad_(False).eval()
 
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )
 
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder"
    ).to(device, dtype=dtype)
    text_encoder.requires_grad_(False).eval()
 
    unet = merge_lora_into_unet(args)
    unet.requires_grad_(False).to(device, dtype=dtype).eval()
 
    img_encoder = vqvae_encoder(args).to(device, dtype=dtype)
    img_encoder.requires_grad_(False).eval()
 
    embedding_change = TwoLayerConv1x1(512, 1024)
    embedding_change.load_state_dict(
        torch.load(
            os.path.join(args.ckpt_path, "embedding_change_weights.pth"),
            weights_only=False,
        )
    )
    embedding_change.to(device, dtype=dtype).requires_grad_(False).eval()
 
    conditioner = TextConditioner(unet, text_dim=1024)
    conditioner.load(args.conditioner_ckpt, map_location=device)
    conditioner.register_hooks(unet)
    conditioner.to(device).eval()
 
    # ── Save layer names ──────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, "layer_names.txt"), "w") as f:
        for layer_name, prefix in LAYER_GROUPS.items():
            result = get_last_transformer_block(unet, prefix)
            if result:
                f.write(f"{layer_name} → {result[0]}\n")
            else:
                f.write(f"{layer_name} → NOT FOUND\n")
 
    # ── Load test images ──────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
 
    with open(args.prompts_json) as f:
        prompts_data = json.load(f)
 
    prompt_lookup = {}
    for item in prompts_data:
        stem, ext = os.path.splitext(item["image"])
        for key in [item["image"], f"{stem}_LQ{ext}", f"{stem}_lq{ext}"]:
            prompt_lookup[key] = item
 
    all_lq = sorted(glob.glob(os.path.join(args.lq_dir, "*")))
    random.seed(args.seed)
    random.shuffle(all_lq)
    selected = [p for p in all_lq if os.path.basename(p) in prompt_lookup][:args.n_images]
 
    if not selected:
        raise ValueError(f"No matched images found in {args.lq_dir}. Check prompts_json.")
 
    lq_tensors, lq_pil, pos_prompts, na_prompts = [], [], [], []
    for lq_path in selected:
        fname = os.path.basename(lq_path)
        entry = prompt_lookup[fname]
        img   = Image.open(lq_path).convert("RGB")
        lq_pil.append(img)
        lq_tensors.append(transform(img) * 2.0 - 1.0)
        pos_prompts.append(strip_preamble(entry.get("pos", "")))
        na_prompts.append(strip_preamble(entry.get("na",  "")))
 
    lq_batch = torch.stack(lq_tensors).to(device, dtype=dtype)
    B = lq_batch.shape[0]
    print(f"Running on {B} images...")
 
    # ── Shared inputs ─────────────────────────────────────────────────────
    with torch.no_grad():
        visual_embeds = embedding_change(
            img_encoder(lq_batch).reshape(B, 77, -1)
        )
        lq_latent = vae.encode(lq_batch).latent_dist.sample() * vae.config.scaling_factor
 
    pos_emb = encode_text_mean_pool(pos_prompts, tokenizer, text_encoder, device, dtype)
    na_emb  = encode_text_mean_pool(na_prompts,  tokenizer, text_encoder, device, dtype)
 
    # Swapped: pos↔na reversed
    pos_emb_sw = encode_text_mean_pool(na_prompts,  tokenizer, text_encoder, device, dtype)
    na_emb_sw  = encode_text_mean_pool(pos_prompts, tokenizer, text_encoder, device, dtype)
 
    # ── Extract features: 3 conditions ────────────────────────────────────
    print("\n[A] Baseline (no FiLM)...")
    feats_base = extract_features_single_pass(
        unet, lq_latent, timestep, visual_embeds,
        conditioner, None, None, args.film_neg_weight, LAYER_GROUPS,
    )
 
    print("[B] Correct prompt...")
    feats_correct = extract_features_single_pass(
        unet, lq_latent, timestep, visual_embeds,
        conditioner, pos_emb, na_emb, args.film_neg_weight, LAYER_GROUPS,
    )
 
    print("[C] Swapped prompt...")
    feats_swapped = extract_features_single_pass(
        unet, lq_latent, timestep, visual_embeds,
        conditioner, pos_emb_sw, na_emb_sw, args.film_neg_weight, LAYER_GROUPS,
    )
 
    # ── Analysis 1: PCA grids ─────────────────────────────────────────────
    print("\n=== Analysis 1: PCA Feature Maps ===")
    for layer_name, (H, W) in LAYER_SPATIAL.items():
        hb = feats_base[layer_name]
        hc = feats_correct[layer_name]
        hs = feats_swapped[layer_name]
        if hb is None or hc is None or hs is None:
            print(f"  Skipping {layer_name} — missing features")
            continue
 
        hb_s, H_, W_ = reshape_to_spatial(hb, H, W)
        hc_s, _,  _  = reshape_to_spatial(hc, H_, W_)
        hs_s, _,  _  = reshape_to_spatial(hs, H_, W_)
 
        pca_b = pca_to_rgb(hb_s)
        pca_c = pca_to_rgb(hc_s)
        pca_s = pca_to_rgb(hs_s)
 
        out_path = os.path.join(args.output_dir, f"pca_grid_{layer_name}.png")
        save_pca_grid_4row(pca_b, pca_c, pca_s, lq_pil, layer_name, out_path)
 
    # ── Analysis 2: Three-condition norm ratios ────────────────────────────
    print("\n=== Analysis 2: Three-Condition Norm Ratios ===")
    norm_results = compute_three_condition_norms(feats_base, feats_correct, feats_swapped)
 
    with open(os.path.join(args.output_dir, "norm_ratios.json"), "w") as f:
        json.dump(norm_results, f, indent=2)
    print(f"  Saved → {args.output_dir}/norm_ratios.json")
 
    save_norm_ratio_chart(
        norm_results,
        os.path.join(args.output_dir, "norm_ratios_table.png"),
    )
 
    # ── Analysis 3: Top-activating image grid ────────────────────────────
    print(f"\n=== Analysis 3: Top-Activating Images (scanning {args.n_top_scan} images) ===")
    all_lq_for_scan = [p for p in all_lq if os.path.basename(p) in prompt_lookup]
    all_lq_for_scan = all_lq_for_scan[:args.n_top_scan]
 
    top_items = compute_delta_norms_full_set(
        unet, vae, img_encoder, embedding_change,
        conditioner, tokenizer, text_encoder,
        all_lq_for_scan, prompt_lookup,
        device, dtype, timestep,
        args.film_neg_weight, n_top=args.n_top,
    )
 
    save_top_activating_grid(
        top_items,
        os.path.join(args.output_dir, "top_activating_grid.png"),
        n_top=args.n_top,
    )
 
    # Save delta scores
    delta_scores = [{"image": os.path.basename(p), "delta": round(d, 4)}
                    for d, p, _ in top_items]
    with open(os.path.join(args.output_dir, "top_activating_scores.json"), "w") as f:
        json.dump(delta_scores, f, indent=2)
 
    # ── Analysis 4: Label purity (optional) ───────────────────────────────
    if args.attributes_json and args.probe_attribute:
        print(f"\n=== Analysis 4: Label Purity — attribute: {args.probe_attribute} ===")
        with open(args.attributes_json) as f:
            attr_data = json.load(f)
 
        attr_lookup = {}
        for item in attr_data:
            stem, ext = os.path.splitext(item["image"])
            for key in [item["image"], f"{stem}_LQ{ext}", f"{stem}_lq{ext}"]:
                attr_lookup[key] = item
 
        labels = []
        valid_idx = []
        for i, lq_path in enumerate(selected):
            fname = os.path.basename(lq_path)
            if fname in attr_lookup and args.probe_attribute in attr_lookup[fname]:
                labels.append(int(attr_lookup[fname][args.probe_attribute]))
                valid_idx.append(i)
 
        if len(set(labels)) < 2:
            print(f"  Skipping — attribute '{args.probe_attribute}' has only one class "
                  f"in this {B}-image sample. Try --n_images larger or a different attribute.")
        else:
            labels = np.array(labels)
 
            def filter_feats(feats_dict, idx):
                out = {}
                for k, v in feats_dict.items():
                    out[k] = v[idx] if v is not None else None
                return out
 
            purity = compute_label_purity(
                filter_feats(feats_base,    valid_idx),
                filter_feats(feats_correct, valid_idx),
                filter_feats(feats_swapped, valid_idx),
                labels, LAYER_GROUPS,
            )
            with open(os.path.join(args.output_dir, "label_purity.json"), "w") as f:
                json.dump(purity, f, indent=2)
            print(f"  Saved → {args.output_dir}/label_purity.json")
 
    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("\nKey outputs:")
    print("  pca_grid_up_ft1.png       — main figure (4 rows)")
    print("  norm_ratios_table.png     — 3-condition bar chart")
    print("  top_activating_grid.png   — Revelio Fig 1/3 style")
    print("\nWhat to look for:")
    print("  PCA:   color shift between 'correct' and 'swapped' rows = FiLM effect")
    print("  Norms: asymmetry (Δ) at up_ft1 > other layers = layer-specific conditioning")
    print("  Top-k: high-delta images = where FiLM matters most (cite in paper)")
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lq_dir",           required=True)
    p.add_argument("--prompts_json",     required=True)
    p.add_argument("--pretrained_model", default="pretrained/sd21")
    p.add_argument("--img_encoder_weight", default="pretrained/associate_2.ckpt")
    p.add_argument("--ckpt_path",        required=True)
    p.add_argument("--conditioner_ckpt", required=True)
    p.add_argument("--output_dir",       default="results/experiment_full")
    p.add_argument("--n_images",         type=int,   default=8,
                   help="Images for PCA/norm analysis")
    p.add_argument("--n_top_scan",       type=int,   default=200,
                   help="Images to scan for top-activating grid")
    p.add_argument("--n_top",            type=int,   default=10,
                   help="How many top images to show in grid")
    p.add_argument("--film_neg_weight",  type=float, default=0.5)
    p.add_argument("--mixed_precision",  choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--attributes_json",  type=str,   default=None,
                   help="Optional: path to binary attributes JSON for label purity")
    p.add_argument("--probe_attribute",  type=str,   default="eyeglasses",
                   help="Which binary attribute to use for label purity analysis")
    p.add_argument("--pretrained_lora_rank",  type=int,   default=16)
    p.add_argument("--pretrained_lora_alpha", type=float, default=16)
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