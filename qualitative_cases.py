"""
qualitative_cases.py — Per-image case study visualization for OSDFace paper

For each specified image, produces a single figure with 7 columns:
    LQ input | HQ reference | OSD output | OSD+Text output |
    PCA baseline | PCA correct prompt | PCA swapped prompt

Images are grouped into labelled categories you define in --cases_json.

cases_json format:
    [
      {
        "filename":   "00020_LQ.png",
        "category":   "Text guided correctly",
        "note":       "Suppressed hallucinated glasses"
      },
      {
        "filename":   "00045_LQ.png",
        "category":   "Text deformed output",
        "note":       "Over-suppressed face structure"
      },
      {
        "filename":   "00078_LQ.png",
        "category":   "Both agreed",
        "note":       "Consistent restoration, no conflict"
      },
      {
        "filename":   "00102_LQ.png",
        "category":   "Text had no effect",
        "note":       "OSDFace prior dominated"
      }
    ]

Usage:
    python qualitative_cases.py \
        --cases_json          cases.json \
        --lq_dir              /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2 \
        --hq_dir              /projectnb/cs585/projects/craft/data/test/CelebA/CelebA_Validation/celeba_512_validation\
        --osd_dir             eval_outputs_final/quantitative/osd \
        --osd_text_dir        eval_outputs_final/quantitative/osd_text  \
        --prompts_json        /projectnb/cs585/projects/craft/data/test/CelebA/celeba_prompts_output.json \
        --pretrained_model    /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \
        --img_encoder_weight  /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
        --ckpt_path           /projectnb/cs585/projects/craft/osdface/pretrained \
        --conditioner_ckpt    checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
        --output_dir          results/presentation \
        --film_neg_weight     0.1 \
        --mixed_precision     fp16

Output:
    results/qualitative/
        case_00020_LQ.png     — one figure per image (7 columns)
        all_cases.png         — all cases stacked into one publication figure
"""

import os
import json
import glob
import argparse
 
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from safetensors import safe_open
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from transformers import CLIPTextModel, CLIPTokenizer
 
from lq_embed import vqvae_encoder, TwoLayerConv1x1
from text_conditioner import TextConditioner
 
 
# ---------------------------------------------------------------------------
# Category colors for row labels
# ---------------------------------------------------------------------------
 
CATEGORY_COLORS = {
    "Text guided correctly": "#2ca02c",   # green
    "Text deformed output":  "#d62728",   # red
    "Both agreed":           "#1f77b4",   # blue
    "Text had no effect":    "#ff7f0e",   # orange
}
DEFAULT_COLOR = "#7f7f7f"
 
COL_TITLES = [
    "LQ Input",
    "HQ Reference",
    "OSD Output",
    "OSD + Text",
    "PCA Baseline",
    "Channel Δ",
    "Cosine Δ",
]
 
 
# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
 
def strip_preamble(text: str) -> str:
    for marker in ["in the description of ", "not in the description of "]:
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
    return text.strip()
 
 
def merge_lora_into_unet(args):
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
            W_A, W_B   = state_dict[key], state_dict[lora_b_key]
            orig       = sd_unet[unet_key]
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
            W_up, W_down  = state_dict[key], state_dict[lora_down_key]
            orig          = sd_unet[unet_key]
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
    hidden = text_encoder(ids).last_hidden_state
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
    eos_emb = hidden[torch.arange(hidden.shape[0], device=device), eos_positions]
    eos_emb = eos_emb.unsqueeze(1).expand(-1, tokenizer.model_max_length, -1)
    return eos_emb.to(dtype)
 
 
# ── Swap this alias to change encoding mode across the entire script ──────
# encode_text_eos       → trained with --text_embed_mode eos  (current)
# encode_text_mean_pool → trained with --text_embed_mode mean_pool
encode_text = encode_text_eos
 
 
# ---------------------------------------------------------------------------
# Feature extraction at up_ft1
# ---------------------------------------------------------------------------
 
def get_last_up_ft1_block(unet):
    found = None
    for name, module in unet.named_modules():
        if isinstance(module, BasicTransformerBlock) and name.startswith("up_blocks.1"):
            found = (name, module)
    assert found is not None, "Could not find up_blocks.1 transformer block"
    return found
 
 
def extract_up_ft1(unet, lq_latent, timestep, visual_embeds,
                    conditioner, pos_emb, na_emb, neg_weight):
    """
    Run one UNet forward pass and capture the last up_ft1 transformer block output.
    Returns (1, seq_len, C) float CPU tensor.
    """
    _, block = get_last_up_ft1_block(unet)
    captured = []
 
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().float().cpu())
 
    handle = block.register_forward_hook(hook)
 
    if pos_emb is not None:
        conditioner.set_text_embedding(pos_emb, na_emb, neg_weight=neg_weight)
    else:
        conditioner.clear_text_embedding()
 
    with torch.no_grad():
        _ = unet(lq_latent, timestep, encoder_hidden_states=visual_embeds)
 
    conditioner.clear_text_embedding()
    handle.remove()
 
    return captured[0]  # (1, seq_len, C)
 
 
# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
 
def pca_to_rgb_single(feat: torch.Tensor, H: int = 16, W: int = 16,
                       pca_basis=None):
    """
    feat: (1, seq_len, C)
    pca_basis: optional (3, C) tensor — if provided, reuse this basis
               so colors are comparable across conditions for the same image.
    Returns:
        rgb: (H, W, 3) uint8
        basis: (3, C) tensor  — pass back in for subsequent conditions
    """
    seq_len = feat.shape[1]
    if seq_len != H * W:
        side = int(seq_len ** 0.5)
        H = W = side
 
    x = feat.reshape(H * W, feat.shape[-1]).float()
    x = x - x.mean(dim=0)
 
    if pca_basis is None:
        try:
            _, _, Vt = torch.linalg.svd(x, full_matrices=False)
        except Exception:
            _, _, Vt = torch.svd(x); Vt = Vt.T
        basis = Vt[:3]  # (3, C)
    else:
        basis = pca_basis
 
    projected = (x @ basis.T).reshape(H, W, 3)
 
    # Normalize to [0,1] using the range of THIS projection
    # (consistent within an image — same basis means same color scale)
    for c in range(3):
        ch = projected[..., c]
        projected[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
 
    return (projected.numpy() * 255).astype(np.uint8), basis
 
 
def pca_diff_to_rgb(feat_base: torch.Tensor, feat_other: torch.Tensor,
                     H: int = 16, W: int = 16) -> np.ndarray:
    """
    PCA on the *difference* (feat_other - feat_base) at up_ft1.
 
    This surfaces WHERE in the spatial map FiLM made changes, rather than
    the dominant face-vs-background structure that raw PCA shows.
 
    Bright regions = spatial locations most changed by FiLM conditioning.
    For glasses suppression you expect activity around the eye region.
 
    feat_base:  (1, seq_len, C) — baseline (no FiLM)
    feat_other: (1, seq_len, C) — correct or swapped prompt
    Returns:    (H, W, 3) uint8
    """
    seq_len = feat_base.shape[1]
    if seq_len != H * W:
        H = W = int(seq_len ** 0.5)
 
    diff = (feat_other - feat_base).reshape(H * W, -1).float()  # (HW, C)
    diff = diff - diff.mean(dim=0)
 
    # Magnitude map — norm of difference per spatial position
    # Used to create an alpha-weighted overlay so bright = high change
    mag = diff.norm(dim=-1)  # (HW,)
 
    try:
        _, _, Vt = torch.linalg.svd(diff, full_matrices=False)
    except Exception:
        _, _, Vt = torch.svd(diff); Vt = Vt.T
 
    projected = (diff @ Vt[:3].T).reshape(H, W, 3)  # (H, W, 3)
 
    # Normalize each channel
    for c in range(3):
        ch = projected[..., c]
        projected[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
 
    # Blend with a white background weighted by magnitude
    # Low-change regions fade to white so high-change areas pop
    mag_map = mag.reshape(H, W).numpy()
    mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-8)
    mag_map = mag_map[..., np.newaxis]  # (H, W, 1)
 
    white = np.ones((H, W, 3), dtype=np.float32)
    blended = projected.numpy() * mag_map + white * (1 - mag_map)
    blended = np.clip(blended, 0, 1)
 
    return (blended * 255).astype(np.uint8)
 
 
 
 
def channel_diff_to_rgb(feat_base: torch.Tensor, feat_other: torch.Tensor,
                         H: int = 16, W: int = 16) -> np.ndarray:
    """
    Top-3 channel activation difference map (Revelio Sec 4.2 style).
 
    Finds the 3 channels with the highest mean absolute difference between
    baseline and FiLM conditions, then visualizes their spatial activation.
    These channels are the specific feature dimensions FiLM is targeting —
    bypasses PCA's variance-maximization which ignores attribute-specific dims.
 
    Red/bright regions = spatial locations where FiLM's target channels fire.
    """
    seq_len = feat_base.shape[1]
    if seq_len != H * W:
        H = W = int(seq_len ** 0.5)
 
    b = feat_base.squeeze(0).reshape(H * W, -1).float()   # (HW, C)
    o = feat_other.squeeze(0).reshape(H * W, -1).float()  # (HW, C)
    diff = (o - b).abs()  # (HW, C)
 
    # Find top-3 channels by mean absolute difference
    channel_importance = diff.mean(dim=0)  # (C,)
    top3 = channel_importance.argsort(descending=True)[:3]
 
    spatial = diff[:, top3].reshape(H, W, 3).numpy()  # (H, W, 3)
    for c in range(3):
        ch = spatial[..., c]
        spatial[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
 
    return (spatial * 255).astype(np.uint8)
 
 
def cosine_diff_to_heatmap(feat_base: torch.Tensor, feat_other: torch.Tensor,
                            H: int = 16, W: int = 16) -> np.ndarray:
    """
    Per-spatial-position cosine dissimilarity heatmap (Revelio Sec 4.3 style).
 
    For each spatial location, computes 1 - cosine_similarity(baseline, FiLM).
    Red (hot) = FiLM strongly changed the representation here.
    Black/dark = FiLM left this region unchanged.
 
    Most interpretable for a paper — directly shows WHICH face region
    FiLM modulated (e.g. eye region for glasses suppression).
    """
    seq_len = feat_base.shape[1]
    if seq_len != H * W:
        H = W = int(seq_len ** 0.5)
 
    b = feat_base.squeeze(0).reshape(H, W, -1).float()   # (H, W, C)
    o = feat_other.squeeze(0).reshape(H, W, -1).float()  # (H, W, C)
 
    sim = F.cosine_similarity(b, o, dim=-1)   # (H, W)  range [-1, 1]
    change = (1.0 - sim).clamp(0, 2)          # (H, W)  0 = unchanged
 
    change_np = change.numpy()
    change_np = (change_np - change_np.min()) / (change_np.max() - change_np.min() + 1e-8)
 
    # Hot colormap: black → red → yellow → white
    heatmap = plt.cm.hot(change_np)[:, :, :3]  # (H, W, 3)
    return (heatmap * 255).astype(np.uint8)
 
# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------
 
def find_image_in_dir(directory, filename):
    """
    Find an image in directory matching filename.
 
    Strategy (in order):
      1. Exact filename match — handles same-name-different-dir case
      2. Variant suffixes (_LQ, _HQ, etc.)
      3. Strip _LQ/_lq suffix — handles lq_dir name → hq_dir name
      4. Glob on base stem as last resort
    """
    # 1. Exact match first — most common case when HQ/LQ share the same filename
    exact = os.path.join(directory, filename)
    if os.path.exists(exact):
        return Image.open(exact).convert("RGB")
 
    stem, ext = os.path.splitext(filename)
 
    # 2. Variant suffixes
    variants = [
        f"{stem}_LQ{ext}", f"{stem}_lq{ext}",
        f"{stem}_HQ{ext}", f"{stem}_hq{ext}",
    ]
    for name in variants:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
 
    # 3. Strip _LQ/_lq to find counterpart in HQ dir
    base_stem = stem.replace("_LQ", "").replace("_lq", "")
    if base_stem != stem:
        stripped = os.path.join(directory, base_stem + ext)
        if os.path.exists(stripped):
            return Image.open(stripped).convert("RGB")
 
    # 4. Glob fallback
    matches = glob.glob(os.path.join(directory, f"{base_stem}*"))
    if matches:
        return Image.open(sorted(matches)[0]).convert("RGB")
 
    return None
 
 
def load_resize(img_or_path, size=256):
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert("RGB")
    else:
        img = img_or_path
    return img.resize((size, size), Image.BICUBIC)
 
 
# ---------------------------------------------------------------------------
# Per-image figure
# ---------------------------------------------------------------------------
 
def make_case_figure(case, models, prompt_lookup, device, dtype,
                     args, img_size=160):
    """
    Builds a (1 row × 7 col) figure for one case.
    Returns matplotlib Figure.
    """
    filename  = case["filename"]
    category  = case.get("category", "")
    note      = case.get("note", "")
    color     = CATEGORY_COLORS.get(category, DEFAULT_COLOR)
 
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
 
    # ── Load images ───────────────────────────────────────────────────────
    lq_pil  = find_image_in_dir(args.lq_dir,      filename)
    hq_pil  = find_image_in_dir(args.hq_dir,      filename)
    osd_pil = find_image_in_dir(args.osd_dir,     filename)
    txt_pil = find_image_in_dir(args.osd_text_dir, filename)
 
    missing = [n for n, p in [("LQ", lq_pil), ("HQ", hq_pil),
                               ("OSD", osd_pil), ("OSD+Text", txt_pil)]
               if p is None]
    if missing:
        print(f"  WARNING [{filename}]: missing {missing} — will show blank panels")
 
    # ── Get prompts ───────────────────────────────────────────────────────
    entry = prompt_lookup.get(filename, {"pos": "", "na": ""})
    pos_text = strip_preamble(entry.get("pos", ""))
    na_text  = strip_preamble(entry.get("na",  ""))
 
    # ── Encode text ───────────────────────────────────────────────────────
    unet, vae, img_encoder, embedding_change, conditioner, tokenizer, text_encoder = models
 
    pos_emb    = encode_text([pos_text], tokenizer, text_encoder, device, dtype)
    na_emb     = encode_text([na_text],  tokenizer, text_encoder, device, dtype)
    pos_emb_sw = encode_text([na_text],  tokenizer, text_encoder, device, dtype)
    na_emb_sw  = encode_text([pos_text], tokenizer, text_encoder, device, dtype)
 
    # ── Prepare LQ latent ─────────────────────────────────────────────────
    if lq_pil is not None:
        lq_t = transform(lq_pil) * 2.0 - 1.0
        lq_batch = lq_t.unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            visual_embeds = embedding_change(
                img_encoder(lq_batch).reshape(1, 77, -1)
            )
            lq_latent = (
                vae.encode(lq_batch).latent_dist.sample()
                * vae.config.scaling_factor
            )
        timestep = 399
 
        # ── Extract PCA features (3 conditions, shared basis) ─────────────
        feat_base    = extract_up_ft1(unet, lq_latent, timestep, visual_embeds,
                                       conditioner, None, None, args.film_neg_weight)
        feat_correct = extract_up_ft1(unet, lq_latent, timestep, visual_embeds,
                                       conditioner, pos_emb, na_emb, args.film_neg_weight)
        feat_swapped = extract_up_ft1(unet, lq_latent, timestep, visual_embeds,
                                       conditioner, pos_emb_sw, na_emb_sw, args.film_neg_weight)
 
        # Baseline PCA — raw features to show spatial structure
        pca_base, _ = pca_to_rgb_single(feat_base)
        # Channel diff — top-3 most-changed feature channels
        chan_correct = channel_diff_to_rgb(feat_base, feat_correct)
        chan_swapped = channel_diff_to_rgb(feat_base, feat_swapped)
        # Cosine dissimilarity — per-spatial-position change heatmap
        cos_correct  = cosine_diff_to_heatmap(feat_base, feat_correct)
        cos_swapped  = cosine_diff_to_heatmap(feat_base, feat_swapped)
    else:
        blank = np.zeros((16, 16, 3), dtype=np.uint8)
        pca_base = chan_correct = chan_swapped = cos_correct = cos_swapped = blank
 
    # ── Build figure ──────────────────────────────────────────────────────
    n_cols = 7
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.2, 2.8))
 
    panels = [
        lq_pil,
        hq_pil,
        osd_pil,
        txt_pil,
        Image.fromarray(pca_base),
        Image.fromarray(chan_correct),
        Image.fromarray(cos_correct),
    ]
 
    for col_idx, (ax, panel, col_title) in enumerate(zip(axes, panels, COL_TITLES)):
        if panel is not None:
            ax.imshow(load_resize(panel, img_size))
        else:
            ax.set_facecolor("#eeeeee")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#999999")
        ax.axis("off")
        ax.set_title(col_title, fontsize=8, pad=3)
 
    # Category label on left as colored row label
    axes[0].set_ylabel(
        f"{category}\n{note}",
        fontsize=8, rotation=0, labelpad=75, va="center",
        color=color, fontweight="bold",
    )
 
    fig.tight_layout()
    return fig
 
 
# ---------------------------------------------------------------------------
# Combined figure (all cases stacked)
# ---------------------------------------------------------------------------
 
def make_combined_figure(cases, models, prompt_lookup, device, dtype,
                          args, img_size=140):
    """
    One figure with N rows (one per case) × 7 columns.
    Each row has a colored category label on the left.
    """
    n_rows = len(cases)
    n_cols = 7
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0, n_rows * 2.4),
    )
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)
 
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
 
    unet, vae, img_encoder, embedding_change, conditioner, tokenizer, text_encoder = models
 
    # Column titles on top row only
    for col_idx, title in enumerate(COL_TITLES):
        axes[0, col_idx].set_title(title, fontsize=9, pad=4, fontweight="bold")
 
    for row_idx, case in enumerate(cases):
        filename = case["filename"]
        category = case.get("category", "")
        note     = case.get("note", "")
        color    = CATEGORY_COLORS.get(category, DEFAULT_COLOR)
 
        print(f"  Processing [{row_idx+1}/{n_rows}]: {filename} — {category}")
 
        # Load images
        lq_pil  = find_image_in_dir(args.lq_dir,       filename)
        hq_pil  = find_image_in_dir(args.hq_dir,       filename)
        osd_pil = find_image_in_dir(args.osd_dir,      filename)
        txt_pil = find_image_in_dir(args.osd_text_dir,  filename)
 
        # Get prompts
        entry    = prompt_lookup.get(filename, {"pos": "", "na": ""})
        pos_text = strip_preamble(entry.get("pos", ""))
        na_text  = strip_preamble(entry.get("na",  ""))
 
        pos_emb    = encode_text([pos_text], tokenizer, text_encoder, device, dtype)
        na_emb     = encode_text([na_text],  tokenizer, text_encoder, device, dtype)
        pos_emb_sw = encode_text([na_text],  tokenizer, text_encoder, device, dtype)
        na_emb_sw  = encode_text([pos_text], tokenizer, text_encoder, device, dtype)
 
        # PCA features
        if lq_pil is not None:
            lq_t   = transform(lq_pil) * 2.0 - 1.0
            lq_b   = lq_t.unsqueeze(0).to(device, dtype=dtype)
            with torch.no_grad():
                vis = embedding_change(img_encoder(lq_b).reshape(1, 77, -1))
                lat = vae.encode(lq_b).latent_dist.sample() * vae.config.scaling_factor
 
            fb = extract_up_ft1(unet, lat, 399, vis, conditioner,
                                 None, None, args.film_neg_weight)
            fc = extract_up_ft1(unet, lat, 399, vis, conditioner,
                                 pos_emb, na_emb, args.film_neg_weight)
            fs = extract_up_ft1(unet, lat, 399, vis, conditioner,
                                 pos_emb_sw, na_emb_sw, args.film_neg_weight)
 
            pca_b,  _   = pca_to_rgb_single(fb)
            chan_c       = channel_diff_to_rgb(fb, fc)
            chan_s       = channel_diff_to_rgb(fb, fs)
            cos_c        = cosine_diff_to_heatmap(fb, fc)
            cos_s        = cosine_diff_to_heatmap(fb, fs)
        else:
            blank  = np.zeros((16, 16, 3), dtype=np.uint8)
            pca_b  = chan_c = chan_s = cos_c = cos_s = blank
 
        panels = [
            lq_pil,
            hq_pil,
            osd_pil,
            txt_pil,
            Image.fromarray(pca_b),
            Image.fromarray(chan_c),
            Image.fromarray(cos_c),
        ]
 
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if panel is not None:
                ax.imshow(load_resize(panel, img_size))
            else:
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#aaaaaa")
            ax.axis("off")
 
        # Row label: category + note
        axes[row_idx, 0].set_ylabel(
            f"{category}\n({note})",
            fontsize=7.5, rotation=0, labelpad=80,
            va="center", color=color, fontweight="bold",
        )
 
    # Legend for categories
    legend_patches = [
        mpatches.Patch(color=c, label=cat)
        for cat, c in CATEGORY_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(CATEGORY_COLORS),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
 
    fig.suptitle(
        "Qualitative Case Studies — OSDFace vs OSDFace+Text\n"
        "PCA baseline | Channel Δ | Cosine Δ at up_ft1 (Revelio Sec 4.2/4.3 style)\nRed in cosine cols = where FiLM changed the representation most",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    return fig
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if args.mixed_precision == "fp16" else torch.float32
 
    # ── Load cases ────────────────────────────────────────────────────────
    with open(args.cases_json) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {args.cases_json}")
 
    # ── Load prompts ──────────────────────────────────────────────────────
    with open(args.prompts_json) as f:
        prompts_raw = json.load(f)
    prompt_lookup = {}
    for item in prompts_raw:
        stem, ext = os.path.splitext(item["image"])
        for key in [item["image"], f"{stem}_LQ{ext}", f"{stem}_lq{ext}"]:
            prompt_lookup[key] = item
 
    # ── Load models ───────────────────────────────────────────────────────
    print("Loading models...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae"
    ).to(device, dtype=dtype)
    vae.requires_grad_(False).eval()
 
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
 
    models = (unet, vae, img_encoder, embedding_change,
              conditioner, tokenizer, text_encoder)
 
    # ── Individual figures ────────────────────────────────────────────────
    print("\nGenerating individual case figures...")
    for case in cases:
        fname = case["filename"]
        print(f"  {fname} — {case.get('category', '')}")
        fig = make_case_figure(case, models, prompt_lookup, device, dtype, args)
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(args.output_dir, f"case_{stem}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved → {out_path}")
 
    # ── Combined figure ───────────────────────────────────────────────────
    print("\nGenerating combined figure (all cases)...")
    fig_all = make_combined_figure(cases, models, prompt_lookup, device, dtype, args)
    out_all = os.path.join(args.output_dir, "all_cases.png")
    fig_all.savefig(out_all, dpi=150, bbox_inches="tight")
    plt.close(fig_all)
    print(f"Saved → {out_all}")
 
    print("\nDone. Key output: all_cases.png")
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases_json",       required=True,
                   help="JSON file listing filenames and categories")
    p.add_argument("--lq_dir",           required=True)
    p.add_argument("--hq_dir",           required=True)
    p.add_argument("--osd_dir",          required=True,
                   help="Directory with baseline OSDFace outputs")
    p.add_argument("--osd_text_dir",     required=True,
                   help="Directory with OSDFace+Text outputs")
    p.add_argument("--prompts_json",     required=True)
    p.add_argument("--pretrained_model", default="pretrained/sd21")
    p.add_argument("--img_encoder_weight", default="pretrained/associate_2.ckpt")
    p.add_argument("--ckpt_path",        required=True)
    p.add_argument("--conditioner_ckpt", required=True)
    p.add_argument("--output_dir",       default="results/qualitative")
    p.add_argument("--film_neg_weight",  type=float, default=0.5)
    p.add_argument("--mixed_precision",  choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--pretrained_lora_rank",  type=int,   default=16)
    p.add_argument("--pretrained_lora_alpha", type=float, default=16)
    p.add_argument("--cat_prompt_embedding", action="store_true")
    p.add_argument("--use_pos_embedding",    action="store_true")
    p.add_argument("--use_att_pool",         action="store_true")
    p.add_argument("--learnable_pos_emb",    action="store_true")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    run(args)