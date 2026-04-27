"""
face_codebook_panel.py - Per-face CLIP interpretability panels on the test set.

Goal:
    Show on actual test faces that region-aware codebooks are working.
    Each output figure proves visually that:
      - the eyes codebook lights up on eye patches and stays dim elsewhere,
      - the lips codebook lights up on lip patches, etc.
    plus annotates the codebooks with their top CLIP text matches from data.json.

Per face the figure contains:
    Row 1 (context, 3 panels):
        (a) Original HQ face
        (b) Region segmentation overlay (5-color, from BiSeNet)
        (c) Reconstruction through the region-aware VQ
    Row 2 (5 panels, one per region):
        (d-h) CLIP patch -> codebook-R similarity heatmap, 16x16 upsampled
              and alpha-blended on the face. The face-parser region for R is
              outlined so the viewer can immediately see whether high-similarity
              cells fall inside or outside the native region. Globally normalized
              so panels are comparable.
    Caption:
        Top-N CLIP text labels per codebook (best phrases from data.json).
        Optional: pre-computed diagonal-dominance numbers from
        clip_codebook_summary.json (existing aggregate scores) for context.

This script does NOT modify any existing file. It only reads:
    - the model checkpoint
    - the face parser checkpoint
    - the test data root
    - data.json (vocabulary)
    - optionally clip_codebook_summary.json (existing aggregate scores)

Usage:
    python face_codebook_panel.py \
        --craft_ckpt   /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
        --parser_ckpt  /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
        --test_root    /projectnb/cs585/projects/craft/data/test \
        --vocab_json   /projectnb/cs585/projects/craft/data/data.json \
        --existing_summary clip_interpretability/clip_codebook_summary.json \
        --n_faces 16 \
        --bank_images 200 \
        --top_k_codes 16 \
        --top_k_texts 3 \
        --out_dir /projectnb/cs585/projects/craft/clip_face_panels
"""
import argparse, json, os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import CLIPModel, CLIPProcessor

from models.vqvae           import build_lq_vqvae
from models.region_aware_vq import RegionAwareVQ
from models.face_parser     import REGION_NAMES
from data.dataset           import FFHQPairedDataset


# ---------------------------------------------------------------------------
# Constants for the 5-region overlay (matches REGION_NAMES order).
# ---------------------------------------------------------------------------
REGION_COLORS_RGB = {
    "eyes": (255,  87, 87),    # red
    "skin": (255, 196, 87),    # amber
    "hair": (113, 184, 255),   # blue
    "lips": (170, 110, 220),   # purple
    "bg":   (128, 128, 128),   # grey
}


# ---------------------------------------------------------------------------
# Helpers (small, intentionally duplicated from existing scripts so this
# file is standalone and edits to existing files are not required).
# ---------------------------------------------------------------------------
def load_vocab(vocab_json, prompt_prefix="a close-up photo of "):
    with open(vocab_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab, raw_phrases = {}, {}
    for region, phrases in raw.items():
        if isinstance(phrases, list) and phrases:
            vocab[region]       = [prompt_prefix + p for p in phrases]
            raw_phrases[region] = list(phrases)
    return vocab, raw_phrases


def crop_patch(image_01, row, col, cell=32, context=64):
    _, H, W = image_01.shape
    cy, cx = row * cell + cell // 2, col * cell + cell // 2
    y0 = max(0, cy - context // 2); y0 = min(y0, H - context)
    x0 = max(0, cx - context // 2); x0 = min(x0, W - context)
    return image_01[:, y0:y0 + context, x0:x0 + context]


@torch.no_grad()
def clip_embed_images(patches, clip, clip_mean, clip_std, device, batch=64):
    embs = []
    for i in range(0, patches.shape[0], batch):
        p = patches[i:i+batch].to(device)
        p = F.interpolate(p, size=(224, 224), mode="bilinear", align_corners=False)
        p = (p - clip_mean) / clip_std
        e = F.normalize(clip.get_image_features(pixel_values=p), dim=-1).float()
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)


# ---------------------------------------------------------------------------
# Pass 1: build per-code CLIP image centroids using a bank of test images.
# (Same pattern as cross_region_clip_eval.py; reproduced here so we don't
#  edit existing files.)
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_assignments(model, loader, device, regions):
    ravq = model.quantizer
    assignments = {r: defaultdict(lambda: defaultdict(list)) for r in regions}
    images_kept = []
    global_idx  = 0

    for batch in loader:
        hq   = batch["hq"].to(device)
        hq01 = batch["hq_01"].to(device)
        B    = hq.shape[0]
        z    = model.encode(hq)
        _, _, H, W = z.shape
        masks  = ravq.face_parser.get_region_masks(hq01, target_h=H, target_w=W)
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, ravq.e_dim)

        for name in regions:
            if name not in ravq.region_codebooks: continue
            mask = masks[name].reshape(B, H * W)
            if mask.sum() == 0: continue
            feats = z_flat[mask]
            b_idx, p_idx = mask.nonzero(as_tuple=True)
            rows = (p_idx // W).cpu().tolist()
            cols = (p_idx %  W).cpu().tolist()
            imgs = (b_idx + global_idx).cpu().tolist()
            rq   = ravq.region_codebooks[name]
            residual = F.normalize(feats.float(), dim=1)
            for lvl, vq in enumerate(rq.levels):
                emb = F.normalize(vq.embedding.weight.float(), dim=1)
                sims = F.normalize(residual, dim=1) @ emb.t()
                max_sim, idx = sims.max(dim=-1)
                for i, cid in enumerate(idx.cpu().tolist()):
                    assignments[name][lvl][cid].append(
                        (imgs[i], rows[i], cols[i], max_sim[i].item())
                    )
                residual = residual - vq.embedding(idx).float()

        images_kept.append(hq01.cpu())
        global_idx += B

    return (
        {r: {l: dict(d) for l, d in lvls.items()} for r, lvls in assignments.items()},
        torch.cat(images_kept, dim=0),
    )


@torch.no_grad()
def build_code_clip_centroids(assignments, images_01, clip, clip_mean, clip_std,
                              device, top_k=16, context=64, min_hits=4):
    code_embs = {}
    for region, region_levels in assignments.items():
        flat = []
        for lvl, code_dict in region_levels.items():
            for cid, hits in code_dict.items():
                if len(hits) < min_hits: continue
                hits_s = sorted(hits, key=lambda x: -x[3])[:top_k]
                patches = torch.stack([
                    crop_patch(images_01[i], r, c, context=context)
                    for (i, r, c, _) in hits_s
                ])
                embs     = clip_embed_images(patches, clip, clip_mean, clip_std, device)
                centroid = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
                flat.append(centroid)
        code_embs[region] = torch.cat(flat, dim=0) if flat else torch.zeros((0, 768))
        print(f"  codebook[{region}]: {code_embs[region].shape[0]} code centroids")
    return code_embs


# ---------------------------------------------------------------------------
# Pass 2: for each region's codebook, find the top-N best CLIP text matches
# across all phrases in data.json (mean-max scoring on the codebook centroid).
# ---------------------------------------------------------------------------
@torch.no_grad()
def codebook_top_text_matches(code_embs, vocab, raw_phrases, clip, proc, device,
                              top_n=3):
    labels = {}
    for region, codes in code_embs.items():
        if codes.shape[0] == 0:
            labels[region] = []
            continue
        codebook_centroid = F.normalize(codes.mean(dim=0, keepdim=True), dim=-1)
        # Score every phrase from EVERY region (cross-region) so the labels are
        # a real argmax, not restricted to the region's own vocab subset.
        all_phrases, all_prefix = [], []
        for src_region, phrases in vocab.items():
            for p_full, p_raw in zip(phrases, raw_phrases[src_region]):
                all_phrases.append(p_full)
                all_prefix.append((src_region, p_raw))
        ins = proc(text=all_phrases, return_tensors="pt",
                   padding=True, truncation=True).to(device)
        emb = F.normalize(clip.get_text_features(**ins), dim=-1).float().cpu()
        sims = (codebook_centroid @ emb.t()).squeeze(0)              # (N_phrases,)
        top_idx = sims.argsort(descending=True)[:top_n].tolist()
        labels[region] = [
            (all_prefix[i][0], all_prefix[i][1], float(sims[i])) for i in top_idx
        ]
    return labels


# ---------------------------------------------------------------------------
# Per-face: build the 16x16xR similarity tensor (CLIP patch -> codebook).
# ---------------------------------------------------------------------------
@torch.no_grad()
def per_face_clip_similarity(image_01, code_embs, regions, clip, clip_mean,
                             clip_std, device, H=16, W=16, context=64, cell=32):
    """
    For one face, return a tensor (R, H, W) where entry [r, i, j] is the
    cosine sim between the CLIP embedding of the (i,j) face patch and the
    nearest code in codebook R. This is the spatial proof of region specificity.
    """
    patches = torch.stack([
        crop_patch(image_01, i, j, cell=cell, context=context)
        for i in range(H) for j in range(W)
    ])                                                                # (H*W, 3, ctx, ctx)
    patch_embs = clip_embed_images(patches, clip, clip_mean, clip_std, device)
    sim_per_region = []
    for r in regions:
        cb = code_embs.get(r, torch.zeros((0, patch_embs.shape[1])))
        if cb.shape[0] == 0:
            sim_per_region.append(np.full((H, W), np.nan))
            continue
        sims = (patch_embs @ cb.t()).max(dim=-1).values.numpy()       # (H*W,)
        sim_per_region.append(sims.reshape(H, W))
    return np.stack(sim_per_region, axis=0)                           # (R, H, W)


# ---------------------------------------------------------------------------
# Rendering helpers.
# ---------------------------------------------------------------------------
def to_uint8(img_01):
    """(3, H, W) tensor in [0,1] -> (H, W, 3) uint8 numpy."""
    return (img_01.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def make_region_overlay(image_01, region_indices_full, alpha=0.45):
    """
    image_01:           (3, H, W) tensor in [0,1]
    region_indices_full:(H, W) int tensor with values 0..len(REGION_NAMES)-1
    Returns:            (H, W, 3) uint8
    """
    base = to_uint8(image_01).astype(np.float32)
    overlay = np.zeros_like(base)
    for idx, name in enumerate(REGION_NAMES):
        m = (region_indices_full.cpu().numpy() == idx)
        overlay[m] = REGION_COLORS_RGB[name]
    blended = (1 - alpha) * base + alpha * overlay
    return blended.clip(0, 255).astype(np.uint8)


def upsample_heatmap(h_lowres, target_size=512):
    """h_lowres: (H, W) numpy in [0,1] -> (target, target) numpy bilinear-upsampled."""
    t = torch.from_numpy(h_lowres).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(target_size, target_size), mode="bilinear",
                       align_corners=False)
    return up.squeeze().numpy()


def overlay_heatmap_on_face(face_uint8, heat_01, cmap_name="magma", alpha=0.55):
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = (cmap(heat_01)[..., :3] * 255).astype(np.float32)
    gray = face_uint8.mean(axis=-1, keepdims=True)
    grayscale = np.repeat(gray, 3, axis=-1).astype(np.float32)
    out = (1 - alpha) * grayscale + alpha * heat_rgb
    return out.clip(0, 255).astype(np.uint8)


def draw_region_outline(ax, region_mask_lowres, color, lw=2.0, target_size=512):
    """region_mask_lowres: (H, W) bool. Draws cell outlines at upsampled scale."""
    H, W = region_mask_lowres.shape
    cell_h = target_size / H
    cell_w = target_size / W
    for i in range(H):
        for j in range(W):
            if not region_mask_lowres[i, j]: continue
            # only draw the borders that face the outside of the region
            if i == 0 or not region_mask_lowres[i-1, j]:
                ax.plot([j*cell_w, (j+1)*cell_w], [i*cell_h, i*cell_h],
                        color=color, linewidth=lw)
            if i == H-1 or not region_mask_lowres[i+1, j]:
                ax.plot([j*cell_w, (j+1)*cell_w], [(i+1)*cell_h, (i+1)*cell_h],
                        color=color, linewidth=lw)
            if j == 0 or not region_mask_lowres[i, j-1]:
                ax.plot([j*cell_w, j*cell_w], [i*cell_h, (i+1)*cell_h],
                        color=color, linewidth=lw)
            if j == W-1 or not region_mask_lowres[i, j+1]:
                ax.plot([(j+1)*cell_w, (j+1)*cell_w], [i*cell_h, (i+1)*cell_h],
                        color=color, linewidth=lw)


# ---------------------------------------------------------------------------
# Main per-face figure.
# ---------------------------------------------------------------------------
def render_face_panel(out_path, image_01, recon_01, region_idx_full,
                      region_masks_16, sim_RHW, regions, code_text_labels,
                      diag_dom_blurb=""):
    R = len(regions)
    fig = plt.figure(figsize=(4 * max(3, R), 11.5))
    gs  = fig.add_gridspec(
        nrows=2, ncols=max(3, R),
        height_ratios=[1.0, 1.0],
        hspace=0.25, wspace=0.10,
    )

    # ------------------- Row 1: context -------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(to_uint8(image_01)); ax.axis("off")
    ax.set_title("Original face", fontsize=12, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(make_region_overlay(image_01, region_idx_full)); ax.axis("off")
    ax.set_title("BiSeNet regions", fontsize=12, fontweight="bold")
    legend = [mpatches.Patch(color=np.array(REGION_COLORS_RGB[r]) / 255.0,
                              label=r) for r in REGION_NAMES]
    ax.legend(handles=legend, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=len(REGION_NAMES),
              fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(to_uint8(recon_01)); ax.axis("off")
    ax.set_title("Region-aware VQ reconstruction", fontsize=12, fontweight="bold")

    # ------------------- Row 2: per-codebook similarity heatmaps -------------------
    valid = sim_RHW[~np.isnan(sim_RHW)]
    vmin, vmax = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)

    face_uint8 = to_uint8(image_01)
    for k, region in enumerate(regions):
        ax = fig.add_subplot(gs[1, k])
        sim_lo = sim_RHW[k]
        sim_norm = (sim_lo - vmin) / max(vmax - vmin, 1e-6)
        sim_hi   = upsample_heatmap(sim_norm, target_size=face_uint8.shape[0])
        ax.imshow(overlay_heatmap_on_face(face_uint8, sim_hi))
        # Overlay the parser region outline so the viewer can see the match.
        draw_region_outline(
            ax, region_masks_16[region].cpu().numpy(),
            color=tuple(c / 255.0 for c in REGION_COLORS_RGB[region]),
            lw=2.5, target_size=face_uint8.shape[0],
        )
        ax.axis("off")
        title = f"CLIP patch -> {region} codebook"
        sub   = f"native diag={sim_lo[region_masks_16[region].cpu().numpy()].mean():.3f}" \
                if region_masks_16[region].any() else "native diag=n/a"
        ax.set_title(f"{title}\n{sub}", fontsize=10)

    # ------------------- Caption: top text labels per region -------------------
    label_lines = []
    for r in regions:
        labs = code_text_labels.get(r, [])
        if not labs:
            label_lines.append(f"  {r}: (no codes)"); continue
        formatted = "; ".join(
            f"\"{txt}\" [{src}, {sim:.3f}]" for (src, txt, sim) in labs
        )
        label_lines.append(f"  {r}:  {formatted}")
    caption = "Top CLIP text matches per codebook (from data.json):\n" + \
              "\n".join(label_lines)
    if diag_dom_blurb:
        caption = diag_dom_blurb + "\n\n" + caption

    fig.suptitle(
        "Region-aware codebooks on a single test face\n"
        "(heatmap inside the colored outline should be brighter than outside)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.text(0.01, 0.005, caption, fontsize=8.5, family="monospace",
             ha="left", va="bottom", wrap=True)

    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Existing-summary loading.
# ---------------------------------------------------------------------------
def load_existing_summary(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f:
        s = json.load(f)
    return s


def diag_dominance_blurb(summary):
    if summary is None: return ""
    dd = summary.get("diag_dominance", {})
    parts = ["Aggregate scores (precomputed in clip_codebook_summary.json):"]
    for k in ("patch", "centroid", "meanmax"):
        if k in dd:
            parts.append(f"  {k:<9} diag-wins={dd[k]['win_rate']*100:.0f}%, "
                         f"avg margin={dd[k]['avg_margin']:+.4f}")
    return "\n".join(parts) if len(parts) > 1 else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft_ckpt",  required=True)
    ap.add_argument("--parser_ckpt", required=True)
    ap.add_argument("--test_root",   required=True,
                    help="Path to the test split (FFHQPairedDataset layout).")
    ap.add_argument("--vocab_json",  required=True)
    ap.add_argument("--existing_summary", default="",
                    help="Optional clip_codebook_summary.json to print "
                         "precomputed diagonal-dominance numbers in the caption.")
    ap.add_argument("--n_faces",     type=int, default=16,
                    help="How many test faces to visualize.")
    ap.add_argument("--bank_images", type=int, default=200,
                    help="How many test images to use to build per-code CLIP "
                         "centroids. Larger = more stable code centroids.")
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--top_k_codes", type=int, default=16,
                    help="Top-k activating patches used to embed each code.")
    ap.add_argument("--top_k_texts", type=int, default=3,
                    help="Top-N text labels printed per codebook.")
    ap.add_argument("--context",     type=int, default=64)
    ap.add_argument("--min_hits",    type=int, default=4)
    ap.add_argument("--clip_model",  default="openai/clip-vit-large-patch14")
    ap.add_argument("--out_dir",     required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Vocab ---
    print(f"Loading vocabulary from {args.vocab_json}")
    vocab, raw_phrases = load_vocab(args.vocab_json)
    for r, phrases in vocab.items():
        print(f"  vocab[{r}]: {len(phrases)} phrases")

    # --- Model ---
    print(f"Loading CRAFT checkpoint from {args.craft_ckpt}")
    ckpt  = torch.load(args.craft_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("lq_model"))
    has_mag = any("magnitude_head" in k for k in state.keys())
    ravq  = RegionAwareVQ(e_dim=512, n_levels=3,
                          parser_ckpt=args.parser_ckpt,
                          use_magnitude_head=has_mag)
    model = build_lq_vqvae(ravq, embed_dim=512)
    model.load_state_dict(state)
    model = model.to(device).eval()
    regions = [r for r in REGION_NAMES if r in model.quantizer.region_codebooks]
    print(f"Active regions: {regions}")

    # --- Test data ---
    # `--test_root` is treated as a flat directory of PNGs (no images512x512/
    # subfolder). Pass hq_folder="" so the dataset globs the root directly.
    ds = FFHQPairedDataset(args.test_root, hq_folder="", hq_only=True, masks_folder="")
    bank_size = min(args.bank_images, len(ds))
    bank_ds = Subset(ds, range(bank_size))
    bank_loader = DataLoader(bank_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    # --- CLIP ---
    print(f"Loading CLIP model: {args.clip_model}")
    clip = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.clip_model)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=device).view(1, 3, 1, 1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                             device=device).view(1, 3, 1, 1)

    # --- Pass 1: build per-code CLIP centroids on the test bank ---
    print(f"[1/3] Building per-code CLIP centroids on {bank_size} test images...")
    assignments, images_01_bank = collect_assignments(model, bank_loader, device, regions)
    code_embs = build_code_clip_centroids(
        assignments, images_01_bank, clip, clip_mean, clip_std, device,
        top_k=args.top_k_codes, context=args.context, min_hits=args.min_hits,
    )

    # --- Pass 2: top text matches per codebook ---
    print(f"[2/3] Finding top-{args.top_k_texts} text matches per codebook...")
    code_text_labels = codebook_top_text_matches(
        code_embs, vocab, raw_phrases, clip, proc, device, top_n=args.top_k_texts,
    )
    for r, labs in code_text_labels.items():
        for src, txt, sim in labs:
            print(f"    {r:<5} <- [{src:<5}] '{txt}'  ({sim:.3f})")

    # --- Existing aggregate scores (optional) ---
    summary    = load_existing_summary(args.existing_summary)
    diag_blurb = diag_dominance_blurb(summary)

    # --- Pass 3: render n_faces panels (faces NOT used in the bank) ---
    n_faces = min(args.n_faces, len(ds))
    if n_faces == 0:
        raise RuntimeError(f"No images found at {args.test_root}.")

    # Pick faces from the *end* of the dataset so they don't overlap the bank.
    # If the test set is too small, fall back to early indices.
    start = max(bank_size, len(ds) - n_faces) if len(ds) > bank_size else 0
    face_indices = list(range(start, start + n_faces))
    face_ds = Subset(ds, face_indices)
    face_loader = DataLoader(face_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"[3/3] Rendering {n_faces} per-face panels (indices {face_indices[0]}..{face_indices[-1]})...")
    ravq_for_parse = model.quantizer

    for k, batch in enumerate(face_loader):
        idx_global = face_indices[k]
        stem = batch["filename"][0]
        hq   = batch["hq"].to(device)
        hq01 = batch["hq_01"].to(device)

        # Encode + native quantize + decode
        with torch.no_grad():
            z = model.encode(hq)
            _, _, H, W = z.shape
            masks_16 = ravq_for_parse.face_parser.get_region_masks(
                hq01, target_h=H, target_w=W,
            )
            z_q, _, _ = ravq_for_parse(z, hq01, masks=masks_16)
            recon = model.decode(z_q)
            # Full-res region map for the colored overlay. Use the 19-class
            # labels at input resolution and remap with the parser's LUT,
            # which avoids the divisibility assertion in _any_hit_downsample.
            labels_full = ravq_for_parse.face_parser.get_full_segmentation(hq01)
            full_idx = ravq_for_parse.face_parser.region_lut[labels_full][0]  # (H_img, W_img)

        # Bring everything back to [0,1] for rendering
        recon01 = (recon[0].clamp(-1, 1) + 1.0) / 2.0

        # Per-face CLIP similarity
        sim_RHW = per_face_clip_similarity(
            hq01[0].cpu(), code_embs, regions, clip, clip_mean, clip_std,
            device, H=H, W=W, context=args.context, cell=hq01.shape[-1] // H,
        )

        # 16x16 region masks for outlines
        masks_16_for_render = {r: masks_16[r][0] for r in regions}

        out_path = os.path.join(args.out_dir, f"face_panel_{idx_global:05d}_{stem}.png")
        render_face_panel(
            out_path,
            image_01=hq01[0].cpu(),
            recon_01=recon01.cpu(),
            region_idx_full=full_idx,
            region_masks_16=masks_16_for_render,
            sim_RHW=sim_RHW,
            regions=regions,
            code_text_labels=code_text_labels,
            diag_dom_blurb=diag_blurb,
        )
        print(f"  saved {out_path}")

    # --- Save a small JSON describing this run for later inspection ---
    run_meta = {
        "n_faces":            n_faces,
        "bank_images":        bank_size,
        "regions":            regions,
        "top_k_codes":        args.top_k_codes,
        "top_k_texts":        args.top_k_texts,
        "code_centroid_counts": {r: int(code_embs[r].shape[0]) for r in regions},
        "code_text_labels":   {r: [{"src": s, "phrase": t, "sim": v}
                                    for s, t, v in code_text_labels.get(r, [])]
                                for r in regions},
        "existing_summary":   args.existing_summary or None,
    }
    with open(os.path.join(args.out_dir, "face_panel_run.json"), "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"\nDone -> {args.out_dir}")


if __name__ == "__main__":
    main()
