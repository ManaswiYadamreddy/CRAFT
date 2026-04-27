"""
code_face_atlas.py - Per-codebook atlas of top codes on test faces.

Goal:
    Answer "what does each code in each region's codebook actually represent?"
    by rendering a grid for every region:
        rows  = top-K most-frequently-used codes (across the test bank)
        cols  = the top-N face crops that most strongly activate that code,
                plus the best-matching CLIP text phrase from data.json
                printed to the left of the row.

    Coupled with face_codebook_panel.py, this is the per-code half of the
    "are the region-aware codebooks working?" story:
        face_codebook_panel.py   shows per-face spatial specificity
        code_face_atlas.py       shows per-code semantic consistency

This script does NOT modify any existing file. It only reads:
    - the model checkpoint
    - the face parser checkpoint
    - the test data root
    - data.json (vocabulary)

Usage:
    python code_face_atlas.py \
        --craft_ckpt   /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
        --parser_ckpt  /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
        --test_root    /projectnb/cs585/projects/craft/data/test \
        --vocab_json   /projectnb/cs585/projects/craft/data/data.json \
        --bank_images  300 \
        --top_k_codes  10 \
        --top_n_patches 8 \
        --context 80 \
        --out_dir /projectnb/cs585/projects/craft/clip_code_atlas
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


REGION_COLORS_RGB = {
    "eyes": (255,  87, 87),
    "skin": (255, 196, 87),
    "hair": (113, 184, 255),
    "lips": (170, 110, 220),
    "bg":   (128, 128, 128),
}


# ---------------------------------------------------------------------------
# Helpers (kept local so this script is self-contained).
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
# Pass 1: collect (region, level, code) -> [(img_idx, r, c, sim), ...]
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
                emb  = F.normalize(vq.embedding.weight.float(), dim=1)
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


# ---------------------------------------------------------------------------
# Per-(region, level, code) CLIP centroid + best CLIP text label.
# ---------------------------------------------------------------------------
@torch.no_grad()
def label_top_codes(assignments, images_01, vocab, raw_phrases, clip, proc,
                    clip_mean, clip_std, device, top_k_codes=10, top_k_patches=16,
                    context=80, min_hits=4):
    """
    For each region, returns a sorted list (most-used first) of dicts:
        { region, level, code_id, count, label_src, label_text, label_sim,
          patch_hits: [(img_idx, row, col, sim), ...] }
    with at most top_k_codes entries per region.
    """
    # --- Embed every text phrase once. ---
    all_phrases, all_prefix = [], []
    for src_region, phrases in vocab.items():
        for p_full, p_raw in zip(phrases, raw_phrases[src_region]):
            all_phrases.append(p_full)
            all_prefix.append((src_region, p_raw))
    ins = proc(text=all_phrases, return_tensors="pt",
               padding=True, truncation=True).to(device)
    text_emb = F.normalize(clip.get_text_features(**ins), dim=-1).float().cpu()

    region_top = {}
    for region, region_levels in assignments.items():
        # Rank codes within this region by total usage across all levels.
        ranked = []
        for lvl, code_dict in region_levels.items():
            for cid, hits in code_dict.items():
                if len(hits) < min_hits: continue
                ranked.append((lvl, cid, hits))
        ranked.sort(key=lambda x: -len(x[2]))
        ranked = ranked[:top_k_codes]

        entries = []
        for lvl, cid, hits in ranked:
            hits_s = sorted(hits, key=lambda x: -x[3])[:top_k_patches]
            patches = torch.stack([
                crop_patch(images_01[i], r, c, context=context)
                for (i, r, c, _) in hits_s
            ])
            embs     = clip_embed_images(patches, clip, clip_mean, clip_std, device)
            centroid = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
            sims     = (centroid @ text_emb.t()).squeeze(0)
            best     = int(torch.argmax(sims).item())
            entries.append({
                "region":      region,
                "level":       int(lvl),
                "code_id":     int(cid),
                "count":       len(hits),
                "label_src":   all_prefix[best][0],
                "label_text":  all_prefix[best][1],
                "label_sim":   float(sims[best].item()),
                "patch_hits":  [(int(i), int(r), int(c), float(s))
                                 for (i, r, c, s) in hits_s],
            })
        region_top[region] = entries
        print(f"  region[{region}]: {len(entries)} top codes labelled")
    return region_top


# ---------------------------------------------------------------------------
# Rendering: one figure per region.
# ---------------------------------------------------------------------------
def to_uint8(img_01):
    return (img_01.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def render_region_atlas(out_path, region, entries, images_01, top_n_patches,
                        context=80):
    if not entries:
        print(f"  [{region}] no entries -> skipping figure"); return

    n_rows = len(entries)
    n_cols = top_n_patches + 1                  # +1 leftmost column for label/text
    fig_w  = 1.6 * n_cols
    fig_h  = 1.6 * n_rows + 1.2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              gridspec_kw={"wspace": 0.05, "hspace": 0.20})
    if n_rows == 1: axes = np.array([axes])

    region_color = tuple(c / 255.0 for c in REGION_COLORS_RGB.get(region, (200, 200, 200)))

    for row_i, entry in enumerate(entries):
        # Leftmost column = label panel
        ax = axes[row_i, 0]
        ax.axis("off")
        ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                         color=region_color, alpha=0.20))
        ax.text(
            0.04, 0.78,
            f"L{entry['level']} c{entry['code_id']:>3}\n(used {entry['count']} px)",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            family="monospace", va="top",
        )
        ax.text(
            0.04, 0.50,
            f"\"{entry['label_text']}\"\n[{entry['label_src']}, "
            f"sim={entry['label_sim']:.3f}]",
            transform=ax.transAxes, fontsize=8.5, va="top", wrap=True,
        )

        # Patch columns
        hits = entry["patch_hits"][:top_n_patches]
        for col_j in range(top_n_patches):
            ax = axes[row_i, col_j + 1]
            ax.axis("off")
            if col_j >= len(hits): continue
            i, r, c, sim = hits[col_j]
            patch = crop_patch(images_01[i], r, c, context=context)
            ax.imshow(to_uint8(patch))
            ax.text(0.5, -0.06, f"sim={sim:.2f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7,
                    color="dimgray")

    fig.suptitle(
        f"Codebook atlas: {region}    "
        f"(top {len(entries)} most-used codes; rows = codes, cols = top activating face patches)",
        fontsize=12, fontweight="bold",
    )
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined index figure: one summary row per region with top-1 code only.
# ---------------------------------------------------------------------------
def render_overview(out_path, region_top, images_01, top_n_patches=6,
                    context=80, n_codes_per_region=3):
    rows = []
    for r in REGION_NAMES:
        entries = region_top.get(r, [])
        for e in entries[:n_codes_per_region]:
            rows.append((r, e))
    if not rows: return

    n_rows = len(rows)
    n_cols = top_n_patches + 1
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(1.55 * n_cols, 1.55 * n_rows + 0.6),
                              gridspec_kw={"wspace": 0.06, "hspace": 0.30})
    if n_rows == 1: axes = np.array([axes])

    for row_i, (region, entry) in enumerate(rows):
        region_color = tuple(c / 255.0 for c in REGION_COLORS_RGB.get(region, (200, 200, 200)))
        ax = axes[row_i, 0]
        ax.axis("off")
        ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                         color=region_color, alpha=0.22))
        ax.text(0.04, 0.85, f"{region.upper()}", transform=ax.transAxes,
                fontsize=10, fontweight="bold", family="monospace", va="top")
        ax.text(0.04, 0.65,
                f"L{entry['level']} c{entry['code_id']:>3}\n({entry['count']} px)",
                transform=ax.transAxes, fontsize=8, family="monospace", va="top")
        ax.text(0.04, 0.32,
                f"\"{entry['label_text']}\"\n[{entry['label_src']}, "
                f"sim={entry['label_sim']:.3f}]",
                transform=ax.transAxes, fontsize=7.5, va="top", wrap=True)

        hits = entry["patch_hits"][:top_n_patches]
        for col_j in range(top_n_patches):
            ax = axes[row_i, col_j + 1]
            ax.axis("off")
            if col_j >= len(hits): continue
            i, r, c, sim = hits[col_j]
            patch = crop_patch(images_01[i], r, c, context=context)
            ax.imshow(to_uint8(patch))
    fig.suptitle(
        "Region-aware codebook overview: top codes across regions, with CLIP text label",
        fontsize=12, fontweight="bold",
    )
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft_ckpt",  required=True)
    ap.add_argument("--parser_ckpt", required=True)
    ap.add_argument("--test_root",   required=True,
                    help="Test data root (FFHQPairedDataset layout).")
    ap.add_argument("--vocab_json",  required=True)
    ap.add_argument("--bank_images",   type=int, default=300)
    ap.add_argument("--batch_size",    type=int, default=8)
    ap.add_argument("--top_k_codes",   type=int, default=10,
                    help="How many of the most-used codes to render per region.")
    ap.add_argument("--top_n_patches", type=int, default=8,
                    help="How many activating face patches to render per code.")
    ap.add_argument("--context",       type=int, default=80,
                    help="Crop size around the cell center (in HQ pixels).")
    ap.add_argument("--min_hits",      type=int, default=4)
    ap.add_argument("--clip_model",    default="openai/clip-vit-large-patch14")
    ap.add_argument("--out_dir",       required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Vocab ---
    print(f"Loading vocabulary from {args.vocab_json}")
    vocab, raw_phrases = load_vocab(args.vocab_json)

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

    # --- Test data bank ---
    ds = FFHQPairedDataset(args.test_root, hq_only=True, masks_folder="")
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

    # --- Pass 1: assignments ---
    print(f"[1/2] Collecting code assignments on {bank_size} test images...")
    assignments, images_01 = collect_assignments(model, bank_loader, device, regions)
    for r in regions:
        n = sum(len(d) for d in assignments[r].values())
        print(f"  {r}: {n} active (level,code) pairs")

    # --- Pass 2: label & rank top codes ---
    print(f"[2/2] Labelling top-{args.top_k_codes} codes per region with CLIP text...")
    region_top = label_top_codes(
        assignments, images_01, vocab, raw_phrases, clip, proc,
        clip_mean, clip_std, device,
        top_k_codes=args.top_k_codes, top_k_patches=args.top_n_patches,
        context=args.context, min_hits=args.min_hits,
    )

    # --- Save the label JSON for later inspection / reuse ---
    serializable = {
        r: [{k: v for k, v in e.items()} for e in entries]
        for r, entries in region_top.items()
    }
    with open(os.path.join(args.out_dir, "code_atlas_labels.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    # --- Per-region figures ---
    print("Rendering atlas figures...")
    for region in regions:
        out_path = os.path.join(args.out_dir, f"atlas_{region}.png")
        render_region_atlas(
            out_path, region, region_top.get(region, []), images_01,
            top_n_patches=args.top_n_patches, context=args.context,
        )
        print(f"  saved {out_path}")

    # --- Combined overview ---
    overview_path = os.path.join(args.out_dir, "atlas_overview.png")
    render_overview(overview_path, region_top, images_01,
                    top_n_patches=min(6, args.top_n_patches),
                    context=args.context, n_codes_per_region=3)
    print(f"  saved {overview_path}")

    print(f"\nDone -> {args.out_dir}")


if __name__ == "__main__":
    main()
