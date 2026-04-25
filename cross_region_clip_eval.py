"""
cross_region_clip_eval.py - CLIP-space cross-region codebook specificity.

Asks: in CLIP's perceptual space, do patches from region R_a look more
similar to codebook R_b's representative patches when R_a == R_b?

Method:
    1. Encode images, get 16x16 latents and face-parser region masks.
    2. Native-quantize each region's patches -> (region, level, code_id)
       assignment (same replay as interpret_codebook.py).
    3. For each code, embed its top-K activating image patches with CLIP
       and mean-pool -> one 768-d CLIP centroid per code.
    4. For each source region R_a, CLIP-embed a random sample of its
       patches (NOT aggregated by code) -> per-patch embeddings.
    5. Build two 5x5 matrices:
         (a) patch-to-codebook: mean over R_a patches of
             max over codes in R_b of cos(patch_clip, code_clip)
         (b) centroid-to-centroid: cos(mean R_a patch CLIP,
             mean R_b code CLIP)

Outputs in --out_dir:
    clip_cross_region_patch.csv       - matrix (a)
    clip_cross_region_centroid.csv    - matrix (b)
    clip_cross_region_heatmap.png     - 2-panel heatmap
    clip_cross_region_summary.json    - raw numbers + diag dominance stats
    clip_cross_region_report.md       - readable tables

Usage:
    python cross_region_clip_eval.py \
        --craft_ckpt  /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
        --parser_ckpt /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
        --data_root   /projectnb/cs585/projects/craft/data/train \
        --n_images    500 \
        --top_k       16 \
        --sample_patches_per_region 400 \
        --out_dir     /projectnb/cs585/projects/craft/clip_cross_region_results
"""
import argparse, json, os
from collections import defaultdict
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import CLIPModel, CLIPProcessor

from models.vqvae           import build_lq_vqvae
from models.region_aware_vq import RegionAwareVQ
from models.face_parser     import REGION_NAMES
from data.dataset           import FFHQPairedDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def crop_patch(image_01, row, col, cell=32, context=64):
    _, H, W = image_01.shape
    cy, cx = row * cell + cell // 2, col * cell + cell // 2
    y0 = max(0, cy - context // 2); y0 = min(y0, H - context)
    x0 = max(0, cx - context // 2); x0 = min(x0, W - context)
    return image_01[:, y0:y0 + context, x0:x0 + context]


@torch.no_grad()
def clip_embed_patches(patches, clip, clip_mean, clip_std, device, batch=64):
    """patches: (N, 3, H, W) in [0,1]. Returns (N, D) unit-norm CLIP embeddings."""
    embs = []
    for i in range(0, patches.shape[0], batch):
        p = patches[i:i+batch].to(device)
        p = F.interpolate(p, size=(224, 224), mode="bilinear", align_corners=False)
        p = (p - clip_mean) / clip_std
        e = clip.get_image_features(pixel_values=p)
        e = F.normalize(e, dim=-1).float().cpu()
        embs.append(e)
    return torch.cat(embs, dim=0)


# ---------------------------------------------------------------------------
# Pass 1: native-quantize every region patch and record (img, r, c, code, sim).
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_assignments(model, loader, device, regions):
    ravq = model.quantizer
    # assignments[region][level][code] = [(img_idx, r, c, sim), ...]
    assignments = {r: defaultdict(lambda: defaultdict(list)) for r in regions}
    # per-region sampled patches (img_idx, r, c) for evaluation side
    region_samples = {r: [] for r in regions}
    images_kept = []
    global_idx = 0

    for batch in loader:
        hq   = batch["hq"].to(device)
        hq01 = batch["hq_01"].to(device)
        B = hq.shape[0]

        z = model.encode(hq)
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

            # Record every patch for sampling later
            for i in range(len(imgs)):
                region_samples[name].append((imgs[i], rows[i], cols[i]))

            # Native residual-VQ replay to get codes
            rq = ravq.region_codebooks[name]
            residual = F.normalize(feats.float(), dim=1)
            for lvl, vq in enumerate(rq.levels):
                emb = F.normalize(vq.embedding.weight.float(), dim=1)
                sims = F.normalize(residual, dim=1) @ emb.t()
                max_sim, idx = sims.max(dim=-1)
                idx_l = idx.cpu().tolist()
                sim_l = max_sim.cpu().tolist()
                for i, cid in enumerate(idx_l):
                    assignments[name][lvl][cid].append(
                        (imgs[i], rows[i], cols[i], sim_l[i])
                    )
                residual = residual - vq.embedding(idx).float()

        images_kept.append(hq01.cpu())
        global_idx += B

    return (
        {r: {l: dict(d) for l, d in lvls.items()} for r, lvls in assignments.items()},
        region_samples,
        torch.cat(images_kept, dim=0),
    )


# ---------------------------------------------------------------------------
# Build per-code CLIP centroids (one per code, all levels, all regions).
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_code_clip_centroids(
    assignments, images_01, clip, proc, device, top_k=16, context=64,
    min_hits=4,
):
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                              device=device).view(1, 3, 1, 1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                              device=device).view(1, 3, 1, 1)
    # code_emb[region] = (N_codes, D)  — stacks across all levels for that region
    # We also keep a per-region "all codes" list for matrix (a).
    code_embs = {}
    for region, region_levels in assignments.items():
        flat = []
        for lvl, code_dict in region_levels.items():
            for cid, hits in code_dict.items():
                if len(hits) < min_hits: continue
                hits_sorted = sorted(hits, key=lambda x: -x[3])[:top_k]
                patches = torch.stack([
                    crop_patch(images_01[i], r, c, context=context)
                    for (i, r, c, _) in hits_sorted
                ])
                embs = clip_embed_patches(patches, clip, clip_mean, clip_std, device)
                centroid = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
                flat.append(centroid)
        if flat:
            code_embs[region] = torch.cat(flat, dim=0)         # (N_region_codes, D)
        else:
            code_embs[region] = torch.zeros((0, 768))
        print(f"  codebook[{region}]: {code_embs[region].shape[0]} code centroids")
    return code_embs


# ---------------------------------------------------------------------------
# Sample patches per region and CLIP-embed them (the "test set").
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_region_patch_embeddings(
    region_samples, images_01, clip, proc, device,
    n_per_region=400, context=64, seed=0,
):
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                              device=device).view(1, 3, 1, 1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                              device=device).view(1, 3, 1, 1)
    rng = random.Random(seed)
    region_embs = {}
    for region, samples in region_samples.items():
        if not samples:
            region_embs[region] = torch.zeros((0, 768)); continue
        chosen = rng.sample(samples, min(n_per_region, len(samples)))
        patches = torch.stack([
            crop_patch(images_01[i], r, c, context=context)
            for (i, r, c) in chosen
        ])
        region_embs[region] = clip_embed_patches(
            patches, clip, clip_mean, clip_std, device
        )
        print(f"  region_patches[{region}]: {region_embs[region].shape[0]} samples")
    return region_embs


# ---------------------------------------------------------------------------
# Compute the two 5x5 matrices.
# ---------------------------------------------------------------------------
def build_matrices(region_embs, code_embs, regions):
    n = len(regions)
    patch_mat    = np.full((n, n), np.nan)
    centroid_mat = np.full((n, n), np.nan)
    for i, ra in enumerate(regions):
        pa = region_embs.get(ra, torch.zeros((0, 768)))
        if pa.shape[0] == 0: continue
        ra_centroid = F.normalize(pa.mean(dim=0, keepdim=True), dim=-1)
        for j, rb in enumerate(regions):
            cb = code_embs.get(rb, torch.zeros((0, 768)))
            if cb.shape[0] == 0: continue
            # Matrix (a): per patch -> max cos over codes
            sims = pa @ cb.t()                           # (N_patches, N_codes)
            patch_mat[i, j] = sims.max(dim=-1).values.mean().item()
            # Matrix (b): centroid-to-centroid
            rb_centroid = F.normalize(cb.mean(dim=0, keepdim=True), dim=-1)
            centroid_mat[i, j] = (ra_centroid @ rb_centroid.t()).item()
    return patch_mat, centroid_mat


def diagonal_dominance(m):
    n = m.shape[0]; wins = 0; margins = []
    for i in range(n):
        row = m[i].copy()
        if np.all(np.isnan(row)): continue
        diag = row[i]
        off  = np.concatenate([row[:i], row[i+1:]])
        best_off = np.nanmax(off)
        if diag > best_off: wins += 1
        margins.append(float(diag - best_off))
    return wins / n, float(np.nanmean(margins))


def save_csv(path, mat, regions):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src\\tgt"] + regions)
        for i, r in enumerate(regions):
            row = [r] + [f"{mat[i, j]:.4f}" if not np.isnan(mat[i, j]) else "NaN"
                         for j in range(len(regions))]
            w.writerow(row)


def plot_heatmap(path, patch_mat, centroid_mat, regions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m, title in zip(
        axes, [patch_mat, centroid_mat],
        ["CLIP patch→codebook (best-match mean)",
         "CLIP region centroid ↔ codebook centroid"],
    ):
        im = ax.imshow(m, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions)
        ax.set_yticks(range(len(regions))); ax.set_yticklabels(regions)
        ax.set_xlabel("target codebook"); ax.set_ylabel("source region")
        ax.set_title(title)
        for i in range(len(regions)):
            for j in range(len(regions)):
                v = m[i, j]
                if np.isnan(v): continue
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="w", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft_ckpt",  required=True)
    ap.add_argument("--parser_ckpt", required=True)
    ap.add_argument("--data_root",   required=True)
    ap.add_argument("--n_images",   type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--top_k",      type=int, default=16)
    ap.add_argument("--context",    type=int, default=64)
    ap.add_argument("--min_hits",   type=int, default=4)
    ap.add_argument("--sample_patches_per_region", type=int, default=400)
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out_dir",    required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    ckpt  = torch.load(args.craft_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("lq_model"))
    has_mag = any("magnitude_head" in k for k in state.keys())
    ravq  = RegionAwareVQ(
        e_dim=512, n_levels=3,
        parser_ckpt=args.parser_ckpt,
        use_magnitude_head=has_mag,
    )
    model = build_lq_vqvae(ravq, embed_dim=512)
    model.load_state_dict(state)
    model = model.to(device).eval()
    regions = [r for r in REGION_NAMES if r in model.quantizer.region_codebooks]
    print(f"Evaluating regions: {regions}")

    # --- Data ---
    ds = FFHQPairedDataset(args.data_root, hq_only=True, masks_folder="")
    if args.n_images < len(ds):
        ds = Subset(ds, range(args.n_images))
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    # --- Pass 1: assignments ---
    print("[1/3] Collecting per-code assignments...")
    assignments, region_samples, images_01 = collect_assignments(
        model, loader, device, regions
    )
    for r in regions:
        n_codes = sum(len(d) for d in assignments[r].values())
        print(f"  {r}: {n_codes} (level,code) pairs, {len(region_samples[r])} patches")

    # --- Pass 2: CLIP code centroids ---
    print("[2/3] Embedding code centroids with CLIP...")
    clip = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.clip_model)
    code_embs = build_code_clip_centroids(
        assignments, images_01, clip, proc, device,
        top_k=args.top_k, context=args.context, min_hits=args.min_hits,
    )

    # --- Pass 3: CLIP region patch sample embeddings ---
    print("[3/3] Embedding sampled region patches with CLIP...")
    region_embs = build_region_patch_embeddings(
        region_samples, images_01, clip, proc, device,
        n_per_region=args.sample_patches_per_region, context=args.context,
    )

    # --- Matrices ---
    patch_mat, centroid_mat = build_matrices(region_embs, code_embs, regions)

    # --- Save ---
    save_csv(os.path.join(args.out_dir, "clip_cross_region_patch.csv"),
             patch_mat, regions)
    save_csv(os.path.join(args.out_dir, "clip_cross_region_centroid.csv"),
             centroid_mat, regions)
    plot_heatmap(os.path.join(args.out_dir, "clip_cross_region_heatmap.png"),
                 patch_mat, centroid_mat, regions)

    fracP, marP = diagonal_dominance(patch_mat)
    fracC, marC = diagonal_dominance(centroid_mat)
    summary = {
        "regions":   regions,
        "patch_mat": patch_mat.tolist(),
        "centroid_mat": centroid_mat.tolist(),
        "diag_dominance": {
            "patch":    {"win_rate": fracP, "avg_margin": marP},
            "centroid": {"win_rate": fracC, "avg_margin": marC},
        },
    }
    with open(os.path.join(args.out_dir, "clip_cross_region_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    md = ["# CLIP-space cross-region codebook specificity\n"]
    md.append(f"Regions: `{', '.join(regions)}`  |  "
              f"n_images={args.n_images}  |  "
              f"top_k={args.top_k}  |  "
              f"sample_patches={args.sample_patches_per_region}\n")
    md.append("## Diagonal dominance\n")
    md.append(f"- Patch→codebook (best-match)   : "
              f"{fracP*100:.0f}% rows win  avg margin={marP:+.3f}")
    md.append(f"- Centroid↔centroid             : "
              f"{fracC*100:.0f}% rows win  avg margin={marC:+.3f}\n")

    for title, mat in [
        ("Patch-to-codebook (per-patch max cos, averaged)", patch_mat),
        ("Centroid-to-centroid cosine",                     centroid_mat),
    ]:
        md.append(f"## {title}\n")
        md.append("| src \\ tgt | " + " | ".join(regions) + " |")
        md.append("|" + "---|" * (len(regions) + 1))
        for i, r in enumerate(regions):
            best_j = int(np.nanargmax(mat[i]))
            cells = []
            for j in range(len(regions)):
                v = mat[i, j]
                s = "n/a" if np.isnan(v) else f"{v:.3f}"
                if j == best_j: s = f"**{s}**"
                cells.append(s)
            md.append(f"| **{r}** | " + " | ".join(cells) + " |")
        md.append("")
    with open(os.path.join(args.out_dir, "clip_cross_region_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Done -> {args.out_dir}")
    print(f"  Patch diagonal wins : {fracP*100:.0f}%  margin={marP:+.3f}")
    print(f"  Centroid diagonal wins: {fracC*100:.0f}%  margin={marC:+.3f}")


if __name__ == "__main__":
    main()
