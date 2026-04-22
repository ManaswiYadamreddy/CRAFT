"""
cross_region_eval.py - Cross-region codebook specificity test.

Proves that each region's codebook actually specializes in its region:
for every pair (source_region R_a, target_codebook R_b), we quantize patches
from R_a using codebook R_b and measure reconstruction quality. If codebooks
are truly region-specific, the diagonal (R_a == R_b) dominates.

Metrics per (R_a, R_b) pair:
    1. Cosine similarity to nearest code (level 0)     - higher = better
    2. Full residual reconstruction cosine similarity  - higher = better
    3. L2 reconstruction error (after full residual)   - lower  = better

Outputs (in --out_dir):
    cross_region_cosine.csv          - numeric matrix
    cross_region_recon_cos.csv       - numeric matrix
    cross_region_recon_l2.csv        - numeric matrix
    cross_region_heatmap.png         - visual matrix
    cross_region_summary.json        - per-cell detail + diagonal dominance stat
    cross_region_report.md           - human-readable write-up

Usage:
    python cross_region_eval.py \
        --craft_ckpt  /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
        --parser_ckpt /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
        --data_root   /projectnb/cs585/projects/craft/data/train \
        --n_images    500 \
        --out_dir     /projectnb/cs585/projects/craft/cross_region_results
"""
import argparse, json, os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.vqvae           import build_lq_vqvae
from models.region_aware_vq import RegionAwareVQ
from models.face_parser     import REGION_NAMES
from data.dataset           import FFHQPairedDataset


# ---------------------------------------------------------------------------
# Quantize a batch of features through a target ResidualVQ (no grad).
# Returns:
#   level0_cos: (N,) cosine similarity to nearest code at level 0
#   final_cos : (N,) cosine similarity of full residual reconstruction to input
#   final_l2  : (N,) L2 distance of full residual reconstruction to input
# ---------------------------------------------------------------------------
@torch.no_grad()
def quantize_through(feats, target_rq):
    """
    feats:     (N, 512) raw feature vectors from source region.
    target_rq: a ResidualVQ module (possibly from a DIFFERENT region).
    """
    z_normed = F.normalize(feats.float(), dim=1)          # unit sphere

    # --- Level 0 cosine-sim to nearest code in target codebook ---
    emb0 = F.normalize(target_rq.levels[0].embedding.weight.float(), dim=1)
    sims0 = z_normed @ emb0.t()                           # (N, n_codes)
    level0_cos = sims0.max(dim=-1).values                 # (N,)

    # --- Full residual replay through target codebook (hard assignment) ---
    residual = z_normed
    z_q_sum  = torch.zeros_like(z_normed)
    for vq in target_rq.levels:
        emb_l = F.normalize(vq.embedding.weight.float(), dim=1)
        sims  = F.normalize(residual, dim=1) @ emb_l.t()
        idx   = sims.argmax(dim=-1)
        z_q   = vq.embedding(idx).float()                 # unit-norm
        z_q_sum  = z_q_sum + z_q
        residual = residual - z_q

    # Scale is per-region; use target_rq's scale on the hard sum
    scale = target_rq.codebook_scale.abs().detach()
    recon = z_q_sum * scale
    target = z_normed * scale

    final_cos = F.cosine_similarity(recon, target, dim=-1)
    final_l2  = (recon - target).norm(dim=-1)

    return level0_cos, final_cos, final_l2


# ---------------------------------------------------------------------------
# Run one full pass: for every (source_region, target_codebook) accumulate
# the three metrics across the whole dataset subset.
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_matrix(model, loader, device, regions):
    ravq = model.quantizer

    # Accumulators: sum and count per (src, tgt) pair
    n_regions = len(regions)
    sum_cos0  = np.zeros((n_regions, n_regions))
    sum_cosF  = np.zeros((n_regions, n_regions))
    sum_l2F   = np.zeros((n_regions, n_regions))
    count     = np.zeros((n_regions, n_regions))

    for batch in loader:
        hq   = batch["hq"].to(device)
        hq01 = batch["hq_01"].to(device)
        B    = hq.shape[0]

        z = model.encode(hq)                               # (B, 512, 16, 16)
        _, _, H, W = z.shape
        masks = ravq.face_parser.get_region_masks(hq01, target_h=H, target_w=W)
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, ravq.e_dim)

        # For every source region, quantize its features through every target codebook
        for si, src in enumerate(regions):
            mask = masks[src].reshape(B, H * W)
            if mask.sum() == 0:
                continue
            feats = z_flat[mask]                           # (N_src, 512)

            for ti, tgt in enumerate(regions):
                if tgt not in ravq.region_codebooks:
                    continue
                target_rq = ravq.region_codebooks[tgt]
                c0, cF, lF = quantize_through(feats, target_rq)
                sum_cos0[si, ti] += c0.sum().item()
                sum_cosF[si, ti] += cF.sum().item()
                sum_l2F[si, ti]  += lF.sum().item()
                count[si, ti]    += feats.shape[0]

    # Per-cell means
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_cos0 = sum_cos0 / count
        mean_cosF = sum_cosF / count
        mean_l2F  = sum_l2F  / count

    return {
        "level0_cos": mean_cos0,
        "final_cos":  mean_cosF,
        "final_l2":   mean_l2F,
        "counts":     count,
    }


def save_csv(path, matrix, regions):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src\\tgt"] + regions)
        for i, r in enumerate(regions):
            row = [r] + [f"{matrix[i, j]:.4f}" if not np.isnan(matrix[i, j]) else "NaN"
                         for j in range(len(regions))]
            w.writerow(row)


def plot_heatmap(out_path, mats, regions):
    """3-panel heatmap: level0 cos, final cos, final L2."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = [
        ("level0_cos", "Level-0 cosine sim\n(higher=better)",  "viridis"),
        ("final_cos",  "Full-residual cos sim\n(higher=better)", "viridis"),
        ("final_l2",   "Full-residual L2 error\n(lower=better)", "magma_r"),
    ]
    for ax, (key, title, cmap) in zip(axes, titles):
        m  = mats[key]
        im = ax.imshow(m, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions)
        ax.set_yticks(range(len(regions))); ax.set_yticklabels(regions)
        ax.set_xlabel("target codebook"); ax.set_ylabel("source region")
        ax.set_title(title)
        for i in range(len(regions)):
            for j in range(len(regions)):
                v = m[i, j]
                if np.isnan(v): continue
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="w" if cmap != "magma_r" else "k", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def diagonal_dominance(matrix, higher_is_better=True):
    """For each row, is the diagonal the best? Returns fraction and margin."""
    n = matrix.shape[0]
    wins = 0
    margins = []
    for i in range(n):
        row = matrix[i].copy()
        if np.all(np.isnan(row)): continue
        diag = row[i]
        off  = np.concatenate([row[:i], row[i+1:]])
        best_off = np.nanmax(off) if higher_is_better else np.nanmin(off)
        if (higher_is_better and diag > best_off) or \
           (not higher_is_better and diag < best_off):
            wins += 1
        margins.append(float(diag - best_off))
    return wins / n, float(np.nanmean(margins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft_ckpt",  required=True)
    ap.add_argument("--parser_ckpt", required=True)
    ap.add_argument("--data_root",   required=True)
    ap.add_argument("--n_images",   type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir",    default="cross_region_results")
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

    # Only evaluate regions that actually have codebooks
    regions = [r for r in REGION_NAMES if r in model.quantizer.region_codebooks]
    print(f"Evaluating regions: {regions}")

    # --- Data ---
    ds = FFHQPairedDataset(args.data_root, hq_only=True, masks_folder="")
    if args.n_images < len(ds):
        ds = Subset(ds, range(args.n_images))
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    # --- Compute matrix ---
    print(f"Running cross-region evaluation over {args.n_images} images...")
    mats = compute_matrix(model, loader, device, regions)

    # --- Save CSVs ---
    save_csv(os.path.join(args.out_dir, "cross_region_cosine.csv"),
             mats["level0_cos"], regions)
    save_csv(os.path.join(args.out_dir, "cross_region_recon_cos.csv"),
             mats["final_cos"], regions)
    save_csv(os.path.join(args.out_dir, "cross_region_recon_l2.csv"),
             mats["final_l2"], regions)

    # --- Heatmap ---
    plot_heatmap(os.path.join(args.out_dir, "cross_region_heatmap.png"),
                 mats, regions)

    # --- Diagonal-dominance stats ---
    frac0, mar0 = diagonal_dominance(mats["level0_cos"], higher_is_better=True)
    fracF, marF = diagonal_dominance(mats["final_cos"],  higher_is_better=True)
    fracL, marL = diagonal_dominance(mats["final_l2"],   higher_is_better=False)

    summary = {
        "regions":    regions,
        "level0_cos": mats["level0_cos"].tolist(),
        "final_cos":  mats["final_cos"].tolist(),
        "final_l2":   mats["final_l2"].tolist(),
        "counts":     mats["counts"].tolist(),
        "diag_dominance": {
            "level0_cos": {"win_rate": frac0, "avg_margin": mar0},
            "final_cos":  {"win_rate": fracF, "avg_margin": marF},
            "final_l2":   {"win_rate": fracL, "avg_margin": marL},
        },
    }
    with open(os.path.join(args.out_dir, "cross_region_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Markdown report ---
    md = []
    md.append("# Cross-region codebook specificity\n")
    md.append(f"Evaluated {args.n_images} images over regions: "
              f"`{', '.join(regions)}`\n")
    md.append("## Diagonal dominance (rows where native codebook wins)\n")
    md.append(f"- **Level-0 cosine sim**      : {frac0*100:.0f}% of rows "
              f"win on diagonal, avg margin = {mar0:+.3f}")
    md.append(f"- **Full-residual cosine sim**: {fracF*100:.0f}% of rows "
              f"win on diagonal, avg margin = {marF:+.3f}")
    md.append(f"- **Full-residual L2 error**  : {fracL*100:.0f}% of rows "
              f"win on diagonal (lower is better), avg margin = {marL:+.3f}\n")
    for title, key, up in [
        ("Level-0 cosine similarity",          "level0_cos", True),
        ("Full-residual cosine similarity",     "final_cos",  True),
        ("Full-residual L2 reconstruction error", "final_l2", False),
    ]:
        md.append(f"## {title}\n")
        md.append("| src \\ tgt | " + " | ".join(regions) + " |")
        md.append("|" + "---|" * (len(regions) + 1))
        m = mats[key]
        for i, r in enumerate(regions):
            row_vals = []
            best_j = (np.nanargmax(m[i]) if up else np.nanargmin(m[i]))
            for j in range(len(regions)):
                v = m[i, j]
                s = "n/a" if np.isnan(v) else f"{v:.3f}"
                if j == best_j: s = f"**{s}**"
                row_vals.append(s)
            md.append(f"| **{r}** | " + " | ".join(row_vals) + " |")
        md.append("")
    with open(os.path.join(args.out_dir, "cross_region_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Done -> {args.out_dir}")
    print(f"  Diagonal wins (level0_cos): {frac0*100:.0f}%  avg margin={mar0:+.3f}")
    print(f"  Diagonal wins (final_cos) : {fracF*100:.0f}%  avg margin={marF:+.3f}")
    print(f"  Diagonal wins (final_l2)  : {fracL*100:.0f}%  avg margin={marL:+.3f}")


if __name__ == "__main__":
    main()
