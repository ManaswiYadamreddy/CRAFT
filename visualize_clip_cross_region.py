"""
visualize_clip_cross_region.py - Publication-quality figures for CLIP cross-region results.

Produces:
  1. clip_heatmap.png        - Clean annotated heatmap with diagonal highlighted
  2. clip_intra_inter.png    - Bar chart: diagonal vs best off-diagonal per region
  3. clip_combined.png       - Combined figure (heatmap + bar chart side by side)

Usage:
    python visualize_clip_cross_region.py \
        --summary /projectnb/cs585/projects/craft/clip_cross_region_results/clip_cross_region_summary.json \
        --out_dir /projectnb/cs585/projects/craft/clip_cross_region_results
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def plot_heatmap(ax, mat, regions, title, cmap, vmin, vmax):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    n = len(regions)
    ax.set_xticks(range(n)); ax.set_xticklabels(regions, fontsize=11)
    ax.set_yticks(range(n)); ax.set_yticklabels(regions, fontsize=11)
    ax.set_xlabel("Target Codebook", fontsize=12)
    ax.set_ylabel("Source Region Patches", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            color = "white" if v < (vmin + vmax) / 2 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight=weight)

    # Highlight diagonal with a colored border
    for i in range(n):
        rect = mpatches.FancyBboxPatch(
            (i - 0.48, i - 0.48), 0.96, 0.96,
            boxstyle="round,pad=0.02",
            linewidth=2.5, edgecolor="#FF4444", facecolor="none",
        )
        ax.add_patch(rect)

    return im


def plot_intra_inter(ax, mat, regions, title):
    n = len(regions)
    intra, inter_best = [], []
    for i in range(n):
        row = mat[i].copy()
        intra.append(row[i])
        off = np.concatenate([row[:i], row[i+1:]])
        inter_best.append(np.max(off))

    x = np.arange(n)
    w = 0.35
    bars1 = ax.bar(x - w/2, intra,     width=w, label="Same region (diagonal)",
                   color="#2196F3", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + w/2, inter_best, width=w, label="Best other region (off-diag)",
                   color="#FF7043", edgecolor="white", linewidth=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(regions, fontsize=11)
    ax.set_ylabel("CLIP Cosine Similarity", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(intra + inter_best) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    s = load_summary(args.summary)
    regions   = s["regions"]
    patch_mat = np.array(s["patch_mat"])
    cent_mat  = np.array(s["centroid_mat"])

    cmap = LinearSegmentedColormap.from_list(
        "craft", ["#f7fbff", "#2171b5", "#08306b"]
    )
    vmin = min(patch_mat.min(), cent_mat.min()) - 0.005
    vmax = max(patch_mat.max(), cent_mat.max()) + 0.005

    # --- Figure 1: standalone heatmaps ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    im = plot_heatmap(axes[0], patch_mat, regions,
                      "Patch → Codebook Similarity", cmap, vmin, vmax)
    plot_heatmap(axes[1], cent_mat, regions,
                 "Centroid ↔ Centroid Similarity", cmap, vmin, vmax)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="CLIP cosine similarity")
    fig.suptitle("CLIP-Space Cross-Region Codebook Specificity", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = os.path.join(args.out_dir, "clip_heatmap.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    # --- Figure 2: intra vs inter bar charts ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_intra_inter(axes[0], patch_mat, regions,
                     "Patch → Codebook: Same vs Best-Other Region")
    plot_intra_inter(axes[1], cent_mat, regions,
                     "Centroid ↔ Centroid: Same vs Best-Other Region")
    fig.suptitle("CLIP Similarity: Native vs Cross-Region Codebook", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(args.out_dir, "clip_intra_inter.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    # --- Figure 3: combined (heatmap + bar) ---
    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    im = plot_heatmap(ax1, patch_mat, regions, "Patch → Codebook Similarity", cmap, vmin, vmax)
    plot_heatmap(ax2, cent_mat, regions, "Centroid ↔ Centroid Similarity", cmap, vmin, vmax)
    plot_intra_inter(ax3, patch_mat, regions, "Patch: Same vs Best-Other")
    plot_intra_inter(ax4, cent_mat,  regions, "Centroid: Same vs Best-Other")

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.6, label="CLIP cosine similarity")
    fig.suptitle("CLIP-Space Cross-Region Codebook Specificity\n(red boxes = native codebook)",
                 fontsize=15, fontweight="bold")
    p = os.path.join(args.out_dir, "clip_combined.png")
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
