"""
text_interpretability.py - Text-to-codebook semantic alignment.

Tests whether human language descriptions of each facial region
align with that region's learned codebook in CLIP space.

Method:
    1. Load the trained CRAFT model and collect per-code CLIP image centroids
       for each region (same as cross_region_clip_eval.py).
    2. Load text phrases from --vocab_json (data.json) for each region.
    3. Embed all text phrases using CLIP's text encoder.
    4. For each (source region text) x (target codebook) pair, compute:
         - mean-max similarity: average over text phrases of the max
           cosine sim to any code centroid in the target codebook.
         - centroid similarity: cosine sim between the mean text embedding
           and the mean code centroid of the target codebook.
    5. Build two NxN matrices and show diagonal dominance.

A diagonal-dominant result proves the codebooks are semantically aligned
with human language descriptions — not just visually similar patches.

Outputs in --out_dir:
    text_codebook_summary.json
    text_codebook_report.md
    text_codebook_heatmap.png
    text_codebook_bar.png
    text_codebook_combined.png

Usage:
    python text_interpretability.py \
        --craft_ckpt  /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
        --parser_ckpt /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
        --data_root   /projectnb/cs585/projects/craft/data/train \
        --vocab_json  /projectnb/cs585/projects/craft/data/data.json \
        --n_images    500 \
        --top_k       16 \
        --out_dir     /projectnb/cs585/projects/craft/text_interpretability_results
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
from matplotlib.colors import LinearSegmentedColormap

from transformers import CLIPModel, CLIPProcessor

from models.vqvae            import build_lq_vqvae
from models.region_aware_vq  import RegionAwareVQ
from models.face_parser       import REGION_NAMES
from data.dataset             import FFHQPairedDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_vocab(vocab_json, prompt_prefix="a close-up photo of "):
    with open(vocab_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab = {}
    for region, phrases in raw.items():
        if isinstance(phrases, list) and phrases:
            vocab[region] = [prompt_prefix + p for p in phrases]
    return vocab


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
# Pass 1: collect code assignments and build CLIP image centroids per code
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
# Pass 2: embed text phrases and build NxN similarity matrices
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_text_embeddings(vocab, clip, proc, device):
    text_embs = {}
    for region, phrases in vocab.items():
        inputs = proc(text=phrases, return_tensors="pt",
                      padding=True, truncation=True).to(device)
        emb = F.normalize(clip.get_text_features(**inputs), dim=-1).float().cpu()
        text_embs[region] = emb
        print(f"  text[{region}]: {emb.shape[0]} phrase embeddings")
    return text_embs


def build_matrices(text_embs, code_embs, regions):
    n = len(regions)
    meanmax_mat  = np.full((n, n), np.nan)
    centroid_mat = np.full((n, n), np.nan)

    for i, ra in enumerate(regions):
        ta = text_embs.get(ra)
        if ta is None or ta.shape[0] == 0: continue
        ta_centroid = F.normalize(ta.mean(dim=0, keepdim=True), dim=-1)

        for j, rb in enumerate(regions):
            cb = code_embs.get(rb)
            if cb is None or cb.shape[0] == 0: continue

            # Mean-max: for each text phrase, find max sim to any code centroid
            sims = ta @ cb.t()                          # (N_phrases, N_codes)
            meanmax_mat[i, j] = sims.max(dim=-1).values.mean().item()

            # Centroid-to-centroid
            rb_centroid = F.normalize(cb.mean(dim=0, keepdim=True), dim=-1)
            centroid_mat[i, j] = (ta_centroid @ rb_centroid.t()).item()

    return meanmax_mat, centroid_mat


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


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_heatmap(out_path, meanmax_mat, centroid_mat, regions):
    cmap = LinearSegmentedColormap.from_list(
        "craft", ["#f7fbff", "#2171b5", "#08306b"]
    )
    vmin = min(np.nanmin(meanmax_mat), np.nanmin(centroid_mat)) - 0.005
    vmax = max(np.nanmax(meanmax_mat), np.nanmax(centroid_mat)) + 0.005

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, mat, title in zip(
        axes,
        [meanmax_mat, centroid_mat],
        ["Text → Codebook (mean-max cosine)",
         "Text Centroid ↔ Codebook Centroid"],
    ):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        n  = len(regions)
        ax.set_xticks(range(n)); ax.set_xticklabels(regions, fontsize=11)
        ax.set_yticks(range(n)); ax.set_yticklabels(regions, fontsize=11)
        ax.set_xlabel("Target Codebook", fontsize=12)
        ax.set_ylabel("Source Text Descriptions", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")

        best_j = [int(np.nanargmax(mat[i])) for i in range(n)]
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if np.isnan(v): continue
                color  = "white" if v < (vmin + vmax) / 2 else "black"
                weight = "bold" if i == j else "normal"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight=weight)
        for i in range(n):
            rect = mpatches.FancyBboxPatch(
                (i - 0.48, i - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                linewidth=2.5, edgecolor="#FF4444", facecolor="none",
            )
            ax.add_patch(rect)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="CLIP cosine similarity")

    fig.suptitle("Text-to-Codebook Semantic Alignment\n"
                 "(Do language descriptions of each region match that region's codebook?)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_bar(out_path, meanmax_mat, centroid_mat, regions):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mat, title in zip(
        axes,
        [meanmax_mat, centroid_mat],
        ["Text → Codebook (mean-max)", "Text Centroid ↔ Codebook Centroid"],
    ):
        n = len(regions)
        native   = [mat[i, i] for i in range(n)]
        best_off = [max((mat[i, j] for j in range(n) if j != i), default=0)
                    for i in range(n)]
        x = np.arange(n); w = 0.35
        b1 = ax.bar(x - w/2, native,   width=w, label="Native codebook",
                    color="#2196F3", edgecolor="white")
        b2 = ax.bar(x + w/2, best_off, width=w, label="Best other codebook",
                    color="#FF7043", edgecolor="white")
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(regions, fontsize=11)
        ax.set_ylabel("CLIP cosine similarity", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(0, max(native + best_off) * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Text Descriptions: Native vs Best Other Codebook",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_combined(out_path, meanmax_mat, centroid_mat, regions):
    cmap = LinearSegmentedColormap.from_list(
        "craft", ["#f7fbff", "#2171b5", "#08306b"]
    )
    vmin = min(np.nanmin(meanmax_mat), np.nanmin(centroid_mat)) - 0.005
    vmax = max(np.nanmax(meanmax_mat), np.nanmax(centroid_mat)) + 0.005
    n    = len(regions)

    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    axes_heat = [fig.add_subplot(gs[0, k]) for k in range(2)]
    axes_bar  = [fig.add_subplot(gs[1, k]) for k in range(2)]

    for ax, mat, title in zip(
        axes_heat,
        [meanmax_mat, centroid_mat],
        ["Text → Codebook (mean-max)", "Text Centroid ↔ Codebook Centroid"],
    ):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(regions, fontsize=10)
        ax.set_yticks(range(n)); ax.set_yticklabels(regions, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if np.isnan(v): continue
                color  = "white" if v < (vmin + vmax) / 2 else "black"
                weight = "bold" if i == j else "normal"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight=weight)
            rect = mpatches.FancyBboxPatch(
                (i - 0.48, i - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                linewidth=2.5, edgecolor="#FF4444", facecolor="none",
            )
            ax.add_patch(rect)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax, mat, title in zip(
        axes_bar,
        [meanmax_mat, centroid_mat],
        ["Text → Codebook: Native vs Best Other",
         "Centroid: Native vs Best Other"],
    ):
        native   = [mat[i, i] for i in range(n)]
        best_off = [max((mat[i, j] for j in range(n) if j != i), default=0)
                    for i in range(n)]
        x = np.arange(n); w = 0.35
        ax.bar(x - w/2, native,   width=w, label="Native", color="#2196F3", edgecolor="white")
        ax.bar(x + w/2, best_off, width=w, label="Best other", color="#FF7043", edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(regions, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(native + best_off) * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Text-to-Codebook Semantic Alignment (red boxes = native codebook)",
                 fontsize=14, fontweight="bold")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft_ckpt",  required=True)
    ap.add_argument("--parser_ckpt", required=True)
    ap.add_argument("--data_root",   required=True)
    ap.add_argument("--vocab_json",  required=True)
    ap.add_argument("--n_images",   type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--top_k",      type=int, default=16)
    ap.add_argument("--context",    type=int, default=64)
    ap.add_argument("--min_hits",   type=int, default=4)
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out_dir",    required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Vocabulary ---
    print(f"Loading vocabulary from {args.vocab_json}")
    vocab = load_vocab(args.vocab_json)
    for r, phrases in vocab.items():
        print(f"  vocab[{r}]: {len(phrases)} phrases")

    # --- Model ---
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
    active  = [r for r in regions if r in vocab]
    print(f"Regions with vocab: {active}")

    # --- Data ---
    ds = FFHQPairedDataset(args.data_root, hq_only=True, masks_folder="")
    if args.n_images < len(ds):
        ds = Subset(ds, range(args.n_images))
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    # --- CLIP ---
    clip = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.clip_model)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                              device=device).view(1, 3, 1, 1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                              device=device).view(1, 3, 1, 1)

    # --- Pass 1: build image code centroids ---
    print("[1/3] Collecting code assignments and building CLIP image centroids...")
    assignments, images_01 = collect_assignments(model, loader, device, active)
    code_embs = build_code_clip_centroids(
        assignments, images_01, clip, clip_mean, clip_std, device,
        top_k=args.top_k, context=args.context, min_hits=args.min_hits,
    )

    # --- Pass 2: embed text phrases ---
    print("[2/3] Embedding text descriptions with CLIP...")
    text_embs = build_text_embeddings(vocab, clip, proc, device)

    # --- Pass 3: build matrices ---
    print("[3/3] Building text-to-codebook similarity matrices...")
    meanmax_mat, centroid_mat = build_matrices(text_embs, code_embs, active)

    fracM, marM = diagonal_dominance(meanmax_mat)
    fracC, marC = diagonal_dominance(centroid_mat)
    print(f"\nMean-max diagonal wins : {fracM*100:.0f}%  margin={marM:+.4f}")
    print(f"Centroid diagonal wins : {fracC*100:.0f}%  margin={marC:+.4f}")

    # --- Save JSON ---
    summary = {
        "regions":      active,
        "meanmax_mat":  meanmax_mat.tolist(),
        "centroid_mat": centroid_mat.tolist(),
        "diag_dominance": {
            "meanmax":  {"win_rate": fracM, "avg_margin": marM},
            "centroid": {"win_rate": fracC, "avg_margin": marC},
        },
    }
    with open(os.path.join(args.out_dir, "text_codebook_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Markdown report ---
    lines = ["# Text-to-Codebook Semantic Alignment\n",
             f"Regions: `{', '.join(active)}`  |  n_images={args.n_images}  |  top_k={args.top_k}\n",
             "## Diagonal Dominance\n",
             f"- Mean-max (text→codebook) : {fracM*100:.0f}% rows win  avg margin={marM:+.4f}",
             f"- Centroid↔centroid        : {fracC*100:.0f}% rows win  avg margin={marC:+.4f}\n"]

    for title, mat in [
        ("Text → Codebook Mean-Max Cosine Similarity", meanmax_mat),
        ("Text Centroid ↔ Codebook Centroid Cosine", centroid_mat),
    ]:
        lines.append(f"## {title}\n")
        lines.append("| src \\ tgt | " + " | ".join(active) + " |")
        lines.append("|" + "---|" * (len(active) + 1))
        for i, r in enumerate(active):
            best_j = int(np.nanargmax(mat[i]))
            cells  = []
            for j in range(len(active)):
                v = mat[i, j]
                s = "n/a" if np.isnan(v) else f"{v:.4f}"
                if j == best_j: s = f"**{s}**"
                cells.append(s)
            lines.append(f"| **{r}** | " + " | ".join(cells) + " |")
        lines.append("")

    with open(os.path.join(args.out_dir, "text_codebook_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --- Plots ---
    plot_heatmap(os.path.join(args.out_dir, "text_codebook_heatmap.png"),
                 meanmax_mat, centroid_mat, active)
    plot_bar(os.path.join(args.out_dir, "text_codebook_bar.png"),
             meanmax_mat, centroid_mat, active)
    plot_combined(os.path.join(args.out_dir, "text_codebook_combined.png"),
                  meanmax_mat, centroid_mat, active)

    print(f"\nDone -> {args.out_dir}")


if __name__ == "__main__":
    main()
