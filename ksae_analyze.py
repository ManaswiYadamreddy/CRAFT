"""
ksae_analyze.py — Stage 3: Neuron analysis and visualization

For each k-SAE neuron:
  1. Compute activation on test set
  2. Find top-N most highly activating images
  3. Measure attribute purity (σ_label equivalent from Revelio)
  4. Identify monosemantic neurons — those that activate for a specific attribute

Then produces:
  - neuron_purity.csv       — per-neuron attribute stats
  - attr_summary.csv        — per-attribute: best neuron, purity, top-k images
  - top_neurons_grid.png    — visual grid of top-9 images for the most
                              attribute-pure neurons (one panel per attribute)
  - attribute_neuron_map.png — heatmap: neurons x attributes activation correlation

The σ_label metric: for a neuron's top-10 activating images, compute the
std of each attribute's binary label across those images. Low σ = monosemantic
(the neuron consistently fires for images with that attribute present/absent).

Usage:
    python ksae_analyze.py \
        --ksae_path checkpoints/ksae_up2a1/ksae_latest.pt \
        --features_path features/ksae_up2a1/features_test.npy \
        --attrs_path    features/ksae_up2a1/attrs_test.npy \
        --filenames_path features/ksae_up2a1/filenames_test.json \
        --image_dir /path/to/LQ_images_test \
        --output_dir results/ksae_analysis \
        --top_n 10 \
        --top_neurons_to_visualize 10 \
        --gpu_ids 0
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Re-use attribute vocabulary
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CELEBA_40 = [
    '5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes',
    'Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair',
    'Blurry','Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin',
    'Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones',
    'Male','Mouth_Slightly_Open','Mustache','Narrow_Eyes','No_Beard',
    'Oval_Face','Pale_Skin','Pointy_Nose','Receding_Hairline','Rosy_Cheeks',
    'Sideburns','Smiling','Straight_Hair','Wavy_Hair','Wearing_Earrings',
    'Wearing_Hat','Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie','Young',
]
PAPER_28_TO_CELEBA40 = {
    'Black Hair':'Black_Hair', 'Blond Hair':'Blond_Hair', 'Blurry':'Blurry',
    'Brown Hair':'Brown_Hair', 'Eyeglasses':'Eyeglasses', 'Gray Hair':'Gray_Hair',
    'Heavy Makeup':'Heavy_Makeup', 'Mouth Slightly Open':'Mouth_Slightly_Open',
    'Mustache':'Mustache', 'Big Eyes':'Narrow_Eyes',
    'No Beard':'No_Beard', 'Receding Hairline':'Receding_Hairline',
    'Sideburns':'Sideburns', 'Smiling':'Smiling', 'Straight Hair':'Straight_Hair',
    'Wearing Earrings':'Wearing_Earrings', 'Wearing Hat':'Wearing_Hat', 'Male':'Male',
    'Wearing Necklace':'Wearing_Necklace', 'Big Nose':'Big_Nose',
    'Wearing Lipstick':'Wearing_Lipstick', 'Young':'Young', 'Wavy Hair':'Wavy_Hair',
    'Big Lips':'Big_Lips', 'Bald':'Bald', 'Bangs':'Bangs',
    'Chubby':'Chubby', 'Double Chin':'Double_Chin',
}
PAPER_28 = list(PAPER_28_TO_CELEBA40.keys())
N_ATTRS  = len(PAPER_28)


# ---------------------------------------------------------------------------
# k-SAE (re-defined here so this script is self-contained)
# ---------------------------------------------------------------------------

class kSAE(nn.Module):
    def __init__(self, d, n, k):
        super().__init__()
        self.d, self.n, self.k = d, n, k
        self.W_enc = nn.Linear(d, n, bias=True)
        self.W_dec = nn.Linear(n, d, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d))

    def encode(self, x):
        x_centered = x - self.b_pre
        pre_acts   = self.W_enc(x_centered)
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        z = torch.zeros_like(pre_acts)
        z.scatter_(1, topk_idx, torch.relu(topk_vals))
        return z

    def forward(self, x):
        z = self.encode(x)
        return self.W_dec(z) + self.b_pre, z


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_activations(model, features, device, batch_size=4096):
    """Run features through k-SAE encoder, return activation matrix (N, n)."""
    model.eval()
    all_z = []
    with torch.no_grad():
        for start in tqdm(range(0, len(features), batch_size), desc="Encoding"):
            x = torch.from_numpy(features[start:start+batch_size]).float().to(device)
            z = model.encode(x)
            all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0)   # (N, n_latents)


def neuron_purity(top_attrs: np.ndarray) -> tuple:
    """
    Given attrs of top-N images for a neuron (N, 28),
    compute the mean σ across attributes (lower = more monosemantic).
    Also returns the most consistent attribute and its std.
    """
    stds = top_attrs.std(axis=0)          # (28,)
    mean_std = stds.mean()
    best_attr_idx = int(stds.argmin())    # attribute with lowest std
    return mean_std, best_attr_idx, stds[best_attr_idx]


def analyze(args):
    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load k-SAE
    print("Loading k-SAE...")
    ckpt = torch.load(args.ksae_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    d, n, k = cfg["d"], cfg["n"], cfg["k"]
    model = kSAE(d=d, n=n, k=k).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"  Loaded k-SAE: d={d}, n={n}, k={k}")

    # Load test set data
    print("Loading test features...")
    features  = np.load(args.features_path)    # (N, d)
    attrs     = np.load(args.attrs_path)        # (N, 28)
    with open(args.filenames_path) as f:
        filenames = json.load(f)               # list of N strings

    N = len(features)
    print(f"  Test set: {N} images, feature dim {d}, n_latents {n}")

    # Compute activations
    Z = compute_activations(model, features, device)   # (N, n_latents)

    # --- Per-neuron analysis ---
    print("\nAnalyzing neuron purity...")

    n_latents  = Z.shape[1]
    top_n      = args.top_n   # default 10, matching Revelio

    records = []
    for neuron_idx in tqdm(range(n_latents), desc="Neurons"):
        acts = Z[:, neuron_idx]

        # Skip dead neurons (never activated)
        if acts.max() == 0:
            records.append({
                "neuron_idx": neuron_idx,
                "is_dead": True,
                "mean_activation": 0.0,
                "mean_sigma": np.nan,
                "best_attr": "",
                "best_attr_idx": -1,
                "best_attr_sigma": np.nan,
                "n_active_images": 0,
            })
            continue

        # Top-N most highly activating images
        top_indices = np.argsort(acts)[-top_n:][::-1]
        top_attrs   = attrs[top_indices]       # (top_n, 28)

        mean_std, best_attr_idx, best_attr_std = neuron_purity(top_attrs)

        records.append({
            "neuron_idx":       neuron_idx,
            "is_dead":          False,
            "mean_activation":  float(acts[acts > 0].mean()) if (acts > 0).any() else 0.0,
            "mean_sigma":       float(mean_std),
            "best_attr":        PAPER_28[best_attr_idx],
            "best_attr_idx":    int(best_attr_idx),
            "best_attr_sigma":  float(best_attr_std),
            "n_active_images":  int((acts > 0).sum()),
        })

    df = pd.DataFrame(records)
    alive_df = df[~df["is_dead"]].copy()
    dead_frac = df["is_dead"].mean()
    print(f"\nDead neurons: {df['is_dead'].sum()}/{n_latents} ({dead_frac:.1%})")
    print(f"Alive neurons: {len(alive_df)}")
    print(f"Mean σ across alive neurons: {alive_df['mean_sigma'].mean():.4f}")

    # Save full purity table
    purity_path = os.path.join(args.output_dir, "neuron_purity.csv")
    df.to_csv(purity_path, index=False)
    print(f"Saved neuron purity → {purity_path}")

    # --- Diagnose dominant neurons before per-attribute summary ---
    print("\nTop-10 most active neurons by n_active_images (collapse check):")
    top_active = alive_df.nlargest(10, "n_active_images")[
        ["neuron_idx", "n_active_images", "mean_activation", "best_attr"]
    ]
    for _, r in top_active.iterrows():
        frac = r["n_active_images"] / N
        print(f"  neuron={int(r['neuron_idx']):<6} "
              f"active_on={int(r['n_active_images'])}/{N} ({frac:.1%})  "
              f"mean_act={r['mean_activation']:.4f}  best_attr={r['best_attr']}")
    n_dominant = (alive_df["n_active_images"] > N * 0.5).sum()
    if n_dominant > 0:
        print(f"\n  WARNING: {n_dominant} neurons activate on >50% of the dataset -- "
              f"these are not monosemantic. Per-attribute analysis will be skewed.\n")

    # --- Per-attribute summary: best neuron per attribute ---
    print("Per-attribute best neuron summary:")
    attr_records = []
    for attr_idx, attr_name in enumerate(PAPER_28):
        attr_stds = []
        for neuron_idx in range(n_latents):
            if df.iloc[neuron_idx]["is_dead"]:
                attr_stds.append(np.nan)
                continue
            acts = Z[:, neuron_idx]
            top_indices = np.argsort(acts)[-top_n:][::-1]
            top_attr_col = attrs[top_indices, attr_idx]
            attr_stds.append(float(top_attr_col.std()))

        attr_stds = np.array(attr_stds)
        valid = ~np.isnan(attr_stds)
        if valid.any():
            best_neuron = int(np.nanargmin(attr_stds))
            best_sigma  = float(attr_stds[best_neuron])
            acts = Z[:, best_neuron]
            top_indices = np.argsort(acts)[-top_n:][::-1]
            top_mean = float(attrs[top_indices, attr_idx].mean())
            n_active = int((acts > 0).sum())
            active_frac = n_active / N
        else:
            best_neuron, best_sigma, top_mean, n_active, active_frac = -1, np.nan, np.nan, 0, 0.0

        attr_records.append({
            "attribute":        attr_name,
            "best_neuron":      best_neuron,
            "best_sigma":       best_sigma,
            "top10_mean_label": top_mean,
            "fires_for":        "present" if top_mean >= 0.5 else "absent",
            "n_active_images":  n_active,
            "active_frac":      active_frac,
        })
        flag = " *** DOMINANT (>50% dataset)" if active_frac > 0.5 else ""
        print(f"  {attr_name:<25} neuron={best_neuron:<6} sigma={best_sigma:.3f}  "
              f"top10_mean={top_mean:.2f}  active={active_frac:.1%}{flag}")

    attr_df = pd.DataFrame(attr_records)
    attr_df_path = os.path.join(args.output_dir, "attr_summary.csv")
    attr_df.to_csv(attr_df_path, index=False)
    print(f"\nSaved attribute summary → {attr_df_path}")

    # --- Visualization: top-9 images per attribute's best neuron ---
    if args.image_dir:
        print("\nGenerating top-image grids...")
        _make_top_image_grids(
            Z, attrs, filenames, attr_df, args.image_dir,
            args.output_dir, args.top_neurons_to_visualize, top_n=9
        )

    # --- Neuron x Attribute activation correlation heatmap ---
    print("Generating neuron-attribute correlation heatmap...")
    _make_neuron_attr_heatmap(Z, attrs, alive_df, args.output_dir)

    # --- σ_label summary bar chart (Revelio Table 1 equivalent) ---
    _make_sigma_bar(attr_df, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}")


def _load_image(image_dir, filename, size=128):
    """Load image as numpy array, try common suffix variants."""
    stem, ext = os.path.splitext(filename)
    candidates = [
        filename,
        f"{stem}_LQ{ext}",
        f"{stem.replace('_LQ','')}{ext}",
    ]
    for name in candidates:
        path = os.path.join(image_dir, name)
        if os.path.exists(path):
            return np.array(Image.open(path).convert("RGB").resize((size, size)))
    return np.zeros((size, size, 3), dtype=np.uint8)  # blank if not found


def _make_top_image_grids(Z, attrs, filenames, attr_df, image_dir,
                           output_dir, n_attrs_to_show, top_n=9):
    """
    For each of the top n_attrs_to_show attributes (by lowest σ),
    create a 3x3 grid of top activating images for that attribute's best neuron.
    """
    # Sort attributes by best_sigma
    sorted_attr_df = attr_df.dropna(subset=["best_sigma"]).sort_values("best_sigma")
    attrs_to_show  = sorted_attr_df.head(n_attrs_to_show)

    import math
    cols  = 3
    rows  = 3
    n_panels = len(attrs_to_show)
    panel_w  = 4
    fig_w    = panel_w * min(n_panels, 4)
    fig_h    = panel_w * math.ceil(n_panels / 4)
    n_cols_outer = min(n_panels, 4)
    n_rows_outer = math.ceil(n_panels / n_cols_outer)

    fig = plt.figure(figsize=(panel_w * n_cols_outer * cols / 3,
                               panel_w * n_rows_outer))
    outer_gs = gridspec.GridSpec(n_rows_outer, n_cols_outer, figure=fig,
                                  hspace=0.4, wspace=0.3)

    for panel_idx, (_, row_data) in enumerate(attrs_to_show.iterrows()):
        attr_name   = row_data["attribute"]
        neuron_idx  = int(row_data["best_neuron"])
        sigma       = row_data["best_sigma"]
        top_mean    = row_data["top10_mean_label"]
        fires_for   = row_data["fires_for"]

        acts = Z[:, neuron_idx]
        top_indices = np.argsort(acts)[-top_n:][::-1]

        r_out = panel_idx // n_cols_outer
        c_out = panel_idx %  n_cols_outer
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            rows, cols, subplot_spec=outer_gs[r_out, c_out],
            hspace=0.05, wspace=0.05
        )

        for img_i, data_idx in enumerate(top_indices[:rows*cols]):
            r = img_i // cols
            c = img_i %  cols
            ax = fig.add_subplot(inner_gs[r, c])
            img_arr = _load_image(image_dir, filenames[data_idx])
            ax.imshow(img_arr)
            ax.axis("off")
            # Show activation value
            ax.set_title(f"{acts[data_idx]:.2f}", fontsize=5, pad=1)

        # Panel title
        color = "green" if fires_for == "present" else "red"
        title = (f"Neuron {neuron_idx}\n{attr_name}\n"
                 f"σ={sigma:.3f}  fires_for={fires_for}\n"
                 f"top-{top_n} mean label={top_mean:.2f}")
        # Add title above inner grid
        ax_title = fig.add_subplot(outer_gs[r_out, c_out])
        ax_title.set_title(title, fontsize=7, color=color, fontweight="bold", pad=2)
        ax_title.axis("off")

    out_path = os.path.join(output_dir, "top_neurons_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved top-image grid → {out_path}")

    # Also save individual grids per attribute for close-up inspection
    grid_dir = os.path.join(output_dir, "per_attr_grids")
    os.makedirs(grid_dir, exist_ok=True)
    for _, row_data in attrs_to_show.iterrows():
        attr_name  = row_data["attribute"].replace(" ", "_")
        neuron_idx = int(row_data["best_neuron"])
        sigma      = row_data["best_sigma"]
        fires_for  = row_data["fires_for"]

        acts = Z[:, neuron_idx]
        top_indices = np.argsort(acts)[-top_n:][::-1]

        fig2, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        for img_i, data_idx in enumerate(top_indices[:rows*cols]):
            r, c = img_i // cols, img_i % cols
            ax   = axes[r][c]
            img_arr = _load_image(image_dir, filenames[data_idx], size=256)
            ax.imshow(img_arr)
            ax.set_title(f"act={acts[data_idx]:.2f}", fontsize=7)
            ax.axis("off")

        fig2.suptitle(
            f"Neuron {neuron_idx}  |  {row_data['attribute']}\n"
            f"σ={sigma:.3f}  fires_for='{fires_for}'  "
            f"top-{top_n} mean label={row_data['top10_mean_label']:.2f}",
            fontsize=9, fontweight="bold"
        )
        fig2.tight_layout()
        fig2.savefig(os.path.join(grid_dir, f"{attr_name}_neuron{neuron_idx}.png"),
                     dpi=120, bbox_inches="tight")
        plt.close(fig2)

    print(f"  Per-attribute grids saved → {grid_dir}/")


def _make_neuron_attr_heatmap(Z, attrs, alive_df, output_dir):
    """
    Heatmap: for each attribute, compute the point-biserial correlation between
    neuron activations and attribute binary labels.
    Rows = top-50 most active neurons, Cols = 28 attributes.
    """
    from scipy.stats import pointbiserialr

    # Select top-50 neurons by mean activation
    top_neurons = alive_df.nlargest(50, "mean_activation")["neuron_idx"].values.astype(int)

    corr_matrix = np.zeros((len(top_neurons), N_ATTRS))
    for i, neuron_idx in enumerate(tqdm(top_neurons, desc="Correlation")):
        acts = Z[:, neuron_idx]
        for j in range(N_ATTRS):
            labels = attrs[:, j]
            if labels.std() == 0:
                corr_matrix[i, j] = 0.0
                continue
            try:
                r, _ = pointbiserialr(labels, acts)
                corr_matrix[i, j] = r if not np.isnan(r) else 0.0
            except Exception:
                corr_matrix[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(14, max(8, len(top_neurons) * 0.25)))
    im = ax.imshow(corr_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(N_ATTRS))
    ax.set_xticklabels(PAPER_28, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top_neurons)))
    ax.set_yticklabels([f"N{i}" for i in top_neurons], fontsize=6)
    ax.set_xlabel("CelebA Attribute", fontsize=10)
    ax.set_ylabel("k-SAE Neuron (top-50 by mean activation)", fontsize=10)
    ax.set_title("Point-biserial correlation: neuron activations × attribute labels\n"
                 "(Red = fires for attribute present, Blue = fires for attribute absent)",
                 fontsize=10)
    plt.colorbar(im, ax=ax, label="Correlation (r)")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "attribute_neuron_map.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved neuron-attribute heatmap → {heatmap_path}")


def _make_sigma_bar(attr_df, output_dir):
    """Bar chart of best σ per attribute — equivalent to Revelio Table 1."""
    valid = attr_df.dropna(subset=["best_sigma"]).sort_values("best_sigma")

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["green" if f == "present" else "tomato"
              for f in valid["fires_for"]]
    bars = ax.bar(range(len(valid)), valid["best_sigma"], color=colors, alpha=0.8)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(valid["attribute"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("σ of attribute label in top-10 activating images", fontsize=9)
    ax.set_title("Attribute purity per best k-SAE neuron\n"
                 "(lower σ = more monosemantic)\n"
                 "Green = neuron fires when attribute PRESENT  |  Red = fires when ABSENT",
                 fontsize=10)
    ax.axhline(0.3, color="gray", linestyle="--", alpha=0.5, label="σ=0.3 threshold")
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.3, label="σ=0.5 (random)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, valid["best_sigma"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", fontsize=6)

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "sigma_per_attribute.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved σ bar chart → {bar_path}")


if __name__ == "__main__":
    import math  # needed inside functions
    parser = argparse.ArgumentParser()
    parser.add_argument("--ksae_path",      required=True, help="Path to ksae_latest.pt")
    parser.add_argument("--features_path",  required=True, help="Path to features_test.npy")
    parser.add_argument("--attrs_path",     required=True, help="Path to attrs_test.npy")
    parser.add_argument("--filenames_path", required=True, help="Path to filenames_test.json")
    parser.add_argument("--image_dir",      default=None,  help="Dir of test LQ images for visualization")
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--top_n",          type=int, default=10, help="Top-N images per neuron for purity")
    parser.add_argument("--top_neurons_to_visualize", type=int, default=10,
                        help="Number of most-pure neurons to visualize")
    parser.add_argument("--gpu_ids",        nargs="+", type=int, default=[0])
    args = parser.parse_args()

    import math
    analyze(args)