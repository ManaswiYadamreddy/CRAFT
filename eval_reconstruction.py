"""
eval_reconstruction.py — Phase C reconstruction evaluation for CRAFT and OSDFace.

Loads the Phase C checkpoints for both models, runs inference on a held-out
subset of the training data (or a separate val split), computes quality metrics,
and saves a rich set of visual comparisons so you can see exactly where each
model succeeds or fails before committing to Stage 2.

Metrics (computed in [0,1] image range):
    Global  : PSNR, SSIM, L1, perceptual distance (VGG19)
    Regional: per-region L1 for CRAFT  (eyes / skin / hair / lips / bg)
              measured against the same HQ ground truth, masked by the
              face-parser segmentation of the HQ image.

Outputs (saved to --out_dir):
    visuals/
        sample_000.png   — 7-panel figure per image
                           [LQ input | HQ GT | HQ rec | CRAFT rec | OSDFace rec
                            | CRAFT Δ×5 | OSDFace Δ×5]
        region_000.png   — CRAFT region overlay + per-region L1 heatmap
    summary/
        metrics.json     — all per-image and aggregate numbers
        summary.png      — bar/violin chart of global metrics
        region_l1.png    — per-region L1 bar chart (CRAFT)

Usage:
    python eval_reconstruction.py \\
        --hq_ckpt   /path/to/checkpoints/phase_a/final.pt \\
        --craft_ckpt /path/to/checkpoints/phase_c/final.pt \\
        --osdface_ckpt /path/to/checkpoints_osdface/phase_c/final.pt \\
        --parser_ckpt /path/to/pretrained/79999_iter.pth \\
        --data_root /path/to/data/train \\
        --n_images 128 \\
        --n_visuals 16 \\
        --out_dir eval_results/phase_c
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

# ── project imports ────────────────────────────────────────────────────────
from models.vqvae import build_hq_vqvae, build_lq_vqvae
from models.region_aware_vq import RegionAwareVQ
from models.face_parser import FaceParser, REGION_NAMES
from losses.losses import VGGPerceptualLoss
from data.dataset import FFHQPairedDataset


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

REGION_COLORS = {          # RGBA, matches diagnose_masks.py
    "eyes": (1.0, 0.2, 0.2, 0.85),
    "skin": (1.0, 0.85, 0.7, 0.85),
    "hair": (0.1, 0.8, 0.8, 0.85),
    "lips": (0.9, 0.3, 0.8, 0.85),
    "bg":   (0.2, 0.2, 0.9, 0.85),
}

REGION_CMAP = ListedColormap([REGION_COLORS[n] for n in REGION_NAMES])


# ═══════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════

def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Peak signal-to-noise ratio in dB.
    pred / target expected in [0, 1], shape (B, C, H, W) or (C, H, W).
    """
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse < eps:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean SSIM over a batch, computed in numpy via skimage.
    Falls back to PSNR-based approximation if skimage is unavailable.
    pred / target in [0, 1], shape (B, C, H, W).
    """
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        pred_np = pred.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
        tgt_np  = target.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
        scores = [
            sk_ssim(pred_np[i], tgt_np[i], data_range=1.0, channel_axis=-1)
            for i in range(pred_np.shape[0])
        ]
        return float(np.mean(scores))
    except ImportError:
        # Rough approximation when skimage not installed
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu1 = pred.mean(); mu2 = target.mean()
        s1  = pred.var();  s2  = target.var()
        s12 = ((pred - mu1) * (target - mu2)).mean()
        return float(
            ((2*mu1*mu2 + c1) * (2*s12 + c2)) /
            ((mu1**2 + mu2**2 + c1) * (s1 + s2 + c2))
        )


def l1_per_region(
    pred: torch.Tensor,
    target: torch.Tensor,
    region_map: torch.Tensor,   # (B, H, W) int64 region indices
) -> dict:
    """
    Compute mean-absolute-error for each face region independently.

    Args:
        pred:       (B, C, H, W) in [0, 1]
        target:     (B, C, H, W) in [0, 1]
        region_map: (B, H, W) int64 region indices (REGION_NAMES order)
                    at the same spatial resolution as pred/target.

    Returns:
        dict mapping region_name → mean L1 over masked pixels (float).
        Regions with zero pixels get value = None.
    """
    # Upsample region_map to match pred spatial size if needed
    B, C, H, W = pred.shape
    if region_map.shape[-2:] != (H, W):
        region_map = F.interpolate(
            region_map.unsqueeze(1).float(),
            size=(H, W), mode="nearest"
        ).squeeze(1).long()

    abs_err = (pred.float() - target.float()).abs()  # (B, C, H, W)
    result = {}
    for ridx, name in enumerate(REGION_NAMES):
        mask = (region_map == ridx)                   # (B, H, W)
        n_pix = mask.sum().item()
        if n_pix == 0:
            result[name] = None
        else:
            mask_bc = mask.unsqueeze(1).expand_as(abs_err)
            result[name] = float(abs_err[mask_bc].mean().item())
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_hq_model(ckpt_path: str, device, embed_dim=512, n_codes=1024):
    """Load HQ VQVAE (Phase A checkpoint)."""
    model = build_hq_vqvae(n_codes=n_codes, embed_dim=embed_dim)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_craft_model(ckpt_path: str, device, parser_ckpt: str,
                     embed_dim=512, rq_levels=3):
    """Load CRAFT LQ VQVAE (Phase C or Phase D checkpoint)."""
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Phase C saves under 'model' key; handle both 'model' and 'lq_model'
    key = "model" if "model" in ckpt else "lq_model"
    state = ckpt[key]

    # Auto-detect Phase D magnitude head from the checkpoint.
    has_mag_head = any("magnitude_head" in k for k in state.keys())
    if has_mag_head:
        print("  Detected magnitude_head in checkpoint → Phase D model")

    ravq  = RegionAwareVQ(
        e_dim=embed_dim,
        n_levels=rq_levels,
        parser_ckpt=parser_ckpt,
        use_magnitude_head=has_mag_head,
    )
    model = build_lq_vqvae(ravq, embed_dim=embed_dim)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_osdface_model(ckpt_path: str, device, embed_dim=512, lq_n_codes=1024):
    """Load OSDFace LQ VQVAE (Phase C checkpoint, flat GlobalVQ)."""
    model = build_hq_vqvae(n_codes=lq_n_codes, embed_dim=embed_dim)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    key   = "model" if "model" in ckpt else "lq_model"
    model.load_state_dict(ckpt[key])
    return model.to(device).eval()


# ═══════════════════════════════════════════════════════════════════════════
# Inference helpers
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def reconstruct(model, x_11: torch.Tensor, x_01: torch.Tensor,
                masks: dict = None):
    """
    Run full encode → quantize → decode.

    Args:
        x_11:  (B, 3, H, W) in [-1, 1]  (encoder input)
        x_01:  (B, 3, H, W) in [0, 1]   (face parser input, CRAFT only)
        masks: optional precomputed region masks dict

    Returns:
        x_rec_01: (B, 3, H, W) in [0, 1]  clipped reconstruction
    """
    x_rec, *_ = model(x_11, images_01=x_01, masks=masks)
    # Decoder output is in [-1, 1]; convert to [0, 1] for metrics
    return ((x_rec.clamp(-1, 1) + 1) / 2).float()


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_np(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) tensor in [0,1] → (H,W,C) uint8 numpy."""
    return (t.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)


def _diff_np(pred: torch.Tensor, gt: torch.Tensor, amp: float = 5.0) -> np.ndarray:
    """Amplified absolute-error map → (H,W,C) uint8."""
    err = (pred - gt).abs().mean(dim=0, keepdim=True) * amp
    err = err.expand(3, -1, -1).clamp(0, 1)
    return _to_np(err)


def save_sample_figure(
    idx: int,
    lq_01: torch.Tensor,
    hq_01: torch.Tensor,
    hq_rec_01: torch.Tensor,
    craft_rec_01: torch.Tensor,
    osdface_rec_01: torch.Tensor,
    metrics: dict,
    out_path: str,
):
    """
    7-panel figure:
      LQ input | HQ GT | HQ rec | CRAFT rec | OSDFace rec
      | CRAFT Δ×5 | OSDFace Δ×5

    metrics: {craft_psnr, craft_ssim, osdface_psnr, osdface_ssim, hq_psnr}
    """
    panels = [
        (_to_np(lq_01),           "LQ input"),
        (_to_np(hq_01),           "HQ GT"),
        (_to_np(hq_rec_01),       f"HQ rec\nPSNR {metrics.get('hq_psnr', 0):.1f}"),
        (_to_np(craft_rec_01),    f"CRAFT\nPSNR {metrics.get('craft_psnr', 0):.1f} "
                                  f"SSIM {metrics.get('craft_ssim', 0):.3f}"),
        (_to_np(osdface_rec_01),  f"OSDFace\nPSNR {metrics.get('osdface_psnr', 0):.1f} "
                                  f"SSIM {metrics.get('osdface_ssim', 0):.3f}"),
        (_diff_np(craft_rec_01,   hq_01), "CRAFT Δ×5"),
        (_diff_np(osdface_rec_01, hq_01), "OSDFace Δ×5"),
    ]

    fig, axes = plt.subplots(1, 7, figsize=(28, 4.5))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")
    fig.suptitle(f"Sample {idx:04d}", fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_region_figure(
    idx: int,
    hq_01: torch.Tensor,
    craft_rec_01: torch.Tensor,
    region_map: torch.Tensor,       # (H, W) int64 on CPU
    region_l1_craft: dict,
    region_l1_osdface: dict,
    out_path: str,
):
    """
    3-panel region figure:
      HQ GT + region overlay  |  CRAFT L1 heatmap  |  per-region L1 bar chart
    """
    H, W = hq_01.shape[-2:]
    # upscale region_map to image resolution for display
    rmap_display = F.interpolate(
        region_map.unsqueeze(0).unsqueeze(0).float(),
        size=(H, W), mode="nearest"
    ).squeeze().long().numpy()

    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.4], wspace=0.3)

    # Panel 1: HQ + region overlay
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(_to_np(hq_01))
    ax0.imshow(rmap_display, cmap=REGION_CMAP, alpha=0.35,
               vmin=0, vmax=len(REGION_NAMES) - 1, interpolation="nearest")
    ax0.set_title("HQ GT + region map", fontsize=9)
    ax0.axis("off")

    # Panel 2: CRAFT per-pixel L1 heatmap (mean across channels)
    ax1 = fig.add_subplot(gs[1])
    l1_map = (craft_rec_01 - hq_01).abs().mean(dim=0).cpu().numpy()
    im = ax1.imshow(l1_map, cmap="hot", vmin=0, vmax=0.15)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title("CRAFT L1 error map", fontsize=9)
    ax1.axis("off")

    # Panel 3: Per-region L1 bar chart (CRAFT vs OSDFace)
    ax2 = fig.add_subplot(gs[2])
    names   = REGION_NAMES
    craft_v = [region_l1_craft.get(n) or 0   for n in names]
    osd_v   = [region_l1_osdface.get(n) or 0 for n in names]
    x       = np.arange(len(names))
    w       = 0.35
    ax2.bar(x - w/2, craft_v,  w, label="CRAFT",   color="#2196F3", alpha=0.85)
    ax2.bar(x + w/2, osd_v,    w, label="OSDFace", color="#FF5722", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("Mean L1 error"); ax2.set_title("Per-region L1", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, max(max(craft_v), max(osd_v)) * 1.35 + 1e-6)

    fig.suptitle(f"Sample {idx:04d} — region analysis", fontsize=9)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_summary_figure(all_metrics: list, out_dir: str):
    """Bar chart comparing CRAFT vs OSDFace on global metrics."""
    craft_psnr = [m["craft_psnr"] for m in all_metrics if m.get("craft_psnr") is not None]
    osd_psnr   = [m["osdface_psnr"] for m in all_metrics if m.get("osdface_psnr") is not None]
    craft_ssim = [m["craft_ssim"] for m in all_metrics if m.get("craft_ssim") is not None]
    osd_ssim   = [m["osdface_ssim"] for m in all_metrics if m.get("osdface_ssim") is not None]
    craft_l1   = [m["craft_l1"] for m in all_metrics if m.get("craft_l1") is not None]
    osd_l1     = [m["osdface_l1"] for m in all_metrics if m.get("osdface_l1") is not None]
    craft_per  = [m["craft_perceptual"] for m in all_metrics if m.get("craft_perceptual") is not None]
    osd_per    = [m["osdface_perceptual"] for m in all_metrics if m.get("osdface_perceptual") is not None]

    def _mean(lst): return float(np.mean(lst)) if lst else 0.0
    def _std(lst):  return float(np.std(lst))  if lst else 0.0

    metrics_to_plot = {
        "PSNR ↑":       ([_mean(craft_psnr), _mean(osd_psnr)],
                          [_std(craft_psnr),  _std(osd_psnr)]),
        "SSIM ↑":       ([_mean(craft_ssim), _mean(osd_ssim)],
                          [_std(craft_ssim),  _std(osd_ssim)]),
        "L1 ↓ (×100)":  ([_mean(craft_l1)*100, _mean(osd_l1)*100],
                          [_std(craft_l1)*100,  _std(osd_l1)*100]),
        "Perceptual ↓": ([_mean(craft_per),  _mean(osd_per)],
                          [_std(craft_per),   _std(osd_per)]),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (metric, (means, stds)) in zip(axes, metrics_to_plot.items()):
        colors = ["#2196F3", "#FF5722"]
        bars = ax.bar(["CRAFT", "OSDFace"], means, color=colors,
                      yerr=stds, capsize=6, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_ylabel("")
        # Annotate bar tops with value
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[bars.index(bar)] * 0.05,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Phase C Reconstruction Quality — CRAFT vs OSDFace", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.join(out_dir, 'summary.png')}")


def save_region_summary(all_region_craft: list, all_region_osdface: list, out_dir: str):
    """Per-region L1 summary bar chart aggregated across all images."""
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(REGION_NAMES))
    w = 0.35

    def _safe_mean(lst_of_dicts, key):
        vals = [d[key] for d in lst_of_dicts if d.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    def _safe_std(lst_of_dicts, key):
        vals = [d[key] for d in lst_of_dicts if d.get(key) is not None]
        return float(np.std(vals)) if vals else 0.0

    craft_means = [_safe_mean(all_region_craft,   n) for n in REGION_NAMES]
    craft_stds  = [_safe_std(all_region_craft,    n) for n in REGION_NAMES]
    osd_means   = [_safe_mean(all_region_osdface, n) for n in REGION_NAMES]
    osd_stds    = [_safe_std(all_region_osdface,  n) for n in REGION_NAMES]

    ax.bar(x - w/2, craft_means, w, yerr=craft_stds, capsize=5,
           label="CRAFT", color="#2196F3", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, osd_means, w, yerr=osd_stds, capsize=5,
           label="OSDFace", color="#FF5722", alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(REGION_NAMES, fontsize=11)
    ax.set_ylabel("Mean L1 error", fontsize=11)
    ax.set_title("Per-region reconstruction L1 — CRAFT vs OSDFace", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "region_l1.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.join(out_dir, 'region_l1.png')}")


# ═══════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Phase C reconstruction evaluation — CRAFT vs OSDFace"
    )
    # Checkpoints
    ap.add_argument("--hq_ckpt",      type=str, required=True,
                    help="Phase A HQ checkpoint (final.pt)")
    ap.add_argument("--craft_ckpt",   type=str, default="",
                    help="CRAFT Phase C LQ checkpoint (final.pt or latest.pt). "
                         "Leave empty to skip CRAFT.")
    ap.add_argument("--osdface_ckpt", type=str, default="",
                    help="OSDFace Phase C LQ checkpoint (final.pt or latest.pt). "
                         "Leave empty to skip OSDFace.")

    # Architecture (must match what was used during training)
    ap.add_argument("--embed_dim",    type=int, default=512)
    ap.add_argument("--hq_n_codes",   type=int, default=1024)
    ap.add_argument("--lq_n_codes",   type=int, default=1024,
                    help="OSDFace LQ codebook size (flat GlobalVQ)")
    ap.add_argument("--rq_levels",    type=int, default=3,
                    help="Residual VQ levels (CRAFT)")

    # Face parser
    ap.add_argument("--parser_ckpt",  type=str,
                    default="/projectnb/cs585/projects/craft/pretrained/79999_iter.pth",
                    help="BiSeNet checkpoint for region masks")

    # Data
    ap.add_argument("--data_root",    type=str, required=True,
                    help="Root directory containing images512x512/ and LQ_images_512x512/")
    ap.add_argument("--hq_folder",    type=str, default="images512x512")
    ap.add_argument("--lq_folder",    type=str, default="LQ_images_512x512")
    ap.add_argument("--n_images",     type=int, default=128,
                    help="Number of images to evaluate (deterministic first-N)")
    ap.add_argument("--batch_size",   type=int, default=8)

    # Output
    ap.add_argument("--n_visuals",    type=int, default=16,
                    help="Number of individual sample figures to save")
    ap.add_argument("--out_dir",      type=str, default="eval_results/phase_c")
    ap.add_argument("--device",       type=str, default="cuda")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── output directories ──────────────────────────────────────────────
    vis_dir     = os.path.join(args.out_dir, "visuals")
    region_dir  = os.path.join(args.out_dir, "region_visuals")
    summary_dir = os.path.join(args.out_dir, "summary")
    for d in [vis_dir, region_dir, summary_dir]:
        os.makedirs(d, exist_ok=True)

    # ── models ─────────────────────────────────────────────────────────
    print("\nLoading models...")
    hq_model = load_hq_model(
        args.hq_ckpt, device, embed_dim=args.embed_dim, n_codes=args.hq_n_codes
    )
    print(f"  HQ model loaded from {args.hq_ckpt}")

    craft_model = None
    if args.craft_ckpt:
        craft_model = load_craft_model(
            args.craft_ckpt, device,
            parser_ckpt=args.parser_ckpt,
            embed_dim=args.embed_dim, rq_levels=args.rq_levels,
        )
        print(f"  CRAFT LQ model loaded from {args.craft_ckpt}")
    else:
        print("  CRAFT skipped (no --craft_ckpt)")

    osdface_model = None
    if args.osdface_ckpt:
        osdface_model = load_osdface_model(
            args.osdface_ckpt, device,
            embed_dim=args.embed_dim, lq_n_codes=args.lq_n_codes,
        )
        print(f"  OSDFace LQ model loaded from {args.osdface_ckpt}")
    else:
        print("  OSDFace skipped (no --osdface_ckpt)")

    # ── face parser for per-region metrics ─────────────────────────────
    # If CRAFT is loaded, reuse its parser. Otherwise load a standalone one.
    if craft_model is not None:
        face_parser = craft_model.quantizer.face_parser
    else:
        print(f"  Loading standalone face parser from {args.parser_ckpt}")
        face_parser = FaceParser(checkpoint_path=args.parser_ckpt).to(device)
    face_parser.eval()

    # ── perceptual loss ─────────────────────────────────────────────────
    perceptual_fn = VGGPerceptualLoss(pretrained=True).to(device).eval()

    # ── dataset ─────────────────────────────────────────────────────────
    print(f"\nLoading dataset from {args.data_root} (first {args.n_images} images)...")
    dataset = FFHQPairedDataset(
        data_root=args.data_root,
        hq_folder=args.hq_folder,
        lq_folder=args.lq_folder,
        masks_folder="",           # don't need precomputed masks for eval
    )
    # Take first n_images (deterministic, reproducible)
    n = min(args.n_images, len(dataset))
    indices = list(range(n))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(str(device) != "cpu"),
    )
    print(f"  {n} images, {len(loader)} batches")

    # ── evaluation loop ─────────────────────────────────────────────────
    all_metrics        = []
    all_region_craft   = []
    all_region_osdface = []
    vis_saved          = 0
    img_idx            = 0

    print("\nRunning evaluation...")
    for batch_num, batch in enumerate(loader):
        hq_11 = batch["hq"].to(device)      # [-1, 1]
        lq_11 = batch["lq"].to(device)
        hq_01 = batch["hq_01"].to(device)   # [0, 1]
        lq_01 = batch["lq_01"].to(device)

        B = hq_11.shape[0]

        with torch.no_grad():
            # ── reconstructions ────────────────────────────────────────
            hq_rec_01     = reconstruct(hq_model,     hq_11, hq_01)
            craft_rec_01  = reconstruct(craft_model,  lq_11, lq_01) if craft_model  else None
            osd_rec_01    = reconstruct(osdface_model, lq_11, lq_01) if osdface_model else None

            # ── face parser region map at image resolution for metrics ─
            # We parse the HQ image so the GT region boundaries are correct.
            region_indices_hw = face_parser.get_region_indices(
                hq_01, target_h=hq_01.shape[-2], target_w=hq_01.shape[-1]
            )  # (B, H, W) – full resolution

            # ── per-image metrics ──────────────────────────────────────
            for i in range(B):
                m = {"sample_idx": img_idx + i}

                # HQ self-reconstruction sanity check
                m["hq_psnr"] = psnr(hq_rec_01[i:i+1], hq_01[i:i+1])

                # CRAFT
                if craft_rec_01 is not None:
                    m["craft_psnr"]       = psnr(craft_rec_01[i:i+1], hq_01[i:i+1])
                    m["craft_ssim"]       = ssim_batch(craft_rec_01[i:i+1], hq_01[i:i+1])
                    m["craft_l1"]         = float(F.l1_loss(craft_rec_01[i], hq_01[i]).item())
                    m["craft_perceptual"] = float(
                        perceptual_fn(
                            craft_rec_01[i:i+1],   # VGGPerceptualLoss handles [0,1]
                            hq_01[i:i+1],
                        ).item()
                    )
                    region_l1_craft = l1_per_region(
                        craft_rec_01[i:i+1], hq_01[i:i+1], region_indices_hw[i:i+1]
                    )
                    m["craft_region_l1"] = region_l1_craft
                    all_region_craft.append(region_l1_craft)

                # OSDFace
                if osd_rec_01 is not None:
                    m["osdface_psnr"]       = psnr(osd_rec_01[i:i+1], hq_01[i:i+1])
                    m["osdface_ssim"]       = ssim_batch(osd_rec_01[i:i+1], hq_01[i:i+1])
                    m["osdface_l1"]         = float(F.l1_loss(osd_rec_01[i], hq_01[i]).item())
                    m["osdface_perceptual"] = float(
                        perceptual_fn(
                            osd_rec_01[i:i+1],
                            hq_01[i:i+1],
                        ).item()
                    )
                    region_l1_osdface = l1_per_region(
                        osd_rec_01[i:i+1], hq_01[i:i+1], region_indices_hw[i:i+1]
                    )
                    m["osdface_region_l1"] = region_l1_osdface
                    all_region_osdface.append(region_l1_osdface)
                else:
                    region_l1_osdface = {n: 0.0 for n in REGION_NAMES}

                all_metrics.append(m)

                # ── per-sample visuals (first n_visuals images) ────────
                if vis_saved < args.n_visuals:
                    sample_path = os.path.join(vis_dir, f"sample_{img_idx + i:04d}.png")
                    save_sample_figure(
                        idx=img_idx + i,
                        lq_01=lq_01[i].cpu(),
                        hq_01=hq_01[i].cpu(),
                        hq_rec_01=hq_rec_01[i].cpu(),
                        craft_rec_01=(craft_rec_01[i].cpu() if craft_rec_01 is not None
                                      else torch.zeros_like(hq_01[i].cpu())),
                        osdface_rec_01=(osd_rec_01[i].cpu() if osd_rec_01 is not None
                                        else torch.zeros_like(hq_01[i].cpu())),
                        metrics=m,
                        out_path=sample_path,
                    )

                    region_path = os.path.join(region_dir, f"region_{img_idx + i:04d}.png")
                    save_region_figure(
                        idx=img_idx + i,
                        hq_01=hq_01[i].cpu(),
                        craft_rec_01=(craft_rec_01[i].cpu() if craft_rec_01 is not None
                                      else torch.zeros_like(hq_01[i].cpu())),
                        region_map=region_indices_hw[i].cpu(),
                        region_l1_craft=m.get("craft_region_l1", {}),
                        region_l1_osdface=m.get("osdface_region_l1", region_l1_osdface),
                        out_path=region_path,
                    )
                    vis_saved += 1

        img_idx += B
        print(f"  [{img_idx:4d}/{n}]  "
              + (f"CRAFT PSNR {m.get('craft_psnr', 0):.2f} dB  " if craft_rec_01 is not None else "")
              + (f"OSD PSNR {m.get('osdface_psnr', 0):.2f} dB" if osd_rec_01 is not None else ""))

    # ── aggregate summary ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("AGGREGATE RESULTS")
    print("=" * 65)

    def agg(key):
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    summary = {}
    for model_tag in ("hq", "craft", "osdface"):
        for metric in ("psnr", "ssim", "l1", "perceptual"):
            k = f"{model_tag}_{metric}"
            mu, sd = agg(k)
            if mu is not None:
                summary[k] = {"mean": round(mu, 4), "std": round(sd, 4)}
                print(f"  {k:<30s}  {mu:8.4f}  ±{sd:.4f}")

    # Per-region (CRAFT)
    print("\n  CRAFT per-region L1 (mean ± std):")
    region_summary_craft   = {}
    region_summary_osdface = {}
    for name in REGION_NAMES:
        craft_vals = [d[name] for d in all_region_craft   if d.get(name) is not None]
        osd_vals   = [d[name] for d in all_region_osdface if d.get(name) is not None]
        if craft_vals:
            mu, sd = float(np.mean(craft_vals)), float(np.std(craft_vals))
            region_summary_craft[name] = {"mean": round(mu, 5), "std": round(sd, 5)}
            print(f"    CRAFT  [{name:<5s}]  {mu:.5f} ± {sd:.5f}")
        if osd_vals:
            mu, sd = float(np.mean(osd_vals)), float(np.std(osd_vals))
            region_summary_osdface[name] = {"mean": round(mu, 5), "std": round(sd, 5)}
            print(f"    OSDFace[{name:<5s}]  {mu:.5f} ± {sd:.5f}")

    summary["region_craft"]   = region_summary_craft
    summary["region_osdface"] = region_summary_osdface

    # ── save JSON ────────────────────────────────────────────────────────
    metrics_path = os.path.join(summary_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"summary": summary, "per_image": all_metrics}, f, indent=2)
    print(f"\n  Saved metrics → {metrics_path}")

    # ── summary figures ──────────────────────────────────────────────────
    if craft_model is not None or osdface_model is not None:
        save_summary_figure(all_metrics, summary_dir)

    if all_region_craft or all_region_osdface:
        save_region_summary(all_region_craft, all_region_osdface, summary_dir)

    print(f"\nAll outputs saved to {args.out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
