"""
diagnose_bottleneck.py — Step 1 diagnostics for CRAFT Stage 1.

Goal: decide whether the ~20 PSNR ceiling is caused by the quantizer, the
encoder/decoder, or both — *without training anything*.

Three checks, all run on the held-out first --n_images of the training set:

    (A1) Bypass-VQ reconstruction
         Run encoder → post_quant_conv → decoder, skipping the quantizer.
         If HQ bypass PSNR is ~35+ dB while normal HQ PSNR is ~20 dB, the
         quantizer is the ceiling. If bypass is also ~20 dB, the
         encoder/decoder pair itself is the ceiling.

    (A2) Cross-feed: HQ encoder features → CRAFT decoder, and LQ encoder
         features → HQ decoder, both skipping quantization.
         Separates decoder capacity from encoder capacity and quantizer
         damage. If CRAFT decoder fed with HQ features reconstructs well,
         the decoder is fine — the LQ encoder/quantizer is the problem.

    (A3) Codebook usage across the full eval set
         Count unique (region, level, code) triples that are actually
         selected during inference. Complements the training-time EMA
         dead-code fraction (which is a one-batch snapshot). A code that
         is never picked at eval time is effectively dead regardless of
         what the EMA counter says.

Outputs
    <out_dir>/report.txt      text report with all numbers
    <out_dir>/bypass.png      side-by-side: HQ input | HQ normal | HQ bypass
                                           | CRAFT normal | CRAFT bypass
                                           | HQ→CRAFT dec | LQ→HQ dec
    stdout                    same report printed live

Usage
    python diagnose_bottleneck.py \
        --hq_ckpt    checkpoints/phase_a/final.pt \
        --craft_ckpt checkpoints/phase_c/final.pt \
        --parser_ckpt pretrained/79999_iter.pth \
        --data_root  data/train \
        --n_images 128 \
        --out_dir diagnostics/bottleneck
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataset import FFHQPairedDataset
from models.vqvae import build_hq_vqvae, build_lq_vqvae
from models.region_aware_vq import RegionAwareVQ
from models.face_parser import FaceParser, REGION_NAMES


# ======================================================================
# Metrics (duplicated from eval_reconstruction.py to stay standalone)
# ======================================================================

def _psnr(pred, target, eps=1e-8):
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse < eps:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _ssim(pred, target):
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        p = pred.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
        t = target.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
        return float(np.mean([
            sk_ssim(p[i], t[i], data_range=1.0, channel_axis=-1)
            for i in range(p.shape[0])
        ]))
    except ImportError:
        return float("nan")


def _to01(x_rec_11):
    """Decoder output is in [-1, 1]; clamp and shift to [0, 1]."""
    return ((x_rec_11.clamp(-1, 1) + 1) * 0.5).float()


# ======================================================================
# Model loading
# ======================================================================

def _load_hq(ckpt_path, device, embed_dim, n_codes):
    model = build_hq_vqvae(n_codes=n_codes, embed_dim=embed_dim)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def _load_craft(ckpt_path, device, parser_ckpt, embed_dim, rq_levels):
    ravq = RegionAwareVQ(
        e_dim=embed_dim, n_levels=rq_levels, parser_ckpt=parser_ckpt,
    )
    model = build_lq_vqvae(ravq, embed_dim=embed_dim)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    key = "model" if "model" in ckpt else "lq_model"
    model.load_state_dict(ckpt[key])
    return model.to(device).eval()


# ======================================================================
# Accumulator for a single reconstruction variant
# ======================================================================

class _Acc:
    """Running mean of PSNR / SSIM / L1 for one variant."""

    def __init__(self, name):
        self.name = name
        self.psnr = []
        self.ssim = []
        self.l1 = []

    def add(self, pred_01, target_01):
        B = pred_01.shape[0]
        for i in range(B):
            self.psnr.append(_psnr(pred_01[i:i+1], target_01[i:i+1]))
        self.ssim.append(_ssim(pred_01, target_01))
        self.l1.append(F.l1_loss(pred_01, target_01).item())

    def _mean(self, xs):
        xs = [v for v in xs if not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(xs)) if xs else float("nan")

    def _std(self, xs):
        xs = [v for v in xs if not (isinstance(v, float) and math.isnan(v))]
        return float(np.std(xs)) if xs else float("nan")

    def summary(self):
        return {
            "psnr_mean": self._mean(self.psnr),
            "psnr_std": self._std(self.psnr),
            "ssim_mean": self._mean(self.ssim),
            "l1_mean":  self._mean(self.l1),
        }


# ======================================================================
# Code-usage tracker (A3)
# ======================================================================

class _CodeUsage:
    """
    Accumulates, for each (region, level), the set of codebook indices
    actually selected at eval time.

        usage[region][level] = set(int)        codes ever hit
        counts[region][level] = Counter()      how many times each was hit
    """

    def __init__(self, n_codes_per_region):
        self.n_codes = dict(n_codes_per_region)
        self.usage = {r: defaultdict(set) for r in n_codes_per_region}
        self.hits = {r: defaultdict(int) for r in n_codes_per_region}

    def update(self, region, level, indices_tensor):
        idxs = indices_tensor.detach().cpu().tolist()
        self.usage[region][level].update(idxs)
        self.hits[region][level] += len(idxs)

    def summary(self):
        """For each (region, level): unique codes used, total, fraction."""
        rows = []
        for r in self.n_codes:
            n = self.n_codes[r]
            for lvl in sorted(self.usage[r].keys()):
                used = len(self.usage[r][lvl])
                hits = self.hits[r][lvl]
                rows.append({
                    "region": r,
                    "level": lvl,
                    "n_codes": n,
                    "used": used,
                    "frac_used": used / n if n else 0.0,
                    "hits": hits,
                })
        return rows


# ======================================================================
# Forward passes
# ======================================================================

@torch.no_grad()
def _normal_reconstruct(model, x_11, x_01):
    """Full encode → quantize → decode."""
    x_rec, *_ = model(x_11, images_01=x_01)
    return _to01(x_rec)


@torch.no_grad()
def _bypass_reconstruct(model, x_11):
    """Encode → post_quant_conv → decode (no quantization at all)."""
    z = model.encode(x_11)              # (B, embed, 16, 16)
    return _to01(model.decode(z))       # decode applies post_quant_conv + decoder


@torch.no_grad()
def _cross_feed(src_model, dst_model, x_11):
    """Encode with src, decode with dst (bypassing quantization on both sides)."""
    z = src_model.encode(x_11)
    return _to01(dst_model.decode(z))


@torch.no_grad()
def _collect_code_usage(craft_model, face_parser, x_11, x_01, tracker):
    """
    Run the CRAFT encoder, parse regions, and record which codebook
    indices each region+level selects. Does NOT modify EMA buffers —
    we use ResidualVQ.encode which is @torch.no_grad.
    """
    z = craft_model.encode(x_11)  # (B, 512, 16, 16)
    B, C, H, W = z.shape

    masks = face_parser.get_region_masks(x_01, target_h=H, target_w=W)
    z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C)

    for name in REGION_NAMES:
        mask = masks[name].reshape(B, H * W)
        feats = z_flat[mask]
        if feats.shape[0] == 0:
            continue
        rq = craft_model.quantizer.region_codebooks[name]
        indices_list = rq.encode(feats)       # list of (N_r,) per level
        for lvl, idxs in enumerate(indices_list):
            tracker.update(name, lvl, idxs)


# ======================================================================
# Visual grid (one batch)
# ======================================================================

def _np(t):
    return (t.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)


def _save_visuals(out_path, row_tensors, col_titles, n_rows=4):
    """
    row_tensors: list of dicts, each {col_name: (3,H,W) tensor in [0,1]}
                 one dict per sample.
    col_titles:  list of column header strings, in display order.
    """
    n_rows = min(n_rows, len(row_tensors))
    n_cols = len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols, 2.6 * n_rows),
                             squeeze=False)
    for i in range(n_rows):
        for j, title in enumerate(col_titles):
            ax = axes[i][j]
            ax.imshow(_np(row_tensors[i][title]))
            if i == 0:
                ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="CRAFT Stage 1 bottleneck diagnostics")
    ap.add_argument("--hq_ckpt",    type=str, required=True)
    ap.add_argument("--craft_ckpt", type=str, required=True)
    ap.add_argument("--parser_ckpt", type=str,
                    default="/projectnb/cs585/projects/craft/pretrained/79999_iter.pth")
    ap.add_argument("--data_root",  type=str, required=True)
    ap.add_argument("--hq_folder",  type=str, default="images512x512")
    ap.add_argument("--lq_folder",  type=str, default="LQ_images_512x512")
    ap.add_argument("--embed_dim",  type=int, default=512)
    ap.add_argument("--hq_n_codes", type=int, default=1024)
    ap.add_argument("--rq_levels",  type=int, default=3)
    ap.add_argument("--n_images",   type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_visuals",  type=int, default=4,
                    help="rows saved in bypass.png")
    ap.add_argument("--out_dir",    type=str, default="diagnostics/bottleneck")
    ap.add_argument("--device",     type=str, default="cuda")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Models ---
    print("Loading models...")
    hq_model = _load_hq(args.hq_ckpt, device, args.embed_dim, args.hq_n_codes)
    print(f"  HQ     <- {args.hq_ckpt}")
    craft_model = _load_craft(
        args.craft_ckpt, device, args.parser_ckpt,
        args.embed_dim, args.rq_levels,
    )
    print(f"  CRAFT  <- {args.craft_ckpt}")

    # Parser: reuse CRAFT's (already frozen)
    face_parser = craft_model.quantizer.face_parser
    face_parser.eval()

    # --- Data ---
    print(f"\nDataset: {args.data_root}  (first {args.n_images})")
    dataset = FFHQPairedDataset(
        data_root=args.data_root,
        hq_folder=args.hq_folder,
        lq_folder=args.lq_folder,
        masks_folder="",
    )
    n = min(args.n_images, len(dataset))
    loader = DataLoader(
        Subset(dataset, list(range(n))),
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # --- Accumulators ---
    accs = {
        "hq_normal":      _Acc("hq_normal"),
        "hq_bypass":      _Acc("hq_bypass"),
        "craft_normal":   _Acc("craft_normal"),
        "craft_bypass":   _Acc("craft_bypass"),
        "hq_enc_craft_dec": _Acc("hq_enc_craft_dec"),
        "lq_enc_hq_dec":    _Acc("lq_enc_hq_dec"),
    }

    n_codes_per_region = {
        r: craft_model.quantizer.region_codebooks[r].n_codes
        for r in REGION_NAMES
    }
    tracker = _CodeUsage(n_codes_per_region)

    vis_rows = []  # store tensors from the first few samples for the grid

    # --- Loop ---
    print("\nRunning diagnostics...")
    seen = 0
    for batch in loader:
        hq_11 = batch["hq"].to(device)
        lq_11 = batch["lq"].to(device)
        hq_01 = batch["hq_01"].to(device)
        lq_01 = batch["lq_01"].to(device)

        # (A1) Normal + bypass
        hq_normal    = _normal_reconstruct(hq_model, hq_11, hq_01)
        hq_bypass    = _bypass_reconstruct(hq_model, hq_11)
        craft_normal = _normal_reconstruct(craft_model, lq_11, lq_01)
        craft_bypass = _bypass_reconstruct(craft_model, lq_11)

        # (A2) Cross-feed (no quantization on either side)
        hq_to_craft  = _cross_feed(hq_model, craft_model, hq_11)  # HQ encoder on HQ image
        lq_to_hq     = _cross_feed(craft_model, hq_model, lq_11)  # LQ encoder on LQ image

        # (A3) Code usage (eval-time, on LQ inputs like normal inference)
        _collect_code_usage(craft_model, face_parser, lq_11, lq_01, tracker)

        # Metrics — everything compared to HQ ground truth in [0,1]
        accs["hq_normal"]       .add(hq_normal,    hq_01)
        accs["hq_bypass"]       .add(hq_bypass,    hq_01)
        accs["craft_normal"]    .add(craft_normal, hq_01)
        accs["craft_bypass"]    .add(craft_bypass, hq_01)
        accs["hq_enc_craft_dec"].add(hq_to_craft,  hq_01)
        accs["lq_enc_hq_dec"]   .add(lq_to_hq,     hq_01)

        # Visuals from the first batch
        if len(vis_rows) < args.n_visuals:
            B = hq_11.shape[0]
            for i in range(min(B, args.n_visuals - len(vis_rows))):
                vis_rows.append({
                    "HQ GT":            hq_01[i].cpu(),
                    "HQ normal":        hq_normal[i].cpu(),
                    "HQ bypass-VQ":     hq_bypass[i].cpu(),
                    "CRAFT normal":     craft_normal[i].cpu(),
                    "CRAFT bypass-VQ":  craft_bypass[i].cpu(),
                    "HQ enc→CRAFT dec": hq_to_craft[i].cpu(),
                    "LQ enc→HQ dec":    lq_to_hq[i].cpu(),
                })

        seen += hq_11.shape[0]
        print(f"  [{seen:4d}/{n}]  "
              f"hq={_psnr(hq_normal, hq_01):5.2f}  "
              f"hq_bypass={_psnr(hq_bypass, hq_01):5.2f}  "
              f"craft={_psnr(craft_normal, hq_01):5.2f}  "
              f"craft_bypass={_psnr(craft_bypass, hq_01):5.2f}")

    # --- Report ---
    report_lines = []
    def _log(s=""):
        print(s)
        report_lines.append(s)

    _log("\n" + "=" * 72)
    _log("STEP 1 DIAGNOSTICS — CRAFT Stage 1 bottleneck")
    _log("=" * 72)
    _log(f"Images scored: {seen}")
    _log("")
    _log(f"{'Variant':<22s}  {'PSNR':>10s}  {'SSIM':>7s}  {'L1':>7s}")
    _log("-" * 52)
    for k in ("hq_normal", "hq_bypass", "craft_normal", "craft_bypass",
              "hq_enc_craft_dec", "lq_enc_hq_dec"):
        s = accs[k].summary()
        _log(f"{k:<22s}  {s['psnr_mean']:6.2f}±{s['psnr_std']:4.2f}  "
             f"{s['ssim_mean']:7.4f}  {s['l1_mean']:7.4f}")

    _log("")
    _log("Interpretation key:")
    _log("  VQ cost (HQ)    = hq_bypass − hq_normal        "
         "(dB lost to Phase A quantizer)")
    _log("  VQ cost (CRAFT) = craft_bypass − craft_normal  "
         "(dB lost to RegionAwareVQ)")
    _log("  Decoder ceiling = hq_enc_craft_dec              "
         "(CRAFT dec fed HQ features; bottleneck = CRAFT decoder)")
    _log("  Encoder ceiling = lq_enc_hq_dec                 "
         "(HQ dec fed LQ features; bottleneck = LQ encoder on degraded input)")
    _log("")
    dhq  = accs["hq_bypass"].summary()["psnr_mean"] - accs["hq_normal"].summary()["psnr_mean"]
    dcr  = accs["craft_bypass"].summary()["psnr_mean"] - accs["craft_normal"].summary()["psnr_mean"]
    _log(f"  →  HQ VQ cost:     {dhq:+.2f} dB")
    _log(f"  →  CRAFT VQ cost:  {dcr:+.2f} dB")

    # --- Code usage ---
    _log("")
    _log("Code usage across full eval set (unique indices selected):")
    _log(f"{'region':<6s} {'lvl':>3s} {'used':>6s} {'/':>1s} {'total':>6s} "
         f"{'frac':>6s} {'hits':>8s}")
    _log("-" * 44)
    usage_rows = tracker.summary()
    for row in usage_rows:
        _log(f"{row['region']:<6s} {row['level']:>3d} "
             f"{row['used']:>6d} / {row['n_codes']:>6d} "
             f"{row['frac_used']*100:>5.1f}% {row['hits']:>8d}")

    # --- Save report + json + visuals ---
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write("\n".join(report_lines) + "\n")
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump({
            "n_images": seen,
            "variants": {k: v.summary() for k, v in accs.items()},
            "hq_vq_cost_db": dhq,
            "craft_vq_cost_db": dcr,
            "code_usage": usage_rows,
        }, f, indent=2)

    if vis_rows:
        col_titles = [
            "HQ GT", "HQ normal", "HQ bypass-VQ",
            "CRAFT normal", "CRAFT bypass-VQ",
            "HQ enc→CRAFT dec", "LQ enc→HQ dec",
        ]
        _save_visuals(
            os.path.join(args.out_dir, "bypass.png"),
            vis_rows, col_titles, n_rows=args.n_visuals,
        )
        _log("")
        _log(f"Saved {os.path.join(args.out_dir, 'bypass.png')}")

    _log(f"Saved {os.path.join(args.out_dir, 'report.txt')}")
    _log(f"Saved {os.path.join(args.out_dir, 'report.json')}")


if __name__ == "__main__":
    main()
