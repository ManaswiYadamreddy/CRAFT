"""
diagnose_masks.py — Sanity-check the RegionAwareVQ mask pipeline.

Answers three questions:
  [1] Histogram: how many latent tokens does each region get per image,
      across N sampled images? Report mean / median / p10 / p90 / %zero.
  [2] Overlay: dump a figure showing (image | full-res parsing mask |
      current 16x16 region map | any-hit 16x16 region map) for K samples,
      so you can see eyes/hair disappearing visually.
  [3] Parser sanity: at full 512x512 resolution (before any downsample),
      what fraction of pixels does each region occupy? If eyes are
      < 0.1% at full res, the parser itself is broken. If eyes are ~2%
      at full res but ~0.01 tokens/img at 16x16, it's the downsample.

Also compares the current nearest-neighbor downsample against an
"any-hit" downsample with priority (eyes > lips > hair > skin) so you
can see how much the fix would recover before touching training code.

Usage:
    python diagnose_masks.py \
        --data_root /projectnb/cs585/projects/craft/data/train \
        --parser_ckpt pretrained/79999_iter.pth \
        --n_images 64 --n_overlays 6 \
        --out_dir diagnostics/phase_b_masks
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models.face_parser import FaceParser, REGION_NAMES, REGION_MAP, N_CLASSES


# Priority order for any-hit downsample: rarest regions win ties.
# eyes=0, lips=3, hair=2, skin=1 in REGION_NAMES; we encode priority
# as a rank so that lower rank = higher priority.
ANY_HIT_PRIORITY = ["eyes", "lips", "hair", "skin"]

# Fixed colors for the 4 regions (RGBA)
REGION_COLORS = {
    "eyes": (1.0, 0.2, 0.2, 1.0),   # red
    "skin": (1.0, 0.85, 0.7, 1.0),  # tan
    "hair": (0.3, 0.3, 0.3, 1.0),   # dark gray
    "lips": (0.9, 0.3, 0.8, 1.0),   # magenta
}


def build_region_cmap():
    """Colormap indexed by region index in REGION_NAMES order."""
    colors = [REGION_COLORS[n] for n in REGION_NAMES]
    return ListedColormap(colors)


def any_hit_downsample(region_indices_full, target_h, target_w):
    """
    Downsample a (B, H, W) region-index map to (B, target_h, target_w)
    using priority-based any-hit pooling instead of nearest-neighbor.

    If any pixel in the pooling window belongs to a higher-priority
    region, the target cell gets that region.

    Args:
        region_indices_full: (B, H, W) int64 in {0..3}, REGION_NAMES order.
    Returns:
        (B, target_h, target_w) int64 in {0..3}.
    """
    B, H, W = region_indices_full.shape
    # Map region index -> priority rank (0 = highest priority)
    idx_to_priority = torch.zeros(len(REGION_NAMES), dtype=torch.long,
                                  device=region_indices_full.device)
    for rank, name in enumerate(ANY_HIT_PRIORITY):
        idx_to_priority[REGION_NAMES.index(name)] = rank

    priority_map = idx_to_priority[region_indices_full]  # (B, H, W), lower=better

    # Min-pool priority over each (H/target_h, W/target_w) window.
    # Use -max-pool on negated values because PyTorch has no min_pool.
    kh, kw = H // target_h, W // target_w
    assert H % target_h == 0 and W % target_w == 0, \
        f"Full res ({H}x{W}) must be divisible by target ({target_h}x{target_w})"

    neg = (-priority_map.float()).unsqueeze(1)  # (B, 1, H, W)
    pooled_neg = F.max_pool2d(neg, kernel_size=(kh, kw), stride=(kh, kw))
    best_priority = (-pooled_neg).squeeze(1).long()  # (B, target_h, target_w)

    # Map priority rank back to region index.
    priority_to_idx = torch.zeros(len(REGION_NAMES), dtype=torch.long,
                                  device=region_indices_full.device)
    for rank, name in enumerate(ANY_HIT_PRIORITY):
        priority_to_idx[rank] = REGION_NAMES.index(name)

    return priority_to_idx[best_priority]


def compute_region_indices_full(parser, images):
    """Run parser and map CelebAMask labels -> region indices at FULL res (512x512)."""
    labels = parser.parse(images)          # (B, 512, 512) in [0..18]
    region_full = parser.region_lut[labels]  # (B, 512, 512) in [0..3]
    return region_full


def summarize_counts(counts_per_image, region_names):
    """counts_per_image: (N, R) numpy. Returns per-region summary dict."""
    out = {}
    for r, name in enumerate(region_names):
        c = counts_per_image[:, r]
        out[name] = dict(
            mean=float(c.mean()),
            median=float(np.median(c)),
            p10=float(np.percentile(c, 10)),
            p90=float(np.percentile(c, 90)),
            min=int(c.min()),
            max=int(c.max()),
            frac_zero=float((c == 0).mean()),
        )
    return out


def print_summary(title, summary, total_tokens):
    print(f"\n=== {title} ===")
    print(f"{'region':<8} {'mean':>8} {'median':>8} {'p10':>6} {'p90':>8} "
          f"{'min':>5} {'max':>6} {'%zero':>7} {'%tokens':>8}")
    for name in REGION_NAMES:
        s = summary[name]
        pct = 100.0 * s["mean"] / total_tokens
        print(f"{name:<8} {s['mean']:>8.2f} {s['median']:>8.1f} "
              f"{s['p10']:>6.1f} {s['p90']:>8.1f} "
              f"{s['min']:>5d} {s['max']:>6d} "
              f"{100*s['frac_zero']:>6.1f}% {pct:>7.2f}%")


def draw_overlay(image, region_full, region_cur, region_any, cmap, out_path):
    """
    image:       (3, H, W) tensor in [0,1]
    region_full: (H, W) in {0..3}
    region_cur:  (target_h, target_w) in {0..3}  (current nearest)
    region_any:  (target_h, target_w) in {0..3}  (any-hit)
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    img_np = image.permute(1, 2, 0).cpu().numpy()

    axes[0].imshow(img_np)
    axes[0].set_title("input")

    axes[1].imshow(img_np)
    axes[1].imshow(region_full.cpu().numpy(), cmap=cmap, alpha=0.45,
                   vmin=0, vmax=len(REGION_NAMES) - 1, interpolation="nearest")
    axes[1].set_title("full-res parsing (overlay)")

    axes[2].imshow(region_cur.cpu().numpy(), cmap=cmap,
                   vmin=0, vmax=len(REGION_NAMES) - 1, interpolation="nearest")
    axes[2].set_title(f"16x16 NEAREST (current)\n"
                      f"eyes={int((region_cur==0).sum())} "
                      f"hair={int((region_cur==2).sum())} "
                      f"lips={int((region_cur==3).sum())}")

    axes[3].imshow(region_any.cpu().numpy(), cmap=cmap,
                   vmin=0, vmax=len(REGION_NAMES) - 1, interpolation="nearest")
    axes[3].set_title(f"16x16 ANY-HIT (proposed)\n"
                      f"eyes={int((region_any==0).sum())} "
                      f"hair={int((region_any==2).sum())} "
                      f"lips={int((region_any==3).sum())}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--hq_folder", type=str, default="images512x512")
    ap.add_argument("--parser_ckpt", type=str, required=True)
    ap.add_argument("--n_images", type=int, default=64,
                    help="How many images to sample for the histogram.")
    ap.add_argument("--n_overlays", type=int, default=6,
                    help="How many overlay figures to dump.")
    ap.add_argument("--target_h", type=int, default=16)
    ap.add_argument("--target_w", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default="diagnostics/phase_b_masks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load images
    hq_dir = os.path.join(args.data_root, args.hq_folder)
    paths = sorted(glob.glob(os.path.join(hq_dir, "*.png")))
    if not paths:
        raise SystemExit(f"No PNGs found in {hq_dir}")
    rng = np.random.RandomState(args.seed)
    paths = list(rng.choice(paths, size=min(args.n_images, len(paths)), replace=False))
    print(f"Sampling {len(paths)} images from {hq_dir}")

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize(
        (512, 512),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )

    def load_batch(batch_paths):
        xs = [resize(to_tensor(Image.open(p).convert("RGB"))) for p in batch_paths]
        return torch.stack(xs, dim=0)

    # Load parser
    print(f"Loading face parser from {args.parser_ckpt}")
    parser = FaceParser(checkpoint_path=args.parser_ckpt).to(device)

    total_tokens = args.target_h * args.target_w
    R = len(REGION_NAMES)

    counts_cur = np.zeros((len(paths), R), dtype=np.int64)
    counts_any = np.zeros((len(paths), R), dtype=np.int64)
    full_pixel_counts = np.zeros(R, dtype=np.int64)
    full_pixel_total = 0

    cmap = build_region_cmap()
    overlay_left = args.n_overlays

    i = 0
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start:start + args.batch_size]
        images = load_batch(batch_paths).to(device)            # (B, 3, 512, 512)

        region_full = compute_region_indices_full(parser, images)  # (B, 512, 512)

        # Current: nearest-neighbor (exactly what the parser currently does)
        region_cur = F.interpolate(
            region_full.unsqueeze(1).float(),
            size=(args.target_h, args.target_w),
            mode="nearest",
        ).squeeze(1).long()                                     # (B, h, w)

        # Proposed: priority any-hit
        region_any = any_hit_downsample(region_full, args.target_h, args.target_w)

        # [1] per-image latent-token counts
        for r in range(R):
            counts_cur[i:i + images.size(0), r] = (region_cur == r).sum(dim=(1, 2)).cpu().numpy()
            counts_any[i:i + images.size(0), r] = (region_any == r).sum(dim=(1, 2)).cpu().numpy()

        # [3] full-res pixel coverage
        for r in range(R):
            full_pixel_counts[r] += int((region_full == r).sum().item())
        full_pixel_total += region_full.numel()

        # [2] overlays
        while overlay_left > 0 and (images.size(0) - (args.n_overlays - overlay_left + i - start)) > 0:
            local_idx = args.n_overlays - overlay_left
            if local_idx >= images.size(0):
                break
            out_path = os.path.join(args.out_dir, f"overlay_{args.n_overlays - overlay_left:02d}.png")
            draw_overlay(
                images[local_idx], region_full[local_idx],
                region_cur[local_idx], region_any[local_idx],
                cmap, out_path,
            )
            overlay_left -= 1
            if overlay_left == 0 or local_idx + 1 >= images.size(0):
                break

        i += images.size(0)
        print(f"  processed {i}/{len(paths)}")

    # -------- Report --------
    print("\n" + "=" * 68)
    print("[3] PARSER SANITY — full-res (512x512) pixel coverage")
    print("=" * 68)
    for r, name in enumerate(REGION_NAMES):
        frac = 100.0 * full_pixel_counts[r] / full_pixel_total
        print(f"  {name:<8} {frac:6.3f}%  ({full_pixel_counts[r]:>12d} px)")
    print("  Expected order of magnitude on FFHQ:")
    print("    skin ~40-60%, hair ~10-30%, eyes ~1-3%, lips ~0.5-2%")
    print("  If eyes < 0.1% here, the parser is broken (wrong ckpt, bad")
    print("  alignment, missing ImageNet norm, etc). If eyes ~1-3% here")
    print("  but ~0 tokens/img below, the DOWNSAMPLE is the culprit.")

    summary_cur = summarize_counts(counts_cur, REGION_NAMES)
    summary_any = summarize_counts(counts_any, REGION_NAMES)

    print_summary(
        f"[1a] CURRENT downsample (nearest) — counts per image "
        f"(out of {total_tokens} tokens)",
        summary_cur, total_tokens,
    )
    print_summary(
        f"[1b] PROPOSED downsample (priority any-hit) — counts per image "
        f"(out of {total_tokens} tokens)",
        summary_any, total_tokens,
    )

    # Save raw counts for later inspection
    np.savez(
        os.path.join(args.out_dir, "counts.npz"),
        counts_cur=counts_cur,
        counts_any=counts_any,
        region_names=np.array(REGION_NAMES),
        full_pixel_counts=full_pixel_counts,
        full_pixel_total=full_pixel_total,
    )

    # Histogram figure
    fig, axes = plt.subplots(2, R, figsize=(4 * R, 6), sharey="row")
    for r, name in enumerate(REGION_NAMES):
        axes[0, r].hist(counts_cur[:, r], bins=30, color="tab:red", alpha=0.8)
        axes[0, r].set_title(f"{name}  (nearest)")
        axes[0, r].set_xlabel("tokens/img")
        axes[1, r].hist(counts_any[:, r], bins=30, color="tab:green", alpha=0.8)
        axes[1, r].set_title(f"{name}  (any-hit)")
        axes[1, r].set_xlabel("tokens/img")
    axes[0, 0].set_ylabel("# images")
    axes[1, 0].set_ylabel("# images")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "histogram.png"), dpi=120)
    plt.close(fig)

    print(f"\nSaved histogram, overlays, and counts.npz to {args.out_dir}")

    # -------- Verdict --------
    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    eyes_full_pct = 100.0 * full_pixel_counts[REGION_NAMES.index("eyes")] / full_pixel_total
    eyes_tok_cur = summary_cur["eyes"]["mean"]
    eyes_tok_any = summary_any["eyes"]["mean"]
    if eyes_full_pct < 0.1:
        print("  PARSER looks broken: eyes < 0.1% of full-res pixels.")
        print("  Check: parser_ckpt path, input normalization, crop alignment.")
    elif eyes_tok_cur < 2 and eyes_tok_any > 3 * max(eyes_tok_cur, 1e-6):
        print("  DOWNSAMPLE is the culprit. Current nearest-neighbor loses")
        print(f"  small regions: eyes={eyes_tok_cur:.2f} tok/img now vs")
        print(f"  {eyes_tok_any:.2f} tok/img under any-hit. Switch the")
        print("  downsample in models/face_parser.py:get_region_indices to")
        print("  priority any-hit, re-run precompute_masks.py, resume Phase B.")
    else:
        print("  Inconclusive from counts alone — inspect the overlay PNGs.")


if __name__ == "__main__":
    main()
