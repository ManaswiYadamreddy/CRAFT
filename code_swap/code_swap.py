"""
code_swap.py — Code-swap reconstruction experiment for CRAFT Stage-1.

Goal
----
Test whether a single discrete code id in the region-aware codebook is
*causally* responsible for a visual property of the reconstruction.

For every region we:
    1. Encode the input face → 16×16 feature map.
    2. Run the face parser → region masks at 16×16.
    3. For each region, quantize the masked features through the per-region
       ResidualVQ → get 3 levels of code indices per spatial position.
    4. (Baseline) Decode → original reconstruction.
    5. (Swap)    Replace **all** indices at a chosen level for that region
                  with one donor code id (or several donors, one per panel
                  column). Decode again. The other regions are untouched.

If the codebook carries real visual semantics, swapping (e.g.) an "eye L0"
code id should change *only the eye region* of the reconstruction. If the
output is identical, the code id was redundant. If the output is garbled,
the codebook is not smoothly swappable.

Outputs (written under --out_dir):
    original.png                  Input face (resized to 512×512)
    recon_baseline.png            Reconstruction with no swaps
    swap_panel_<region>.png       baseline | donor_1 | donor_2 | …  for that region
    swap_indices.json             Donor ids tried, top-K most-used codes per region

Usage example
-------------
    python code_swap/code_swap.py \
        --stage1_ckpt /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
        --parser_ckpt /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
        --image       /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2/00219_LQ.png \
        --regions     eyes lips hair \
        --level       0 \
        --n_donors    4 \
        --out_dir     /projectnb/cs585/projects/craft/code_swap_outputs/face_00219
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# --- Repo root on path so `models.*` imports resolve ---
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.region_aware_vq import RegionAwareVQ
from models.face_parser import REGION_NAMES
from models.vqvae import build_lq_vqvae


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_stage1(ckpt_path, parser_ckpt, device, embed_dim=512, rq_levels=3):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt["lq_model"]
    has_mag = any("magnitude_head" in k for k in state.keys())
    if has_mag:
        print("  detected magnitude_head → Phase D checkpoint")
    ravq = RegionAwareVQ(
        e_dim=embed_dim, n_levels=rq_levels,
        parser_ckpt=parser_ckpt, use_magnitude_head=has_mag,
    )
    model = build_lq_vqvae(ravq, embed_dim=embed_dim)
    model.load_state_dict(state)
    return model.to(device).eval()


# ----------------------------------------------------------------------
# Core: encode → (optional swap) → decode
# ----------------------------------------------------------------------

@torch.no_grad()
def encode_quantize_decode(model, x_neg11, x_01, override=None):
    """
    Run the full LQ pipeline with optional per-region index override.

    Args
    ----
    override : dict | None
        If given, maps region_name -> {level_idx: replacement}, where
        `replacement` is either an int (replace every position with that
        code id) or a 1-D tensor of shape (N_r,) (one id per masked
        position, in mask-order).

    Returns
    -------
    x_rec  : (1, 3, 512, 512) reconstruction in [-1, 1]
    info   : dict with original index lists per region and per level.
    """
    quant = model.quantizer
    z = model.encode(x_neg11)
    B, C, H, W = z.shape
    masks = quant.face_parser.get_region_masks(x_01, target_h=H, target_w=W)

    z_flat   = z.permute(0, 2, 3, 1).reshape(B, H * W, C)
    z_q_flat = torch.zeros_like(z_flat)
    masks_flat = {n: m.reshape(B, H * W) for n, m in masks.items()}

    indices_per_region = {}

    for name in REGION_NAMES:
        if name not in quant.region_codebooks:
            continue
        mask = masks_flat[name]
        if mask.sum() == 0:
            continue
        rq = quant.region_codebooks[name]
        region_features = z_flat[mask]                  # (N_r, C)
        idx_list = rq.encode(region_features)           # list of n_levels tensors

        # Apply override
        if override and name in override:
            for lvl, repl in override[name].items():
                if isinstance(repl, int):
                    idx_list[lvl] = torch.full_like(idx_list[lvl], repl)
                else:
                    idx_list[lvl] = repl.to(idx_list[lvl].device,
                                             dtype=idx_list[lvl].dtype)

        indices_per_region[name] = [i.detach().cpu().tolist() for i in idx_list]
        z_q_region = rq.decode(idx_list)                # (N_r, C) scaled
        z_q_flat[mask] = z_q_region.to(z_q_flat.dtype)

    z_q = z_q_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    if quant.use_magnitude_head:
        mag = F.softplus(quant.magnitude_head(z))
        z_q = z_q * mag

    x_rec = model.decode(z_q)
    return x_rec, {
        "indices_per_region": indices_per_region,
        "n_positions": {k: int(v.sum().item()) for k, v in masks_flat.items()},
    }


def get_top_codes(rq, level, k=8):
    """Top-k most-used code ids at a given level, by EMA count."""
    counts = rq.levels[level].ema_count.detach().cpu()
    return counts.argsort(descending=True)[:k].tolist()


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------

def tensor_neg11_to_pil(x):
    """(1, 3, H, W) in [-1, 1] → PIL.Image (RGB)."""
    x01 = ((x.clamp(-1, 1) + 1) / 2).float().squeeze(0).cpu()
    return transforms.ToPILImage()(x01)


def _load_font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_panel(images_with_labels, header, cell=384, label_h=28,
               header_h=44, gap=10):
    """
    images_with_labels : list[(PIL.Image, str)]
    header             : str  (drawn across the top of the panel)
    """
    n = len(images_with_labels)
    W = n * cell + (n - 1) * gap
    H = header_h + label_h + cell
    panel = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    h_font = _load_font(20)
    l_font = _load_font(15)

    # Header
    try:
        bbox = draw.textbbox((0, 0), header, font=h_font)
        tw = bbox[2] - bbox[0]
    except AttributeError:
        tw = len(header) * 10
    draw.text(((W - tw) // 2, 12), header, fill=(40, 40, 40), font=h_font)

    # Cells
    for i, (img, lbl) in enumerate(images_with_labels):
        x = i * (cell + gap)
        # column label
        try:
            bbox = draw.textbbox((0, 0), lbl, font=l_font)
            lw = bbox[2] - bbox[0]
        except AttributeError:
            lw = len(lbl) * 7
        draw.text((x + (cell - lw) // 2, header_h + 5),
                  lbl, fill=(60, 60, 60), font=l_font)
        # image
        img_resized = img.resize((cell, cell), Image.BICUBIC)
        panel.paste(img_resized, (x, header_h + label_h))
    return panel


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

@torch.no_grad()
def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading CRAFT Stage-1 from {args.stage1_ckpt}")
    model = load_stage1(args.stage1_ckpt, args.parser_ckpt, device)
    print(f"  active regions: {list(model.quantizer.region_codebooks.keys())}")

    # --- Load + prep image ---
    img = Image.open(args.image).convert("RGB")
    to_tensor = transforms.ToTensor()
    x01 = to_tensor(img).unsqueeze(0).to(device)
    if x01.shape[-2:] != (args.resolution, args.resolution):
        x01 = F.interpolate(x01, size=(args.resolution, args.resolution),
                            mode="bilinear", align_corners=False)
    x11 = x01 * 2.0 - 1.0

    os.makedirs(args.out_dir, exist_ok=True)
    tensor_neg11_to_pil(x11).save(os.path.join(args.out_dir, "original.png"))

    # --- Baseline reconstruction (no swap) ---
    x_baseline, baseline_info = encode_quantize_decode(model, x11, x01, override=None)
    baseline_pil = tensor_neg11_to_pil(x_baseline)
    baseline_pil.save(os.path.join(args.out_dir, "recon_baseline.png"))

    # --- Choose donor codes per region ---
    user_donor_ids = (
        [int(x) for x in args.donor_codes.split(",") if x.strip()]
        if args.donor_codes else None
    )

    summary = {
        "image": os.path.abspath(args.image),
        "level": args.level,
        "regions": {},
        "n_positions_per_region": baseline_info["n_positions"],
    }

    quant = model.quantizer
    for region in args.regions:
        if region not in quant.region_codebooks:
            print(f"  [skip] region '{region}' not in codebook")
            continue
        if baseline_info["n_positions"].get(region, 0) == 0:
            print(f"  [skip] region '{region}' had 0 positions for this face")
            continue

        rq = quant.region_codebooks[region]
        max_codes = rq.levels[args.level].n_codes

        if user_donor_ids is not None:
            donor_ids = [c for c in user_donor_ids if 0 <= c < max_codes]
        else:
            donor_ids = get_top_codes(rq, args.level, k=args.n_donors)

        # Track + render
        cells = [(baseline_pil, "baseline (no swap)")]
        for d in donor_ids:
            override = {region: {args.level: int(d)}}
            x_swap, _ = encode_quantize_decode(model, x11, x01, override=override)
            swap_pil = tensor_neg11_to_pil(x_swap)
            cells.append((swap_pil, f"L{args.level} ← code {d}"))

        header = (f"Region: {region}    swap level {args.level}    "
                  f"({baseline_info['n_positions'][region]} positions)")
        panel = make_panel(cells, header)
        panel_path = os.path.join(args.out_dir, f"swap_panel_{region}.png")
        panel.save(panel_path)
        print(f"  saved {panel_path}  (donors: {donor_ids})")

        summary["regions"][region] = {
            "donor_ids": donor_ids,
            "top_used_codes_at_this_level": get_top_codes(rq, args.level, k=16),
            "original_indices": baseline_info["indices_per_region"][region],
        }

    with open(os.path.join(args.out_dir, "swap_indices.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone -> {args.out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Code-swap reconstruction experiment for CRAFT Stage-1."
    )
    ap.add_argument("--stage1_ckpt", required=True,
                    help="Phase-D checkpoint (final.pt).")
    ap.add_argument("--parser_ckpt", required=True,
                    help="BiSeNet checkpoint (79999_iter.pth).")
    ap.add_argument("--image", required=True, help="Input face image.")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--regions", nargs="+",
                    default=["eyes", "lips", "hair", "skin"],
                    help="Which regions to swap codes in.")
    ap.add_argument("--level", type=int, default=0,
                    help="Which RQ level to swap (0 = coarsest, 2 = finest).")
    ap.add_argument("--n_donors", type=int, default=4,
                    help="How many donor code ids to try per region "
                         "(top-N most-used codes by default).")
    ap.add_argument("--donor_codes", type=str, default="",
                    help="Comma-separated explicit donor code ids "
                         "(overrides --n_donors).")

    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
