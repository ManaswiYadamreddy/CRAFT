"""
code_swap_batch.py — Run the code-swap reconstruction experiment over a folder
of LQ face images (with optional matching HQ ground truth for the panel).

Self-contained — does not import from any other script in code_swap/ or any
other interpretability tool. Only depends on the CRAFT model code under
models/.

Pipeline (same as code_swap.py, applied to every image):

    encode → ResidualVQ.encode (3 levels of code ids per masked position)
           → for each region:  replace level-L ids with one donor code id
           → ResidualVQ.decode (look up new code vectors)
           → magnitude head + decoder → swapped reconstruction

For every image we save (under <out_dir>/<stem>/):

    original_lq.png            input  (resized to 512×512)
    original_hq.png            ground truth (only if --hq_dir given; same filename)
    recon_baseline.png         baseline reconstruction (no swap)
    swap_panel_<region>.png    HQ | LQ | baseline | donor_1 | … | donor_N
    swap_indices.json          donors used + original code-id grid per region

LQ ↔ HQ matching:
    Filenames are assumed identical between --input_dir and --hq_dir.
    No prefix or suffix stripping is applied.

Example:
    python code_swap/code_swap_batch.py \
        --stage1_ckpt /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
        --parser_ckpt /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
        --input_dir   /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2/ \
        --hq_dir      /projectnb/cs585/projects/craft/data/test/CelebA/CelebA_Validation/celeba_512_validation/ \
        --out_dir     /projectnb/cs585/projects/craft/code_swap_outputs/celeba_lq \
        --regions     eyes lips hair \
        --level       0 \
        --n_donors    4 \
        --max_images  20
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

# --- Repo root on path so `models.*` imports resolve ---
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.region_aware_vq import RegionAwareVQ
from models.face_parser import REGION_NAMES
from models.vqvae import build_lq_vqvae


IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def list_images(input_dir):
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(input_dir, ext))
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(input_dir, "*", ext))
    return sorted(set(paths))


def find_match(other_dir, filename):
    """Same filename in `other_dir` (no prefix/suffix mangling). None if absent."""
    if not other_dir:
        return None
    cand = os.path.join(other_dir, filename)
    return cand if os.path.isfile(cand) else None


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


def tensor_neg11_to_pil(x):
    x01 = ((x.clamp(-1, 1) + 1) / 2).float().squeeze(0).cpu()
    return transforms.ToPILImage()(x01)


def _load_font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_panel(images_with_labels, header, cell=320, label_h=26,
               header_h=44, gap=8):
    n = len(images_with_labels)
    W = n * cell + (n - 1) * gap
    H = header_h + label_h + cell
    panel = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    h_font = _load_font(20)
    l_font = _load_font(14)

    # Header
    try:
        bbox = draw.textbbox((0, 0), header, font=h_font)
        tw = bbox[2] - bbox[0]
    except AttributeError:
        tw = len(header) * 10
    draw.text(((W - tw) // 2, 12), header, fill=(40, 40, 40), font=h_font)

    for i, (img, lbl) in enumerate(images_with_labels):
        x = i * (cell + gap)
        try:
            bbox = draw.textbbox((0, 0), lbl, font=l_font)
            lw = bbox[2] - bbox[0]
        except AttributeError:
            lw = len(lbl) * 7
        draw.text((x + (cell - lw) // 2, header_h + 4),
                  lbl, fill=(60, 60, 60), font=l_font)
        panel.paste(img.resize((cell, cell), Image.BICUBIC),
                    (x, header_h + label_h))
    return panel


# ----------------------------------------------------------------------
# Core: encode → (optional swap) → decode
# ----------------------------------------------------------------------

@torch.no_grad()
def encode_quantize_decode(model, x_neg11, x_01, override=None, parser_x_01=None):
    """
    Run encode → per-region quantize (with optional index override) → decode.

    override     : dict | None
        region_name -> {level_idx: int_or_tensor}.  int = replace every
        position with that code id; tensor = per-position replacement.
    parser_x_01  : tensor | None
        Optional separate [0,1] image fed to the face parser. Defaults to
        x_01. Useful when the encoder input is LQ (so BiSeNet would fail)
        but you have a paired HQ image with valid region masks.

    Returns (x_rec, info).
    """
    quant = model.quantizer
    z = model.encode(x_neg11)
    B, C, H, W = z.shape
    parser_input = parser_x_01 if parser_x_01 is not None else x_01
    masks = quant.face_parser.get_region_masks(parser_input, target_h=H, target_w=W)

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
        region_features = z_flat[mask]
        idx_list = rq.encode(region_features)

        if override and name in override:
            for lvl, repl in override[name].items():
                if isinstance(repl, int):
                    idx_list[lvl] = torch.full_like(idx_list[lvl], repl)
                else:
                    idx_list[lvl] = repl.to(idx_list[lvl].device,
                                             dtype=idx_list[lvl].dtype)

        indices_per_region[name] = [i.detach().cpu().tolist() for i in idx_list]
        z_q_region = rq.decode(idx_list)
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
    """Top-k most-used code ids at this level (by EMA count)."""
    counts = rq.levels[level].ema_count.detach().cpu()
    return counts.argsort(descending=True)[:k].tolist()


# ----------------------------------------------------------------------
# Per-image driver
# ----------------------------------------------------------------------

@torch.no_grad()
def process_one(
    model, lq_path, out_root, args, hq_path=None, user_donor_ids=None,
):
    device = next(model.parameters()).device
    stem = os.path.splitext(os.path.basename(lq_path))[0]
    out_dir = os.path.join(out_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    # Resume support
    if args.skip_existing and os.path.isfile(
        os.path.join(out_dir, "swap_indices.json")
    ):
        return "skipped"

    # --- Load LQ ---
    lq_pil = Image.open(lq_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    x01 = to_tensor(lq_pil).unsqueeze(0).to(device)
    if x01.shape[-2:] != (args.resolution, args.resolution):
        x01 = F.interpolate(x01, size=(args.resolution, args.resolution),
                            mode="bilinear", align_corners=False)
    x11 = x01 * 2.0 - 1.0
    lq_disp = lq_pil.resize((args.resolution, args.resolution), Image.BICUBIC)
    lq_disp.save(os.path.join(out_dir, "original_lq.png"))

    # --- HQ (optional, for the panel and/or the parser) ---
    hq_disp = None
    parser_x01 = None
    if hq_path:
        hq_pil = Image.open(hq_path).convert("RGB")
        hq_disp = hq_pil.resize((args.resolution, args.resolution), Image.BICUBIC)
        hq_disp.save(os.path.join(out_dir, "original_hq.png"))
        if args.use_hq_for_parser:
            hq_t = to_tensor(hq_pil).unsqueeze(0).to(device)
            if hq_t.shape[-2:] != (args.resolution, args.resolution):
                hq_t = F.interpolate(hq_t, size=(args.resolution, args.resolution),
                                     mode="bilinear", align_corners=False)
            parser_x01 = hq_t

    # --- Baseline ---
    x_baseline, baseline_info = encode_quantize_decode(
        model, x11, x01, override=None, parser_x_01=parser_x01,
    )
    baseline_pil = tensor_neg11_to_pil(x_baseline)
    baseline_pil.save(os.path.join(out_dir, "recon_baseline.png"))

    # --- Per region: swap panels ---
    summary = {
        "lq_image": os.path.abspath(lq_path),
        "hq_image": os.path.abspath(hq_path) if hq_path else None,
        "level": args.level,
        "regions": {},
        "n_positions_per_region": baseline_info["n_positions"],
    }

    quant = model.quantizer
    for region in args.regions:
        if region not in quant.region_codebooks:
            continue
        if baseline_info["n_positions"].get(region, 0) == 0:
            continue

        rq = quant.region_codebooks[region]
        max_codes = rq.levels[args.level].n_codes
        if user_donor_ids is not None:
            donor_ids = [c for c in user_donor_ids if 0 <= c < max_codes]
        else:
            donor_ids = get_top_codes(rq, args.level, k=args.n_donors)

        cells = []
        if hq_disp is not None:
            cells.append((hq_disp, "HQ (ground truth)"))
        cells.append((lq_disp,       "LQ (input)"))
        cells.append((baseline_pil,  "baseline (no swap)"))

        for d in donor_ids:
            x_swap, _ = encode_quantize_decode(
                model, x11, x01,
                override={region: {args.level: int(d)}},
                parser_x_01=parser_x01,
            )
            cells.append((tensor_neg11_to_pil(x_swap),
                          f"L{args.level} ← code {d}"))

        header = (f"Region: {region}    swap level {args.level}    "
                  f"({baseline_info['n_positions'][region]} positions)")
        panel = make_panel(cells, header)
        panel.save(os.path.join(out_dir, f"swap_panel_{region}.png"))

        summary["regions"][region] = {
            "donor_ids": donor_ids,
            "top_used_codes_at_this_level": get_top_codes(rq, args.level, k=16),
            "original_indices": baseline_info["indices_per_region"][region],
        }

    with open(os.path.join(out_dir, "swap_indices.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return "done"


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

    paths = list_images(args.input_dir)
    if not paths:
        raise FileNotFoundError(f"No images in {args.input_dir}")
    if args.max_images > 0:
        paths = paths[:args.max_images]
    print(f"Found {len(paths)} images.")

    user_donor_ids = (
        [int(x) for x in args.donor_codes.split(",") if x.strip()]
        if args.donor_codes else None
    )

    os.makedirs(args.out_dir, exist_ok=True)
    n_done, n_skipped, n_missing_hq = 0, 0, 0

    for path in tqdm(paths, desc="code-swap"):
        fname = os.path.basename(path)
        hq_path = find_match(args.hq_dir, fname) if args.hq_dir else None
        if args.hq_dir and hq_path is None:
            n_missing_hq += 1
        status = process_one(
            model, path, args.out_dir, args,
            hq_path=hq_path, user_donor_ids=user_donor_ids,
        )
        if status == "skipped":
            n_skipped += 1
        else:
            n_done += 1

    print(f"\nDone: {n_done} processed, {n_skipped} skipped (already existed)")
    if args.hq_dir and n_missing_hq:
        print(f"  [warn] {n_missing_hq} images had no HQ counterpart in {args.hq_dir}")
    print(f"Outputs under: {args.out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Batch code-swap reconstruction over a folder of LQ images."
    )
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--parser_ckpt", required=True)
    ap.add_argument("--input_dir",   required=True,
                    help="Folder of LQ images.")
    ap.add_argument("--hq_dir",      default="",
                    help="Optional folder of HQ ground-truth images "
                         "(same filenames as in --input_dir, no prefix/suffix).")
    ap.add_argument("--out_dir",     required=True,
                    help="Each image gets its own subfolder under this root.")

    ap.add_argument("--regions", nargs="+",
                    default=["eyes", "lips", "hair", "skin"])
    ap.add_argument("--level",     type=int, default=0)
    ap.add_argument("--n_donors",  type=int, default=4,
                    help="Top-N most-used codes (by EMA count) at the chosen "
                         "level are used as donors (per region).")
    ap.add_argument("--donor_codes", type=str, default="",
                    help="Comma-separated explicit donor ids (overrides --n_donors).")

    ap.add_argument("--max_images",     type=int, default=0,
                    help="Cap how many images to process (0 = no cap).")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip images whose output folder already has swap_indices.json.")
    ap.add_argument("--use_hq_for_parser", action="store_true",
                    help="Feed the HQ image (from --hq_dir) to the face parser "
                         "while the encoder still sees the LQ image. Recommended "
                         "for LQ inputs where BiSeNet fails. Requires --hq_dir.")

    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--device",     default="cuda")

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
