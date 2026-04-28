"""
code_swap_batch_stage2.py — Code-swap reconstruction experiment over a folder
of LQ face images, using the CRAFT Stage-2 diffusion generator.

This script does ONE thing: for each face it produces a panel that compares
the Stage-2 baseline reconstruction with several reconstructions where the
**entire visual prompt** has been overwritten — every region's codes at
every RQ level are replaced with one donor code id, then the rest of
Stage-2 (prompt projector → SD UNet + LoRA → one-step denoise → VAE
decode) runs as usual.

(Per-region / per-level swaps were tried and produce no visible change
in Stage-2 outputs — the diffusion path leans on the LQ latent z_L for
geometry. Only a near-total prompt swap moves the output, so that's the
only mode this script supports.)

Self-contained — only depends on `models/`. LQ ↔ HQ filenames are assumed
identical between --input_dir and --hq_dir.

Output layout (per image, under <out_dir>/<stem>/):
    original_lq.png
    original_hq.png            (only if --hq_dir given)
    recon_baseline.png         Stage-2 reconstruction with no swap
    swap_panel_all_regions.png HQ | LQ | Stage-2 baseline | donor_1 | …
    swap_indices.json          donors used + original code-id grid per region

Example:
    python code_swap/code_swap_batch_stage2.py \
        --stage1_ckpt      /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
        --stage2_ckpt      /projectnb/cs585/projects/craft/checkpoints_stage2/final.pt \
        --parser_ckpt      /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
        --pretrained_model /projectnb/cs585/projects/craft/pretrained/stable-diffusion-v1-5 \
        --context_dim      768 \
        --input_dir        /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2/ \
        --hq_dir           /projectnb/cs585/projects/craft/data/test/CelebA/CelebA_Validation/celeba_512_validation/ \
        --out_dir          /projectnb/cs585/projects/craft/code_swap_outputs/celeba_lq_stage2_allregions \
        --regions          eyes lips hair skin \
        --n_donors         4 \
        --max_images       10 \
        --use_hq_for_parser \
        --skip_existing
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

# --- Repo root on path so models.* imports resolve ---
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.face_parser import REGION_NAMES
from models.stage2_generator import Stage2Generator


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
    if not other_dir:
        return None
    cand = os.path.join(other_dir, filename)
    return cand if os.path.isfile(cand) else None


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
# Stage-2 generator: load + apply trained checkpoint
# ----------------------------------------------------------------------

def load_stage2_generator(args, device, dtype):
    print(f"Building Stage2Generator (SD = {args.pretrained_model})")
    gen = Stage2Generator(
        stage1_ckpt_path=args.stage1_ckpt,
        parser_ckpt=args.parser_ckpt,
        sd_model_name=args.pretrained_model,
        t_fixed=args.t_fixed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        prompt_dim=args.embed_dim,
        context_dim=args.context_dim,
    )
    print(f"  {gen.describe()}")

    print(f"Loading Stage-2 checkpoint from {args.stage2_ckpt}")
    ckpt = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
    gen_state = ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt

    missing, unexpected = gen.load_state_dict(gen_state, strict=False)
    trainable_missing = [
        k for k in missing
        if "lora" in k.lower() or k.startswith(("prompt_proj", "prompt_ln"))
    ]
    if trainable_missing:
        raise RuntimeError(
            f"Stage-2 ckpt is missing {len(trainable_missing)} trainable keys, "
            f"e.g. {trainable_missing[:3]}"
        )
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys "
              f"(first 3): {unexpected[:3]}")
    print(f"  loaded {len(gen_state)} tensors (iter={ckpt.get('iter', '?')})")

    gen.to(device=device, dtype=dtype)
    gen.eval()
    return gen


# ----------------------------------------------------------------------
# Manual prompt construction with code-swap support
# ----------------------------------------------------------------------

@torch.no_grad()
def build_prompt_with_swap(gen, I_L_11, I_L_01, override=None, parser_x_01=None):
    """
    Replicate `_Stage1VRE.forward` but with optional per-region per-level
    code-id overrides, returning the visual prompt p_L (B, 256, 512).
    """
    vqvae = gen.stage1.vqvae
    quant = vqvae.quantizer

    z = vqvae.encode(I_L_11)                                 # (B, 512, 16, 16)
    B, C, H, W = z.shape
    parser_input = parser_x_01 if parser_x_01 is not None else I_L_01
    masks = quant.face_parser.get_region_masks(
        parser_input, target_h=H, target_w=W,
    )

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

    p_L = z_q.permute(0, 2, 3, 1).reshape(B, H * W, C)
    n_positions = {k: int(v.sum().item()) for k, v in masks_flat.items()}
    return p_L, indices_per_region, n_positions


@torch.no_grad()
def stage2_reconstruct_from_prompt(gen, I_L_11, p_L):
    """One-step diffusion forward, given a precomputed visual prompt p_L."""
    p_L_proj = gen.project_prompt(p_L)                       # (B, 256, 1024)
    z_L      = gen.encode_vae(I_L_11)                        # (B, 4, 64, 64)
    B = I_L_11.shape[0]
    t = torch.full((B,), gen.t_fixed, device=I_L_11.device, dtype=torch.long)
    eps_pred = gen.unet(
        sample=z_L,
        timestep=t,
        encoder_hidden_states=p_L_proj,
    ).sample
    z_hat_H = gen.one_step_denoise(z_L, eps_pred)
    return gen.decode_vae(z_hat_H)                           # (B, 3, 512, 512)


def get_top_codes(rq, level, k=8):
    counts = rq.levels[level].ema_count.detach().cpu()
    return counts.argsort(descending=True)[:k].tolist()


# ----------------------------------------------------------------------
# Per-image driver
# ----------------------------------------------------------------------

@torch.no_grad()
def process_one(gen, lq_path, out_root, args, hq_path=None,
                user_donor_ids=None, dtype=torch.float16):
    device = next(gen.parameters()).device
    stem = os.path.splitext(os.path.basename(lq_path))[0]
    out_dir = os.path.join(out_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    if args.skip_existing and os.path.isfile(
        os.path.join(out_dir, "swap_indices.json")
    ):
        return "skipped"

    # --- Load LQ ---
    lq_pil = Image.open(lq_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    x01 = to_tensor(lq_pil).unsqueeze(0).to(device=device, dtype=dtype)
    if x01.shape[-2:] != (args.resolution, args.resolution):
        x01 = F.interpolate(x01, size=(args.resolution, args.resolution),
                            mode="bilinear", align_corners=False)
    x11 = x01 * 2.0 - 1.0
    lq_disp = lq_pil.resize((args.resolution, args.resolution), Image.BICUBIC)
    lq_disp.save(os.path.join(out_dir, "original_lq.png"))

    # --- HQ (optional, used for the panel and optionally the parser) ---
    hq_disp = None
    parser_x01 = None
    if hq_path:
        hq_pil = Image.open(hq_path).convert("RGB")
        hq_disp = hq_pil.resize((args.resolution, args.resolution), Image.BICUBIC)
        hq_disp.save(os.path.join(out_dir, "original_hq.png"))
        if args.use_hq_for_parser:
            hq_t = to_tensor(hq_pil).unsqueeze(0).to(device=device, dtype=dtype)
            if hq_t.shape[-2:] != (args.resolution, args.resolution):
                hq_t = F.interpolate(hq_t, size=(args.resolution, args.resolution),
                                     mode="bilinear", align_corners=False)
            parser_x01 = hq_t

    # --- Baseline (Stage-2, no swap) ---
    p_L_base, baseline_indices, n_positions = build_prompt_with_swap(
        gen, x11, x01, override=None, parser_x_01=parser_x01,
    )
    x_baseline = stage2_reconstruct_from_prompt(gen, x11, p_L_base)
    baseline_pil = tensor_neg11_to_pil(x_baseline)
    baseline_pil.save(os.path.join(out_dir, "recon_baseline.png"))

    quant = gen.stage1.vqvae.quantizer
    active_regions = [
        r for r in args.regions
        if r in quant.region_codebooks and n_positions.get(r, 0) > 0
    ]
    if not active_regions:
        # Fall back to whatever regions actually have positions on this face.
        active_regions = [
            r for r in REGION_NAMES
            if r in quant.region_codebooks and n_positions.get(r, 0) > 0
        ]

    # Donor ids are picked from the first active region's level-0 codebook.
    # When the swap is applied to a region, the same donor id is used at
    # every RQ level (mod that level's codebook size, in case sizes differ).
    anchor_rq = quant.region_codebooks[active_regions[0]]
    if user_donor_ids is not None:
        donor_ids = [c for c in user_donor_ids
                     if 0 <= c < anchor_rq.levels[0].n_codes]
    else:
        donor_ids = get_top_codes(anchor_rq, 0, k=args.n_donors)

    # ---------------- Panel 1: ALL regions, all RQ levels swapped ----------------
    cells = []
    if hq_disp is not None:
        cells.append((hq_disp, "HQ (ground truth)"))
    cells.append((lq_disp,       "LQ (input)"))
    cells.append((baseline_pil,  "Stage-2 baseline"))

    for d in donor_ids:
        override = {
            r: {
                lvl: int(d) % quant.region_codebooks[r].levels[lvl].n_codes
                for lvl in range(quant.region_codebooks[r].n_levels)
            }
            for r in active_regions
        }
        p_L_swap, _, _ = build_prompt_with_swap(
            gen, x11, x01, override=override, parser_x_01=parser_x01,
        )
        x_swap = stage2_reconstruct_from_prompt(gen, x11, p_L_swap)
        cells.append((tensor_neg11_to_pil(x_swap),
                      f"all regions · all levels ← code {d}"))

    total_pos = sum(n_positions.get(r, 0) for r in active_regions)
    header = (
        f"all regions · all RQ levels swapped    "
        f"({total_pos} positions across {', '.join(active_regions)})    "
        f"[Stage-2 reconstruction]"
    )
    panel = make_panel(cells, header)
    panel.save(os.path.join(out_dir, "swap_panel_all_regions.png"))

    # ---------------- Panel 2..N: one region at a time, all RQ levels swapped ----
    # Keeps other regions' codes original; only this region's three levels move.
    # Same donor ids reused so columns are directly comparable across panels.
    for region in active_regions:
        rq_target = quant.region_codebooks[region]
        cells_r = []
        if hq_disp is not None:
            cells_r.append((hq_disp, "HQ (ground truth)"))
        cells_r.append((lq_disp,       "LQ (input)"))
        cells_r.append((baseline_pil,  "Stage-2 baseline"))

        for d in donor_ids:
            override = {
                region: {
                    lvl: int(d) % rq_target.levels[lvl].n_codes
                    for lvl in range(rq_target.n_levels)
                }
            }
            p_L_swap, _, _ = build_prompt_with_swap(
                gen, x11, x01, override=override, parser_x_01=parser_x01,
            )
            x_swap = stage2_reconstruct_from_prompt(gen, x11, p_L_swap)
            cells_r.append((tensor_neg11_to_pil(x_swap),
                            f"{region} · all levels ← code {d}"))

        header_r = (
            f"region: {region}    all RQ levels swapped    "
            f"({n_positions[region]} positions)    "
            f"[other regions intact · Stage-2 reconstruction]"
        )
        panel_r = make_panel(cells_r, header_r)
        panel_r.save(os.path.join(out_dir, f"swap_panel_{region}.png"))

    summary = {
        "lq_image": os.path.abspath(lq_path),
        "hq_image": os.path.abspath(hq_path) if hq_path else None,
        "stage2_ckpt": os.path.abspath(args.stage2_ckpt),
        "regions_swapped": active_regions,
        "donor_ids": donor_ids,
        "n_positions_per_region": n_positions,
        "original_indices": {r: baseline_indices[r] for r in active_regions
                             if r in baseline_indices},
        "panels_written": (
            ["swap_panel_all_regions.png"]
            + [f"swap_panel_{r}.png" for r in active_regions]
        ),
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
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    print(f"Device: {device}    dtype: {dtype}")

    gen = load_stage2_generator(args, device, dtype)

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
    for path in tqdm(paths, desc="stage-2 swap"):
        fname = os.path.basename(path)
        hq_path = find_match(args.hq_dir, fname) if args.hq_dir else None
        if args.hq_dir and hq_path is None:
            n_missing_hq += 1
        status = process_one(
            gen, path, args.out_dir, args,
            hq_path=hq_path, user_donor_ids=user_donor_ids, dtype=dtype,
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
        description=("Batch code-swap experiment using the Stage-2 diffusion "
                     "generator. Always swaps every region's every RQ level "
                     "with one donor code id (the only mode that produces "
                     "visible Stage-2 differences)."),
    )

    # --- Stage-1 (frozen tokenizer inside Stage-2) ---
    ap.add_argument("--stage1_ckpt", required=True,
                    help="CRAFT Stage-1 checkpoint (phase_d/final.pt).")
    ap.add_argument("--parser_ckpt", required=True,
                    help="BiSeNet face-parser checkpoint (79999_iter.pth).")

    # --- Stage-2 ---
    ap.add_argument("--stage2_ckpt", required=True,
                    help="train_stage2.py final.pt (the bundle dict).")
    ap.add_argument("--pretrained_model",
                    default="sd-legacy/stable-diffusion-v1-5",
                    help="Local SD directory or HF id.")
    ap.add_argument("--mixed_precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--lora_rank",  type=int,   default=16)
    ap.add_argument("--lora_alpha", type=float, default=16)
    ap.add_argument("--t_fixed",    type=int,   default=999)
    ap.add_argument("--embed_dim",  type=int,   default=512,
                    help="Stage-1 visual-prompt channel count.")
    ap.add_argument("--context_dim", type=int,  default=768,
                    help="UNet cross-attention dim "
                         "(768 for SD 1.5, 1024 for SD 2.x).")

    # --- Data ---
    ap.add_argument("--input_dir", required=True,
                    help="Folder of LQ images.")
    ap.add_argument("--hq_dir",    default="",
                    help="Optional folder of HQ images "
                         "(same filenames as in --input_dir).")
    ap.add_argument("--out_dir",   required=True,
                    help="Each image gets its own subfolder under this root.")

    # --- Swap config (kept minimal — only the all-regions, all-levels mode) ---
    ap.add_argument("--regions", nargs="+",
                    default=["eyes", "lips", "hair", "skin"],
                    help="Regions to include in the swap. The donor id is "
                         "applied to every region in this list at every "
                         "RQ level (mod each codebook's size).")
    ap.add_argument("--n_donors", type=int, default=4,
                    help="Top-N most-used codes (by EMA count) at level 0 "
                         "of the first listed region used as donor ids.")
    ap.add_argument("--donor_codes", type=str, default="",
                    help="Comma-separated explicit donor ids (overrides --n_donors).")
    ap.add_argument("--use_hq_for_parser", action="store_true",
                    help="Feed HQ (from --hq_dir) to the face parser while "
                         "the encoder still sees LQ. Recommended for LQ inputs.")

    # --- Run config ---
    ap.add_argument("--max_images",     type=int, default=0,
                    help="Cap how many images to process (0 = no cap).")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--resolution",    type=int, default=512)
    ap.add_argument("--device",        default="cuda")

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
