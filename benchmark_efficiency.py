"""
benchmark_efficiency.py — Step Time / Params / MACs for the CRAFT and OSDFace
final Stage 1 models.

Reports the three columns shown in the OSDFace paper's efficiency table:

    Step Time (s)  : average wall time of one full encode → quantize → decode
                     pass on a single 512×512 image.
    Param (M)      : total parameter count, in millions.
    MACs (G)       : multiply-accumulate operations per forward pass, in giga.
                     (1 MAC == 1 multiply + 1 add ≈ 2 FLOPs.)

Run from inside the CRAFT repo so `models.*` is importable:

    python benchmark_efficiency.py \\
        --craft_ckpt   /path/to/checkpoints/phase_c/final.pt \\
        --osdface_ckpt /path/to/checkpoints_osdface/phase_c/final.pt \\
        --parser_ckpt  /path/to/79999_iter.pth

Both --craft_ckpt and --osdface_ckpt are optional — if omitted, the model is
built with random weights, which gives identical Param / MACs numbers and a
representative Step Time. Use --device cpu to benchmark without a GPU.

Notes
-----
* For CRAFT we precompute the face-parser masks once and reuse them, so the
  parser cost is *excluded* from the inner-loop step time and MACs (mirrors
  how the paper reports it: the parser is an upstream stage, not part of the
  Stage 1 VQVAE itself). Pass --include_parser to fold it in instead.
* MACs are counted with `ptflops`. ptflops hooks standard nn modules (Conv2d,
  Linear, GroupNorm, …) and does NOT count raw tensor matmuls inside the
  attention blocks. This matches what every VQVAE/diffusion paper reports.
"""

import argparse
import time

import torch
import torch.nn as nn

from models.vqvae import build_hq_vqvae, build_lq_vqvae
from models.region_aware_vq import RegionAwareVQ


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_craft(ckpt_path, parser_ckpt, device, embed_dim=512, rq_levels=3):
    """CRAFT Stage 1 LQ branch: VQVAE + RegionAwareVQ (Phase C / Phase D)."""
    state, has_mag_head = None, False
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        key = "model" if "model" in ckpt else "lq_model"
        state = ckpt[key]
        has_mag_head = any("magnitude_head" in k for k in state.keys())
        if has_mag_head:
            print("  CRAFT: detected magnitude_head → Phase D checkpoint")

    ravq = RegionAwareVQ(
        e_dim=embed_dim,
        n_levels=rq_levels,
        parser_ckpt=parser_ckpt,
        use_magnitude_head=has_mag_head,
    )
    model = build_lq_vqvae(ravq, embed_dim=embed_dim)
    if state is not None:
        model.load_state_dict(state)
    return model.to(device).eval()


def load_osdface(ckpt_path, device, embed_dim=512, lq_n_codes=1024):
    """OSDFace Stage 1 LQ branch: VQVAE + flat GlobalVQ (Phase C)."""
    model = build_hq_vqvae(n_codes=lq_n_codes, embed_dim=embed_dim)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        key = "model" if "model" in ckpt else "lq_model"
        model.load_state_dict(ckpt[key])
    return model.to(device).eval()


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper so ptflops / timer see a single-tensor forward
# ──────────────────────────────────────────────────────────────────────────────

class SingleArgWrapper(nn.Module):
    """Bind images_01 / masks so forward(x) is a single-tensor call."""
    def __init__(self, model, images_01=None, masks=None):
        super().__init__()
        self.model = model
        self.images_01 = images_01
        self.masks = masks

    def forward(self, x):
        x_rec, *_ = self.model(x, images_01=self.images_01, masks=self.masks)
        return x_rec


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, input_shape):
    """Returns MACs as a Python int. Uses ptflops in flops-counter mode."""
    from ptflops import get_model_complexity_info
    with torch.no_grad():
        macs, _ = get_model_complexity_info(
            model,
            input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
            backend="pytorch",
        )
    return int(macs)


@torch.no_grad()
def time_step(model, x, n_warmup, n_iter, device):
    is_cuda = str(device).startswith("cuda")
    for _ in range(n_warmup):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    return (time.time() - t0) / n_iter


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft_ckpt",   default=None,
                    help="Path to CRAFT final.pt (Phase C/D). Optional.")
    ap.add_argument("--osdface_ckpt", default=None,
                    help="Path to OSDFace final.pt (Phase C). Optional.")
    ap.add_argument("--parser_ckpt",  default=None,
                    help="Path to BiSeNet 79999_iter.pth. Required for "
                         "--include_parser; otherwise random weights are fine "
                         "since masks are precomputed once.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--res",     type=int, default=512)
    ap.add_argument("--n_warmup", type=int, default=10)
    ap.add_argument("--n_iter",   type=int, default=50)
    ap.add_argument("--include_parser", action="store_true",
                    help="Run the face parser inside the CRAFT measurement "
                         "(default: precompute masks and exclude parser).")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}    Resolution: {args.res}×{args.res}")

    # ── Load both models ────────────────────────────────────────────────────
    print("\nLoading CRAFT...")
    craft = load_craft(args.craft_ckpt, args.parser_ckpt, device)
    print("Loading OSDFace...")
    osdface = load_osdface(args.osdface_ckpt, device)

    # ── Inputs ──────────────────────────────────────────────────────────────
    x_11 = torch.randn(1, 3, args.res, args.res, device=device)
    x_01 = (x_11.clamp(-1, 1) + 1) / 2

    if args.include_parser:
        masks = None
    else:
        with torch.no_grad():
            masks = craft.quantizer.face_parser.get_region_masks(
                x_01, target_h=args.res // 32, target_w=args.res // 32,
            )

    craft_w   = SingleArgWrapper(craft, images_01=x_01, masks=masks).to(device).eval()
    osdface_w = SingleArgWrapper(osdface).to(device).eval()

    # ── Step time ───────────────────────────────────────────────────────────
    print("\nTiming forward passes "
          f"({args.n_warmup} warmup + {args.n_iter} timed)...")
    t_craft   = time_step(craft_w,   x_11, args.n_warmup, args.n_iter, device)
    t_osdface = time_step(osdface_w, x_11, args.n_warmup, args.n_iter, device)

    # ── Params ──────────────────────────────────────────────────────────────
    # We exclude the (frozen) face parser from CRAFT's param count, since the
    # paper only reports parameters of the trainable Stage 1 VQVAE.
    p_craft_total = count_params(craft)
    p_parser      = count_params(craft.quantizer.face_parser)
    p_craft       = p_craft_total - p_parser
    p_osdface     = count_params(osdface)

    # ── MACs ────────────────────────────────────────────────────────────────
    print("Counting MACs (ptflops)...")
    in_shape = (3, args.res, args.res)
    macs_craft   = count_macs(craft_w,   in_shape)
    macs_osdface = count_macs(osdface_w, in_shape)

    # ── Report ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"{'Model':<10} {'Step Time (s)':>16} {'Param (M)':>14} {'MACs (G)':>14}")
    print("-" * 60)
    print(f"{'CRAFT':<10} {t_craft:>16.4f} "
          f"{p_craft/1e6:>14.2f} {macs_craft/1e9:>14.2f}")
    print(f"{'OSDFace':<10} {t_osdface:>16.4f} "
          f"{p_osdface/1e6:>14.2f} {macs_osdface/1e9:>14.2f}")
    print("=" * 60)
    if not args.include_parser:
        print(f"(CRAFT excludes face-parser cost: {p_parser/1e6:.2f} M params, "
              "masks precomputed before timing.)")


if __name__ == "__main__":
    main()
