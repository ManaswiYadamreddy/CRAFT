"""
diagnose_phase_a.py — Sanity checks for the Phase A HQ VQVAE checkpoint.

Run this BEFORE starting Phase B to decide whether Phase A actually converged
or whether it has a latent bug (codebook collapse, norm mismatch, bottleneck
too tight). Four things are checked:

    1. Reconstruction quality on held-out HQ images. Saves a side-by-side grid
       (input | reconstruction | abs diff) to --out_dir.
    2. Codebook health: perplexity, fraction of codes used, dead-code count,
       mean / std of codebook entry norms.
    3. z_norm vs codebook_norm comparison — answers the 'cb: 0.3 vs z: 19.3'
       question. For the L2-normalized codebook used in GlobalVQ, codebook_norm
       should be ~1.0 and encoder output is scaled separately by codebook_scale.
       A gap here means the quantizer isn't functioning as designed.
    4. Loss curve shape over the last --last_n epochs, read from
       checkpoints/phase_a/latest.pt's stored avg_logs history if available.
       Tells you if training is still decreasing or plateaued.

Usage:
    python diagnose_phase_a.py \
        --hq_ckpt checkpoints/phase_a/final.pt \
        --data_root data/train \
        --out_dir diagnostics/phase_a \
        --n_images 8

Output:
    <out_dir>/reconstructions.png   - visual grid (HQ | recon | 5x diff)
    <out_dir>/report.txt            - text report with all numbers
    stdout                          - same report, printed live
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from data.dataset import FFHQPairedDataset
from models.vqvae import build_hq_vqvae


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _fmt(x, p=4):
    if isinstance(x, torch.Tensor):
        x = x.item()
    return f"{x:.{p}f}"


def _load_model(args, device):
    model = build_hq_vqvae(
        n_codes=args.hq_n_codes, embed_dim=args.embed_dim,
    ).to(device)
    ckpt = torch.load(args.hq_ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


# ----------------------------------------------------------------------
# Check 1: reconstructions
# ----------------------------------------------------------------------

def check_reconstructions(model, loader, device, out_dir, n_images, lines):
    lines.append("\n=== [1] Reconstruction quality ===")

    batch = next(iter(loader))
    hq = batch["hq"][:n_images].to(device)       # [-1, 1]
    hq_01 = batch["hq_01"][:n_images].to(device) # [0, 1]

    with torch.no_grad():
        x_rec, z, z_q, vq_losses, vq_info = model(hq, images_01=hq_01)

    rec_01 = (x_rec.clamp(-1, 1) + 1) / 2
    diff = (rec_01 - hq_01).abs()
    diff_vis = (diff * 5.0).clamp(0, 1)  # 5x amplification

    l1 = F.l1_loss(rec_01, hq_01).item()
    psnr = 10 * torch.log10(1.0 / (diff.pow(2).mean() + 1e-10)).item()

    lines.append(f"  n_images      : {n_images}")
    lines.append(f"  L1 [0,1]      : {_fmt(l1)}")
    lines.append(f"  PSNR (dB)     : {_fmt(psnr, 2)}")
    lines.append(f"  vq_commitment : {_fmt(vq_losses['commitment'])}")
    lines.append(f"  vq_entropy    : {_fmt(vq_losses['entropy'])}")
    if "perplexity" in vq_info:
        lines.append(f"  perplexity    : {_fmt(vq_info['perplexity'], 1)} "
                     f"(out of {model.quantizer.n_codes})")
    if "codebook_usage" in vq_info:
        lines.append(f"  codebook_used : "
                     f"{_fmt(vq_info['codebook_usage'] * 100, 1)}%")

    # Build a 3-row grid: hq | rec | 5x diff, all in [0,1]
    os.makedirs(out_dir, exist_ok=True)
    rows = torch.cat([hq_01, rec_01, diff_vis], dim=0)
    grid = make_grid(rows, nrow=n_images, padding=2)
    save_path = os.path.join(out_dir, "reconstructions.png")
    save_image(grid, save_path)
    lines.append(f"  saved grid    : {save_path}")
    lines.append("  (row 1: HQ input | row 2: reconstruction | "
                 "row 3: |diff|×5)")

    return z, z_q, vq_info


# ----------------------------------------------------------------------
# Check 2 + 3: codebook health and z_norm vs cb_norm
# ----------------------------------------------------------------------

def check_codebook(model, z, vq_info, lines):
    lines.append("\n=== [2] Codebook health ===")
    q = model.quantizer

    # Direct weight inspection (bypasses any logging confusion)
    w = q.embedding.weight.detach().float()        # (n_codes, e_dim)
    norms = w.norm(dim=-1)
    ema = q.ema_count.detach().float()

    lines.append(f"  n_codes               : {q.n_codes}")
    lines.append(f"  e_dim                 : {q.e_dim}")
    lines.append(f"  embedding.norm mean   : {_fmt(norms.mean(), 4)}")
    lines.append(f"  embedding.norm std    : {_fmt(norms.std(), 4)}")
    lines.append(f"  embedding.norm min    : {_fmt(norms.min(), 4)}")
    lines.append(f"  embedding.norm max    : {_fmt(norms.max(), 4)}")
    lines.append(f"  (expected ~1.0: GlobalVQ re-normalizes after every "
                 f"EMA update)")

    if hasattr(q, "codebook_scale"):
        lines.append(f"  codebook_scale (learn): "
                     f"{_fmt(q.codebook_scale.abs().item(), 3)}")

    dead_thresh = 1.0
    n_dead = int((ema < dead_thresh).sum().item())
    lines.append(f"  dead codes (ema<1.0)  : {n_dead}/{q.n_codes} "
                 f"({100 * n_dead / q.n_codes:.1f}%)")
    lines.append(f"  ema_count min/mean/max: "
                 f"{_fmt(ema.min(), 3)} / "
                 f"{_fmt(ema.mean(), 3)} / "
                 f"{_fmt(ema.max(), 1)}")

    lines.append("\n=== [3] z_norm vs codebook_norm ===")
    z_norms = z.float().norm(dim=1)  # (B, H, W)
    lines.append(f"  z.norm mean (pre-quant): {_fmt(z_norms.mean(), 3)}")
    lines.append(f"  z.norm std             : {_fmt(z_norms.std(), 3)}")
    lines.append(f"  codebook.norm mean     : {_fmt(norms.mean(), 4)}")
    ratio = z_norms.mean().item() / max(norms.mean().item(), 1e-6)
    lines.append(f"  ratio z / cb           : {_fmt(ratio, 2)}")
    lines.append(f"  (GlobalVQ L2-normalizes z before nearest-neighbor "
                 f"lookup, so a large ratio here is EXPECTED and the "
                 f"learned `codebook_scale` bridges the gap on the "
                 f"decoder side. If codebook.norm is drastically != 1.0, "
                 f"that's the real bug.)")


# ----------------------------------------------------------------------
# Check 4: loss curve shape from checkpoint history
# ----------------------------------------------------------------------

def check_loss_curve(ckpt, last_n, lines):
    lines.append(f"\n=== [4] Loss curve (last {last_n} epochs) ===")
    history = ckpt.get("history") or ckpt.get("avg_logs_history")
    if not history:
        lines.append("  no 'history' / 'avg_logs_history' field in the "
                     "checkpoint.")
        lines.append("  (If you want this, append `avg_logs` to a list in "
                     "run_phase_a and save it alongside the model.)")
        return

    recent = history[-last_n:]
    keys = ["l1", "perceptual", "vq", "total_gen"]
    lines.append(f"  epochs shown : "
                 f"{len(history) - len(recent)}..{len(history) - 1}")
    header = "  epoch  " + "  ".join(f"{k:>10}" for k in keys)
    lines.append(header)
    for i, row in enumerate(recent):
        ep = len(history) - len(recent) + i
        vals = "  ".join(
            f"{row.get(k, float('nan')):>10.4f}" for k in keys
        )
        lines.append(f"  {ep:5d}  {vals}")

    # Trend check: is the last half still decreasing?
    if len(recent) >= 4:
        mid = len(recent) // 2
        for k in keys:
            first = sum(r.get(k, 0) for r in recent[:mid]) / mid
            last = sum(r.get(k, 0) for r in recent[mid:]) / (len(recent) - mid)
            delta = last - first
            trend = ("decreasing" if delta < -1e-4
                     else "increasing" if delta > 1e-4
                     else "flat")
            lines.append(f"  {k:>10}: first_half={_fmt(first)} "
                         f"last_half={_fmt(last)} ({trend}, Δ={_fmt(delta)})")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hq_ckpt", type=str,
                   default="checkpoints/phase_a/final.pt")
    p.add_argument("--data_root", type=str, default="data/train")
    p.add_argument("--out_dir", type=str, default="diagnostics/phase_a")
    p.add_argument("--n_images", type=int, default=8)
    p.add_argument("--hq_n_codes", type=int, default=1024)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--last_n", type=int, default=10,
                   help="Number of recent epochs to summarize in check 4.")
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    lines = []
    lines.append(f"Phase A diagnostics")
    lines.append(f"  checkpoint : {args.hq_ckpt}")
    lines.append(f"  data_root  : {args.data_root}")
    lines.append(f"  device     : {device}")

    # Model + checkpoint
    model, ckpt = _load_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    lines.append(f"  params     : {n_params:,}")
    lines.append(f"  epoch      : {ckpt.get('epoch', 'unknown')}")

    # Held-out-ish batch (we just grab a shuffled batch from the same data;
    # strict held-out isn't critical for a sanity check)
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=True)
    loader = DataLoader(
        dataset, batch_size=args.n_images, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    z, z_q, vq_info = check_reconstructions(
        model, loader, device, args.out_dir, args.n_images, lines,
    )
    check_codebook(model, z, vq_info, lines)
    check_loss_curve(ckpt, args.last_n, lines)

    lines.append("\n" + "=" * 60)
    lines.append("Interpretation cheat-sheet:")
    lines.append("  - Reconstructions look sharp  → Phase A is fine.")
    lines.append("  - PSNR > 28dB, L1 < 0.05       → good reconstruction.")
    lines.append("  - dead codes > 20%             → partial collapse.")
    lines.append("  - perplexity << n_codes/4      → underused codebook.")
    lines.append("  - codebook.norm ~1.0           → normalization working.")
    lines.append("  - loss curve still decreasing  → train more.")
    lines.append("  - loss curve flat + bad recon  → fix quantizer, don't "
                 "just train longer.")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    report_path = os.path.join(args.out_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
