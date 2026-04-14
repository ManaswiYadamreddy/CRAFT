"""
heal_phase_a.py — Revive a Phase A checkpoint whose codebook has collapsed.

The Phase A training had a bug in GlobalVQ._ema_update where dead codes'
ema_weight multiplied by 0.95 per step, underflowed to 0 in fp32 after
~2000 non-selection steps, and F.normalize(0) = 0 baked permanently-dead
zero rows into the embedding. 74% of the 1024 codes ended up with norm 0.

The encoder, decoder, quant_conv, and post_quant_conv are all fine — only
the codebook is damaged. This script:

  1. Loads the old checkpoint.
  2. Runs a batch of HQ images through the encoder to collect real z vectors.
  3. Identifies "dead" codebook rows (norm < --dead_threshold or
     ema_count < 1.0) and re-seeds them with randomly sampled normalized z
     vectors from the collected pool.
  4. Resets their ema_count to the live-rows' mean, so the re-seeded codes
     compete fairly with existing ones.
  5. Saves a healed checkpoint to --out_ckpt.

After running this, resume Phase A training from the healed checkpoint
(the `train_stage1.py` resume path will pick up latest.pt automatically,
or point --hq_ckpt at the healed file) for ~10–15 more epochs. Much faster
than retraining from scratch.

Usage:
    python heal_phase_a.py \
        --in_ckpt checkpoints/phase_a/final.pt \
        --out_ckpt checkpoints/phase_a/healed.pt \
        --data_root data/train \
        --n_heal_batches 8 \
        --dead_threshold 0.5

    # Then copy over as latest.pt so train_stage1 picks it up on resume:
    cp checkpoints/phase_a/healed.pt checkpoints/phase_a/latest.pt
    # And re-run Phase A — it will resume from epoch 50 with a healthy
    # codebook and run for the remaining epochs (bump --hq_epochs if needed).
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import FFHQPairedDataset
from models.vqvae import build_hq_vqvae


@torch.no_grad()
def collect_encoder_features(model, loader, device, n_batches):
    """Collect a pool of normalized encoder features across several batches."""
    model.eval()
    pool = []
    seen = 0
    for batch in loader:
        hq = batch["hq"].to(device)
        z = model.encode(hq)  # (B, C, H, W)
        B, C, H, W = z.shape
        z_flat = z.float().permute(0, 2, 3, 1).reshape(-1, C)
        z_flat = F.normalize(z_flat, dim=1)
        pool.append(z_flat)
        seen += 1
        if seen >= n_batches:
            break
    return torch.cat(pool, dim=0)  # (N_total, C)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_ckpt", type=str,
                   default="checkpoints/phase_a/final.pt")
    p.add_argument("--out_ckpt", type=str,
                   default="checkpoints/phase_a/healed.pt")
    p.add_argument("--data_root", type=str, default="data/train")
    p.add_argument("--hq_n_codes", type=int, default=1024)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_heal_batches", type=int, default=8,
                   help="How many batches to collect encoder features from "
                        "before re-seeding dead codes.")
    p.add_argument("--dead_threshold", type=float, default=0.5,
                   help="Rows with weight norm below this are considered "
                        "dead and re-seeded (in addition to the ema_count<1 "
                        "criterion).")
    p.add_argument("--reset_optimizer", action="store_true", default=True,
                   help="Drop optimizer state from the checkpoint. "
                        "Recommended — the stale Adam moments reference the "
                        "old collapsed codebook.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model + checkpoint ---
    print(f"[heal] Loading checkpoint: {args.in_ckpt}")
    ckpt = torch.load(args.in_ckpt, map_location=device, weights_only=False)
    model = build_hq_vqvae(
        n_codes=args.hq_n_codes, embed_dim=args.embed_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    q = model.quantizer

    # --- Diagnose before ---
    w = q.embedding.weight.detach()
    ema = q.ema_count.detach()
    norms = w.norm(dim=-1)
    n_dead_norm = int((norms < args.dead_threshold).sum().item())
    n_dead_ema = int((ema < 1.0).sum().item())
    dead_mask = (norms < args.dead_threshold) | (ema < 1.0)
    n_dead = int(dead_mask.sum().item())
    n_alive = args.hq_n_codes - n_dead
    print(f"[heal] Before:")
    print(f"       codes              : {args.hq_n_codes}")
    print(f"       dead by norm<{args.dead_threshold} : {n_dead_norm}")
    print(f"       dead by ema<1.0    : {n_dead_ema}")
    print(f"       dead (union)       : {n_dead}")
    print(f"       alive              : {n_alive}")
    print(f"       norm mean (all)    : {norms.mean().item():.4f}")
    if n_alive > 0:
        print(f"       norm mean (alive)  : "
              f"{norms[~dead_mask].mean().item():.4f}")
        print(f"       ema_mean (alive)   : "
              f"{ema[~dead_mask].mean().item():.4f}")

    if n_dead == 0:
        print("[heal] Nothing to heal — codebook is already clean.")
        return

    # --- Collect encoder features ---
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    print(f"[heal] Collecting encoder features from "
          f"{args.n_heal_batches} batches...")
    pool = collect_encoder_features(model, loader, device, args.n_heal_batches)
    print(f"[heal] Pool size: {pool.shape[0]} vectors, dim {pool.shape[1]}")
    assert pool.shape[0] >= n_dead, (
        f"Not enough encoder features ({pool.shape[0]}) to replace "
        f"{n_dead} dead codes. Raise --n_heal_batches or --batch_size."
    )

    # --- Re-seed ---
    # Sample without replacement (avoids duplicate rows)
    perm = torch.randperm(pool.shape[0], device=device)[:n_dead]
    replacements = pool[perm]  # already unit-norm

    # Ema count: set to live-rows' mean so they compete fairly. If there are
    # no live rows (edge case), use 1.0.
    if n_alive > 0:
        target_ema = ema[~dead_mask].mean().item()
    else:
        target_ema = 1.0
    target_ema = max(target_ema, 1.0)  # floor

    with torch.no_grad():
        q.embedding.weight.data[dead_mask] = replacements
        q.ema_weight.data[dead_mask] = replacements
        q.ema_count[dead_mask] = target_ema

    # --- Diagnose after ---
    w = q.embedding.weight.detach()
    norms = w.norm(dim=-1)
    n_dead_after = int((norms < args.dead_threshold).sum().item())
    print(f"[heal] After:")
    print(f"       dead by norm<{args.dead_threshold} : {n_dead_after}")
    print(f"       norm mean (all)    : {norms.mean().item():.4f}")
    print(f"       norm std (all)     : {norms.std().item():.4f}")
    print(f"       norm min / max     : "
          f"{norms.min().item():.4f} / {norms.max().item():.4f}")
    print(f"       ema_count mean     : {q.ema_count.mean().item():.4f}")
    print(f"       reseeded target ema: {target_ema:.4f}")

    # --- Write out ---
    out = {
        "epoch": ckpt.get("epoch", 0),
        "model": model.state_dict(),
    }
    # Carry discriminator forward if present (it's still useful)
    if "discriminator" in ckpt:
        out["discriminator"] = ckpt["discriminator"]
    # Preserve history
    if "history" in ckpt:
        out["history"] = ckpt["history"]
    # Drop optimizer / scheduler / scaler — they reference the broken codebook
    if not args.reset_optimizer:
        for k in ("optimizer_g", "optimizer_d", "scheduler_g",
                  "scheduler_d", "scaler_g", "scaler_d"):
            if k in ckpt:
                out[k] = ckpt[k]
    else:
        print("[heal] Dropping optimizer / scheduler / scaler state "
              "(will be rebuilt on resume).")

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    torch.save(out, args.out_ckpt)
    print(f"[heal] Saved: {args.out_ckpt}")
    print()
    print("Next steps:")
    print(f"  1. cp {args.out_ckpt} "
          f"{os.path.join(os.path.dirname(args.out_ckpt), 'latest.pt')}")
    print(f"  2. Resume Phase A training — it will continue from the "
          f"healed codebook with the underflow-guarded EMA update.")
    print(f"  3. Run ~10–15 more epochs; the decoder needs to re-adapt to "
          f"the now-richer codebook. Watch `cb` logged in the training "
          f"printouts — it should stay ~1.0 and `perplexity` should climb.")
    print(f"  4. Re-run diagnose_phase_a.py on the resulting checkpoint: "
          f"expect PSNR > 28dB, dead_frac < 10%, perplexity > 500.")


if __name__ == "__main__":
    main()
