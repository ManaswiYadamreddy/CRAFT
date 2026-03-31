"""
train_stage1.py — CRAFT Stage 1: Hierarchical Region-Aware VQVAE training.

Three training phases:
    Phase A  (HQ, 50 epochs):  Train HQ encoder + global VQ dict + HQ decoder
                                via self-reconstruction on HQ images.
    Phase B  (LQ, 10 epochs):  Freeze HQ branch. Train LQ encoder + region-aware
                                RQ-VAE + LQ decoder via self-reconstruction on LQ
                                images. λ_assoc = 0.
    Phase C  (LQ, 10 epochs):  Same as Phase B but enable HQ-LQ feature association
                                loss. λ_assoc = 1.

Usage:
    # Phase A: Train HQ VQVAE
    python train_stage1.py --phase A --data_root data/train --batch_size 32

    # Phase B: Train LQ VQVAE (requires Phase A checkpoint)
    python train_stage1.py --phase B --data_root data/train --batch_size 32 \
        --hq_ckpt checkpoints/phase_a/final.pt

    # Phase C: Continue LQ with association loss (requires Phase A + B checkpoints)
    python train_stage1.py --phase C --data_root data/train --batch_size 32 \
        --hq_ckpt checkpoints/phase_a/final.pt \
        --lq_ckpt checkpoints/phase_b/final.pt

    # Run all phases sequentially
    python train_stage1.py --phase all --data_root data/train --batch_size 32

Hyperparameters (from OSDFace):
    Optimizer:  Adam, lr=1.44e-4
    Batch size: 32
    Resolution: 512×512
    β = 0.25, λ_per = 1.0, λ_dis = 0.8, λ_assoc = 0→1
    HQ codebook: 1024 codes, dim 512
    LQ codebooks: eyes=128, skin=256, hair=512, lips=64 (per level, 3 levels)
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FFHQPairedDataset
from vqvae import VQVAE, GlobalVQ, build_hq_vqvae, build_lq_vqvae
from region_aware_vq import RegionAwareVQ
from losses import Stage1VQLoss


# ======================================================================
# Training loop
# ======================================================================

def train_one_epoch(
    model,
    criterion,
    loader,
    optimizer_g,
    optimizer_d,
    device,
    epoch,
    phase,
    hq_model=None,
    expire_every=1000,
    log_every=50,
):
    """
    Train for one epoch with alternating generator/discriminator updates.

    Args:
        model:       VQVAE being trained (HQ or LQ).
        criterion:   Stage1VQLoss instance.
        loader:      DataLoader yielding batches.
        optimizer_g: Optimizer for generator (encoder + decoder).
        optimizer_d: Optimizer for discriminator.
        device:      Torch device.
        epoch:       Current epoch number (for logging).
        phase:       'A', 'B', or 'C'.
        hq_model:    Frozen HQ VQVAE (Phase C only, for association loss).
        expire_every: Run dead code expiry every N steps.
        log_every:   Print logs every N steps.

    Returns:
        avg_logs: dict of average loss values over the epoch.
    """
    model.train()
    criterion.discriminator.train()

    accum = defaultdict(float)
    n_steps = 0

    for step, batch in enumerate(loader):
        # ----- Prepare inputs -----
        if phase == "A":
            # HQ self-reconstruction
            x = batch["hq"].to(device)
            x_01 = batch["hq_01"].to(device)
            target = x
        else:
            # LQ self-reconstruction
            x = batch["lq"].to(device)
            x_01 = batch["hq_01"].to(device)  # use HQ for face parsing (better masks)
            target = x

        # ----- Generator step -----
        optimizer_g.zero_grad()

        x_rec, z, z_q, vq_losses, vq_info = model(x, images_01=x_01)

        # Association features (Phase C only)
        z_H_flat, z_L_flat = None, None
        if phase == "C" and hq_model is not None:
            with torch.no_grad():
                z_H_flat = hq_model.get_features_flat(batch["hq"].to(device))
            B, C, H, W = z.shape
            z_L_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C)

        gen_loss, gen_logs = criterion.generator_loss(
            x_rec, target, vq_losses["total_vq"],
            z_H=z_H_flat, z_L=z_L_flat,
        )
        gen_loss.backward()
        optimizer_g.step()

        # ----- Discriminator step -----
        optimizer_d.zero_grad()
        d_loss, d_logs = criterion.discriminator_loss(target, x_rec.detach())
        d_loss.backward()
        optimizer_d.step()

        # ----- Dead code expiry -----
        if (step + 1) % expire_every == 0 and hasattr(model.quantizer, "expire_dead_codes"):
            with torch.no_grad():
                z_for_expire = model.encode(x)
                expired = model.quantizer.expire_dead_codes(z_for_expire, images=x_01)
                if isinstance(expired, dict):
                    total_expired = sum(expired.values())
                else:
                    total_expired = expired
                if total_expired > 0:
                    print(f"    [Step {step+1}] Expired {total_expired} dead codes")

        # ----- Accumulate logs -----
        all_logs = {**gen_logs, **d_logs}
        for k, v in all_logs.items():
            accum[k] += v.item() if torch.is_tensor(v) else v
        n_steps += 1

        # ----- Print -----
        if (step + 1) % log_every == 0:
            msg = f"  Phase {phase} | Epoch {epoch} | Step {step+1}/{len(loader)}"
            for k in ["total_gen", "l1", "perceptual", "gan_g", "vq", "gan_d"]:
                if k in accum:
                    msg += f" | {k}: {accum[k]/n_steps:.4f}"
            if "association" in accum:
                msg += f" | assoc: {accum['association']/n_steps:.4f}"
            print(msg)

    # Average logs
    avg_logs = {k: v / max(n_steps, 1) for k, v in accum.items()}
    return avg_logs


# ======================================================================
# Phase runners
# ======================================================================

def run_phase_a(args, device):
    """Phase A: Train HQ VQVAE (self-reconstruction, 50 epochs)."""
    print("=" * 60)
    print("Phase A: Training HQ VQVAE")
    print("=" * 60)

    # Dataset (HQ only)
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    print(f"Dataset: {len(dataset)} HQ images, {len(loader)} batches/epoch")

    # Model
    model = build_hq_vqvae(
        n_codes=args.hq_n_codes, embed_dim=args.embed_dim,
    ).to(device)
    print(f"HQ VQVAE params: {sum(p.numel() for p in model.parameters()):,}")

    # Losses
    criterion = Stage1VQLoss(
        lambda_per=args.lambda_per, lambda_dis=args.lambda_dis, lambda_assoc=0.0,
        vgg_pretrained=True,
    ).to(device)

    # Optimizers (separate for generator and discriminator)
    gen_params = list(model.parameters())
    optimizer_g = torch.optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(criterion.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Resume
    start_epoch = 0
    ckpt_dir = os.path.join(args.ckpt_dir, "phase_a")
    os.makedirs(ckpt_dir, exist_ok=True)

    if not args.fresh and os.path.exists(os.path.join(ckpt_dir, "latest.pt")):
        ckpt = torch.load(os.path.join(ckpt_dir, "latest.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        criterion.discriminator.load_state_dict(ckpt["discriminator"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.hq_epochs):
        t0 = time.time()
        avg_logs = train_one_epoch(
            model, criterion, loader, optimizer_g, optimizer_d,
            device, epoch, phase="A",
            expire_every=args.expire_every, log_every=args.log_every,
        )
        elapsed = time.time() - t0
        print(f"Phase A | Epoch {epoch} done in {elapsed:.0f}s | "
              f"L1={avg_logs.get('l1',0):.4f} VQ={avg_logs.get('vq',0):.4f}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "discriminator": criterion.discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    # Save final
    torch.save(ckpt, os.path.join(ckpt_dir, "final.pt"))
    print(f"Phase A complete. Saved to {ckpt_dir}/final.pt")

    return model


def run_phase_b(args, device, hq_model=None):
    """Phase B: Train LQ Region-Aware VQVAE (10 epochs, λ_assoc=0)."""
    print("=" * 60)
    print("Phase B: Training LQ Region-Aware VQVAE")
    print("=" * 60)

    # Dataset (paired HQ/LQ)
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    print(f"Dataset: {len(dataset)} HQ/LQ pairs, {len(loader)} batches/epoch")

    # Model
    ravq = RegionAwareVQ(
        e_dim=args.embed_dim,
        n_levels=args.rq_levels,
        parser_ckpt=args.parser_ckpt,
    ).to(device)
    model = build_lq_vqvae(ravq, embed_dim=args.embed_dim).to(device)
    print(f"LQ VQVAE params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable (excl. parser): "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Losses
    criterion = Stage1VQLoss(
        lambda_per=args.lambda_per, lambda_dis=args.lambda_dis, lambda_assoc=0.0,
        vgg_pretrained=True,
    ).to(device)

    # Optimizers
    gen_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_g = torch.optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(criterion.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Resume
    start_epoch = 0
    ckpt_dir = os.path.join(args.ckpt_dir, "phase_b")
    os.makedirs(ckpt_dir, exist_ok=True)

    if not args.fresh and os.path.exists(os.path.join(ckpt_dir, "latest.pt")):
        ckpt = torch.load(os.path.join(ckpt_dir, "latest.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        criterion.discriminator.load_state_dict(ckpt["discriminator"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.lq_epochs):
        t0 = time.time()
        avg_logs = train_one_epoch(
            model, criterion, loader, optimizer_g, optimizer_d,
            device, epoch, phase="B",
            expire_every=args.expire_every, log_every=args.log_every,
        )
        elapsed = time.time() - t0
        print(f"Phase B | Epoch {epoch} done in {elapsed:.0f}s | "
              f"L1={avg_logs.get('l1',0):.4f} VQ={avg_logs.get('vq',0):.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "discriminator": criterion.discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    torch.save(ckpt, os.path.join(ckpt_dir, "final.pt"))
    print(f"Phase B complete. Saved to {ckpt_dir}/final.pt")

    return model


def run_phase_c(args, device, hq_model=None, lq_model=None):
    """Phase C: Continue LQ training with association loss (10 epochs, λ_assoc=1)."""
    print("=" * 60)
    print("Phase C: LQ VQVAE + Association Loss")
    print("=" * 60)

    # Load HQ model (frozen)
    if hq_model is None:
        if not args.hq_ckpt:
            raise ValueError("Phase C requires --hq_ckpt (Phase A checkpoint)")
        print(f"Loading HQ model from {args.hq_ckpt}")
        hq_model = build_hq_vqvae(
            n_codes=args.hq_n_codes, embed_dim=args.embed_dim,
        ).to(device)
        ckpt = torch.load(args.hq_ckpt, map_location=device, weights_only=False)
        hq_model.load_state_dict(ckpt["model"])

    hq_model.eval()
    for p in hq_model.parameters():
        p.requires_grad_(False)
    print("HQ model frozen.")

    # Load LQ model (continuing training)
    if lq_model is None:
        if not args.lq_ckpt:
            raise ValueError("Phase C requires --lq_ckpt (Phase B checkpoint)")
        print(f"Loading LQ model from {args.lq_ckpt}")
        ravq = RegionAwareVQ(
            e_dim=args.embed_dim,
            n_levels=args.rq_levels,
            parser_ckpt=args.parser_ckpt,
        ).to(device)
        lq_model = build_lq_vqvae(ravq, embed_dim=args.embed_dim).to(device)
        ckpt = torch.load(args.lq_ckpt, map_location=device, weights_only=False)
        lq_model.load_state_dict(ckpt["model"])

    model = lq_model

    # Dataset
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    print(f"Dataset: {len(dataset)} HQ/LQ pairs, {len(loader)} batches/epoch")

    # Losses (association enabled)
    criterion = Stage1VQLoss(
        lambda_per=args.lambda_per, lambda_dis=args.lambda_dis,
        lambda_assoc=args.lambda_assoc,
        vgg_pretrained=True,
    ).to(device)

    # Optimizers
    gen_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_g = torch.optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(criterion.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Resume
    start_epoch = 0
    ckpt_dir = os.path.join(args.ckpt_dir, "phase_c")
    os.makedirs(ckpt_dir, exist_ok=True)

    if not args.fresh and os.path.exists(os.path.join(ckpt_dir, "latest.pt")):
        ckpt = torch.load(os.path.join(ckpt_dir, "latest.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        criterion.discriminator.load_state_dict(ckpt["discriminator"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.assoc_epochs):
        t0 = time.time()
        avg_logs = train_one_epoch(
            model, criterion, loader, optimizer_g, optimizer_d,
            device, epoch, phase="C", hq_model=hq_model,
            expire_every=args.expire_every, log_every=args.log_every,
        )
        elapsed = time.time() - t0
        print(f"Phase C | Epoch {epoch} done in {elapsed:.0f}s | "
              f"L1={avg_logs.get('l1',0):.4f} VQ={avg_logs.get('vq',0):.4f} "
              f"assoc={avg_logs.get('association',0):.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "discriminator": criterion.discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    torch.save(ckpt, os.path.join(ckpt_dir, "final.pt"))
    print(f"Phase C complete. Saved to {ckpt_dir}/final.pt")

    return model


# ======================================================================
# Main
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="CRAFT Stage 1 Training")

    # Phase selection
    parser.add_argument("--phase", type=str, default="all",
                        choices=["A", "B", "C", "all"],
                        help="Training phase: A (HQ), B (LQ), C (assoc), or all")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to train/ directory with images512x512/ and LQ_images_512x512/")
    parser.add_argument("--num_workers", type=int, default=4)

    # Architecture
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--hq_n_codes", type=int, default=1024,
                        help="HQ branch codebook size")
    parser.add_argument("--rq_levels", type=int, default=3,
                        help="Residual quantization levels for LQ branch")

    # Face parser
    parser.add_argument("--parser_ckpt", type=str, default="pretrained/79999_iter.pth",
                        help="Path to BiSeNet face parsing checkpoint")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.44e-4)
    parser.add_argument("--hq_epochs", type=int, default=50, help="Phase A epochs")
    parser.add_argument("--lq_epochs", type=int, default=10, help="Phase B epochs")
    parser.add_argument("--assoc_epochs", type=int, default=10, help="Phase C epochs")

    # Loss weights
    parser.add_argument("--lambda_per", type=float, default=1.0)
    parser.add_argument("--lambda_dis", type=float, default=0.8)
    parser.add_argument("--lambda_assoc", type=float, default=1.0,
                        help="Association loss weight for Phase C")

    # Checkpointing
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--hq_ckpt", type=str, default="",
                        help="Phase A checkpoint (for Phases B, C)")
    parser.add_argument("--lq_ckpt", type=str, default="",
                        help="Phase B checkpoint (for Phase C)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoints and start fresh")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")

    # Misc
    parser.add_argument("--expire_every", type=int, default=1000,
                        help="Dead code expiry interval (steps)")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Print logs every N steps")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(vars(args), indent=2)}")

    if args.phase == "A" or args.phase == "all":
        hq_model = run_phase_a(args, device)

        if args.phase == "all":
            # Set HQ checkpoint for subsequent phases
            args.hq_ckpt = os.path.join(args.ckpt_dir, "phase_a", "final.pt")
    else:
        hq_model = None

    if args.phase == "B" or args.phase == "all":
        lq_model = run_phase_b(args, device, hq_model=hq_model)

        if args.phase == "all":
            args.lq_ckpt = os.path.join(args.ckpt_dir, "phase_b", "final.pt")
    else:
        lq_model = None

    if args.phase == "C" or args.phase == "all":
        run_phase_c(args, device, hq_model=hq_model, lq_model=lq_model)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()