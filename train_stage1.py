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
    # Using config file (recommended):
    python train_stage1.py --config configs/train.yaml
    python train_stage1.py --config configs/train.yaml --phase A
    python train_stage1.py --config configs/train.yaml --phase C --grad_clip_norm 1.0

    # Without config file (CLI-only):
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
import math
import os
import time
import yaml
from collections import defaultdict

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader

from data.dataset import FFHQPairedDataset
from models.vqvae import VQVAE, GlobalVQ, build_hq_vqvae, build_lq_vqvae
from models.region_aware_vq import RegionAwareVQ
from models.face_parser import REGION_NAMES
from losses.losses import Stage1VQLoss


def _indices_to_masks(region_indices, device):
    """Convert (B, H, W) region index tensor to dict of bool masks."""
    region_indices = region_indices.to(device)
    masks = {}
    for idx, name in enumerate(REGION_NAMES):
        masks[name] = region_indices == idx
    return masks


def _is_main_process():
    """Return True if this is rank 0 or not running under DDP."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _build_scheduler(optimizer, args, total_epochs):
    """Build LR scheduler based on args.lr_schedule."""
    if args.lr_schedule == "none":
        return None
    if args.lr_schedule == "cosine":
        if args.warmup_epochs > 0:
            warmup = LinearLR(
                optimizer, start_factor=1e-2, total_iters=args.warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer, T_max=total_epochs - args.warmup_epochs,
            )
            return SequentialLR(
                optimizer, schedulers=[warmup, cosine],
                milestones=[args.warmup_epochs],
            )
        else:
            return CosineAnnealingLR(optimizer, T_max=total_epochs)
    raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")


# ======================================================================
# Training loop
# ======================================================================

def _enable_gradient_checkpointing(model):
    """Enable gradient checkpointing on encoder/decoder blocks."""
    n_blocks = 0
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'enable_gradient_checkpointing'):
        model.encoder.enable_gradient_checkpointing()
        n_blocks += len(list(model.encoder.down)) * 2 + 2  # approx
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'enable_gradient_checkpointing'):
        model.decoder.enable_gradient_checkpointing()
        n_blocks += len(list(model.decoder.up)) * 3 + 2
    return n_blocks


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
    grad_clip_norm=1.0,
    use_amp=False,
    scaler_g=None,
    scaler_d=None,
    grad_accum_steps=1,
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
        grad_clip_norm: Max norm for gradient clipping (0 = disabled).
        use_amp:     Enable mixed precision training.
        scaler_g:    GradScaler for generator (AMP).
        scaler_d:    GradScaler for discriminator (AMP).
        grad_accum_steps: Number of micro-batches to accumulate before optimizer step.

    Returns:
        avg_logs: dict of average loss values over the epoch.
    """
    model.train()
    criterion.discriminator.train()

    accum = defaultdict(float)
    n_steps = 0
    nan_steps = 0

    for step, batch in enumerate(loader):
        # ----- Prepare inputs -----
        if phase == "A":
            x = batch["hq"].to(device)
            x_01 = batch["hq_01"].to(device)
            target = x
            masks = None  # GlobalVQ doesn't use masks
        else:
            x = batch["lq"].to(device)
            x_01 = batch["lq_01"].to(device)
            target = x
            # Use pre-computed masks if available
            if "lq_mask" in batch:
                masks = _indices_to_masks(batch["lq_mask"], device)
            else:
                masks = None

        is_accum_step = (step + 1) % grad_accum_steps != 0
        loss_scale = 1.0 / grad_accum_steps

        # ----- Generator step -----
        if not is_accum_step or step == 0:
            optimizer_g.zero_grad()

        with autocast("cuda", enabled=use_amp):
            x_rec, z, z_q, vq_losses, vq_info = model(x, images_01=x_01, masks=masks)

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
            gen_loss = gen_loss * loss_scale

        # NaN detection — skip this step entirely
        if torch.isnan(gen_loss) or torch.isinf(gen_loss):
            nan_steps += 1
            if nan_steps <= 3 or nan_steps % 100 == 0:
                print(f"  [Step {step+1}] NaN/Inf detected in gen_loss, skipping step "
                      f"(total NaN steps: {nan_steps})")
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            continue

        if scaler_g is not None:
            scaler_g.scale(gen_loss).backward()
        else:
            gen_loss.backward()

        if not is_accum_step:
            if grad_clip_norm > 0:
                if scaler_g is not None:
                    scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=grad_clip_norm,
                )
            if scaler_g is not None:
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                optimizer_g.step()

        # ----- Discriminator step -----
        if not is_accum_step or step == 0:
            optimizer_d.zero_grad()

        with autocast("cuda", enabled=use_amp):
            d_loss, d_logs = criterion.discriminator_loss(target, x_rec.detach())
            d_loss = d_loss * loss_scale

        if scaler_d is not None:
            scaler_d.scale(d_loss).backward()
        else:
            d_loss.backward()

        if not is_accum_step:
            if scaler_d is not None:
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
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
            val = v.item() if torch.is_tensor(v) else v
            if not (math.isnan(val) or math.isinf(val)):
                accum[k] += val
        n_steps += 1

        # ----- Print -----
        if (step + 1) % log_every == 0:
            msg = f"  Phase {phase} | Epoch {epoch} | Step {step+1}/{len(loader)}"
            for k in ["total_gen", "l1", "perceptual", "gan_g", "vq", "gan_d"]:
                if k in accum:
                    msg += f" | {k}: {accum[k]/n_steps:.4f}"
            if "association" in accum:
                msg += f" | assoc: {accum['association']/n_steps:.4f}"
            if nan_steps > 0:
                msg += f" | nan_skips: {nan_steps}"
            print(msg)

    if nan_steps > 0:
        print(f"  WARNING: {nan_steps} NaN/Inf steps skipped this epoch")

    # Average logs
    avg_logs = {k: v / max(n_steps, 1) for k, v in accum.items()}
    return avg_logs


# ======================================================================
# Phase runners
# ======================================================================

def run_phase_a(args, device):
    """Phase A: Train HQ VQVAE (self-reconstruction, 50 epochs)."""
    if _is_main_process():
        print("=" * 60)
        print("Phase A: Training HQ VQVAE")
        print("=" * 60)

    # Dataset (HQ only)
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Dataset: {len(dataset)} HQ images, {len(loader)} batches/epoch")

    # Model
    model = build_hq_vqvae(
        n_codes=args.hq_n_codes, embed_dim=args.embed_dim,
    ).to(device).to(memory_format=torch.channels_last)
    print(f"HQ VQVAE params: {sum(p.numel() for p in model.parameters()):,}")

    # Gradient checkpointing
    if args.grad_ckpt:
        n = _enable_gradient_checkpointing(model)
        print(f"  Gradient checkpointing enabled on ~{n} blocks")

    # AMP scalers
    use_amp = args.amp and device.type == "cuda"
    scaler_g = GradScaler("cuda") if use_amp else None
    scaler_d = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  Mixed precision (AMP) enabled")

    # Losses
    criterion = Stage1VQLoss(
        lambda_per=args.lambda_per, lambda_dis=args.lambda_dis, lambda_assoc=0.0,
        vgg_pretrained=True,
    ).to(device)

    # Optimizers (separate for generator and discriminator)
    gen_params = list(model.parameters())
    optimizer_g = torch.optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(criterion.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # LR schedulers
    scheduler_g = _build_scheduler(optimizer_g, args, args.hq_epochs)
    scheduler_d = _build_scheduler(optimizer_d, args, args.hq_epochs)

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
        if scheduler_g and "scheduler_g" in ckpt:
            scheduler_g.load_state_dict(ckpt["scheduler_g"])
        if scheduler_d and "scheduler_d" in ckpt:
            scheduler_d.load_state_dict(ckpt["scheduler_d"])
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    ckpt = None
    for epoch in range(start_epoch, args.hq_epochs):
        # GAN loss warmup: ramp lambda_dis from 0 to target over first N epochs
        if args.gan_warmup_epochs > 0 and epoch < args.gan_warmup_epochs:
            warmup_frac = (epoch + 1) / args.gan_warmup_epochs
            criterion.lambda_dis = args.lambda_dis * warmup_frac
            print(f"  GAN warmup: lambda_dis = {criterion.lambda_dis:.4f}")
        else:
            criterion.lambda_dis = args.lambda_dis

        t0 = time.time()
        avg_logs = train_one_epoch(
            model, criterion, loader, optimizer_g, optimizer_d,
            device, epoch, phase="A",
            expire_every=args.expire_every, log_every=args.log_every,
            grad_clip_norm=args.grad_clip_norm,
            use_amp=use_amp, scaler_g=scaler_g, scaler_d=scaler_d,
            grad_accum_steps=args.grad_accum_steps,
        )
        if scheduler_g:
            scheduler_g.step()
        if scheduler_d:
            scheduler_d.step()
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
        if scheduler_g:
            ckpt["scheduler_g"] = scheduler_g.state_dict()
        if scheduler_d:
            ckpt["scheduler_d"] = scheduler_d.state_dict()
        torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    # Save final
    if ckpt is None:
        ckpt = {
            "epoch": start_epoch - 1,
            "model": model.state_dict(),
            "discriminator": criterion.discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }
    torch.save(ckpt, os.path.join(ckpt_dir, "final.pt"))
    print(f"Phase A complete. Saved to {ckpt_dir}/final.pt")

    return model


def run_phase_b(args, device, hq_model=None):
    """Phase B: Train LQ Region-Aware VQVAE (10 epochs, λ_assoc=0)."""
    if _is_main_process():
        print("=" * 60)
        print("Phase B: Training LQ Region-Aware VQVAE")
        print("=" * 60)

    # Dataset (paired HQ/LQ)
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Dataset: {len(dataset)} HQ/LQ pairs, {len(loader)} batches/epoch")
    if dataset.masks_dir:
        print(f"  Using pre-computed masks from {dataset.masks_dir}")

    # Model
    ravq = RegionAwareVQ(
        e_dim=args.embed_dim,
        n_levels=args.rq_levels,
        parser_ckpt=args.parser_ckpt,
    ).to(device)
    model = build_lq_vqvae(ravq, embed_dim=args.embed_dim).to(device).to(memory_format=torch.channels_last)
    print(f"LQ VQVAE params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable (excl. parser): "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Gradient checkpointing
    if args.grad_ckpt:
        n = _enable_gradient_checkpointing(model)
        print(f"  Gradient checkpointing enabled on ~{n} blocks")

    # AMP scalers
    use_amp = args.amp and device.type == "cuda"
    scaler_g = GradScaler("cuda") if use_amp else None
    scaler_d = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  Mixed precision (AMP) enabled")

    # Losses
    criterion = Stage1VQLoss(
        lambda_per=args.lambda_per, lambda_dis=args.lambda_dis, lambda_assoc=0.0,
        vgg_pretrained=True,
    ).to(device)

    # Optimizers
    gen_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_g = torch.optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(criterion.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # LR schedulers
    scheduler_g = _build_scheduler(optimizer_g, args, args.lq_epochs)
    scheduler_d = _build_scheduler(optimizer_d, args, args.lq_epochs)

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
        if scheduler_g and "scheduler_g" in ckpt:
            scheduler_g.load_state_dict(ckpt["scheduler_g"])
        if scheduler_d and "scheduler_d" in ckpt:
            scheduler_d.load_state_dict(ckpt["scheduler_d"])
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    ckpt = None
    for epoch in range(start_epoch, args.lq_epochs):
        # GAN loss warmup
        if args.gan_warmup_epochs > 0 and epoch < args.gan_warmup_epochs:
            warmup_frac = (epoch + 1) / args.gan_warmup_epochs
            criterion.lambda_dis = args.lambda_dis * warmup_frac
            print(f"  GAN warmup: lambda_dis = {criterion.lambda_dis:.4f}")
        else:
            criterion.lambda_dis = args.lambda_dis

        t0 = time.time()
        avg_logs = train_one_epoch(
            model, criterion, loader, optimizer_g, optimizer_d,
            device, epoch, phase="B",
            expire_every=args.expire_every, log_every=args.log_every,
            grad_clip_norm=args.grad_clip_norm,
            use_amp=use_amp, scaler_g=scaler_g, scaler_d=scaler_d,
            grad_accum_steps=args.grad_accum_steps,
        )
        if scheduler_g:
            scheduler_g.step()
        if scheduler_d:
            scheduler_d.step()
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
        if scheduler_g:
            ckpt["scheduler_g"] = scheduler_g.state_dict()
        if scheduler_d:
            ckpt["scheduler_d"] = scheduler_d.state_dict()
        torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    if ckpt is None:
        ckpt = {
            "epoch": start_epoch - 1,
            "model": model.state_dict(),
            "discriminator": criterion.discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }
    torch.save(ckpt, os.path.join(ckpt_dir, "final.pt"))
    print(f"Phase B complete. Saved to {ckpt_dir}/final.pt")

    return model


def run_phase_c(args, device, hq_model=None, lq_model=None):
    """Phase C: Continue LQ training with association loss (10 epochs, λ_assoc=1)."""
    if _is_main_process():
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
        hq_ckpt = torch.load(args.hq_ckpt, map_location=device, weights_only=False)
        hq_model.load_state_dict(hq_ckpt["model"])

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
        lq_model = build_lq_vqvae(ravq, embed_dim=args.embed_dim).to(device).to(memory_format=torch.channels_last)
        lq_ckpt = torch.load(args.lq_ckpt, map_location=device, weights_only=False)
        lq_model.load_state_dict(lq_ckpt["model"])

    model = lq_model

    # Dataset
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Dataset: {len(dataset)} HQ/LQ pairs, {len(loader)} batches/epoch")

    # Losses (association enabled)
    # By default, discriminator is freshly initialized so it doesn't resist
    # the association-loss-driven changes the generator needs to make.
    criterion = Stage1VQLoss(
        lambda_per=args.lambda_per, lambda_dis=args.lambda_dis,
        lambda_assoc=args.lambda_assoc,
        vgg_pretrained=True,
    ).to(device)

    # Optionally load Phase B's discriminator for continuity
    if args.phase_c_load_disc and args.lq_ckpt:
        disc_ckpt = torch.load(args.lq_ckpt, map_location=device, weights_only=False)
        criterion.discriminator.load_state_dict(disc_ckpt["discriminator"])
        print("Loaded Phase B discriminator into Phase C.")

    # Gradient checkpointing
    if args.grad_ckpt:
        n = _enable_gradient_checkpointing(model)
        print(f"  Gradient checkpointing enabled on ~{n} blocks")

    # AMP scalers
    use_amp = args.amp and device.type == "cuda"
    scaler_g = GradScaler("cuda") if use_amp else None
    scaler_d = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  Mixed precision (AMP) enabled")

    # Optimizers
    gen_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_g = torch.optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(criterion.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # LR schedulers
    scheduler_g = _build_scheduler(optimizer_g, args, args.assoc_epochs)
    scheduler_d = _build_scheduler(optimizer_d, args, args.assoc_epochs)

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
        if scheduler_g and "scheduler_g" in ckpt:
            scheduler_g.load_state_dict(ckpt["scheduler_g"])
        if scheduler_d and "scheduler_d" in ckpt:
            scheduler_d.load_state_dict(ckpt["scheduler_d"])
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    ckpt = None
    for epoch in range(start_epoch, args.assoc_epochs):
        # GAN loss warmup
        if args.gan_warmup_epochs > 0 and epoch < args.gan_warmup_epochs:
            warmup_frac = (epoch + 1) / args.gan_warmup_epochs
            criterion.lambda_dis = args.lambda_dis * warmup_frac
            print(f"  GAN warmup: lambda_dis = {criterion.lambda_dis:.4f}")
        else:
            criterion.lambda_dis = args.lambda_dis

        t0 = time.time()
        avg_logs = train_one_epoch(
            model, criterion, loader, optimizer_g, optimizer_d,
            device, epoch, phase="C", hq_model=hq_model,
            expire_every=args.expire_every, log_every=args.log_every,
            grad_clip_norm=args.grad_clip_norm,
            use_amp=use_amp, scaler_g=scaler_g, scaler_d=scaler_d,
            grad_accum_steps=args.grad_accum_steps,
        )
        if scheduler_g:
            scheduler_g.step()
        if scheduler_d:
            scheduler_d.step()
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
        if scheduler_g:
            ckpt["scheduler_g"] = scheduler_g.state_dict()
        if scheduler_d:
            ckpt["scheduler_d"] = scheduler_d.state_dict()
        torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    if ckpt is None:
        ckpt = {
            "epoch": start_epoch - 1,
            "model": model.state_dict(),
            "discriminator": criterion.discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }
    torch.save(ckpt, os.path.join(ckpt_dir, "final.pt"))
    print(f"Phase C complete. Saved to {ckpt_dir}/final.pt")

    return model


# ======================================================================
# Main
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="CRAFT Stage 1 Training")

    # Config file (YAML values are used as defaults; CLI flags override them)
    parser.add_argument("--config", type=str, default="",
                        help="Path to YAML config file (e.g. configs/train.yaml)")

    # Phase selection
    parser.add_argument("--phase", type=str, default="all",
                        choices=["A", "B", "C", "all"],
                        help="Training phase: A (HQ), B (LQ), C (assoc), or all")

    # Data
    parser.add_argument("--data_root", type=str, default="/projectnb/cs585/projects/craft/data/train",
                        help="Path to train/ directory with images512x512/ and LQ_images_512x512/")
    parser.add_argument("--num_workers", type=int, default=4)

    # Architecture
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--hq_n_codes", type=int, default=1024,
                        help="HQ branch codebook size")
    parser.add_argument("--rq_levels", type=int, default=3,
                        help="Residual quantization levels for LQ branch")

    # Face parser
    parser.add_argument("--parser_ckpt", type=str, default="/projectnb/cs585/projects/craft/pretrained/79999_iter.pth",
                        help="Path to BiSeNet face parsing checkpoint")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.44e-4)
    parser.add_argument("--lr_schedule", type=str, default="none",
                        choices=["none", "cosine"],
                        help="LR schedule: none (constant) or cosine decay")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Linear warmup epochs before cosine decay (0 = no warmup)")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = disabled, default: 1.0)")
    parser.add_argument("--hq_epochs", type=int, default=50, help="Phase A epochs")
    parser.add_argument("--lq_epochs", type=int, default=10, help="Phase B epochs")
    parser.add_argument("--assoc_epochs", type=int, default=10, help="Phase C epochs")

    # Loss weights
    parser.add_argument("--lambda_per", type=float, default=1.0)
    parser.add_argument("--lambda_dis", type=float, default=0.8)
    parser.add_argument("--lambda_assoc", type=float, default=1.0,
                        help="Association loss weight for Phase C")
    parser.add_argument("--gan_warmup_epochs", type=int, default=2,
                        help="Ramp lambda_dis from 0 to full over first N epochs (0 = disabled)")

    # Checkpointing
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--hq_ckpt", type=str, default="",
                        help="Phase A checkpoint (for Phases B, C)")
    parser.add_argument("--lq_ckpt", type=str, default="",
                        help="Phase B checkpoint (for Phase C)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoints and start fresh")
    parser.add_argument("--phase_c_load_disc", action="store_true",
                        help="Load Phase B discriminator in Phase C instead of fresh init")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")

    # Memory optimization
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision training (AMP)")
    parser.add_argument("--grad_ckpt", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accum)")

    # Misc
    parser.add_argument("--expire_every", type=int, default=1000,
                        help="Dead code expiry interval (steps)")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Print logs every N steps")
    parser.add_argument("--device", type=str, default="cuda")

    # First parse to get --config path, then re-parse with YAML defaults
    args, remaining = parser.parse_known_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # Set YAML values as new defaults (CLI flags still override)
        parser.set_defaults(**cfg)

    return parser.parse_args()


def main():
    args = parse_args()

    # Performance settings
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(vars(args), indent=2)}")
    if args.amp:
        print("Mixed precision (AMP): ENABLED")
    if args.grad_ckpt:
        print("Gradient checkpointing: ENABLED")
    if args.grad_accum_steps > 1:
        print(f"Gradient accumulation: {args.grad_accum_steps} steps "
              f"(effective batch size: {args.batch_size * args.grad_accum_steps})")
    print("Tip: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
          "to reduce CUDA memory fragmentation")

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