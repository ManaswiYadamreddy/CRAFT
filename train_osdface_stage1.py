"""
train_osdface_stage1.py — OSDFace Stage 1 (VRE) training.

Reproduces OSDFace's Stage 1 — Visual Representation Embedder — as a baseline
to compare against CRAFT, in three phases that mirror the OSDFace paper §3.2:

    HQ phase (50 epochs, λ_assoc n/a):
        Train the HQ VQ-VAE on HQ self-reconstruction.
        --> REUSED FROM CRAFT PHASE A. We do not retrain it here. The HQ
            branch is loaded from `hq_ckpt` and frozen.

    Phase B (10 epochs, λ_assoc = 0):
        Train the LQ VQ-VAE on LQ self-reconstruction. HQ is frozen.

    Phase C (10 epochs, λ_assoc = 1):
        Continue training the LQ VQ-VAE with the HQ↔LQ feature association
        loss (OSDFace Eq. 9–10) enabled. HQ is still frozen.

Differences from the paper, intentional, for apples-to-apples comparison
with CRAFT:

    * Codebooks live on the unit hypersphere and quantization uses cosine
      similarity (CRAFT's `GlobalVQ`), not raw-space L2.
    * Otherwise the architecture, losses, and schedule match OSDFace Stage 1.

This file only adds OSDFace Stage 1. The rest of the CRAFT code (models,
losses, dataset) is reused unchanged.

Usage
-----
    # Phases B and C sequentially (10 + 10 epochs)
    python train_osdface_stage1.py --config configs/train_osdface.yaml --phase all

    # Phase B only
    python train_osdface_stage1.py --config configs/train_osdface.yaml --phase B

    # Phase C only (continues from Phase B checkpoint written by --phase B)
    python train_osdface_stage1.py --config configs/train_osdface.yaml --phase C
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
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader

from data.dataset import FFHQPairedDataset
from models.vqvae import build_hq_vqvae
from losses.losses import Stage1VQLoss, AssociationLoss


# ======================================================================
# Helpers
# ======================================================================

def _build_scheduler(optimizer, args, total_epochs):
    if args.lr_schedule == "none":
        return None
    if args.lr_schedule == "cosine":
        if args.warmup_epochs > 0:
            warmup = LinearLR(
                optimizer, start_factor=1e-2, total_iters=args.warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer, T_max=max(1, total_epochs - args.warmup_epochs),
            )
            return SequentialLR(
                optimizer, schedulers=[warmup, cosine],
                milestones=[args.warmup_epochs],
            )
        return CosineAnnealingLR(optimizer, T_max=total_epochs)
    raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")


def _enable_gradient_checkpointing(model):
    if hasattr(model, "encoder") and hasattr(model.encoder, "enable_gradient_checkpointing"):
        model.encoder.enable_gradient_checkpointing()
    if hasattr(model, "decoder") and hasattr(model.decoder, "enable_gradient_checkpointing"):
        model.decoder.enable_gradient_checkpointing()


def _features_flat(z):
    """(B, C, H, W) -> (B, H*W, C)"""
    B, C, H, W = z.shape
    return z.permute(0, 2, 3, 1).reshape(B, H * W, C)


def _load_frozen_hq(args, device):
    """Build a fresh HQ VQ-VAE and load CRAFT Phase A weights into it.

    The HQ branch is identical to CRAFT Phase A (`build_hq_vqvae` factory),
    so the state-dict keys match exactly.
    """
    if not args.hq_ckpt:
        raise ValueError(
            "OSDFace Stage 1 needs an HQ checkpoint. Set `hq_ckpt` in "
            "configs/train_osdface.yaml or pass --hq_ckpt."
        )
    print(f"Loading frozen HQ branch from {args.hq_ckpt}")
    hq_model = build_hq_vqvae(
        n_codes=args.hq_n_codes, embed_dim=args.embed_dim,
    ).to(device).to(memory_format=torch.channels_last)
    ck = torch.load(args.hq_ckpt, map_location=device, weights_only=False)
    hq_model.load_state_dict(ck["model"])
    hq_model.eval()
    for p in hq_model.parameters():
        p.requires_grad_(False)
    print("HQ branch frozen.")
    return hq_model


# ======================================================================
# Training loop (LQ branch only; HQ frozen)
# ======================================================================

def train_one_epoch(
    lq_model,
    hq_model,
    criterion,
    assoc_loss_fn,
    lambda_assoc,
    loader,
    optimizer_g,
    optimizer_d,
    device,
    epoch,
    phase,
    expire_every=1000,
    log_every=50,
    grad_clip_norm=1.0,
    use_amp=False,
    scaler_g=None,
    scaler_d=None,
    grad_accum_steps=1,
):
    lq_model.train()
    criterion.discriminator.train()
    hq_model.eval()  # always frozen

    accum = defaultdict(float)
    n_steps = 0
    nan_steps = 0

    for step, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        x_h = batch["hq"]
        x_l = batch["lq"]
        x_l01 = batch["lq_01"]

        is_accum_step = (step + 1) % grad_accum_steps != 0
        is_first_accum = step % grad_accum_steps == 0
        loss_scale = 1.0 / grad_accum_steps

        if is_first_accum:
            optimizer_g.zero_grad()

        with autocast("cuda", enabled=use_amp):
            # LQ branch (the one being trained)
            x_l_rec, z_l, z_l_q, vq_losses_l, _ = lq_model(
                x_l, images_01=x_l01, masks=None,
            )
            gen_loss, gen_logs = criterion.generator_loss(
                x_l_rec, x_l, vq_losses_l["total_vq"],
            )

            # Association loss (Phase C only)
            if lambda_assoc > 0:
                with torch.no_grad():
                    z_h = hq_model.encode(x_h)
                z_H_flat = _features_flat(z_h.float())
                z_L_flat = _features_flat(z_l.float())
                l_assoc = assoc_loss_fn(z_H_flat, z_L_flat)
                gen_loss = gen_loss + lambda_assoc * l_assoc
            else:
                l_assoc = torch.zeros((), device=device)

            gen_loss = gen_loss * loss_scale

        if torch.isnan(gen_loss) or torch.isinf(gen_loss):
            nan_steps += 1
            print(f"  [Step {step+1}] NaN/Inf in gen_loss (total: {nan_steps}), skipping")
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            if scaler_g is not None:
                scaler_g._scale = torch.tensor(2**10, dtype=torch.float32, device=device)
            if scaler_d is not None:
                scaler_d._scale = torch.tensor(2**10, dtype=torch.float32, device=device)
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
                    [p for p in lq_model.parameters() if p.requires_grad],
                    max_norm=grad_clip_norm,
                )
            if scaler_g is not None:
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                optimizer_g.step()

        # Discriminator step
        if not is_accum_step:
            optimizer_d.zero_grad()
            with autocast("cuda", enabled=use_amp):
                d_loss, d_logs = criterion.discriminator_loss(x_l, x_l_rec.detach())
            if scaler_d is not None:
                scaler_d.scale(d_loss).backward()
                scaler_d.unscale_(optimizer_d)
            else:
                d_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    criterion.discriminator.parameters(), max_norm=grad_clip_norm,
                )
            if scaler_d is not None:
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                optimizer_d.step()

        # Dead code expiry on LQ codebook
        if (step + 1) % expire_every == 0 and hasattr(lq_model.quantizer, "expire_dead_codes"):
            with torch.no_grad():
                z_for_expire = lq_model.encode(x_l)
                n = lq_model.quantizer.expire_dead_codes(z_for_expire, images=x_l01)
                if n:
                    print(f"    [Step {step+1}] LQ expired {n} dead codes")

        if is_accum_step:
            d_logs = {}
        all_logs = {**gen_logs, **d_logs, "assoc": l_assoc.detach()}
        for k, v in all_logs.items():
            val = v.item() if torch.is_tensor(v) else v
            if not (math.isnan(val) or math.isinf(val)):
                accum[k] += val
        n_steps += 1

        if (step + 1) % log_every == 0:
            msg = f"  OSDFace S1 Phase {phase} | Epoch {epoch} | Step {step+1}/{len(loader)}"
            for k in ["l1", "perceptual", "gan_g", "vq", "gan_d", "assoc"]:
                if k in accum:
                    msg += f" | {k}: {accum[k]/n_steps:.4f}"
            if nan_steps > 0:
                msg += f" | nan_skips: {nan_steps}"
            print(msg)

    if nan_steps > 0:
        print(f"  WARNING: {nan_steps} NaN/Inf steps skipped this epoch")
    return {k: v / max(n_steps, 1) for k, v in accum.items()}


# ======================================================================
# Phase runners
# ======================================================================

def _build_lq_branch(args, device):
    lq_model = build_hq_vqvae(
        n_codes=args.lq_n_codes, embed_dim=args.embed_dim,
    ).to(device).to(memory_format=torch.channels_last)
    if args.grad_ckpt:
        _enable_gradient_checkpointing(lq_model)
    print(f"LQ branch params: {sum(p.numel() for p in lq_model.parameters()):,}")
    return lq_model


def _make_loader(args):
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    print(f"Dataset: {len(dataset)} HQ/LQ pairs, {len(loader)} batches/epoch")
    return loader


def _save_ckpt(path, *, epoch, lq_model, criterion, optimizer_g, optimizer_d,
               scheduler_g, scheduler_d, scaler_g, scaler_d):
    ckpt = {
        "epoch": epoch,
        "lq_model": lq_model.state_dict(),
        "discriminator": criterion.discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
    }
    if scheduler_g:
        ckpt["scheduler_g"] = scheduler_g.state_dict()
    if scheduler_d:
        ckpt["scheduler_d"] = scheduler_d.state_dict()
    if scaler_g is not None:
        ckpt["scaler_g"] = scaler_g.state_dict()
    if scaler_d is not None:
        ckpt["scaler_d"] = scaler_d.state_dict()
    torch.save(ckpt, path)
    return ckpt


def _try_resume(path, device, lq_model, criterion, optimizer_g, optimizer_d,
                scheduler_g, scheduler_d, scaler_g, scaler_d):
    if not os.path.exists(path):
        return 0
    ck = torch.load(path, map_location=device, weights_only=False)
    lq_model.load_state_dict(ck["lq_model"])
    try:
        criterion.discriminator.load_state_dict(ck["discriminator"])
    except RuntimeError:
        print("  WARNING: discriminator state incompatible, reinitializing")
    if "optimizer_g" in ck:
        optimizer_g.load_state_dict(ck["optimizer_g"])
    if "optimizer_d" in ck:
        optimizer_d.load_state_dict(ck["optimizer_d"])
    if scheduler_g and "scheduler_g" in ck:
        scheduler_g.load_state_dict(ck["scheduler_g"])
    if scheduler_d and "scheduler_d" in ck:
        scheduler_d.load_state_dict(ck["scheduler_d"])
    if scaler_g is not None and "scaler_g" in ck:
        scaler_g.load_state_dict(ck["scaler_g"])
    if scaler_d is not None and "scaler_d" in ck:
        scaler_d.load_state_dict(ck["scaler_d"])
    print(f"Resumed from epoch {ck['epoch'] + 1}")
    return ck["epoch"] + 1


def run_phase_a(args, device):
    """Phase A: train HQ VQ-VAE from scratch on HQ self-reconstruction (50 epochs).

    Note: this is identical in design to CRAFT Phase A. By default the
    OSDFace pipeline does NOT run this — it reuses CRAFT's Phase A
    checkpoint via `hq_ckpt`. This function is provided for completeness
    so the OSDFace baseline can be trained end-to-end without any CRAFT
    artifacts if desired.
    """
    print("=" * 60)
    print("OSDFace Stage 1 — Phase A: HQ VQ-VAE (self-reconstruction)")
    print("=" * 60)

    # HQ-only dataset (LQ images not needed)
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    print(f"Dataset: {len(dataset)} HQ images, {len(loader)} batches/epoch")

    hq_model = build_hq_vqvae(
        n_codes=args.hq_n_codes, embed_dim=args.embed_dim,
    ).to(device).to(memory_format=torch.channels_last)
    if args.grad_ckpt:
        _enable_gradient_checkpointing(hq_model)
    print(f"HQ branch params: {sum(p.numel() for p in hq_model.parameters()):,}")

    criterion = Stage1VQLoss(
        lambda_per=args.lambda_per, lambda_dis=args.lambda_dis,
        lambda_assoc=0.0, vgg_pretrained=True,
    ).to(device)

    optimizer_g = torch.optim.Adam(
        list(hq_model.parameters()), lr=args.lr, betas=(0.5, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        criterion.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999),
    )
    scheduler_g = _build_scheduler(optimizer_g, args, args.hq_epochs)
    scheduler_d = _build_scheduler(optimizer_d, args, args.hq_epochs)

    use_amp = args.amp and device.type == "cuda"
    scaler_g = GradScaler("cuda", init_scale=2**12, growth_interval=4000) if use_amp else None
    scaler_d = GradScaler("cuda", init_scale=2**12, growth_interval=4000) if use_amp else None

    ckpt_dir = os.path.join(args.ckpt_dir, "phase_a")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Phase A self-reconstructs HQ images, so it doesn't use the LQ-branch
    # train_one_epoch helper. We run a small inline loop here.
    def _save_a(path, epoch):
        ckpt = {
            "epoch": epoch,
            # Use the same key CRAFT Phase A uses so the checkpoint can be
            # loaded directly via _load_frozen_hq() afterwards.
            "model": hq_model.state_dict(),
            "discriminator": criterion.discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }
        if scheduler_g:
            ckpt["scheduler_g"] = scheduler_g.state_dict()
        if scheduler_d:
            ckpt["scheduler_d"] = scheduler_d.state_dict()
        if scaler_g is not None:
            ckpt["scaler_g"] = scaler_g.state_dict()
        if scaler_d is not None:
            ckpt["scaler_d"] = scaler_d.state_dict()
        torch.save(ckpt, path)
        return ckpt

    start_epoch = 0
    latest = os.path.join(ckpt_dir, "latest.pt")
    if not args.fresh and os.path.exists(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)
        hq_model.load_state_dict(ck["model"])
        try:
            criterion.discriminator.load_state_dict(ck["discriminator"])
        except RuntimeError:
            print("  WARNING: discriminator state incompatible, reinitializing")
        if "optimizer_g" in ck:
            optimizer_g.load_state_dict(ck["optimizer_g"])
        if "optimizer_d" in ck:
            optimizer_d.load_state_dict(ck["optimizer_d"])
        if scheduler_g and "scheduler_g" in ck:
            scheduler_g.load_state_dict(ck["scheduler_g"])
        if scheduler_d and "scheduler_d" in ck:
            scheduler_d.load_state_dict(ck["scheduler_d"])
        if scaler_g is not None and "scaler_g" in ck:
            scaler_g.load_state_dict(ck["scaler_g"])
        if scaler_d is not None and "scaler_d" in ck:
            scaler_d.load_state_dict(ck["scaler_d"])
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    ckpt = None
    for epoch in range(start_epoch, args.hq_epochs):
        if args.gan_warmup_epochs > 0 and epoch < args.gan_warmup_epochs:
            criterion.lambda_dis = args.lambda_dis * (epoch / args.gan_warmup_epochs)
        else:
            criterion.lambda_dis = args.lambda_dis
        print(f"  lambda_dis={criterion.lambda_dis:.3f}")

        hq_model.train()
        criterion.discriminator.train()
        accum = defaultdict(float)
        n_steps = 0
        nan_steps = 0
        t0 = time.time()

        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            x = batch["hq"]
            x01 = batch["hq_01"]

            is_accum_step = (step + 1) % args.grad_accum_steps != 0
            is_first_accum = step % args.grad_accum_steps == 0
            loss_scale = 1.0 / args.grad_accum_steps

            if is_first_accum:
                optimizer_g.zero_grad()

            with autocast("cuda", enabled=use_amp):
                x_rec, z, z_q, vq_losses, _ = hq_model(x, images_01=x01, masks=None)
                gen_loss, gen_logs = criterion.generator_loss(
                    x_rec, x, vq_losses["total_vq"],
                )
                gen_loss = gen_loss * loss_scale

            if torch.isnan(gen_loss) or torch.isinf(gen_loss):
                nan_steps += 1
                optimizer_g.zero_grad()
                optimizer_d.zero_grad()
                if scaler_g is not None:
                    scaler_g._scale = torch.tensor(2**10, dtype=torch.float32, device=device)
                if scaler_d is not None:
                    scaler_d._scale = torch.tensor(2**10, dtype=torch.float32, device=device)
                continue

            if scaler_g is not None:
                scaler_g.scale(gen_loss).backward()
            else:
                gen_loss.backward()

            if not is_accum_step:
                if args.grad_clip_norm > 0:
                    if scaler_g is not None:
                        scaler_g.unscale_(optimizer_g)
                    torch.nn.utils.clip_grad_norm_(
                        hq_model.parameters(), max_norm=args.grad_clip_norm,
                    )
                if scaler_g is not None:
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    optimizer_g.step()

            if not is_accum_step:
                optimizer_d.zero_grad()
                with autocast("cuda", enabled=use_amp):
                    d_loss, d_logs = criterion.discriminator_loss(x, x_rec.detach())
                if scaler_d is not None:
                    scaler_d.scale(d_loss).backward()
                    scaler_d.unscale_(optimizer_d)
                else:
                    d_loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        criterion.discriminator.parameters(),
                        max_norm=args.grad_clip_norm,
                    )
                if scaler_d is not None:
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    optimizer_d.step()

            if (step + 1) % args.expire_every == 0 and hasattr(hq_model.quantizer, "expire_dead_codes"):
                with torch.no_grad():
                    z_for_expire = hq_model.encode(x)
                    n = hq_model.quantizer.expire_dead_codes(z_for_expire, images=x01)
                    if n:
                        print(f"    [Step {step+1}] HQ expired {n} dead codes")

            if is_accum_step:
                d_logs = {}
            for k, v in {**gen_logs, **d_logs}.items():
                val = v.item() if torch.is_tensor(v) else v
                if not (math.isnan(val) or math.isinf(val)):
                    accum[k] += val
            n_steps += 1

            if (step + 1) % args.log_every == 0:
                msg = f"  OSDFace S1 Phase A | Epoch {epoch} | Step {step+1}/{len(loader)}"
                for k in ["l1", "perceptual", "gan_g", "vq", "gan_d"]:
                    if k in accum:
                        msg += f" | {k}: {accum[k]/n_steps:.4f}"
                if nan_steps > 0:
                    msg += f" | nan_skips: {nan_steps}"
                print(msg)

        if scheduler_g:
            scheduler_g.step()
        if scheduler_d:
            scheduler_d.step()
        elapsed = time.time() - t0
        print(
            f"OSDFace S1 Phase A | Epoch {epoch} done in {elapsed:.0f}s | "
            f"L1={accum.get('l1', 0)/max(n_steps,1):.4f} "
            f"VQ={accum.get('vq', 0)/max(n_steps,1):.4f}"
        )

        ckpt = _save_a(latest, epoch)
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    if ckpt is not None:
        torch.save(ckpt, os.path.join(ckpt_dir, "final.pt"))
        print(f"OSDFace S1 Phase A complete. Saved to {ckpt_dir}/final.pt")

    return hq_model


def run_phase_b(args, device, hq_model=None, lq_model=None):
    """Phase B: train LQ VQ-VAE, HQ frozen, λ_assoc=0 (10 epochs)."""
    print("=" * 60)
    print("OSDFace Stage 1 — Phase B: LQ VQ-VAE (lambda_assoc = 0)")
    print("=" * 60)

    if hq_model is None:
        hq_model = _load_frozen_hq(args, device)
    if lq_model is None:
        lq_model = _build_lq_branch(args, device)

    loader = _make_loader(args)

    criterion = Stage1VQLoss(
        lambda_per=args.lambda_per, lambda_dis=args.lambda_dis,
        lambda_assoc=0.0, vgg_pretrained=True,
    ).to(device)
    assoc_loss_fn = AssociationLoss().to(device)

    optimizer_g = torch.optim.Adam(
        [p for p in lq_model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.5, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        criterion.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999),
    )
    scheduler_g = _build_scheduler(optimizer_g, args, args.lq_epochs)
    scheduler_d = _build_scheduler(optimizer_d, args, args.lq_epochs)

    use_amp = args.amp and device.type == "cuda"
    scaler_g = GradScaler("cuda", init_scale=2**12, growth_interval=4000) if use_amp else None
    scaler_d = GradScaler("cuda", init_scale=2**12, growth_interval=4000) if use_amp else None

    ckpt_dir = os.path.join(args.ckpt_dir, "phase_b")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    if not args.fresh:
        start_epoch = _try_resume(
            os.path.join(ckpt_dir, "latest.pt"), device, lq_model, criterion,
            optimizer_g, optimizer_d, scheduler_g, scheduler_d, scaler_g, scaler_d,
        )

    ckpt = None
    for epoch in range(start_epoch, args.lq_epochs):
        if args.gan_warmup_epochs > 0 and epoch < args.gan_warmup_epochs:
            criterion.lambda_dis = args.lambda_dis * (epoch / args.gan_warmup_epochs)
        else:
            criterion.lambda_dis = args.lambda_dis
        print(f"  lambda_dis={criterion.lambda_dis:.3f}  lambda_assoc=0.000")

        t0 = time.time()
        avg_logs = train_one_epoch(
            lq_model, hq_model, criterion, assoc_loss_fn, 0.0,
            loader, optimizer_g, optimizer_d, device, epoch, phase="B",
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
        print(
            f"OSDFace S1 Phase B | Epoch {epoch} done in {elapsed:.0f}s | "
            f"L1={avg_logs.get('l1', 0):.4f} VQ={avg_logs.get('vq', 0):.4f}"
        )

        ckpt = _save_ckpt(
            os.path.join(ckpt_dir, "latest.pt"),
            epoch=epoch, lq_model=lq_model, criterion=criterion,
            optimizer_g=optimizer_g, optimizer_d=optimizer_d,
            scheduler_g=scheduler_g, scheduler_d=scheduler_d,
            scaler_g=scaler_g, scaler_d=scaler_d,
        )
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    if ckpt is not None:
        torch.save(ckpt, os.path.join(ckpt_dir, "final.pt"))
        print(f"OSDFace S1 Phase B complete. Saved to {ckpt_dir}/final.pt")

    return hq_model, lq_model


def run_phase_c(args, device, hq_model=None, lq_model=None):
    """Phase C: continue LQ VQ-VAE with λ_assoc=1 (10 epochs)."""
    print("=" * 60)
    print("OSDFace Stage 1 — Phase C: LQ VQ-VAE + Association Loss "
          "(lambda_assoc = 1)")
    print("=" * 60)

    if hq_model is None:
        hq_model = _load_frozen_hq(args, device)
    if lq_model is None:
        lq_model = _build_lq_branch(args, device)
        if not args.lq_ckpt:
            raise ValueError(
                "Phase C requires `lq_ckpt` (Phase B checkpoint). Set it in "
                "the config or pass --lq_ckpt."
            )
        print(f"Loading Phase B LQ branch from {args.lq_ckpt}")
        ck = torch.load(args.lq_ckpt, map_location=device, weights_only=False)
        lq_model.load_state_dict(ck["lq_model"])

    loader = _make_loader(args)

    criterion = Stage1VQLoss(
        lambda_per=args.lambda_per, lambda_dis=args.lambda_dis,
        lambda_assoc=0.0, vgg_pretrained=True,
    ).to(device)
    assoc_loss_fn = AssociationLoss().to(device)

    optimizer_g = torch.optim.Adam(
        [p for p in lq_model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.5, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        criterion.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999),
    )
    scheduler_g = _build_scheduler(optimizer_g, args, args.assoc_epochs)
    scheduler_d = _build_scheduler(optimizer_d, args, args.assoc_epochs)

    use_amp = args.amp and device.type == "cuda"
    scaler_g = GradScaler("cuda", init_scale=2**12, growth_interval=4000) if use_amp else None
    scaler_d = GradScaler("cuda", init_scale=2**12, growth_interval=4000) if use_amp else None

    ckpt_dir = os.path.join(args.ckpt_dir, "phase_c")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    if not args.fresh:
        start_epoch = _try_resume(
            os.path.join(ckpt_dir, "latest.pt"), device, lq_model, criterion,
            optimizer_g, optimizer_d, scheduler_g, scheduler_d, scaler_g, scaler_d,
        )

    ckpt = None
    for epoch in range(start_epoch, args.assoc_epochs):
        if args.gan_warmup_epochs > 0 and epoch < args.gan_warmup_epochs:
            criterion.lambda_dis = args.lambda_dis * (epoch / args.gan_warmup_epochs)
        else:
            criterion.lambda_dis = args.lambda_dis
        print(f"  lambda_dis={criterion.lambda_dis:.3f}  "
              f"lambda_assoc={args.lambda_assoc:.3f}")

        t0 = time.time()
        avg_logs = train_one_epoch(
            lq_model, hq_model, criterion, assoc_loss_fn, args.lambda_assoc,
            loader, optimizer_g, optimizer_d, device, epoch, phase="C",
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
        print(
            f"OSDFace S1 Phase C | Epoch {epoch} done in {elapsed:.0f}s | "
            f"L1={avg_logs.get('l1', 0):.4f} VQ={avg_logs.get('vq', 0):.4f} "
            f"assoc={avg_logs.get('assoc', 0):.4f}"
        )

        ckpt = _save_ckpt(
            os.path.join(ckpt_dir, "latest.pt"),
            epoch=epoch, lq_model=lq_model, criterion=criterion,
            optimizer_g=optimizer_g, optimizer_d=optimizer_d,
            scheduler_g=scheduler_g, scheduler_d=scheduler_d,
            scaler_g=scaler_g, scaler_d=scaler_d,
        )
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    if ckpt is not None:
        torch.save(ckpt, os.path.join(ckpt_dir, "final.pt"))
        print(f"OSDFace S1 Phase C complete. Saved to {ckpt_dir}/final.pt")

    return hq_model, lq_model


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="OSDFace Stage 1 (VRE) Training")

    parser.add_argument("--config", type=str, default="",
                        help="Path to YAML config (e.g. configs/train_osdface.yaml)")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["A", "B", "C", "all"],
                        help="Training phase: A (HQ, 50 ep), B (LQ, lambda_assoc=0, "
                             "10 ep), C (LQ + association, 10 ep), or all. "
                             "NOTE: --phase all runs B and C only and reuses "
                             "`hq_ckpt` for the HQ branch; pass --phase A to "
                             "train the HQ branch from scratch.")

    # Data
    parser.add_argument("--data_root", type=str,
                        default="/projectnb/cs585/projects/craft/data/train")
    parser.add_argument("--num_workers", type=int, default=4)

    # Architecture
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--hq_n_codes", type=int, default=1024)
    parser.add_argument("--lq_n_codes", type=int, default=1024)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.44e-4)
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["none", "cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--hq_epochs", type=int, default=50,
                        help="Phase A epochs (paper: 50)")
    parser.add_argument("--lq_epochs", type=int, default=10,
                        help="Phase B epochs (paper: 10)")
    parser.add_argument("--assoc_epochs", type=int, default=10,
                        help="Phase C epochs (paper: 10)")

    # Loss weights
    parser.add_argument("--lambda_per", type=float, default=1.0)
    parser.add_argument("--lambda_dis", type=float, default=0.8)
    parser.add_argument("--lambda_assoc", type=float, default=1.0)
    parser.add_argument("--gan_warmup_epochs", type=int, default=2)

    # Checkpoints
    parser.add_argument("--hq_ckpt", type=str,
                        default="/projectnb/cs585/projects/craft/checkpoints/phase_a/final.pt",
                        help="CRAFT Phase A checkpoint, used as the frozen HQ branch")
    parser.add_argument("--lq_ckpt", type=str, default="",
                        help="OSDFace Phase B checkpoint, required for --phase C")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_osdface")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)

    # Memory
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    # Misc
    parser.add_argument("--expire_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")

    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(vars(args), indent=2)}")

    hq_model, lq_model = None, None

    if args.phase == "A":
        run_phase_a(args, device)
        print("\nOSDFace Stage 1 Phase A complete!")
        return

    if args.phase in ("B", "all"):
        hq_model, lq_model = run_phase_b(args, device,
                                         hq_model=hq_model, lq_model=lq_model)
        if args.phase == "all":
            args.lq_ckpt = os.path.join(args.ckpt_dir, "phase_b", "final.pt")

    if args.phase in ("C", "all"):
        # For --phase C alone, lq_model is None and Phase C will load from args.lq_ckpt.
        # For --phase all, we already have lq_model in memory and skip the disk reload.
        run_phase_c(args, device,
                    hq_model=hq_model,
                    lq_model=lq_model if args.phase == "all" else None)

    print("\nOSDFace Stage 1 training complete!")


if __name__ == "__main__":
    main()
