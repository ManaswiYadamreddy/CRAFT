"""
train_osdface_stage1.py — OSDFace Stage 1 (VRE) training.

Reproduces OSDFace's Stage 1 — Visual Representation Embedder (VRE) — as a
baseline to compare against CRAFT. It trains two flat VQ-VAEs in parallel:

    HQ branch:  E_H → quant_conv → Q_H (HQ Dict) → post_quant_conv → D_H → I_H
    LQ branch:  E_L → quant_conv → Q_L (LQ Dict) → post_quant_conv → D_L → I_L

and aligns their encoder feature spaces via the HQ↔LQ association loss
(OSDFace Eq. 9–10).

Differences from the paper — intentional, for apples-to-apples comparison
with CRAFT:

    * Codebooks live on the unit hypersphere and quantization uses cosine
      similarity (CRAFT's `GlobalVQ`), not raw-space L2 as in the paper.
    * Otherwise the architecture, losses, and schedule match OSDFace Stage 1.

This file only touches OSDFace Stage 1. The rest of the CRAFT code (models,
losses, dataset) is reused unchanged.

Usage
-----
    # Train from scratch on both branches
    python train_osdface_stage1.py --config configs/train_osdface.yaml

    # Warm-start HQ branch from an already trained CRAFT Phase A checkpoint
    # (recommended — see README). Optionally freeze HQ to save compute.
    python train_osdface_stage1.py --config configs/train_osdface.yaml \
        --hq_ckpt checkpoints/phase_a/final.pt --freeze_hq
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
# Helpers (mirror train_stage1.py)
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
    """(B, C, H, W) → (B, H*W, C)"""
    B, C, H, W = z.shape
    return z.permute(0, 2, 3, 1).reshape(B, H * W, C)


# ======================================================================
# Training loop (joint HQ + LQ)
# ======================================================================

def train_one_epoch(
    hq_model,
    lq_model,
    criterion_h,
    criterion_l,
    assoc_loss_fn,
    lambda_assoc,
    loader,
    optimizer_g,
    optimizer_d,
    device,
    epoch,
    freeze_hq,
    expire_every=1000,
    log_every=50,
    grad_clip_norm=1.0,
    use_amp=False,
    scaler_g=None,
    scaler_d=None,
    grad_accum_steps=1,
):
    """
    One OSDFace Stage 1 epoch.

    For each batch we:
      1) Forward HQ branch (self-recon of HQ image) and LQ branch (self-recon of LQ image).
      2) Combine generator losses: L1+per+gan+vq for each branch + λ_assoc * L_assoc.
      3) Backward through the combined generator loss once.
      4) Alternating discriminator step on both discriminators.
    """
    if not freeze_hq:
        hq_model.train()
    else:
        hq_model.eval()
    lq_model.train()
    criterion_h.discriminator.train()
    criterion_l.discriminator.train()

    accum = defaultdict(float)
    n_steps = 0
    nan_steps = 0

    for step, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        x_h = batch["hq"]
        x_h01 = batch["hq_01"]
        x_l = batch["lq"]
        x_l01 = batch["lq_01"]

        is_accum_step = (step + 1) % grad_accum_steps != 0
        is_first_accum = step % grad_accum_steps == 0
        loss_scale = 1.0 / grad_accum_steps

        # ----- Generator step -----
        if is_first_accum:
            optimizer_g.zero_grad()

        with autocast("cuda", enabled=use_amp):
            # HQ branch
            if freeze_hq:
                with torch.no_grad():
                    x_h_rec, z_h, z_h_q, vq_losses_h, vq_info_h = hq_model(
                        x_h, images_01=x_h01, masks=None,
                    )
                # Re-run encoder under grad=False for assoc features; z_h already detached
                z_h_for_assoc = z_h.detach()
                gen_loss_h = torch.zeros((), device=device)
                gen_logs_h = {}
            else:
                x_h_rec, z_h, z_h_q, vq_losses_h, vq_info_h = hq_model(
                    x_h, images_01=x_h01, masks=None,
                )
                gen_loss_h, gen_logs_h = criterion_h.generator_loss(
                    x_h_rec, x_h, vq_losses_h["total_vq"],
                )
                z_h_for_assoc = z_h

            # LQ branch
            x_l_rec, z_l, z_l_q, vq_losses_l, vq_info_l = lq_model(
                x_l, images_01=x_l01, masks=None,
            )
            gen_loss_l, gen_logs_l = criterion_l.generator_loss(
                x_l_rec, x_l, vq_losses_l["total_vq"],
            )

            # Association loss (OSDFace Eq. 9–10), λ=0 during warmup
            if lambda_assoc > 0:
                z_H_flat = _features_flat(z_h_for_assoc.float())
                z_L_flat = _features_flat(z_l.float())
                l_assoc = assoc_loss_fn(z_H_flat, z_L_flat)
            else:
                l_assoc = torch.zeros((), device=device)

            gen_loss = gen_loss_h + gen_loss_l + lambda_assoc * l_assoc
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
                trainable = [p for p in optimizer_g.param_groups[0]["params"] if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip_norm)
            if scaler_g is not None:
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                optimizer_g.step()

        # ----- Discriminator step (both discriminators) -----
        if not is_accum_step:
            optimizer_d.zero_grad()
            with autocast("cuda", enabled=use_amp):
                d_loss_h, d_logs_h = criterion_h.discriminator_loss(x_h, x_h_rec.detach())
                d_loss_l, d_logs_l = criterion_l.discriminator_loss(x_l, x_l_rec.detach())
                if freeze_hq:
                    d_loss = d_loss_l
                else:
                    d_loss = d_loss_h + d_loss_l

            if scaler_d is not None:
                scaler_d.scale(d_loss).backward()
                scaler_d.unscale_(optimizer_d)
            else:
                d_loss.backward()
            if grad_clip_norm > 0:
                disc_params = list(criterion_l.discriminator.parameters())
                if not freeze_hq:
                    disc_params += list(criterion_h.discriminator.parameters())
                torch.nn.utils.clip_grad_norm_(disc_params, max_norm=grad_clip_norm)
            if scaler_d is not None:
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                optimizer_d.step()

        # ----- Dead code expiry on both branches -----
        if (step + 1) % expire_every == 0:
            with torch.no_grad():
                if not freeze_hq and hasattr(hq_model.quantizer, "expire_dead_codes"):
                    z_for_expire = hq_model.encode(x_h)
                    n = hq_model.quantizer.expire_dead_codes(z_for_expire, images=x_h01)
                    if n:
                        print(f"    [Step {step+1}] HQ expired {n} dead codes")
                if hasattr(lq_model.quantizer, "expire_dead_codes"):
                    z_for_expire = lq_model.encode(x_l)
                    n = lq_model.quantizer.expire_dead_codes(z_for_expire, images=x_l01)
                    if n:
                        print(f"    [Step {step+1}] LQ expired {n} dead codes")

        # ----- Logging -----
        if is_accum_step:
            d_logs_h, d_logs_l = {}, {}
        all_logs = {
            **{f"h_{k}": v for k, v in gen_logs_h.items()},
            **{f"l_{k}": v for k, v in gen_logs_l.items()},
            **{f"h_{k}": v for k, v in (d_logs_h or {}).items()},
            **{f"l_{k}": v for k, v in (d_logs_l or {}).items()},
            "assoc": l_assoc.detach(),
        }
        for k, v in all_logs.items():
            val = v.item() if torch.is_tensor(v) else v
            if not (math.isnan(val) or math.isinf(val)):
                accum[k] += val
        n_steps += 1

        if (step + 1) % log_every == 0:
            msg = f"  OSDFace S1 | Epoch {epoch} | Step {step+1}/{len(loader)}"
            for k in ["h_l1", "l_l1", "h_perceptual", "l_perceptual",
                      "h_vq", "l_vq", "assoc"]:
                if k in accum:
                    msg += f" | {k}: {accum[k]/n_steps:.4f}"
            if nan_steps > 0:
                msg += f" | nan_skips: {nan_steps}"
            print(msg)

    if nan_steps > 0:
        print(f"  WARNING: {nan_steps} NaN/Inf steps skipped this epoch")

    return {k: v / max(n_steps, 1) for k, v in accum.items()}


# ======================================================================
# Main training runner
# ======================================================================

def run(args, device):
    print("=" * 60)
    print("OSDFace Stage 1 (VRE) — joint HQ/LQ VQ-VAE + association loss")
    print("=" * 60)

    # ----- Data -----
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    print(f"Dataset: {len(dataset)} HQ/LQ pairs, {len(loader)} batches/epoch")

    # ----- Models -----
    hq_model = build_hq_vqvae(
        n_codes=args.hq_n_codes, embed_dim=args.embed_dim,
    ).to(device).to(memory_format=torch.channels_last)
    lq_model = build_hq_vqvae(
        n_codes=args.lq_n_codes, embed_dim=args.embed_dim,
    ).to(device).to(memory_format=torch.channels_last)

    print(f"HQ branch params: {sum(p.numel() for p in hq_model.parameters()):,}")
    print(f"LQ branch params: {sum(p.numel() for p in lq_model.parameters()):,}")

    # Warm-start HQ branch from CRAFT Phase A (optional)
    if args.hq_ckpt:
        print(f"Warm-starting HQ branch from {args.hq_ckpt}")
        ck = torch.load(args.hq_ckpt, map_location=device, weights_only=False)
        hq_model.load_state_dict(ck["model"])

    if args.freeze_hq:
        for p in hq_model.parameters():
            p.requires_grad_(False)
        hq_model.eval()
        print("HQ branch: FROZEN")

    if args.grad_ckpt:
        _enable_gradient_checkpointing(hq_model)
        _enable_gradient_checkpointing(lq_model)
        print("Gradient checkpointing enabled")

    # ----- Losses -----
    # One Stage1VQLoss per branch → each owns its own PatchDiscriminator.
    # We keep lambda_assoc=0 inside Stage1VQLoss and add the association term
    # explicitly in the training loop so we can share one assoc loss across
    # both branches and ramp it on a warmup schedule.
    criterion_h = Stage1VQLoss(
        lambda_per=args.lambda_per, lambda_dis=args.lambda_dis,
        lambda_assoc=0.0, vgg_pretrained=True,
    ).to(device)
    criterion_l = Stage1VQLoss(
        lambda_per=args.lambda_per, lambda_dis=args.lambda_dis,
        lambda_assoc=0.0, vgg_pretrained=True,
    ).to(device)
    assoc_loss_fn = AssociationLoss().to(device)

    # ----- Optimizers -----
    gen_params = [p for p in lq_model.parameters() if p.requires_grad]
    if not args.freeze_hq:
        gen_params += [p for p in hq_model.parameters() if p.requires_grad]

    disc_params = list(criterion_l.discriminator.parameters())
    if not args.freeze_hq:
        disc_params += list(criterion_h.discriminator.parameters())

    optimizer_g = torch.optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(disc_params, lr=args.lr, betas=(0.5, 0.999))

    scheduler_g = _build_scheduler(optimizer_g, args, args.epochs)
    scheduler_d = _build_scheduler(optimizer_d, args, args.epochs)

    use_amp = args.amp and device.type == "cuda"
    scaler_g = GradScaler("cuda", init_scale=2**12, growth_interval=4000) if use_amp else None
    scaler_d = GradScaler("cuda", init_scale=2**12, growth_interval=4000) if use_amp else None

    # ----- Resume -----
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0
    latest = os.path.join(ckpt_dir, "latest.pt")
    if not args.fresh and os.path.exists(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)
        hq_model.load_state_dict(ck["hq_model"])
        lq_model.load_state_dict(ck["lq_model"])
        try:
            criterion_h.discriminator.load_state_dict(ck["disc_h"])
            criterion_l.discriminator.load_state_dict(ck["disc_l"])
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

    # ----- Training loop -----
    ckpt = None
    for epoch in range(start_epoch, args.epochs):
        # GAN-loss warmup (both discriminators)
        if args.gan_warmup_epochs > 0 and epoch < args.gan_warmup_epochs:
            frac = epoch / args.gan_warmup_epochs
            criterion_h.lambda_dis = args.lambda_dis * frac
            criterion_l.lambda_dis = args.lambda_dis * frac
        else:
            criterion_h.lambda_dis = args.lambda_dis
            criterion_l.lambda_dis = args.lambda_dis

        # Association-loss warmup
        if epoch < args.assoc_warmup_epochs:
            lambda_assoc = 0.0
        else:
            lambda_assoc = args.lambda_assoc
        print(f"  lambda_dis={criterion_l.lambda_dis:.3f}  lambda_assoc={lambda_assoc:.3f}")

        t0 = time.time()
        avg_logs = train_one_epoch(
            hq_model, lq_model, criterion_h, criterion_l, assoc_loss_fn,
            lambda_assoc, loader, optimizer_g, optimizer_d, device, epoch,
            freeze_hq=args.freeze_hq,
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
            f"OSDFace S1 | Epoch {epoch} done in {elapsed:.0f}s | "
            f"h_L1={avg_logs.get('h_l1', 0):.4f} l_L1={avg_logs.get('l_l1', 0):.4f} "
            f"assoc={avg_logs.get('assoc', 0):.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "hq_model": hq_model.state_dict(),
            "lq_model": lq_model.state_dict(),
            "disc_h": criterion_h.discriminator.state_dict(),
            "disc_l": criterion_l.discriminator.state_dict(),
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

        torch.save(ckpt, os.path.join(ckpt_dir, "latest.pt"))
        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

    if ckpt is not None:
        torch.save(ckpt, os.path.join(ckpt_dir, "final.pt"))
        print(f"OSDFace Stage 1 complete. Saved to {ckpt_dir}/final.pt")


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="OSDFace Stage 1 (VRE) Training")

    parser.add_argument("--config", type=str, default="",
                        help="Path to YAML config (e.g. configs/train_osdface.yaml)")

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
    parser.add_argument("--epochs", type=int, default=50)

    # Loss weights
    parser.add_argument("--lambda_per", type=float, default=1.0)
    parser.add_argument("--lambda_dis", type=float, default=0.8)
    parser.add_argument("--lambda_assoc", type=float, default=1.0)
    parser.add_argument("--assoc_warmup_epochs", type=int, default=5)
    parser.add_argument("--gan_warmup_epochs", type=int, default=2)

    # HQ branch
    parser.add_argument("--hq_ckpt", type=str,
                        default="/projectnb/cs585/projects/craft/checkpoints/phase_a/final.pt",
                        help="CRAFT Phase A checkpoint to warm-start HQ branch (default: reuse CRAFT Phase A)")
    # freeze_hq defaults to True; pass --no_freeze_hq to train HQ jointly.
    parser.add_argument("--freeze_hq", dest="freeze_hq", action="store_true",
                        help="Freeze HQ branch (default)")
    parser.add_argument("--no_freeze_hq", dest="freeze_hq", action="store_false",
                        help="Train HQ branch jointly with LQ branch")
    parser.set_defaults(freeze_hq=True)

    # Checkpointing
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
    run(args, device)
    print("\nOSDFace Stage 1 training complete!")


if __name__ == "__main__":
    main()
