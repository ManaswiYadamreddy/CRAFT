"""
train_stage2_osdface.py — OSDFace Stage 2 baseline training.

Parallel to train_stage2.py (CRAFT). Differences are isolated to:
    - Generator: Stage2GeneratorOSDFace (GlobalVQ prompt, no face parser).
    - No region-aware masks or region-weighted L1 term.
    - Dedicated config (configs/stage2_osdface.yaml) and checkpoint dir.

Everything else — latent-space discriminator, loss module (MSE + EA-DISTS +
ArcFace + adversarial + R1), AdamW + cosine schedule + GAN warmup + TTUR,
bf16 AMP, gradient checkpointing — is shared byte-for-byte via the
`stage2_*` modules and imported helpers from `train_stage2`.

Usage:
    python train_stage2_osdface.py --config configs/stage2_osdface.yaml
    python train_stage2_osdface.py --config configs/stage2_osdface.yaml --fresh
"""

from __future__ import annotations

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
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from data.dataset import FFHQPairedDataset
from models.stage2_generator_osdface import Stage2GeneratorOSDFace
from models.stage2_discriminator import LatentDiscriminator
from losses.stage2_losses import Stage2Loss

# Reuse helpers from the CRAFT trainer — these are pure utility functions
# (no side effects on import) so sharing them keeps the two paths in sync.
from train_stage2 import (
    _cycle,
    _trainable_state_dict,
    _load_trainable_state,
    _lr_schedule_scale,
    _set_lr,
)


# ======================================================================
# OSDFace-specific helpers
# ======================================================================

def _enable_unet_grad_ckpt_osdface(generator: Stage2GeneratorOSDFace) -> bool:
    """Same logic as the CRAFT variant — PEFT / diffusers expose this in
    several places depending on wrapping depth."""
    candidates = [
        generator.unet,
        getattr(generator.unet, "base_model", None),
        getattr(getattr(generator.unet, "base_model", None), "model", None),
    ]
    for obj in candidates:
        if obj is not None and hasattr(obj, "enable_gradient_checkpointing"):
            obj.enable_gradient_checkpointing()
            return True
    return False


# ======================================================================
# Sampling
# ======================================================================

@torch.no_grad()
def dump_samples(
    generator: Stage2GeneratorOSDFace,
    batch: dict,
    device,
    amp_dtype: torch.dtype,
    out_path: str,
):
    """Save a (I_L, Î_H, I_H) grid as rows, batch elements as columns."""
    I_L = batch["lq"].to(device)
    I_H = batch["hq"].to(device)

    was_training = generator.training
    generator.eval()
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
        I_hat, _, _, _ = generator(I_L, I_L_01=None, I_H_11=None)
    generator.train(was_training)

    I_hat = I_hat.clamp(-1, 1)
    def to01(x): return (x.float() + 1.0) * 0.5
    grid_t = torch.cat([to01(I_L), to01(I_hat), to01(I_H)], dim=0)
    grid = make_grid(grid_t, nrow=I_L.shape[0])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)


# ======================================================================
# Training
# ======================================================================

def train(args):
    # ----- device / dtype / seed -----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
    }[args.amp_dtype]
    use_amp = bool(args.amp) and device.type == "cuda"

    print(f"Device: {device}   AMP: {use_amp} ({args.amp_dtype})  [OSDFace baseline]")

    # ----- data -----
    dataset = FFHQPairedDataset(data_root=args.data_root, hq_only=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    print(f"Dataset: {len(dataset)} HQ/LQ pairs, {len(loader)} batches/epoch")
    data_iter = iter(_cycle(loader))

    # Fixed visualisation batch (reproducible across iters)
    vis_loader = DataLoader(
        dataset, batch_size=min(args.num_sample_images, args.batch_size * 4),
        shuffle=False, num_workers=0, pin_memory=True,
    )
    vis_batch = next(iter(vis_loader))

    # ----- generator (OSDFace baseline) -----
    generator = Stage2GeneratorOSDFace(
        stage1_ckpt_path=args.stage1_ckpt,
        sd_model_name=args.sd_model_id,
        sd_cache_dir=args.sd_cache_dir or None,
        t_fixed=args.t_low,
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        prompt_dim=args.stage1_embed_dim,
        context_dim=args.prompt_ctx_dim,
        stage1_n_codes=args.stage1_n_codes,
    ).to(device)
    print(generator.describe())

    if args.grad_ckpt:
        ok = _enable_unet_grad_ckpt_osdface(generator)
        print(f"UNet gradient checkpointing: {'ENABLED' if ok else 'FAILED'}")
    if args.vae_slicing:
        generator.vae.enable_slicing()
    if args.vae_tiling:
        generator.vae.enable_tiling()

    # ----- discriminator -----
    discriminator = LatentDiscriminator(
        in_channels=4,
        base_ch=args.disc_base_ch,
        t_embed_dim=args.disc_t_embed_dim,
        t_mlp_dim=args.disc_t_mlp_dim,
    ).to(device)
    print(f"LatentDiscriminator: {discriminator.num_parameters() / 1e6:.2f} M params")

    # ----- loss (region-L1 is a no-op at λ=0) -----
    loss_module = Stage2Loss(
        alphas_cumprod=generator.alphas_cumprod.clone(),
        lambda_mse=args.lambda_mse,
        lambda_per=args.lambda_per,
        lambda_id=args.lambda_id,
        lambda_dis=args.lambda_dis,
        lambda_region_l1=args.lambda_region_l1,  # expect 0 for OSDFace
        enable_id=args.enable_id,
        t_max_dis=args.t_max_dis,
    ).to(device)
    print(
        f"Stage2Loss | λ_mse={args.lambda_mse} λ_per={args.lambda_per} "
        f"λ_id={args.lambda_id} λ_dis={args.lambda_dis} "
        f"λ_region={args.lambda_region_l1} | "
        f"t_max_dis={args.t_max_dis} | r1_γ={args.r1_gamma} every={args.r1_every}"
    )

    # ----- optimizers -----
    gen_params = [p for p in generator.parameters() if p.requires_grad]
    optim_g = torch.optim.AdamW(
        gen_params, lr=args.lr_g, betas=tuple(args.betas_g),
        weight_decay=args.weight_decay,
    )
    optim_d = torch.optim.AdamW(
        discriminator.parameters(), lr=args.lr_d, betas=tuple(args.betas_d),
        weight_decay=args.weight_decay,
    )

    # ----- dirs -----
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # ----- resume -----
    resume_path = ""
    if args.resume and os.path.exists(args.resume):
        resume_path = args.resume
    elif not args.fresh:
        latest = os.path.join(args.ckpt_dir, "latest.pt")
        if os.path.exists(latest):
            resume_path = latest
            print(f"Auto-resume: found {latest}")

    start_iter = 0
    if resume_path:
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        missing, unexpected = _load_trainable_state(generator, ckpt["generator"])
        print(f"  generator trainable params: "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        discriminator.load_state_dict(ckpt["discriminator"])
        optim_g.load_state_dict(ckpt["optim_g"])
        optim_d.load_state_dict(ckpt["optim_d"])
        start_iter = int(ckpt.get("iter", 0))
        print(f"  resumed at iter {start_iter}")
    else:
        print("Starting fresh (no checkpoint to resume from)")

    # ----- main loop -----
    accum = defaultdict(float)
    nan_steps = 0
    t_start = time.time()
    g_step = 0

    generator.train()
    discriminator.train()

    for it in range(start_iter, args.total_iters):
        # ---- LR schedule & GAN warmup ----
        if args.lr_schedule == "cosine":
            scale = _lr_schedule_scale(
                it, args.total_iters, args.warmup_iters, args.min_lr_ratio,
            )
        else:
            scale = 1.0
        _set_lr(optim_g, args.lr_g, scale)
        _set_lr(optim_d, args.lr_d, scale)

        if args.gan_warmup_iters > 0 and it < args.gan_warmup_iters:
            loss_module.lambda_dis = args.lambda_dis * (it / args.gan_warmup_iters)
        else:
            loss_module.lambda_dis = args.lambda_dis

        # ---- batch ----
        batch = next(data_iter)
        I_L = batch["lq"].to(device, non_blocking=True)
        I_H = batch["hq"].to(device, non_blocking=True)

        # ================================================================
        # Generator step
        # ================================================================
        optim_g.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            # OSDFace generator ignores I_L_01/masks — pass None.
            I_hat, z_hat, z_H, _ = generator(I_L, I_L_01=None, I_H_11=I_H)
            g_loss, g_logs = loss_module.generator_loss(
                I_H_11=I_H,
                I_hat_H_11=I_hat,
                z_hat_H=z_hat,
                discriminator=discriminator,
                region_mask=None,  # OSDFace baseline: no regions
            )

        if torch.isnan(g_loss) or torch.isinf(g_loss):
            nan_steps += 1
            print(f"  [iter {it}] NaN/Inf in g_loss — skipping")
            optim_g.zero_grad(set_to_none=True)
            optim_d.zero_grad(set_to_none=True)
            del I_hat, z_hat, z_H, g_loss, g_logs
            torch.cuda.empty_cache()
            continue

        g_loss.backward()
        if args.grad_clip_norm > 0:
            g_gnorm = torch.nn.utils.clip_grad_norm_(gen_params, args.grad_clip_norm)
            accum["g_gnorm_sum"] += float(g_gnorm)
            accum["g_gnorm_cnt"] += 1.0
        optim_g.step()

        # ================================================================
        # Discriminator step
        # ================================================================
        d_logs = {}
        if (it % args.d_update_every) == 0:
            optim_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                d_loss, d_logs = loss_module.discriminator_loss(
                    z_H=z_H.detach(),
                    z_hat_H=z_hat.detach(),
                    discriminator=discriminator,
                )
            if torch.isnan(d_loss) or torch.isinf(d_loss):
                nan_steps += 1
                print(f"  [iter {it}] NaN/Inf in d_loss — skipping D update")
            else:
                d_loss.backward()

                # R1 gradient penalty — lazy, every r1_every D steps.
                if (
                    args.r1_gamma > 0
                    and args.r1_every > 0
                    and (it % args.r1_every) == 0
                ):
                    r1_loss, r1_logs = loss_module.r1_penalty(
                        z_H=z_H.detach(), discriminator=discriminator,
                        gamma=args.r1_gamma,
                    )
                    if not (torch.isnan(r1_loss) or torch.isinf(r1_loss)):
                        r1_loss.backward()
                        d_logs.update(r1_logs)

                if args.grad_clip_norm > 0:
                    d_gnorm = torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(), args.grad_clip_norm,
                    )
                    accum["d_gnorm_sum"] += float(d_gnorm)
                    accum["d_gnorm_cnt"] += 1.0
                optim_d.step()

        # ---- accumulate logs ----
        for k, v in {**g_logs, **d_logs}.items():
            val = v.item() if torch.is_tensor(v) else float(v)
            if not (math.isnan(val) or math.isinf(val)):
                accum[k] += val
        g_step += 1

        # ---- periodic log ----
        if (it + 1) % args.log_every == 0:
            dt = time.time() - t_start
            rate = g_step / max(dt, 1e-6)
            msg = (
                f"it {it+1:>6d}/{args.total_iters} | "
                f"{rate:.2f} it/s | λ_dis={loss_module.lambda_dis:.3f} | "
                f"lr={optim_g.param_groups[0]['lr']:.1e}"
            )
            for k in ("mse", "ea_dists", "id", "gan_g", "total_gen",
                      "d_real", "d_fake", "r1"):
                if k in accum:
                    msg += f" | {k}={accum[k]/max(g_step,1):+.4f}"
            if accum.get("g_gnorm_cnt", 0) > 0:
                msg += f" | gnG={accum['g_gnorm_sum']/accum['g_gnorm_cnt']:.2f}"
            if accum.get("d_gnorm_cnt", 0) > 0:
                msg += f" | gnD={accum['d_gnorm_sum']/accum['d_gnorm_cnt']:.2f}"
            if nan_steps > 0:
                msg += f" | nan_skips={nan_steps}"
            print(msg, flush=True)
            accum.clear()
            g_step = 0
            t_start = time.time()

        # ---- periodic sample ----
        if (it + 1) % args.sample_every == 0:
            sample_path = os.path.join(args.sample_dir, f"iter_{it+1:07d}.png")
            try:
                dump_samples(generator, vis_batch, device, amp_dtype, sample_path)
                print(f"  [iter {it+1}] saved sample → {sample_path}")
            except Exception as e:
                print(f"  [iter {it+1}] sample dump failed: {e}")

        # ---- periodic checkpoint ----
        if (it + 1) % args.save_every == 0 or (it + 1) == args.total_iters:
            ckpt_path = os.path.join(args.ckpt_dir, f"iter_{it+1:07d}.pt")
            latest = os.path.join(args.ckpt_dir, "latest.pt")
            state = {
                "iter": it + 1,
                "generator": _trainable_state_dict(generator),
                "discriminator": discriminator.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "args": vars(args),
                "backbone": "osdface",
            }
            torch.save(state, ckpt_path)
            torch.save(state, latest)
            print(f"  [iter {it+1}] saved ckpt → {ckpt_path}")

    # ---- final save ----
    final = os.path.join(args.ckpt_dir, "final.pt")
    state = {
        "iter": args.total_iters,
        "generator": _trainable_state_dict(generator),
        "discriminator": discriminator.state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "args": vars(args),
        "backbone": "osdface",
    }
    torch.save(state, final)
    print(f"Training complete (OSDFace baseline). Final checkpoint: {final}")


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CRAFT Stage 2 — OSDFace baseline training")
    p.add_argument("--config", type=str, default="")

    # data
    p.add_argument("--data_root", type=str)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=512)

    # stage1 (no parser for OSDFace)
    p.add_argument("--stage1_ckpt", type=str)
    p.add_argument("--stage1_n_codes", type=int, default=1024)
    p.add_argument("--stage1_embed_dim", type=int, default=512)

    # backbone
    p.add_argument("--sd_model_id", type=str,
                   default="sd-legacy/stable-diffusion-v1-5")
    p.add_argument("--sd_cache_dir", type=str, default="")
    p.add_argument("--t_low", type=int, default=999)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", nargs="+",
                   default=["to_q", "to_k", "to_v", "to_out.0"])
    p.add_argument("--prompt_ctx_dim", type=int, default=768)

    # discriminator
    p.add_argument("--disc_base_ch", type=int, default=64)
    p.add_argument("--disc_t_embed_dim", type=int, default=256)
    p.add_argument("--disc_t_mlp_dim", type=int, default=512)

    # optimization
    p.add_argument("--total_iters", type=int, default=150_000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=4e-5)
    p.add_argument("--betas_g", nargs=2, type=float, default=[0.9, 0.999])
    p.add_argument("--betas_d", nargs=2, type=float, default=[0.5, 0.999])
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--lr_schedule", type=str, default="cosine",
                   choices=["none", "cosine"])
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--gan_warmup_iters", type=int, default=20000)
    p.add_argument("--d_update_every", type=int, default=1)
    p.add_argument("--t_max_dis", type=int, default=500)
    p.add_argument("--r1_gamma", type=float, default=10.0)
    p.add_argument("--r1_every", type=int, default=16)

    # losses (no region-L1 for OSDFace, but arg present so a unified logger works)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_per", type=float, default=1.0)
    p.add_argument("--lambda_id",  type=float, default=0.1)
    p.add_argument("--lambda_dis", type=float, default=0.3)
    p.add_argument("--lambda_region_l1", type=float, default=0.0)
    p.add_argument("--enable_id", action="store_true")
    p.add_argument("--no_id", dest="enable_id", action="store_false")
    p.set_defaults(enable_id=True)

    # memory
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--amp_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--grad_ckpt", action="store_true", default=True)
    p.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    p.add_argument("--vae_slicing", action="store_true", default=True)
    p.add_argument("--vae_tiling", action="store_true", default=False)

    # ckpt/log
    p.add_argument("--ckpt_dir", type=str)
    p.add_argument("--sample_dir", type=str)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_sample_images", type=int, default=8)

    # misc
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda")

    # two-pass YAML override
    args, _ = p.parse_known_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        p.set_defaults(**cfg)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Config (OSDFace baseline):\n{json.dumps(vars(args), indent=2, default=str)}")
    train(args)


if __name__ == "__main__":
    main()
