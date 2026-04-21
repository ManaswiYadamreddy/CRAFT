"""
train_stage2.py — CRAFT Stage 2 training: one-step diffusion generator +
                  D³SR-style latent discriminator.

Pipeline (per iteration):
    1. Load batch: (I_H, I_L, I_L_01, masks).
    2. Generator forward (bf16 autocast):
         p_L = Stage1_VRE(I_L, I_L_01)                # frozen, no_grad
         z_L = VAE_enc(I_L)                           # frozen, no_grad
         ε̂   = UNet(z_L, t=T_L, p_L_proj)             # LoRA + prompt_proj trainable
         ẑ_H = (z_L − √(1−ᾱ_{T_L})·ε̂) / √(ᾱ_{T_L})
         Î_H = VAE_dec(ẑ_H)                           # frozen params, grads flow
       Generator loss (OSDFace Eq. 12):
         L_G = λ_MSE·MSE + λ_per·EA-DISTS + λ_ID·ArcFace + λ_dis·softplus(−D(F(ẑ_H,t)))
    3. Discriminator forward (bf16):
         z_H       = VAE_enc(I_H)                     # recomputed w/ no_grad
         z_noisy_R = F(z_H.detach(),  t)
         z_noisy_F = F(ẑ_H.detach(),  t)
         L_D = softplus(−D(real)) + softplus(D(fake))
    4. Alternating G / D updates with AdamW; grad clip 1.0; bf16 autocast (no GradScaler).

Iteration-based (not epoch-based) to match OSDFace's 150K-iter training budget.

Usage:
    python train_stage2.py --config configs/stage2.yaml
    python train_stage2.py --config configs/stage2.yaml --total_iters 100   # dry run
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
from models.face_parser import REGION_NAMES
from models.stage2_generator import Stage2Generator
from models.stage2_discriminator import LatentDiscriminator
from losses.stage2_losses import Stage2Loss


# ======================================================================
# Helpers
# ======================================================================

def _indices_to_masks(region_indices: torch.Tensor, device) -> dict:
    region_indices = region_indices.to(device)
    return {name: region_indices == idx for idx, name in enumerate(REGION_NAMES)}


def _region_mask_512(
    region_indices: torch.Tensor,
    region_names: list[str],
    device,
    size: int = 512,
) -> torch.Tensor:
    """
    Build a (B, 1, size, size) float mask that is 1 inside any of `region_names`
    and 0 elsewhere. `region_indices` is the (B, 16, 16) int64 label grid from
    the dataset — we NN-upsample (no blurring of boundaries) to image size.
    """
    region_indices = region_indices.to(device)
    idxs = [REGION_NAMES.index(n) for n in region_names if n in REGION_NAMES]
    if not idxs:
        return torch.zeros(
            (region_indices.shape[0], 1, size, size),
            device=device, dtype=torch.float32,
        )
    mask = torch.zeros_like(region_indices, dtype=torch.bool)
    for i in idxs:
        mask = mask | (region_indices == i)
    mask = mask.unsqueeze(1).float()                             # (B, 1, 16, 16)
    mask = F.interpolate(mask, size=(size, size), mode="nearest")
    return mask


def _cycle(loader):
    """Infinite iterator over `loader`, re-shuffling at epoch boundaries."""
    while True:
        for batch in loader:
            yield batch


def _enable_unet_grad_ckpt(generator: Stage2Generator) -> bool:
    """Try the several places PEFT / diffusers may expose gradient ckpt."""
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


def _trainable_state_dict(model: nn.Module) -> dict:
    """State dict of only trainable parameters (saves disk on LoRA ckpts)."""
    return {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def _load_trainable_state(model: nn.Module, state: dict, strict: bool = False):
    """Load a trainable-only state dict back into `model`."""
    own = dict(model.named_parameters())
    missing, unexpected = [], []
    with torch.no_grad():
        for k, v in state.items():
            if k in own:
                own[k].data.copy_(v.to(own[k].device, dtype=own[k].dtype))
            else:
                unexpected.append(k)
        for k in own:
            if own[k].requires_grad and k not in state:
                missing.append(k)
    if strict and (missing or unexpected):
        raise RuntimeError(f"load_trainable_state: missing={missing}, unexpected={unexpected}")
    return missing, unexpected


def _lr_schedule_scale(iter_idx: int, total: int, warmup: int, min_ratio: float) -> float:
    """Linear warmup then cosine decay to `min_ratio`."""
    if iter_idx < warmup:
        return max(iter_idx, 1) / max(warmup, 1)
    if total <= warmup:
        return 1.0
    progress = (iter_idx - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def _set_lr(optimizer, base_lr: float, scale: float):
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


# ======================================================================
# Sampling (dump a (I_L, Î_H, I_H) grid every `sample_every` iters)
# ======================================================================

@torch.no_grad()
def dump_samples(
    generator: Stage2Generator,
    batch: dict,
    device,
    amp_dtype: torch.dtype,
    out_path: str,
):
    """Save a single-file grid: rows of [LQ | pred | HQ]."""
    I_L = batch["lq"].to(device)
    I_L_01 = batch["lq_01"].to(device)
    I_H = batch["hq"].to(device)
    masks = (
        _indices_to_masks(batch["lq_mask"], device)
        if "lq_mask" in batch else None
    )

    was_training = generator.training
    generator.eval()
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
        I_hat, _, _, _ = generator(I_L, I_L_01, I_H_11=None, masks=masks)
    generator.train(was_training)

    # Build (3B, 3, H, W) tensor: rows = [LQ, pred, HQ]
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

    print(f"Device: {device}   AMP: {use_amp} ({args.amp_dtype})")

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

    # Peel off one fixed batch for periodic visualisation (reproducible)
    # (a second DataLoader so we don't disturb the shuffled training iter)
    vis_loader = DataLoader(
        dataset, batch_size=min(args.num_sample_images, args.batch_size * 4),
        shuffle=False, num_workers=0, pin_memory=True,
    )
    vis_batch = next(iter(vis_loader))

    # ----- generator -----
    generator = Stage2Generator(
        stage1_ckpt_path=args.stage1_ckpt,
        parser_ckpt=args.parser_ckpt,
        sd_model_name=args.sd_model_id,
        sd_cache_dir=args.sd_cache_dir or None,
        t_fixed=args.t_low,
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
        prompt_dim=512,
        context_dim=args.prompt_ctx_dim,
    ).to(device)
    print(generator.describe())

    if args.grad_ckpt:
        ok = _enable_unet_grad_ckpt(generator)
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

    # ----- loss -----
    loss_module = Stage2Loss(
        alphas_cumprod=generator.alphas_cumprod.clone(),
        lambda_mse=args.lambda_mse,
        lambda_per=args.lambda_per,
        lambda_id=args.lambda_id,
        lambda_dis=args.lambda_dis,  # full value; we ramp externally via gan_warmup
        lambda_region_l1=args.lambda_region_l1,
        enable_id=args.enable_id,
        t_max_dis=args.t_max_dis,
    ).to(device)
    print(
        f"Stage2Loss | λ_mse={args.lambda_mse} λ_per={args.lambda_per} "
        f"λ_id={args.lambda_id} λ_dis={args.lambda_dis} "
        f"λ_region={args.lambda_region_l1} (regions={args.region_l1_regions}) | "
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
    # Priority: (1) explicit --resume path, (2) auto-resume from ckpt_dir/latest.pt
    # unless --fresh is set. Stage 1 uses the same convention.
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
    g_step = 0  # counts G updates actually taken

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
        I_L_01 = batch["lq_01"].to(device, non_blocking=True)
        I_H = batch["hq"].to(device, non_blocking=True)
        masks = _indices_to_masks(batch["lq_mask"], device) if "lq_mask" in batch else None
        # Region mask (512×512 float) for the region-L1 term. Prefer HQ labels
        # if present, else LQ; else None (term is a no-op at λ=0 anyway).
        region_mask = None
        if args.lambda_region_l1 > 0 and args.region_l1_regions:
            hq_idx = batch.get("hq_mask", batch.get("lq_mask"))
            if hq_idx is not None:
                region_mask = _region_mask_512(
                    hq_idx, list(args.region_l1_regions), device,
                    size=args.image_size,
                )

        # ================================================================
        # Generator step
        # ================================================================
        optim_g.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            I_hat, z_hat, z_H, _ = generator(I_L, I_L_01, I_H_11=I_H, masks=masks)
            g_loss, g_logs = loss_module.generator_loss(
                I_H_11=I_H,
                I_hat_H_11=I_hat,
                z_hat_H=z_hat,
                discriminator=discriminator,
                region_mask=region_mask,
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
        # Discriminator step (every `d_update_every` G updates)
        # ================================================================
        d_logs = {}
        if (it % args.d_update_every) == 0:
            optim_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                # z_H was encoded above (no grads) but we re-use to avoid redundant encode
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

                # -- R1 gradient penalty on D (every r1_every steps, fp32) --
                # Separate backward pass; penalty gradients accumulate into
                # D.grad alongside the main D loss before the optimizer step.
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
            for k in ("mse", "ea_dists", "id", "region_l1", "gan_g",
                      "total_gen", "d_real", "d_fake", "r1"):
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
    }
    torch.save(state, final)
    print(f"Training complete. Final checkpoint: {final}")


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CRAFT Stage 2 training")
    p.add_argument("--config", type=str, default="")

    # data
    p.add_argument("--data_root", type=str)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--image_size", type=int, default=512)

    # stage1 / parser
    p.add_argument("--stage1_ckpt", type=str)
    p.add_argument("--parser_ckpt", type=str)

    # backbone
    p.add_argument("--sd_model_id", type=str,
                   default="stabilityai/stable-diffusion-2-1-base")
    p.add_argument("--sd_cache_dir", type=str, default="",
                   help="Override HuggingFace cache dir (empty = default)")
    p.add_argument("--t_low", type=int, default=999)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", nargs="+",
                   default=["to_q", "to_k", "to_v", "to_out.0"])
    p.add_argument("--prompt_ctx_dim", type=int, default=1024)

    # discriminator
    p.add_argument("--disc_base_ch", type=int, default=64)
    p.add_argument("--disc_t_embed_dim", type=int, default=256)
    p.add_argument("--disc_t_mlp_dim", type=int, default=512)

    # optimization
    p.add_argument("--total_iters", type=int, default=150_000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--betas_g", nargs=2, type=float, default=[0.9, 0.999])
    p.add_argument("--betas_d", nargs=2, type=float, default=[0.5, 0.999])
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--lr_schedule", type=str, default="cosine",
                   choices=["none", "cosine"])
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--gan_warmup_iters", type=int, default=5000)
    p.add_argument("--d_update_every", type=int, default=1)
    p.add_argument("--t_max_dis", type=int, default=1000,
                   help="Adversarial t sampled from U{1, t_max_dis-1}.")
    p.add_argument("--r1_gamma", type=float, default=0.0,
                   help="R1 gradient penalty coefficient on D (0 = disabled).")
    p.add_argument("--r1_every", type=int, default=16,
                   help="Apply R1 every N D steps (lazy regularization).")

    # losses
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--lambda_per", type=float, default=1.0)
    p.add_argument("--lambda_id",  type=float, default=0.1)
    p.add_argument("--lambda_dis", type=float, default=0.8)
    p.add_argument("--lambda_region_l1", type=float, default=0.0,
                   help="Weight for region-L1 term on parsed regions.")
    p.add_argument("--region_l1_regions", nargs="+", default=["lips", "eyes"],
                   help="BiSeNet region names to include in the region-L1 mask.")
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
    p.add_argument("--resume", type=str, default="",
                   help="Explicit checkpoint path to resume from. Overrides auto-resume.")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore ckpt_dir/latest.pt and start from step 0.")
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
    print(f"Config:\n{json.dumps(vars(args), indent=2, default=str)}")
    train(args)


if __name__ == "__main__":
    main()
