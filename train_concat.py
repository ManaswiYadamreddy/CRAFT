"""
train_concat.py — Concatenation-based text conditioning for OSDFace (Option B)

Strategy: merge the pretrained OSDFace LoRA permanently into the SD 2.1 UNet
base weights, then train a fresh LoRA (rank 32) on top of the merged model.

Why Option B over Option A (continuing their LoRA):
  - Their face restoration knowledge is baked into the base weights — it cannot
    be overwritten or interfered with during training.
  - Your new LoRA only needs to learn the incremental task: how to use the
    concatenated visual + text tokens. Simpler task = faster convergence.
  - Higher rank (32 vs 16) gives more capacity to learn text conditioning
    without any extra VRAM cost.

The only architectural change vs. the original forward pass is one line:

    BEFORE:  encoder_hidden_states = visual_embeds              # (B, 77, 1024)
    AFTER:   encoder_hidden_states = cat([visual, text], dim=1) # (B, 154, 1024)

What is frozen vs trained:
    VQ-VAE Encoder (VRE)   — FROZEN
    TwoLayerConv1x1        — FROZEN  (loaded from pretrained checkpoint)
    CLIP Text Encoder      — FROZEN
    SD 2.1 VAE             — FROZEN
    Merged UNet base       — FROZEN  (OSDFace LoRA baked in)
    Your new LoRA (rank 32) — TRAINED

Dataset layout:
    data/train/
        lq/            ← low-quality faces
        hq/            ← high-quality ground truth
        prompts.json   ← {"filename.png": "text prompt", ...}

Usage:
    python train_concat.py \
        --lq_dir data/train/lq \
        --hq_dir data/train/hq \
        --prompts_json data/train/prompts.json \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base \
        --img_encoder_weight pretrained/associate_2.ckpt \
        --ckpt_path pretrained \
        --output_dir checkpoints/concat_v1 \
        --mixed_precision bf16 \
        --batch_size 4 \
        --max_train_steps 50000
"""

import os
import sys
import json
import copy
import glob
import random
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from tqdm import tqdm
from safetensors import safe_open

from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer

from lq_embed import vqvae_encoder, TwoLayerConv1x1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA merge helper  (mirrors merge_Unet from infer.py)
# ---------------------------------------------------------------------------

def merge_lora_into_unet(args):
    """
    Loads the pretrained OSDFace LoRA weights and permanently bakes them into
    a fresh copy of the SD 2.1 UNet. Returns the merged UNet with no LoRA
    adapters attached — their knowledge is now part of the base weights.
    """
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    # Use the PRETRAINED LoRA's rank/alpha for the merge scale —
    # NOT args.lora_rank/alpha which belong to your new LoRA (rank 32).
    # OSDFace was trained at rank=16, alpha=16 → scale = 1.0
    alpha = float(args.pretrained_lora_alpha / args.pretrained_lora_rank)
    processed_keys = set()

    lora_path = os.path.join(args.ckpt_path, "pytorch_lora_weights.safetensors")
    with safe_open(lora_path, framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    sd_unet = unet.state_dict()

    for key in state_dict:
        if "lora_A" in key:
            lora_b_key = key.replace("lora_A", "lora_B")
            unet_key   = key.replace(".lora_A.weight", ".weight").replace("unet.", "")
            W_A = state_dict[key]
            W_B = state_dict[lora_b_key]
            orig = sd_unet[unet_key]
            processed_keys.update([key, lora_b_key])

            if orig.ndim == 4:
                # Conv weights
                rank = W_A.shape[0]
                out_ch, in_ch, kH, kW = orig.shape
                delta = torch.matmul(
                    W_B.view(out_ch, rank),
                    W_A.view(rank, -1),
                ).view(out_ch, in_ch, kH, kW)
            else:
                delta = torch.mm(W_B, W_A)

            sd_unet[unet_key] = orig + alpha * delta

        elif "lora.up.weight" in key:
            lora_down_key = key.replace("lora.up.weight", "lora.down.weight")
            unet_key      = key.replace(".lora.up.weight", ".weight").replace("unet.", "")
            W_up   = state_dict[key]
            W_down = state_dict[lora_down_key]
            orig   = sd_unet[unet_key]
            processed_keys.update([key, lora_down_key])
            if orig.ndim == 2:
                sd_unet[unet_key] = orig + alpha * torch.mm(W_up, W_down)

    remaining = [k for k in state_dict if k not in processed_keys]
    if remaining:
        logger.warning(f"Unprocessed LoRA keys: {remaining}")

    unet.load_state_dict(sd_unet)
    logger.info("OSDFace LoRA merged into UNet base weights.")
    return unet


# ---------------------------------------------------------------------------
# Dataset  (identical to train_fusion.py)
# ---------------------------------------------------------------------------

class FaceRestorationDataset(Dataset):
    def __init__(self, lq_dir, hq_dir, prompts_json, resolution=512):
        self.resolution = resolution

        lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.[jpJP][pnPN]*[gG]")))
        hq_paths = sorted(glob.glob(os.path.join(hq_dir, "*.[jpJP][pnPN]*[gG]")))
        assert len(lq_paths) == len(hq_paths), (
            f"LQ ({len(lq_paths)}) and HQ ({len(hq_paths)}) counts don't match."
        )

        with open(prompts_json, "r") as f:
            raw = json.load(f)
        self.prompts = {item["image"]: item["pos"] for item in raw}
        
        self.pairs = list(zip(lq_paths, hq_paths))

        self.transform = transforms.Compose([
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ])
        logger.info(f"Dataset: {len(self.pairs)} pairs loaded.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lq_path, hq_path = self.pairs[idx]
        lq = Image.open(lq_path).convert("RGB")
        hq = Image.open(hq_path).convert("RGB")

        if random.random() > 0.5:
            lq = TF.hflip(lq)
            hq = TF.hflip(hq)

        lq = self.transform(lq) * 2.0 - 1.0   # [-1, 1]
        hq = self.transform(hq) * 2.0 - 1.0

        filename = os.path.basename(lq_path)
        prompt = self.prompts.get(filename, "")
        return {"lq": lq, "hq": hq, "prompt": prompt, "filename": filename}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t  = 1 - alpha_prod_t
    return (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5


@torch.no_grad()
def encode_text(prompts, tokenizer, text_encoder, device, weight_dtype):
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    return text_encoder(tokens).last_hidden_state.to(weight_dtype)  # (B, 77, 1024)


def save_checkpoint(unet, output_dir, step):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    unet.save_pretrained(ckpt_dir)
    logger.info(f"Checkpoint saved → {ckpt_dir}")


# ---------------------------------------------------------------------------
# Build models
# ---------------------------------------------------------------------------

def build_models(args, device):
    weight_dtype = {"fp32": torch.float32,
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16}[args.mixed_precision]

    logger.info("Loading SD 2.1 VAE...")
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(device, dtype=weight_dtype)
    vae.requires_grad_(False)

    logger.info("Loading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    ).to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    # ── Option B: merge their LoRA into the base UNet, then add a fresh one ──
    # Step 1: merge OSDFace LoRA permanently into the SD 2.1 UNet weights.
    #         Their face restoration knowledge becomes the new "base" —
    #         it cannot be overwritten during your training.
    logger.info("Merging pretrained OSDFace LoRA into UNet base weights...")
    unet = merge_lora_into_unet(args)   # returns a clean UNet with delta baked in
    unet.requires_grad_(False)          # freeze the merged base entirely

    # Step 2: add your own fresh LoRA on top at rank 32.
    #         Only this new LoRA will be trained — learning to use
    #         the concatenated visual + text conditioning.
    logger.info(f"Adding fresh LoRA (rank={args.lora_rank}) on merged UNet...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )
    unet = get_peft_model(unet, lora_config)
    unet.to(device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    logger.info("Loading VQ-VAE image encoder (frozen)...")
    img_encoder = vqvae_encoder(args).to(device, dtype=weight_dtype)
    img_encoder.requires_grad_(False)
    img_encoder.eval()

    # Keep TwoLayerConv1x1 for visual token projection (512 → 1024)
    # This is loaded from the existing checkpoint — also frozen
    embedding_change = TwoLayerConv1x1(512, 1024)
    embedding_change.load_state_dict(
        torch.load(
            os.path.join(args.ckpt_path, "embedding_change_weights.pth"),
            weights_only=False,
        )
    )
    embedding_change.to(device, dtype=weight_dtype)
    embedding_change.requires_grad_(False)
    embedding_change.eval()

    return (
        noise_scheduler, vae, tokenizer, text_encoder,
        unet, img_encoder, embedding_change, weight_dtype,
    )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_step(unet, vae, img_encoder, embedding_change,
              tokenizer, text_encoder, noise_scheduler,
              dataloader, device, weight_dtype, step):
    unet.eval()
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    timestep = 399
    total_loss, n = 0.0, 0

    for batch in dataloader:
        lq      = batch["lq"].to(device, dtype=weight_dtype)
        hq      = batch["hq"].to(device, dtype=weight_dtype)
        prompts = batch["prompt"]

        # Visual tokens → (B, 77, 1024)
        visual_embeds = embedding_change(
            img_encoder(lq).reshape(lq.shape[0], 77, -1)
        )

        # Text tokens → (B, 77, 1024)
        text_embeds = encode_text(prompts, tokenizer, text_encoder, device, weight_dtype)

        # ── Concatenate along sequence dimension ──────────────────────────
        prompt_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, 154, 1024)

        hq_latent = vae.encode(hq).latent_dist.sample() * vae.config.scaling_factor
        lq_latent = vae.encode(lq).latent_dist.sample() * vae.config.scaling_factor

        model_pred = unet(lq_latent, timestep, encoder_hidden_states=prompt_embeds).sample
        x0_pred = get_x0_from_noise(
            lq_latent.double(), model_pred.double(),
            alphas_cumprod.double(), timestep
        ).float().to(weight_dtype)

        total_loss += F.l1_loss(x0_pred, hq_latent).item()
        n += 1
        if n >= 50:
            break

    logger.info(f"[Eval @ step {step}] L1 loss: {total_loss / max(n,1):.5f}")
    unet.train()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    (
        noise_scheduler, vae, tokenizer, text_encoder,
        unet, img_encoder, embedding_change, weight_dtype,
    ) = build_models(args, device)

    # Dataset
    dataset = FaceRestorationDataset(
        args.lq_dir, args.hq_dir, args.prompts_json
    )
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # Only UNet LoRA params are trainable — everything else is frozen
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    total = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable parameters: {total:,}  (UNet LoRA only)")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=1e-2,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    use_amp = args.mixed_precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    timestep = 399

    unet.train()
    img_encoder.eval()
    text_encoder.eval()
    embedding_change.eval()
    vae.eval()

    global_step = 0
    data_iter = iter(train_loader)

    logger.info(f"Starting training for {args.max_train_steps} steps...")
    pbar = tqdm(total=args.max_train_steps, desc="Training")

    while global_step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        lq      = batch["lq"].to(device, dtype=weight_dtype)
        hq      = batch["hq"].to(device, dtype=weight_dtype)
        prompts = batch["prompt"]

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):

            # ── Visual tokens (frozen VRE + projection) ───────────────────
            with torch.no_grad():
                visual_embeds = embedding_change(
                    img_encoder(lq).reshape(lq.shape[0], 77, -1)
                )  # (B, 77, 1024)

            # ── Text embeddings (frozen CLIP) ─────────────────────────────
            with torch.no_grad():
                text_embeds = encode_text(
                    prompts, tokenizer, text_encoder, device, weight_dtype
                )  # (B, 77, 1024)

            # ── Concatenate ───────────────────────────────────────────────
            #   visual tokens come first so the UNet's existing cross-attention
            #   weights still "see" them in the positions they were trained on.
            #   Text tokens occupy the second half (positions 77–153).
            prompt_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            # (B, 154, 1024)

            # ── VAE encode ────────────────────────────────────────────────
            with torch.no_grad():
                hq_latent = (
                    vae.encode(hq).latent_dist.sample()
                    * vae.config.scaling_factor
                )
                lq_latent = (
                    vae.encode(lq).latent_dist.sample()
                    * vae.config.scaling_factor
                )

            # ── UNet forward ──────────────────────────────────────────────
            model_pred = unet(
                lq_latent, timestep,
                encoder_hidden_states=prompt_embeds,
            ).sample

            # ── Loss ──────────────────────────────────────────────────────
            x0_pred = get_x0_from_noise(
                lq_latent.double(), model_pred.double(),
                alphas_cumprod.double(), timestep,
            ).float().to(weight_dtype)

            loss_latent = F.l1_loss(x0_pred, hq_latent)

            # Optional pixel-space loss (decode x0 and compare to HQ)
            if args.pixel_loss_weight > 0:
                with torch.no_grad():
                    x0_decoded = vae.decode(
                        x0_pred / vae.config.scaling_factor
                    ).sample.clamp(-1, 1)
                loss_pixel = F.l1_loss(x0_decoded, hq)
            else:
                loss_pixel = torch.tensor(0.0, device=device)

            loss = loss_latent + args.pixel_loss_weight * loss_pixel

        # ── Backward ──────────────────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        global_step += 1
        pbar.update(1)
        pbar.set_postfix({
            "loss":  f"{loss.item():.4f}",
            "l1":    f"{loss_latent.item():.4f}",
            "pixel": f"{loss_pixel.item():.4f}",
            "lr":    f"{lr_scheduler.get_last_lr()[0]:.2e}",
        })

        if global_step % args.eval_every == 0:
            eval_step(
                unet, vae, img_encoder, embedding_change,
                tokenizer, text_encoder, noise_scheduler,
                eval_loader, device, weight_dtype, global_step,
            )

        if global_step % args.save_every == 0:
            save_checkpoint(unet, args.output_dir, global_step)

    pbar.close()
    save_checkpoint(unet, args.output_dir, global_step)
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--lq_dir",       required=True)
    p.add_argument("--hq_dir",       required=True)
    p.add_argument("--prompts_json", required=True)

    # Model paths
    p.add_argument("--pretrained_model_name_or_path",
                   default="stabilityai/stable-diffusion-2-1-base")
    p.add_argument("--img_encoder_weight", default="pretrained/associate_2.ckpt")
    p.add_argument("--ckpt_path",    required=True)
    p.add_argument("--output_dir",   default="checkpoints/concat_v1")

    # Pretrained OSDFace LoRA — used ONLY for merging into the base UNet.
    # Must match the rank/alpha the original model was trained with.
    p.add_argument("--pretrained_lora_rank",  type=int,   default=16,
                   help="Rank of the pretrained OSDFace LoRA (default: 16)")
    p.add_argument("--pretrained_lora_alpha", type=float, default=16,
                   help="Alpha of the pretrained OSDFace LoRA (default: 16)")

    # Your new LoRA — rank 32 gives more capacity for learning text conditioning
    # without meaningful VRAM cost increase over rank 16
    p.add_argument("--lora_rank",    type=int,   default=32,
                   help="Rank of your new LoRA trained on top of the merged UNet")
    p.add_argument("--lora_alpha",   type=float, default=32,
                   help="Alpha of your new LoRA")

    # Training
    p.add_argument("--batch_size",         type=int,   default=4)
    p.add_argument("--max_train_steps",    type=int,   default=50000)
    p.add_argument("--learning_rate",      type=float, default=1e-4)
    p.add_argument("--warmup_steps",       type=int,   default=500)
    p.add_argument("--pixel_loss_weight",  type=float, default=0.1)
    p.add_argument("--mixed_precision",    type=str,
                   choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--eval_every",  type=int, default=2000)
    p.add_argument("--save_every",  type=int, default=5000)

    # VRE passthrough args
    p.add_argument("--cat_prompt_embedding", action="store_true")
    p.add_argument("--use_pos_embedding",    action="store_true")
    p.add_argument("--use_att_pool",         action="store_true")
    p.add_argument("--learnable_pos_emb",    action="store_true")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)