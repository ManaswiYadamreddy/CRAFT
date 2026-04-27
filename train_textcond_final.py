"""
train_textcond.py — FiLM-based text conditioning for OSDFace (v3)

Changes from v2:
  1. Dual FiLM MLPs — separate film_pos_layers and film_na_layers so pos
     and na learn genuinely different modulation directions. Fixes the
     cancellation problem from v2 where a shared MLP produced γ_pos ≈ γ_na.

  2. Preamble stripping — removes the shared quality prefix from pos and na
     before CLIP encoding so CLIP sees only attribute-specific content.
     Matches what inference does.

  3. Gradient loss — penalizes blur by comparing image gradients between
     predicted and target. Counteracts L1's tendency to encourage smoothing.

  4. Contrastive FiLM loss — explicitly pushes the pos and na modulation
     vectors apart so they learn meaningfully different directions.

  5. Counterfactual ranking loss — compares reconstruction with correct vs.
     wrong prompts and enforces a margin so text must matter during training.

prompts.json format:
    [
      {"image": "0001.png",
       "pos": "A high quality... in the description of young, big eyes",
       "na":  "A high quality... not in the description of old, bald..."},
      ...
    ]

Usage:
    python train_textcond.py \
        --lq_dir data/train/lq \
        --hq_dir data/train/hq \
        --prompts_json data/train/prompts.json \
        --pretrained_model_name_or_path pretrained/sd21 \
        --img_encoder_weight pretrained/associate_2.ckpt \
        --ckpt_path pretrained \
        --output_dir checkpoints/textcond_v3 \
        --mixed_precision bf16 \
        --batch_size 4 \
        --max_train_steps 50000
"""

import os
import json
import glob
import random
import argparse
import logging

import torch
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
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer

from lq_embed import vqvae_encoder, TwoLayerConv1x1
from text_conditioner import TextConditioner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preamble stripping
# ---------------------------------------------------------------------------

def strip_preamble(text: str) -> str:
    """
    Remove the shared quality prefix from pos/na prompts so CLIP sees only
    the attribute-specific content.

    e.g. "A high quality... in the description of young, big eyes"
      →  "young, big eyes"

         "A high quality... not in the description of old, bald"
      →  "old, bald"
    """
    for marker in ["in the description of ", "not in the description of "]:
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# LoRA merge helper
# ---------------------------------------------------------------------------

def merge_lora_into_unet(args):
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
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
# Dataset
# ---------------------------------------------------------------------------

class FaceRestorationDataset(Dataset):
    def __init__(self, lq_dir, hq_dir, prompts_json, resolution=512):
        self.resolution = resolution

        def find_images(directory):
            exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
                    "*.webp", "*.WEBP"]
            paths = []
            for ext in exts:
                paths.extend(glob.glob(os.path.join(directory, ext)))
            paths = sorted(set(paths))
            if not paths:
                all_files = os.listdir(directory) if os.path.isdir(directory) else []
                logger.error(f"No images in {directory}. Contents: {all_files[:20]}")
            return paths

        lq_paths = find_images(lq_dir)
        hq_paths = find_images(hq_dir)
        assert len(lq_paths) == len(hq_paths), (
            f"LQ ({len(lq_paths)}) and HQ ({len(hq_paths)}) counts don't match."
        )

        with open(prompts_json, "r") as f:
            raw = json.load(f)

        # Build lookup with multiple key variants so we match regardless of
        # whether the LQ filename has a suffix like _LQ (e.g. 00020_LQ.png)
        # but the prompts.json uses the clean name (e.g. 00020.png).
        self.prompts = {}
        for item in raw:
            entry = {
                "pos": item.get("pos", ""),
                "na":  item.get("na",  ""),
            }
            name = item["image"]                    # e.g. "00020.png"
            stem, ext = os.path.splitext(name)      # "00020", ".png"
            self.prompts[name]              = entry  # 00020.png
            self.prompts[f"{stem}_LQ{ext}"] = entry  # 00020_LQ.png
            self.prompts[f"{stem}_lq{ext}"] = entry  # 00020_lq.png
            self.prompts[f"{stem}_HQ{ext}"] = entry  # 00020_HQ.png

        # Confirm a few keys so you can verify matching at startup
        sample_keys = list(self.prompts.keys())[:8]
        logger.info(f"Prompt keys sample: {sample_keys}")

        # Verify at least one LQ file actually matches a prompt key
        matched = sum(1 for lq, _ in zip(lq_paths, hq_paths)
                      if os.path.basename(lq) in self.prompts)
        logger.info(f"Prompt matches: {matched}/{len(lq_paths)} LQ files have prompts")
        if matched == 0:
            raise ValueError(
                "No LQ filenames matched any prompt key. "
                "Check that prompts.json image names correspond to your LQ files."
            )

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

        lq = self.transform(lq) * 2.0 - 1.0
        hq = self.transform(hq) * 2.0 - 1.0

        filename = os.path.basename(lq_path)
        entry    = self.prompts.get(filename, {"pos": "", "na": ""})

        return {
            "lq":       lq,
            "hq":       hq,
            "pos":      entry["pos"],
            "na":       entry["na"],
            "filename": filename,
        }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t  = 1 - alpha_prod_t
    return (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Penalizes blur by comparing image gradients between prediction and target.
    Operates in pixel space on decoded images (B, C, H, W) in [-1, 1].
    L1 on gradients encourages the model to preserve edges and fine detail
    rather than smoothing them away.
    """
    pred_dx   = pred[:, :, :, 1:]  - pred[:, :, :, :-1]
    pred_dy   = pred[:, :, 1:, :]  - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def contrastive_film_loss(
    conditioner: TextConditioner,
    pos_embeds: torch.Tensor,
    na_embeds: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Pushes pos and na FiLM modulations apart so they learn different directions.

    For each block, we compute the (γ, β) vectors that film_pos and film_na
    would produce for this batch, then minimize their cosine similarity.
    This ensures the two MLPs diverge rather than collapsing to the same output.

    Loss = mean cosine_similarity(pos_modulation, na_modulation)
    Minimizing this → the two modulations point in different directions.
    """
    pos_pooled = pos_embeds.mean(dim=1).to(dtype)
    na_pooled  = na_embeds.mean(dim=1).to(dtype)

    total = torch.tensor(0.0, device=pos_embeds.device, dtype=dtype)
    n = 0

    for key in conditioner.film_pos_layers:
        gp, bp = conditioner.film_pos_layers[key](pos_pooled)  # (B,1,d)
        gn, bn = conditioner.film_na_layers[key](na_pooled)    # (B,1,d)

        # Flatten to (B, 2d) and compute cosine similarity
        vec_pos = torch.cat([gp, bp], dim=-1).squeeze(1)  # (B, 2d)
        vec_na  = torch.cat([gn, bn], dim=-1).squeeze(1)  # (B, 2d)

        # We want similarity to be LOW (vectors pointing in different directions)
        sim  = F.cosine_similarity(vec_pos, vec_na, dim=-1).mean()
        total = total + sim
        n += 1

    return total / max(n, 1)


def make_counterfactual_perm(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Build a non-identity permutation for in-batch wrong-prompt pairing.
    Falls back to a 1-step roll if random permutation is identity.
    """
    if batch_size <= 1:
        return torch.arange(batch_size, device=device)
    perm = torch.randperm(batch_size, device=device)
    identity = torch.arange(batch_size, device=device)
    if torch.equal(perm, identity):
        perm = torch.roll(identity, shifts=1)
    return perm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_text(prompts, tokenizer, text_encoder, device, weight_dtype):
    """
    Encode prompts using the EOS token embedding broadcast across sequence.

    Mean-pooled CLIP hidden states have ~0.90 cosine similarity to empty
    string — almost no discriminative signal. The EOS token embedding has
    ~-0.04 similarity to empty string — nearly orthogonal, much stronger
    signal for the FiLM layers to learn from.

    Returns (B, 77, 1024) with the EOS embedding broadcast across all
    sequence positions, compatible with the FiLM conditioning pathway.
    """
    input_ids = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    output = text_encoder(input_ids)
    hidden = output.last_hidden_state  # (B, 77, 1024)

    # EOS token position = first occurrence of eos_token_id per sequence
    eos_positions = (input_ids == tokenizer.eos_token_id).float().argmax(dim=1)

    # Extract EOS embedding per batch item → (B, 1024)
    eos_embeds = hidden[
        torch.arange(hidden.shape[0], device=device), eos_positions
    ]

    # Broadcast across sequence → (B, 77, 1024)
    eos_embeds = eos_embeds.unsqueeze(1).expand(-1, tokenizer.model_max_length, -1)

    return eos_embeds.to(weight_dtype)


@torch.no_grad()
def encode_text_mean_pooled(prompts, tokenizer, text_encoder, device, weight_dtype):
    """
    Encode prompts as mean-pooled CLIP hidden states, then broadcast to sequence.
    This preserves information from all tokens and can strengthen attribute signal.
    """
    input_ids = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    hidden = text_encoder(input_ids).last_hidden_state  # (B, 77, 1024)
    pooled = hidden.mean(dim=1)  # (B, 1024)
    pooled = pooled.unsqueeze(1).expand(-1, tokenizer.model_max_length, -1)
    return pooled.to(weight_dtype)


def encode_text_dispatch(prompts, tokenizer, text_encoder, device, weight_dtype, mode):
    if mode == "eos":
        return encode_text(prompts, tokenizer, text_encoder, device, weight_dtype)
    if mode == "mean_pool":
        return encode_text_mean_pooled(prompts, tokenizer, text_encoder, device, weight_dtype)
    raise ValueError(f"Unsupported --text_embed_mode: {mode}")


def save_checkpoint(conditioner, output_dir, step):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    conditioner.save(os.path.join(ckpt_dir, "text_conditioner.pth"))
    training_state = {"global_step": step}
    torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))
    logger.info(f"Checkpoint saved → {ckpt_dir}")


def save_training_state(optimizer, lr_scheduler, scaler, output_dir, step):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    training_state = {
        "global_step": step,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))
    logger.info(f"Training state saved → {ckpt_dir}")


def get_latest_checkpoint(output_dir):
    pattern = os.path.join(output_dir, "checkpoint-*")
    candidates = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-")[-1])
        except ValueError:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def resolve_resume_checkpoint(args):
    if args.resume_from_checkpoint is None:
        return None
    if args.resume_from_checkpoint.lower() == "latest":
        latest = get_latest_checkpoint(args.output_dir)
        if latest is None:
            raise ValueError(
                f"--resume_from_checkpoint=latest but no checkpoints were found in {args.output_dir}"
            )
        return latest
    return args.resume_from_checkpoint


def load_checkpoint_if_available(args, conditioner, optimizer, lr_scheduler, scaler, device):
    resume_path = resolve_resume_checkpoint(args)
    if resume_path is None:
        return 0

    if not os.path.isdir(resume_path):
        raise ValueError(f"Resume checkpoint directory does not exist: {resume_path}")

    conditioner_path = os.path.join(resume_path, "text_conditioner.pth")
    if not os.path.isfile(conditioner_path):
        raise ValueError(f"Missing conditioner checkpoint: {conditioner_path}")
    conditioner.load(conditioner_path, map_location=device)

    state_path = os.path.join(resume_path, "training_state.pt")
    global_step = 0
    if os.path.isfile(state_path):
        state = torch.load(state_path, map_location=device, weights_only=False)
        global_step = int(state.get("global_step", 0))
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            lr_scheduler.load_state_dict(state["lr_scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
    else:
        ckpt_name = os.path.basename(os.path.normpath(resume_path))
        if ckpt_name.startswith("checkpoint-"):
            try:
                global_step = int(ckpt_name.split("-")[-1])
            except ValueError:
                global_step = 0

    logger.info(f"Resumed training from {resume_path} at global step {global_step}")
    return global_step


# ---------------------------------------------------------------------------
# Build models
# ---------------------------------------------------------------------------

def build_models(args, device):
    weight_dtype = {"fp32": torch.float32,
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16}[args.mixed_precision]

    logger.info("Loading scheduler and VAE...")
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(device, dtype=weight_dtype)
    vae.requires_grad_(False)

    logger.info("Loading CLIP text encoder (frozen)...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    ).to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    logger.info("Merging OSDFace LoRA into UNet base weights...")
    unet = merge_lora_into_unet(args)
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    unet.eval()

    logger.info("Loading VQ-VAE image encoder (frozen)...")
    img_encoder = vqvae_encoder(args).to(device, dtype=weight_dtype)
    img_encoder.requires_grad_(False)
    img_encoder.eval()

    logger.info("Loading TwoLayerConv1x1 projection (frozen)...")
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

    logger.info("Building TextConditioner (dual FiLM MLPs)...")
    conditioner = TextConditioner(unet, text_dim=1024)
    conditioner.register_hooks(unet)
    # Keep trainable conditioner params in FP32 for stable AMP + GradScaler behavior.
    conditioner.to(device, dtype=torch.float32)

    return (
        noise_scheduler, vae, tokenizer, text_encoder,
        unet, img_encoder, embedding_change, conditioner, weight_dtype,
    )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_step(unet, vae, img_encoder, embedding_change, conditioner,
              tokenizer, text_encoder, noise_scheduler,
              dataloader, device, weight_dtype, step, film_neg_weight, text_embed_mode):
    conditioner.eval()
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    timestep = 399
    total_loss, n = 0.0, 0

    use_amp = weight_dtype in (torch.float16, torch.bfloat16)
    for batch in dataloader:
        lq  = batch["lq"].to(device, dtype=weight_dtype)
        hq  = batch["hq"].to(device, dtype=weight_dtype)
        pos = [strip_preamble(p) for p in batch["pos"]]
        na  = [strip_preamble(p) for p in batch["na"]]

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=weight_dtype):
            visual_embeds = embedding_change(
                img_encoder(lq).reshape(lq.shape[0], 77, -1)
            )
            pos_embeds = encode_text_dispatch(
                pos, tokenizer, text_encoder, device, weight_dtype, text_embed_mode
            )
            na_embeds  = encode_text_dispatch(
                na, tokenizer, text_encoder, device, weight_dtype, text_embed_mode
            )

            hq_latent = vae.encode(hq).latent_dist.sample() * vae.config.scaling_factor
            lq_latent = vae.encode(lq).latent_dist.sample() * vae.config.scaling_factor

            conditioner.set_text_embedding(pos_embeds, na_embeds, neg_weight=film_neg_weight)
            model_pred = unet(
                lq_latent, timestep,
                encoder_hidden_states=visual_embeds,
            ).sample
            conditioner.clear_text_embedding()

            x0_pred = get_x0_from_noise(
                lq_latent.double(), model_pred.double(),
                alphas_cumprod.double(), timestep,
            ).float().to(weight_dtype)

        total_loss += F.l1_loss(x0_pred, hq_latent).item()
        n += 1
        if n >= 50:
            break

    logger.info(f"[Eval @ step {step}] L1 loss: {total_loss / max(n, 1):.5f}")
    conditioner.train()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    (
        noise_scheduler, vae, tokenizer, text_encoder,
        unet, img_encoder, embedding_change, conditioner, weight_dtype,
    ) = build_models(args, device)

    dataset = FaceRestorationDataset(args.lq_dir, args.hq_dir, args.prompts_json)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    trainable_params = list(conditioner.parameters())
    total = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable parameters: {total:,}  (TextConditioner dual FiLM only)")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=1e-2,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    use_amp   = args.mixed_precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    use_fp16_scaler = args.mixed_precision == "fp16"
    # Torch AMP API differs across versions. Prefer torch.amp, fallback to torch.cuda.amp.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

        def autocast_ctx():
            return torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)

        def autocast_ctx():
            return torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    timestep = 399  # fixed — OSDFace is a one-step diffusion model

    unet.eval()
    img_encoder.eval()
    text_encoder.eval()
    embedding_change.eval()
    vae.eval()
    conditioner.train()

    global_step = load_checkpoint_if_available(
        args, conditioner, optimizer, lr_scheduler, scaler, device
    )
    data_iter   = iter(train_loader)

    logger.info(f"Starting training for {args.max_train_steps} steps...")
    logger.info(
        f"Loss weights — latent: 1.0 | pixel: {args.pixel_loss_weight} | "
        f"grad: {args.grad_loss_weight} | contrastive: {args.contrastive_loss_weight} | "
        f"counterfactual: {args.counterfactual_loss_weight} (margin={args.counterfactual_margin})"
    )
    pbar = tqdm(total=args.max_train_steps, initial=global_step, desc="Training")

    while global_step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch     = next(data_iter)

        lq  = batch["lq"].to(device, dtype=weight_dtype)
        hq  = batch["hq"].to(device, dtype=weight_dtype)

        # Strip preamble so CLIP sees only attribute content
        pos = [strip_preamble(p) for p in batch["pos"]]
        na  = [strip_preamble(p) for p in batch["na"]]

        with autocast_ctx():

            with torch.no_grad():
                visual_embeds = embedding_change(
                    img_encoder(lq).reshape(lq.shape[0], 77, -1)
                )  # (B, 77, 1024)

                pos_embeds = encode_text_dispatch(
                    pos, tokenizer, text_encoder, device, weight_dtype, args.text_embed_mode
                )  # (B, 77, 1024)
                na_embeds = encode_text_dispatch(
                    na, tokenizer, text_encoder, device, weight_dtype, args.text_embed_mode
                )  # (B, 77, 1024)

                hq_latent = (
                    vae.encode(hq).latent_dist.sample()
                    * vae.config.scaling_factor
                )
                lq_latent = (
                    vae.encode(lq).latent_dist.sample()
                    * vae.config.scaling_factor
                )

            # FiLM modulation — separate pos and na MLPs
            conditioner.set_text_embedding(
                pos_embeds, na_embeds,
                neg_weight=args.film_neg_weight,
            )

            model_pred = unet(
                lq_latent, timestep,
                encoder_hidden_states=visual_embeds,
            ).sample

            conditioner.clear_text_embedding()

            # ── Reconstruction losses ──────────────────────────────────────
            x0_pred = get_x0_from_noise(
                lq_latent.double(), model_pred.double(),
                alphas_cumprod.double(), timestep,
            ).float().to(weight_dtype)

            loss_latent = F.l1_loss(x0_pred, hq_latent)

            # ── Counterfactual ranking loss ────────────────────────────────
            # Force text to matter: correct prompts should reconstruct better
            # than wrong/swapped prompts by at least a margin.
            if args.counterfactual_loss_weight > 0:
                bsz = pos_embeds.shape[0]
                if bsz > 1:
                    perm = make_counterfactual_perm(bsz, pos_embeds.device)
                    pos_embeds_cf = pos_embeds[perm]
                    na_embeds_cf = na_embeds[perm]
                else:
                    # Batch size 1 fallback: invert pos/na for a counterfactual.
                    pos_embeds_cf = na_embeds
                    na_embeds_cf = pos_embeds

                conditioner.set_text_embedding(
                    pos_embeds_cf, na_embeds_cf,
                    neg_weight=args.film_neg_weight,
                )
                model_pred_cf = unet(
                    lq_latent, timestep,
                    encoder_hidden_states=visual_embeds,
                ).sample
                conditioner.clear_text_embedding()

                x0_pred_cf = get_x0_from_noise(
                    lq_latent.double(), model_pred_cf.double(),
                    alphas_cumprod.double(), timestep,
                ).float().to(weight_dtype)
                loss_latent_cf = F.l1_loss(x0_pred_cf, hq_latent)
                loss_counterfactual = F.relu(
                    args.counterfactual_margin + loss_latent - loss_latent_cf
                )
            else:
                loss_counterfactual = torch.tensor(0.0, device=device)

            if args.pixel_loss_weight > 0 or args.grad_loss_weight > 0:
                # Keep gradient path: pixel/gradient losses -> x0_pred -> FiLM params.
                # VAE stays frozen since requires_grad_(False) is set during model build.
                x0_decoded = vae.decode(
                    x0_pred / vae.config.scaling_factor
                ).sample.clamp(-1, 1)

                loss_pixel = F.l1_loss(x0_decoded, hq) if args.pixel_loss_weight > 0 \
                             else torch.tensor(0.0, device=device)

                # Gradient loss — penalizes blur by comparing image gradients
                loss_grad = gradient_loss(x0_decoded, hq) if args.grad_loss_weight > 0 \
                            else torch.tensor(0.0, device=device)
            else:
                loss_pixel = torch.tensor(0.0, device=device)
                loss_grad  = torch.tensor(0.0, device=device)

            # ── Contrastive FiLM loss ──────────────────────────────────────
            # Pushes pos and na modulations to learn different directions.
            # Computed outside no_grad so gradients flow to FiLM params.
            if args.contrastive_loss_weight > 0:
                loss_contrast = contrastive_film_loss(
                    conditioner, pos_embeds, na_embeds, amp_dtype
                )
            else:
                loss_contrast = torch.tensor(0.0, device=device)

            loss = (loss_latent
                    + args.pixel_loss_weight      * loss_pixel
                    + args.grad_loss_weight        * loss_grad
                    + args.contrastive_loss_weight * loss_contrast
                    + args.counterfactual_loss_weight * loss_counterfactual)

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
            "loss":     f"{loss.item():.4f}",
            "l1":       f"{loss_latent.item():.4f}",
            "grad":     f"{loss_grad.item():.4f}",
            "contrast": f"{loss_contrast.item():.4f}",
            "cf":       f"{loss_counterfactual.item():.4f}",
            "lr":       f"{lr_scheduler.get_last_lr()[0]:.2e}",
        })

        if global_step % args.eval_every == 0:
            eval_step(
                unet, vae, img_encoder, embedding_change, conditioner,
                tokenizer, text_encoder, noise_scheduler,
                eval_loader, device, weight_dtype, global_step,
                args.film_neg_weight, args.text_embed_mode,
            )

        if global_step % args.save_every == 0:
            save_checkpoint(conditioner, args.output_dir, global_step)
            save_training_state(optimizer, lr_scheduler, scaler, args.output_dir, global_step)

    pbar.close()
    save_checkpoint(conditioner, args.output_dir, global_step)
    save_training_state(optimizer, lr_scheduler, scaler, args.output_dir, global_step)
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
    p.add_argument("--output_dir",   default="checkpoints/textcond_v3")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to checkpoint directory to resume from, or 'latest'.")

    # Pretrained OSDFace LoRA (for merging only)
    p.add_argument("--pretrained_lora_rank",  type=int,   default=16)
    p.add_argument("--pretrained_lora_alpha", type=float, default=16)

    # Training
    p.add_argument("--batch_size",         type=int,   default=4)
    p.add_argument("--max_train_steps",    type=int,   default=50000)
    p.add_argument("--learning_rate",      type=float, default=1e-4)
    p.add_argument("--warmup_steps",       type=int,   default=500)
    p.add_argument("--mixed_precision",    type=str,
                   choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--eval_every",  type=int, default=2000)
    p.add_argument("--save_every",  type=int, default=5000)

    # Loss weights
    p.add_argument("--pixel_loss_weight",       type=float, default=0.1,
                   help="Weight for pixel-space L1 loss (default: 0.1)")
    p.add_argument("--grad_loss_weight",         type=float, default=0.1,
                   help="Weight for gradient loss — penalizes blur (default: 0.1)")
    p.add_argument("--contrastive_loss_weight",  type=float, default=0.05,
                   help="Weight for contrastive FiLM loss — keeps pos/na apart (default: 0.05)")
    p.add_argument("--counterfactual_loss_weight", type=float, default=0.5,
                   help="Weight for ranking loss that prefers correct prompts over wrong prompts.")
    p.add_argument("--counterfactual_margin", type=float, default=0.02,
                   help="Margin for counterfactual ranking: relu(margin + L_pos - L_wrong).")

    # FiLM
    p.add_argument("--film_neg_weight", type=float, default=0.5,
                   help="Weight for na suppression in FiLM modulation (default: 0.5)")
    p.add_argument("--text_embed_mode", type=str, default="mean_pool",
                   choices=["eos", "mean_pool"],
                   help="Text embedding mode for FiLM conditioning.")

    # VRE passthrough
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