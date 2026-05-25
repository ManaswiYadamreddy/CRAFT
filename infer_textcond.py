"""
infer_textcond.py — Inference for FiLM-based text conditioned OSDFace

Loads the frozen OSDFace pipeline and attaches a trained TextConditioner
to guide restoration using per-image pos + na text prompts.

Usage:
    Train
    python infer_textcond.py \
        --input_image /projectnb/cs585/projects/craft/data/train/LQ_images_512x512/00020_LQ.png \
        --output_dir results/textcondfinal_test \
        --prompts_json /projectnb/cs585/projects/craft/prompts_output_final.json \
        --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21\
        --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
        --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
        --conditioner_path checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
        --film_neg_weight 0.1 \
        --gpu_ids 0 \
        --mixed_precision fp16
    Test
    python infer_textcond.py \
        --input_image /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2/00000037.png \
        --output_dir results/textcondfinal_test/ \
        --prompts_json /projectnb/cs585/projects/craft/data/test/CelebA/celeba_prompts_output.json \
        --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21\
        --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
        --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
        --conditioner_path checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
        --film_neg_weight 0.1 \
        --gpu_ids 0 \
        --mixed_precision fp16

    # Run without text conditioning (pure OSDFace baseline):
    python infer_textcond.py \
        --input_image /projectnb/cs585/projects/craft/data/train/LQ_images_512x512 \
        --output_dir results/osdface_baseline \
        --ckpt_path pretrained \
        --no_text_cond

    python3 -c "
import json
with open('/projectnb/cs585/projects/craft/data/test/CelebA/celeba_prompts_output.json') as f:
    data = json.load(f)
match = next(item for item in data if item['image'] == '00000037.png')
swapped = [{
    'image': '00000037.png',
    'pos': match['na'],
    'na':  match['pos']
}]
with open('/tmp/swapped_prompt.json', 'w') as f:
    json.dump(swapped, f)
print('Done')
"

    python3 -c "
import json
with open('/projectnb/cs585/projects/craft/prompts_output_final.json') as f:
    data = json.load(f)
match = next(item for item in data if item['image'] == '00020.png')
swapped = [{
    'image': '00020_LQ.png',
    'pos': match['na'],
    'na':  match['pos']
}]
with open('/tmp/swapped_prompt.json', 'w') as f:
    json.dump(swapped, f)
print('Done')
"

python infer_textcond.py \
    --input_image /projectnb/cs585/projects/craft/data/train/LQ_images_512x512/00020_LQ.png \
    --output_dir results/textcondv5_test/swapped \
    --prompts_json /tmp/swapped_prompt.json \
    --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21\
    --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
    --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
    --conditioner_path checkpoints/textcond_v5/checkpoint-50000/text_conditioner.pth \
    --film_neg_weight 0.5 \
    --gpu_ids 0 --mixed_precision fp16
"""

import os
import sys
import json
import glob
import copy
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as Fun
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F
from safetensors import safe_open
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)

from utils.vaehook import perfcount
from utils.others import get_x0_from_noise
from lq_embed import vqvae_encoder, TwoLayerConv1x1
from text_conditioner import TextConditioner



# ---------------------------------------------------------------------------
# Preamble stripping (matches train_textcond.py)
# ---------------------------------------------------------------------------

def strip_preamble(text: str) -> str:
    """Remove shared quality prefix so CLIP sees only attribute content."""
    for marker in ["in the description of ", "not in the description of "]:
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
    return text.strip()
    # return text


# ---------------------------------------------------------------------------
# LoRA merge (same as train_textcond.py)
# ---------------------------------------------------------------------------

def merge_unet(args):
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    alpha = float(args.lora_alpha / args.lora_rank)
    processed_keys = set()

    with safe_open(
        os.path.join(args.ckpt_path, "pytorch_lora_weights.safetensors"),
        framework="pt",
    ) as f:
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

    unet.load_state_dict(sd_unet)
    print("LoRA merge done.")
    return unet


# ---------------------------------------------------------------------------
# Image discovery helper (handles all common extensions + cases)
# ---------------------------------------------------------------------------

def find_images(path):
    """Find all images under path — works on a directory or a single file."""
    if not os.path.isdir(path):
        return [path]
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG",
            "*.webp", "*.WEBP"]
    results = []
    for ext in exts:
        results.extend(glob.glob(os.path.join(path, ext)))
    return sorted(set(results))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class OSDFace_TextCond(nn.Module):
    """
    OSDFace with FiLM-based text conditioning.

    The UNet runs exactly as in the original OSDFace (visual tokens only in
    cross-attention). The TextConditioner modulates internal UNet features
    via forward hooks using both pos and na text embeddings.

    Setting --no_text_cond skips the conditioner for a clean baseline.
    """

    def __init__(self, args, gpu_id, unet_merged):
        super().__init__()

        self.args   = args
        self.device = torch.device(f"cuda:{gpu_id}")

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        # ── SD 2.1 components ──────────────────────────────────────────────
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae"
        ).to(self.device, dtype=self.weight_dtype)

        self.unet = copy.deepcopy(unet_merged).to(self.device, dtype=self.weight_dtype)
        self.unet.eval()
        self.unet.requires_grad_(False)

        # ── CLIP text encoder ─────────────────────────────────────────────
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device, dtype=self.weight_dtype)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        # ── VQ-VAE image encoder + projection ─────────────────────────────
        self.img_encoder = vqvae_encoder(args).to(self.device, dtype=self.weight_dtype)
        self.img_encoder.eval()

        self.embedding_change = TwoLayerConv1x1(512, 1024)
        self.embedding_change.load_state_dict(
            torch.load(
                os.path.join(args.ckpt_path, "embedding_change_weights.pth"),
                weights_only=False,
            )
        )
        self.embedding_change.to(self.device, dtype=self.weight_dtype)
        self.embedding_change.eval()

        # ── TextConditioner ───────────────────────────────────────────────
        self.use_text_cond = not args.no_text_cond
        self.film_neg_weight = args.film_neg_weight

        if self.use_text_cond:
            self.conditioner = TextConditioner(self.unet, text_dim=1024)
            self.conditioner.register_hooks(self.unet)
            self.conditioner.load(args.conditioner_path, map_location="cpu")
            self.conditioner.to(self.device, dtype=self.weight_dtype)
            self.conditioner.eval()
            print(f"TextConditioner loaded from {args.conditioner_path}")
            print(f"film_neg_weight = {self.film_neg_weight}")
        else:
            self.conditioner = None
            print("Running without text conditioning (pure OSDFace baseline).")

        # ── Prompts lookup: stores both pos and na per filename ───────────
        # Indexes by multiple key variants to handle mismatches between
        # prompts.json ("00020.png") and actual LQ filenames ("00020_LQ.png").
        # Raises an error at inference time if the filename is not found.
        self.prompts: dict = {}
        if hasattr(args, "prompts_json") and args.prompts_json \
                and os.path.exists(args.prompts_json):
            with open(args.prompts_json, "r") as f:
                raw = json.load(f)
            for item in raw:
                entry = {
                    "pos": item.get("pos", ""),
                    "na":  item.get("na",  ""),
                }
                name = item["image"]                     # e.g. "00020.png"
                stem, ext = os.path.splitext(name)       # "00020", ".png"
                self.prompts[name]              = entry  # 00020.png
                self.prompts[f"{stem}_LQ{ext}"] = entry  # 00020_LQ.png
                self.prompts[f"{stem}_lq{ext}"] = entry  # 00020_lq.png
            print(f"Loaded {len(raw)} prompts from {args.prompts_json}")
        elif self.use_text_cond:
            raise ValueError(
                "--prompts_json is required when using text conditioning. "
                "Pass --no_text_cond to run without prompts."
            )

        self.timesteps = 399

    # @torch.no_grad()
    # def _encode_text(self, prompts: list) -> torch.Tensor:
    #     input_ids = self.tokenizer(
    #         prompts,
    #         padding="max_length",
    #         max_length=self.tokenizer.model_max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     ).input_ids.to(self.device)

    #     hidden = self.text_encoder(input_ids).last_hidden_state  # (B, 77, 1024)
    #     pooled = hidden.mean(dim=1)  # (B, 1024) — matches train_textcond.py
    #     pooled = pooled.unsqueeze(1).expand(-1, self.tokenizer.model_max_length, -1)
    #     return pooled.to(self.weight_dtype)

    @torch.no_grad()
    def _encode_text(self, prompts: list) -> torch.Tensor:
        """EOS token embedding — far more discriminative than mean pooling."""
        input_ids = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        output = self.text_encoder(input_ids)
        hidden = output.last_hidden_state  # (B, 77, 1024)

        eos_positions = (
            input_ids == self.tokenizer.eos_token_id
        ).float().argmax(dim=1)

        eos_embeds = hidden[
            torch.arange(hidden.shape[0], device=self.device), eos_positions
        ]  # (B, 1024)

        # Broadcast across sequence → (B, 77, 1024)
        eos_embeds = eos_embeds.unsqueeze(1).expand(
            -1, self.tokenizer.model_max_length, -1
        )
        return eos_embeds.to(self.weight_dtype)

    @perfcount
    @torch.no_grad()
    def forward(self, lq: torch.Tensor, filename: str = "") -> torch.Tensor:
        B = lq.shape[0]

        # Visual tokens — identical to original OSDFace
        visual_embeds = self.embedding_change(
            self.img_encoder(lq).reshape(B, 77, -1)
        )  # (B, 77, 1024)

        # VAE encode
        lq_latent = (
            self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()
            * self.vae.config.scaling_factor
        )

        # Activate text conditioning via FiLM hooks using both pos and na
        if self.use_text_cond and self.conditioner is not None:
            if filename not in self.prompts:
                raise KeyError(
                    f"No prompt found for '{filename}'. "
                    f"Check that prompts.json contains an entry whose image "
                    f"name matches this file (with or without _LQ suffix)."
                )
            entry = self.prompts[filename]

            pos_embeds = self._encode_text(
                [strip_preamble(entry["pos"])] * B
            )  # (B, 77, 1024)
            na_embeds  = self._encode_text(
                [strip_preamble(entry["na"])]  * B
            )  # (B, 77, 1024)

            self.conditioner.set_text_embedding(
                pos_embeds,
                na_embeds,
                neg_weight=self.film_neg_weight,
            )

        # UNet forward — hooks fire here if text cond is active
        model_pred = self.unet(
            lq_latent, self.timesteps,
            encoder_hidden_states=visual_embeds,
        ).sample

        if self.use_text_cond and self.conditioner is not None:
            self.conditioner.clear_text_embedding()

        # Decode
        x_0 = get_x0_from_noise(
            lq_latent.double(),
            model_pred.double(),
            self.alphas_cumprod.double(),
            self.timesteps,
        ).float()

        output_image = self.vae.decode(
            x_0.to(self.weight_dtype) / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)

        return (output_image * 0.5 + 0.5).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def main_worker(unet_merged, rank, gpu_id, image_names, weight_dtype, args):
    torch.cuda.set_device(gpu_id)
    model = OSDFace_TextCond(args, gpu_id, unet_merged).to(gpu_id)

    for image_name in tqdm(image_names, desc=f"GPU {gpu_id}"):
        out_path = os.path.join(args.output_dir, os.path.basename(image_name))
        if os.path.exists(out_path):
            continue

        img = Image.open(image_name).convert("RGB")
        lq  = F.to_tensor(img).unsqueeze(0).to(gpu_id, dtype=weight_dtype) * 2 - 1
        if lq.shape[2] == lq.shape[3]:
            lq = Fun.interpolate(lq, (512, 512), mode="bilinear", align_corners=True)

        output = model(lq, filename=os.path.basename(image_name))
        transforms.ToPILImage()(output[0].cpu()).save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_inference(args, unet_merged):
    # FIX: use find_images() instead of fragile glob pattern
    image_names = find_images(args.input_image)

    existing = {
        os.path.basename(p)
        for p in find_images(args.output_dir)
    }
    image_names = [p for p in image_names if os.path.basename(p) not in existing]
    random.shuffle(image_names)

    print(f"Processing {len(image_names)} images...")

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    num_gpus     = len(args.gpu_ids)
    per_gpu      = len(image_names) // num_gpus
    processes    = []

    for rank, gpu_id in enumerate(args.gpu_ids):
        start = rank * per_gpu
        end   = start + per_gpu if rank != num_gpus - 1 else len(image_names)
        p = mp.Process(
            target=main_worker,
            args=(unet_merged, rank, gpu_id, image_names[start:end], weight_dtype, args),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image",  "-i", required=True)
    parser.add_argument("--output_dir",   "-o", required=True)
    parser.add_argument("--prompts_json", type=str, default=None)
    parser.add_argument("--pretrained_model_name_or_path",
                        default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--img_encoder_weight",
                        default="pretrained/associate_2.ckpt")
    parser.add_argument("--ckpt_path",        required=True)
    parser.add_argument("--conditioner_path", type=str, default=None,
                        help="Path to text_conditioner.pth checkpoint")
    parser.add_argument("--no_text_cond",     action="store_true",
                        help="Run without text conditioning (pure OSDFace baseline)")
    parser.add_argument("--film_neg_weight",  type=float, default=0.5,
                        help="How strongly na suppresses pos in FiLM (default: 0.5)")
    parser.add_argument("--mixed_precision",  choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--gpu_ids",          nargs="+", type=int, default=[0])
    parser.add_argument("--lora_rank",        type=int,   default=16)
    parser.add_argument("--lora_alpha",       type=float, default=16)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--cat_prompt_embedding", action="store_true")
    parser.add_argument("--use_pos_embedding",    action="store_true")
    parser.add_argument("--use_att_pool",         action="store_true")
    parser.add_argument("--learnable_pos_emb",    action="store_true")

    args = parser.parse_args()

    if not args.no_text_cond and args.conditioner_path is None:
        parser.error("--conditioner_path is required unless --no_text_cond is set")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    mp.set_start_method("spawn", force=True)

    unet_merged = merge_unet(args)
    run_inference(args, unet_merged)