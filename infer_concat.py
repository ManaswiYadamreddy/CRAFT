"""
infer_concat.py — OSDFace inference with concatenated visual + text tokens

Identical to infer.py except for one change in forward():

    BEFORE:  encoder_hidden_states = visual_embeds          # (B, 77, 1024)
    AFTER:   encoder_hidden_states = cat([visual, text], 1) # (B, 154, 1024)

Usage:
    python infer_concat.py \
        --input_image  data/WebPhoto-Test \
        --output_dir   results/WebPhoto-concat \
        --prompts_json data/WebPhoto-Test/prompts.json \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base \
        --img_encoder_weight pretrained/associate_2.ckpt \
        --ckpt_path    checkpoints/concat_v1/checkpoint-50000 \
        --merge_lora \
        --mixed_precision fp16

    # Run without prompts (falls back to empty string — same as original):
    python infer_concat.py \
        --input_image data/WebPhoto-Test \
        --output_dir  results/WebPhoto-concat \
        --ckpt_path   pretrained \
        --merge_lora
"""

import copy
import os
import sys
import json
import glob
import argparse
import random

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
    StableDiffusionPipeline,
)

from utils.vaehook import perfcount
from utils.others import get_x0_from_noise
from lq_embed import vqvae_encoder, TwoLayerConv1x1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class OSDFace_Concat(nn.Module):
    """
    OSDFace with text prompt conditioning via sequence concatenation.

    The only difference from OSDFace_test (infer.py) is in forward():
        prompt_embeds = cat([visual_embeds, text_embeds], dim=1)  # (B,154,1024)

    Visual tokens are placed first so the UNet's existing cross-attention
    weights still see them at the positions they were originally trained on.
    """

    def __init__(self, args, gpu_id, unet_merged=None):
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

        if args.merge_lora and unet_merged is not None:
            self.unet = copy.deepcopy(unet_merged)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet"
            )
            from peft import PeftModel
            self.unet = PeftModel.from_pretrained(self.unet, args.ckpt_path)
            self.unet = self.unet.merge_and_unload()
        self.unet.to(self.device, dtype=self.weight_dtype)

        # ── CLIP text encoder (frozen) ─────────────────────────────────────
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device, dtype=self.weight_dtype)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # ── VQ-VAE image encoder + projection (frozen) ────────────────────
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

        # ── Prompts lookup ────────────────────────────────────────────────
        self.prompts: dict = {}
        if hasattr(args, "prompts_json") and args.prompts_json \
                and os.path.exists(args.prompts_json):
            with open(args.prompts_json, "r") as f:
                self.prompts = json.load(f)
            print(f"Loaded {len(self.prompts)} prompts from {args.prompts_json}")
        else:
            print("No prompts_json provided — using empty string for all images.")

        self.timesteps = 399

    @torch.no_grad()
    def _encode_text(self, prompts: list) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens).last_hidden_state.to(self.weight_dtype)
        # (B, 77, 1024)

    @perfcount
    @torch.no_grad()
    def forward(self, lq: torch.Tensor, filename: str = "") -> torch.Tensor:
        B = lq.shape[0]

        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        with torch.cuda.stream(stream1):
            # Visual tokens — same as original OSDFace
            visual_embeds = self.embedding_change(
                self.img_encoder(lq).reshape(B, 77, -1)
            )  # (B, 77, 1024)

            # Text tokens
            prompt = self.prompts.get(filename, "")
            text_embeds = self._encode_text([prompt] * B)
            # (B, 77, 1024)

            # ── The one change: concatenate instead of using visual alone ──
            prompt_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            # (B, 154, 1024)

        with torch.cuda.stream(stream2):
            lq_latent = (
                self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()
                * self.vae.config.scaling_factor
            )

        torch.cuda.synchronize()

        model_pred = self.unet(
            lq_latent, self.timesteps,
            encoder_hidden_states=prompt_embeds,
        ).sample

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
# LoRA merge  (unchanged from infer.py)
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
            sd_unet[unet_key] = sd_unet[unet_key] + alpha * torch.mm(W_B, W_A)
            processed_keys.update([key, lora_b_key])

    unet.load_state_dict(sd_unet)
    print("LoRA merge done.")
    return unet


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def main_worker(unet_merged, rank, gpu_id, image_names, weight_dtype, args):
    torch.cuda.set_device(gpu_id)
    model = OSDFace_Concat(args, gpu_id, unet_merged).to(gpu_id)

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
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f"{args.input_image}/*.[jpJP][pnPN]*[gG]"))
    else:
        image_names = [args.input_image]

    existing = {
        os.path.basename(p)
        for p in glob.glob(f"{args.output_dir}/*.[jpJP][pnPN]*[gG]")
    }
    image_names = [p for p in image_names if os.path.basename(p) not in existing]
    random.shuffle(image_names)

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    num_gpus     = len(args.gpu_ids)
    per_gpu      = len(image_names) // num_gpus
    processes    = []

    for rank, gpu_id in enumerate(args.gpu_ids):
        start  = rank * per_gpu
        end    = start + per_gpu if rank != num_gpus - 1 else len(image_names)
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
    parser.add_argument("--prompts_json", type=str, default=None,
                        help="JSON mapping filename → text prompt. "
                             "Omit to run without text conditioning.")
    parser.add_argument("--pretrained_model_name_or_path",
                        default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--img_encoder_weight",
                        default="pretrained/associate_2.ckpt")
    parser.add_argument("--ckpt_path",        required=True)
    parser.add_argument("--mixed_precision",  choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--gpu_ids",          nargs="+", type=int, default=[0])
    parser.add_argument("--merge_lora",       action="store_true")
    parser.add_argument("--lora_rank",        type=int,   default=16)
    parser.add_argument("--lora_alpha",       type=float, default=16)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--cat_prompt_embedding", action="store_true")
    parser.add_argument("--use_pos_embedding",    action="store_true")
    parser.add_argument("--use_att_pool",         action="store_true")
    parser.add_argument("--learnable_pos_emb",    action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    mp.set_start_method("spawn", force=True)

    unet_merged = merge_unet(args) if args.merge_lora else None
    print(f"Processing {len(glob.glob(f'{args.input_image}/*'))} images...")
    run_inference(args, unet_merged)
