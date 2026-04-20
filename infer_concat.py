"""
infer_concat.py — OSDFace inference with concatenated visual + text tokens

Forward pass:
    encoder_hidden_states = cat([visual, pos, na], dim=1)  # (B, 231, 1024)

Fixes vs original:
  1. EOS token embedding instead of mean-pooled hidden states
     (pos vs empty: 0.91 → -0.04 cosine similarity)
  2. Both pos and na prompts used (not just pos)
  3. Prompt lookup handles _LQ filename suffix mismatch
  4. Hard error if prompt not found
  5. Preamble stripped before CLIP encoding
  6. Robust image glob replaces fragile character-class pattern

Usage:
    python infer_concat.py \
        --input_image /projectnb/cs585/projects/craft/data/train/LQ_images_512x512/66123_LQ.png \
        --output_dir results/concat_testv2 \
        --prompts_json /projectnb/cs585/projects/craft/prompts_output_final.json \
        --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \
        --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
        --ckpt_path checkpoints/concat_v2/checkpoint-15000 \
        --pretrained_ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
        --merge_lora \
        --mixed_precision fp16 \
        --gpu_ids 0

    # Baseline — no text conditioning:
    python infer_concat.py \
        --input_image data/test/LQ \
        --output_dir  results/concat_baseline \
        --ckpt_path   pretrained \
        --merge_lora \
        --no_text_cond
"""

import copy
import os
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
)
 
from utils.vaehook import perfcount
from utils.others import get_x0_from_noise
from lq_embed import vqvae_encoder, TwoLayerConv1x1
 
 
# ---------------------------------------------------------------------------
# Preamble stripping
# ---------------------------------------------------------------------------
 
def strip_preamble(text: str) -> str:
    """Remove shared quality prefix so CLIP sees only attribute content."""
    for marker in ["in the description of ", "not in the description of "]:
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
    return text.strip()
 
 
# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------
 
def find_images(path):
    """Find all images under path — handles directory or single file."""
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
 
class OSDFace_Concat(nn.Module):
    """
    OSDFace with text conditioning via sequence concatenation.
 
    encoder_hidden_states = cat([visual, pos, na], dim=1)  # (B, 231, 1024)
 
    Uses EOS token embedding for CLIP encoding — far more discriminative
    than mean-pooled hidden states.
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
        self.unet.eval()
 
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
        # embedding_change lives in pretrained dir, not the LoRA checkpoint dir
        emb_path = os.path.join(args.pretrained_ckpt_path, "embedding_change_weights.pth")
        self.embedding_change.load_state_dict(
            torch.load(emb_path, weights_only=False)
        )
        self.embedding_change.to(self.device, dtype=self.weight_dtype)
        self.embedding_change.eval()
 
        # ── Prompts lookup ────────────────────────────────────────────────
        self.use_text_cond = not args.no_text_cond
        self.prompts: dict = {}
 
        if self.use_text_cond:
            if not args.prompts_json or not os.path.exists(args.prompts_json):
                raise ValueError(
                    "--prompts_json is required unless --no_text_cond is set."
                )
            with open(args.prompts_json, "r") as f:
                raw = json.load(f)
 
            # Index by multiple key variants to handle _LQ suffix mismatch
            for item in raw:
                entry = {
                    "pos": item.get("pos", ""),
                    "na":  item.get("na",  ""),
                }
                name = item["image"]
                stem, ext = os.path.splitext(name)
                self.prompts[name]              = entry  # 00020.png
                self.prompts[f"{stem}_LQ{ext}"] = entry  # 00020_LQ.png
                self.prompts[f"{stem}_lq{ext}"] = entry  # 00020_lq.png
 
            print(f"Loaded {len(raw)} prompts from {args.prompts_json}")
        else:
            print("Running without text conditioning (pure OSDFace baseline).")
 
        self.timesteps = 399
 
    @torch.no_grad()
    def _encode_text(self, prompts: list) -> torch.Tensor:
        """
        EOS token embedding broadcast across sequence.
        Far more discriminative than mean pooling:
            mean-pool: pos vs empty ~0.91 (near identical)
            EOS:       pos vs empty ~-0.04 (near orthogonal)
        """
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
        return eos_embeds.unsqueeze(1).expand(
            -1, self.tokenizer.model_max_length, -1
        ).to(self.weight_dtype)
 
    @perfcount
    @torch.no_grad()
    def forward(self, lq: torch.Tensor, filename: str = "") -> torch.Tensor:
        B = lq.shape[0]
 
        # Visual tokens — identical to original OSDFace
        visual_embeds = self.embedding_change(
            self.img_encoder(lq).reshape(B, 77, -1)
        )  # (B, 77, 1024)
 
        if self.use_text_cond:
            if filename not in self.prompts:
                raise KeyError(
                    f"No prompt found for '{filename}'. "
                    f"Check prompts.json contains a matching entry."
                )
            entry = self.prompts[filename]
 
            pos_embeds = self._encode_text(
                [strip_preamble(entry["pos"])] * B
            )  # (B, 77, 1024)
            na_embeds = self._encode_text(
                [strip_preamble(entry["na"])] * B
            )  # (B, 77, 1024)
 
            # [visual | pos | na] → (B, 231, 1024)
            prompt_embeds = torch.cat([visual_embeds, pos_embeds, na_embeds], dim=1)
        else:
            # Baseline — visual tokens only, same as original OSDFace
            prompt_embeds = visual_embeds  # (B, 77, 1024)
 
        lq_latent = (
            self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()
            * self.vae.config.scaling_factor
        )
 
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
# LoRA merge
# ---------------------------------------------------------------------------
 
def merge_unet(args):
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
 
    # Handle both filename conventions:
    #   pytorch_lora_weights.safetensors  — original OSDFace checkpoint
    #   adapter_model.safetensors         — PEFT save_pretrained() format
    ckpt_dir = args.ckpt_path
    candidates = [
        os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors"),
        os.path.join(ckpt_dir, "adapter_model.safetensors"),
    ]
    lora_path = next((p for p in candidates if os.path.exists(p)), None)
    if lora_path is None:
        raise FileNotFoundError(
            f"No LoRA weights found in {ckpt_dir}. "
            f"Expected one of: {[os.path.basename(c) for c in candidates]}"
        )
    print(f"Loading LoRA weights from: {lora_path}")
 
    alpha = float(args.lora_alpha / args.lora_rank)
    processed_keys = set()
 
    with safe_open(lora_path, framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
 
    sd_unet = unet.state_dict()
 
    def clean_unet_key(key):
        """
        Normalize LoRA key to match UNet state_dict key.
        Handles both formats:
          - Original:  "unet.down_blocks.0...lora_A.weight"
          - PEFT:      "base_model.model.down_blocks.0...lora_A.weight"
        """
        key = key.replace("base_model.model.", "")  # PEFT prefix
        key = key.replace("unet.", "")               # original prefix
        key = key.replace(".lora_A.weight", ".weight")
        key = key.replace(".lora_B.weight", ".weight")
        key = key.replace(".lora.up.weight", ".weight")
        key = key.replace(".lora.down.weight", ".weight")
        return key
 
    # Non-weight keys saved by PEFT alongside the actual LoRA weights
    # These are metadata/config entries, not LoRA parameters — safe to skip
    PEFT_NON_WEIGHT_SUFFIXES = (
        ".base_layer.weight", ".base_layer.bias",
        ".modules_to_save.", "lora_embedding_A", "lora_embedding_B",
    )
 
    for key in state_dict:
        if "lora_A" in key:
            # Skip PEFT metadata entries
            if any(s in key for s in PEFT_NON_WEIGHT_SUFFIXES):
                processed_keys.add(key)
                continue
 
            lora_b_key = key.replace("lora_A", "lora_B")
            if lora_b_key not in state_dict:
                raise KeyError(
                    f"Found lora_A key '{key}' but missing matching lora_B key. "
                    f"Checkpoint may be corrupted."
                )
            unet_key = clean_unet_key(key)
            if unet_key not in sd_unet:
                raise KeyError(
                    f"LoRA key '{key}' mapped to UNet key '{unet_key}' "
                    f"which does not exist in the UNet state dict. "
                    f"Check --lora_rank and --lora_alpha match your training config."
                )
            W_A = state_dict[key]
            W_B = state_dict[lora_b_key]
            orig = sd_unet[unet_key]
            processed_keys.update([key, lora_b_key])
            if orig.ndim == 4:
                rank = W_A.shape[0]
                out_ch, in_ch, kH, kW = orig.shape
                delta = torch.matmul(
                    W_B.view(out_ch, rank), W_A.view(rank, -1)
                ).view(out_ch, in_ch, kH, kW)
            else:
                delta = torch.mm(W_B, W_A)
            sd_unet[unet_key] = orig + alpha * delta
 
        elif "lora.up.weight" in key:
            if any(s in key for s in PEFT_NON_WEIGHT_SUFFIXES):
                processed_keys.add(key)
                continue
 
            lora_down_key = key.replace("lora.up.weight", "lora.down.weight")
            if lora_down_key not in state_dict:
                raise KeyError(
                    f"Found lora.up key '{key}' but missing lora.down key."
                )
            unet_key = clean_unet_key(key)
            if unet_key not in sd_unet:
                raise KeyError(
                    f"LoRA key '{key}' mapped to UNet key '{unet_key}' "
                    f"which does not exist in the UNet state dict."
                )
            W_up   = state_dict[key]
            W_down = state_dict[lora_down_key]
            orig   = sd_unet[unet_key]
            processed_keys.update([key, lora_down_key])
            if orig.ndim == 2:
                sd_unet[unet_key] = orig + alpha * torch.mm(W_up, W_down)
 
    remaining = [k for k in state_dict if k not in processed_keys]
    if remaining:
        # These are truly unrecognized keys — warn but list them all
        print(f"WARNING: {len(remaining)} unprocessed keys in checkpoint:")
        for k in remaining[:10]:
            print(f"  {k}")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")
 
    n_merged = len(processed_keys) // 2  # each layer has A and B
    print(f"LoRA merge done — {n_merged} layers merged.")
    unet.load_state_dict(sd_unet)
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
    image_names = find_images(args.input_image)
 
    existing = {os.path.basename(p) for p in find_images(args.output_dir)}
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
    parser.add_argument("--ckpt_path",           required=True,
                        help="Path to trained LoRA checkpoint folder")
    parser.add_argument("--pretrained_ckpt_path", default="pretrained",
                        help="Path to OSDFace pretrained dir containing "
                             "embedding_change_weights.pth (default: pretrained)")
    parser.add_argument("--merge_lora",      action="store_true")
    parser.add_argument("--no_text_cond",    action="store_true",
                        help="Run without text conditioning (pure OSDFace baseline)")
    parser.add_argument("--mixed_precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--gpu_ids",         nargs="+", type=int, default=[0])
    parser.add_argument("--lora_rank",       type=int,   default=16)
    parser.add_argument("--lora_alpha",      type=float, default=16)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--cat_prompt_embedding", action="store_true")
    parser.add_argument("--use_pos_embedding",    action="store_true")
    parser.add_argument("--use_att_pool",         action="store_true")
    parser.add_argument("--learnable_pos_emb",    action="store_true")
 
    args = parser.parse_args()
 
    if not args.no_text_cond and not args.prompts_json:
        parser.error("--prompts_json is required unless --no_text_cond is set")
 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
 
    os.makedirs(args.output_dir, exist_ok=True)
    mp.set_start_method("spawn", force=True)
 
    unet_merged = merge_unet(args) if args.merge_lora else None
    run_inference(args, unet_merged)