"""
interp_film_magnitude.py — Block-level FiLM activation analysis

Runs inference on a single image and logs the L1 magnitude of the net
gamma modulation (gamma_pos - neg_weight * gamma_na) per BasicTransformerBlock.

This answers: WHICH blocks in the UNet are actually being modulated by the
text prompt (e.g. "no glasses", "mouth slightly open")?

Revelio shows fine-grained semantic info lives in up_ft1 of SD-based U-Nets.
This script lets you verify whether your FiLM modulations are concentrated
there or spread across the network.

Usage:
    python interp_film_magnitude.py \
        --input_image /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2/00001645.png \
        --prompts_json /projectnb/cs585/projects/craft/data/test/CelebA/celeba_prompts_output.json \
        --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21\
        --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
        --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
        --conditioner_path checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
        --film_neg_weight 0.1 \
        --output_dir results/interp_film_magnitude \
        --gpu_ids 0 \
        --mixed_precision fp16
"""

import os
import sys
import json
import copy
import glob
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as Fun
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
import torchvision.transforms.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock

# -- adjust sys.path so local modules resolve --
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.others import get_x0_from_noise
from lq_embed import vqvae_encoder, TwoLayerConv1x1
from text_conditioner import TextConditioner


# ---------------------------------------------------------------------------
# Helpers (copied from infer_textcond.py)
# ---------------------------------------------------------------------------

def strip_preamble(text: str) -> str:
    for marker in ["in the description of ", "not in the description of "]:
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
    return text.strip()


def merge_unet(args):
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    alpha = float(args.lora_alpha / args.lora_rank)
    with safe_open(
        os.path.join(args.ckpt_path, "pytorch_lora_weights.safetensors"),
        framework="pt",
    ) as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    sd_unet = unet.state_dict()
    for key in state_dict:
        if "lora_A" in key:
            lora_b_key = key.replace("lora_A", "lora_B")
            unet_key = key.replace(".lora_A.weight", ".weight").replace("unet.", "")
            W_A, W_B = state_dict[key], state_dict[lora_b_key]
            orig = sd_unet[unet_key]
            if orig.ndim == 4:
                rank = W_A.shape[0]
                out_ch, in_ch, kH, kW = orig.shape
                delta = torch.matmul(W_B.view(out_ch, rank), W_A.view(rank, -1)).view(orig.shape)
            else:
                delta = torch.mm(W_B, W_A)
            sd_unet[unet_key] = orig + alpha * delta
        elif "lora.up.weight" in key:
            lora_down_key = key.replace("lora.up.weight", "lora.down.weight")
            unet_key = key.replace(".lora.up.weight", ".weight").replace("unet.", "")
            W_up, W_down = state_dict[key], state_dict[lora_down_key]
            orig = sd_unet[unet_key]
            if orig.ndim == 2:
                sd_unet[unet_key] = orig + alpha * torch.mm(W_up, W_down)
    unet.load_state_dict(sd_unet)
    return unet


# ---------------------------------------------------------------------------
# Instrumented TextConditioner: records gamma magnitudes per block
# ---------------------------------------------------------------------------

class InstrumentedTextConditioner(TextConditioner):
    """
    Subclass of TextConditioner that records the L1 magnitude of gamma_net
    and beta_net at each BasicTransformerBlock during forward.
    """
    def __init__(self, unet, text_dim=1024):
        super().__init__(unet, text_dim)
        self.gamma_log: dict = {}   # key -> mean |gamma_net|
        self.beta_log: dict  = {}   # key -> mean |beta_net|
        self.block_order: list = [] # insertion order

    def _make_hook(self, key: str):
        def hook(module, input, output):
            if self._pos_emb is None:
                return output

            if isinstance(output, tuple):
                h, rest = output[0], output[1:]
            else:
                h, rest = output, None

            device, dtype = h.device, h.dtype
            film_dtype = next(self.film_pos_layers[key].parameters()).dtype

            pos_pooled = self._pos_emb.mean(dim=1).to(device, dtype=film_dtype)
            gamma_pos, beta_pos = self.film_pos_layers[key](pos_pooled)
            gamma_pos = gamma_pos.to(dtype=dtype)
            beta_pos  = beta_pos.to(dtype=dtype)

            if self._na_emb is not None and self._neg_weight > 0:
                na_pooled = self._na_emb.mean(dim=1).to(device, dtype=film_dtype)
                gamma_na, beta_na = self.film_na_layers[key](na_pooled)
                gamma_na = gamma_na.to(dtype=dtype)
                beta_na  = beta_na.to(dtype=dtype)
                gamma_net = gamma_pos - self._neg_weight * gamma_na
                beta_net  = beta_pos  - self._neg_weight * beta_na
            else:
                gamma_net = gamma_pos
                beta_net  = beta_pos

            # --- Record magnitudes ---
            self.gamma_log[key] = gamma_net.abs().mean().item()
            self.beta_log[key]  = beta_net.abs().mean().item()
            if key not in self.block_order:
                self.block_order.append(key)

            h = h * (1 + gamma_net) + beta_net
            if rest is not None:
                return (h,) + rest
            return h
        return hook

    def clear_logs(self):
        self.gamma_log.clear()
        self.beta_log.clear()
        self.block_order.clear()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(args):
    device = torch.device(f"cuda:{args.gpu_ids[0]}")
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    unet_merged = merge_unet(args)

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    alphas_cumprod  = noise_scheduler.alphas_cumprod.to(device)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae"
                                        ).to(device, dtype=weight_dtype)

    unet = copy.deepcopy(unet_merged).to(device, dtype=weight_dtype)
    unet.eval().requires_grad_(False)

    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder"
                                                 ).to(device, dtype=weight_dtype)
    text_encoder.eval().requires_grad_(False)

    img_encoder = vqvae_encoder(args).to(device, dtype=weight_dtype)
    img_encoder.eval()

    embedding_change = TwoLayerConv1x1(512, 1024)
    embedding_change.load_state_dict(
        torch.load(os.path.join(args.ckpt_path, "embedding_change_weights.pth"), weights_only=False)
    )
    embedding_change.to(device, dtype=weight_dtype).eval()

    # Instrumented conditioner
    conditioner = InstrumentedTextConditioner(unet, text_dim=1024)
    conditioner.register_hooks(unet)
    conditioner.load(args.conditioner_path, map_location="cpu")
    conditioner.to(device, dtype=weight_dtype).eval()

    # Load prompt
    with open(args.prompts_json) as f:
        raw = json.load(f)
    prompts = {}
    for item in raw:
        name = item["image"]
        stem, ext = os.path.splitext(name)
        entry = {"pos": item.get("pos",""), "na": item.get("na","")}
        prompts[name] = prompts[f"{stem}_LQ{ext}"] = prompts[f"{stem}_lq{ext}"] = entry

    filename = os.path.basename(args.input_image)
    if filename not in prompts:
        raise KeyError(f"No prompt found for '{filename}'")
    entry = prompts[filename]

    print(f"\nImage: {filename}")
    print(f"POS: {strip_preamble(entry['pos'])[:120]}")
    print(f"NA:  {strip_preamble(entry['na'])[:120]}")
    print(f"film_neg_weight: {args.film_neg_weight}\n")

    # Encode text
    def encode_text(text):
        ids = tokenizer([text], padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        hidden = text_encoder(ids).last_hidden_state
        eos_pos = (ids == tokenizer.eos_token_id).float().argmax(dim=1)
        eos = hidden[torch.arange(1, device=device), eos_pos]
        return eos.unsqueeze(1).expand(-1, tokenizer.model_max_length, -1).to(weight_dtype)

    pos_emb = encode_text(strip_preamble(entry["pos"]))
    na_emb  = encode_text(strip_preamble(entry["na"]))

    # Load and preprocess image
    img = Image.open(args.input_image).convert("RGB")
    lq  = F.to_tensor(img).unsqueeze(0).to(device, dtype=weight_dtype) * 2 - 1
    lq  = Fun.interpolate(lq, (512, 512), mode="bilinear", align_corners=True)

    timestep = 399

    with torch.no_grad():
        visual_embeds = embedding_change(img_encoder(lq).reshape(1, 77, -1))
        lq_latent = vae.encode(lq).latent_dist.sample() * vae.config.scaling_factor

        conditioner.clear_logs()
        conditioner.set_text_embedding(pos_emb, na_emb, neg_weight=args.film_neg_weight)
        _ = unet(lq_latent, timestep, encoder_hidden_states=visual_embeds).sample
        conditioner.clear_text_embedding()

    gamma_log = conditioner.gamma_log
    beta_log  = conditioner.beta_log
    keys      = conditioner.block_order

    print(f"Total blocks instrumented: {len(keys)}")
    print(f"\n{'Block':<55} {'|gamma_net|':>12}  {'|beta_net|':>10}")
    print("-" * 80)
    for k in keys:
        print(f"  {k:<53} {gamma_log[k]:12.6f}  {beta_log[k]:10.6f}")

    # --- Identify top-5 most activated blocks ---
    sorted_by_gamma = sorted(keys, key=lambda k: gamma_log[k], reverse=True)
    print(f"\nTop 5 blocks by |gamma_net| (film_neg_weight={args.film_neg_weight}):")
    for rank, k in enumerate(sorted_by_gamma[:5], 1):
        print(f"  {rank}. {k}  →  gamma={gamma_log[k]:.6f}  beta={beta_log[k]:.6f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(keys) * 0.35), 9))
    x = range(len(keys))
    short_keys = [k.replace("down_blocks_", "d").replace("up_blocks_", "u")
                   .replace("mid_block_", "mid").replace("attentions_", "a")
                   .replace("transformer_blocks_", "t") for k in keys]

    for ax, log, label, color in [
        (axes[0], gamma_log, "|gamma_net| (scale)", "steelblue"),
        (axes[1], beta_log,  "|beta_net|  (shift)", "darkorange"),
    ]:
        vals = [log[k] for k in keys]
        bars = ax.bar(x, vals, color=color, alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(short_keys, rotation=90, fontsize=6)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"FiLM {label} per BasicTransformerBlock  "
                     f"(film_neg_weight={args.film_neg_weight})", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        # Highlight top-3
        top3_idx = sorted(range(len(keys)), key=lambda i: vals[i], reverse=True)[:3]
        for i in top3_idx:
            bars[i].set_edgecolor("red")
            bars[i].set_linewidth(2)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "film_magnitude_per_block.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    # Save raw numbers
    csv_path = os.path.join(args.output_dir, "film_magnitude.csv")
    with open(csv_path, "w") as f:
        f.write("block,gamma_net_abs_mean,beta_net_abs_mean\n")
        for k in keys:
            f.write(f"{k},{gamma_log[k]:.8f},{beta_log[k]:.8f}\n")
    print(f"CSV saved  → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image",  "-i", required=True)
    parser.add_argument("--prompts_json", required=True)
    parser.add_argument("--output_dir",   "-o", default="results/interp_film_magnitude")
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--img_encoder_weight", default="pretrained/associate_2.ckpt")
    parser.add_argument("--ckpt_path",        required=True)
    parser.add_argument("--conditioner_path", required=True)
    parser.add_argument("--film_neg_weight",  type=float, default=0.1)
    parser.add_argument("--mixed_precision",  choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--gpu_ids",          nargs="+", type=int, default=[0])
    parser.add_argument("--lora_rank",        type=int, default=16)
    parser.add_argument("--lora_alpha",       type=float, default=16)
    parser.add_argument("--cat_prompt_embedding", action="store_true")
    parser.add_argument("--use_pos_embedding",    action="store_true")
    parser.add_argument("--use_att_pool",         action="store_true")
    parser.add_argument("--learnable_pos_emb",    action="store_true")
    args = parser.parse_args()
    run_analysis(args)