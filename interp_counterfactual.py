"""
interp_counterfactual.py — Swapped prompt counterfactual comparison

Generates 4 outputs from the same LQ image and assembles a side-by-side panel:

  [LQ Input] | [No Text (baseline)] | [Correct Prompt] | [Swapped Prompt (pos<->na)]

If the text is causally responsible for generating "no glasses / open mouth",
the swapped prompt should REVERSE those attributes.

This is the strongest qualitative interpretability result for your paper:
it shows the model is NOT just a good restorer — the text is actively
steering the output toward specific facial attributes.

Usage:
    python interp_counterfactual.py \
        --input_image /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2/00001327.png \
        --prompts_json /projectnb/cs585/projects/craft/data/test/CelebA/celeba_prompts_output.json \
        --hq_image /projectnb/cs585/projects/craft/data/test/CelebA/CelebA_Validation/celeba_512_validation/00001327.png \
        --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21\
        --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
        --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
        --conditioner_path checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
        --film_neg_weight 0.1 \
        --output_dir results/interp_counterfactual \
        --gpu_ids 0 \
        --mixed_precision fp16


Note: --hq_image is optional. If provided, it adds the HQ reference to the panel.
"""

import os
import sys
import json
import copy
import argparse

import torch
import torch.nn.functional as Fun
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from safetensors import safe_open
import torchvision.transforms.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.others import get_x0_from_noise
from lq_embed import vqvae_encoder, TwoLayerConv1x1
from text_conditioner import TextConditioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_preamble(text: str) -> str:
    for marker in ["in the description of ", "not in the description of "]:
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
    return text.strip()


def merge_unet(args):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    alpha = float(args.lora_alpha / args.lora_rank)
    with safe_open(os.path.join(args.ckpt_path, "pytorch_lora_weights.safetensors"), framework="pt") as f:
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
                out_ch = orig.shape[0]
                delta = torch.matmul(W_B.view(out_ch, rank), W_A.view(rank, -1)).view(orig.shape)
            else:
                delta = torch.mm(W_B, W_A)
            sd_unet[unet_key] = orig + alpha * delta
        elif "lora.up.weight" in key:
            lora_down_key = key.replace("lora.up.weight", "lora.down.weight")
            unet_key = key.replace(".lora.up.weight", ".weight").replace("unet.", "")
            orig = sd_unet[unet_key]
            if orig.ndim == 2:
                sd_unet[unet_key] = orig + alpha * torch.mm(state_dict[key], state_dict[lora_down_key])
    unet.load_state_dict(sd_unet)
    return unet


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_forward(unet, vae, img_encoder, embedding_change, conditioner,
                lq, alphas_cumprod, weight_dtype, device,
                pos_emb=None, na_emb=None, neg_weight=0.1):
    """Single forward pass. Pass pos_emb=None for baseline (no text cond)."""
    visual_embeds = embedding_change(img_encoder(lq).reshape(1, 77, -1))
    lq_latent = vae.encode(lq).latent_dist.sample() * vae.config.scaling_factor

    if pos_emb is not None:
        conditioner.set_text_embedding(pos_emb, na_emb, neg_weight=neg_weight)

    model_pred = unet(lq_latent, 399, encoder_hidden_states=visual_embeds).sample

    if pos_emb is not None:
        conditioner.clear_text_embedding()

    x0 = get_x0_from_noise(
        lq_latent.double(), model_pred.double(),
        alphas_cumprod.double(), 399,
    ).float()

    output = vae.decode(x0.to(weight_dtype) / vae.config.scaling_factor).sample.clamp(-1, 1)
    return ((output * 0.5 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(args):
    device = torch.device(f"cuda:{args.gpu_ids[0]}")
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    unet_merged = merge_unet(args)
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    alphas_cumprod  = noise_scheduler.alphas_cumprod.to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device, dtype=weight_dtype)
    unet = copy.deepcopy(unet_merged).to(device, dtype=weight_dtype)
    unet.eval().requires_grad_(False)
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device, dtype=weight_dtype)
    text_encoder.eval().requires_grad_(False)
    img_encoder = vqvae_encoder(args).to(device, dtype=weight_dtype)
    img_encoder.eval()
    embedding_change = TwoLayerConv1x1(512, 1024)
    embedding_change.load_state_dict(
        torch.load(os.path.join(args.ckpt_path, "embedding_change_weights.pth"), weights_only=False)
    )
    embedding_change.to(device, dtype=weight_dtype).eval()
    conditioner = TextConditioner(unet, text_dim=1024)
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
        for k in [name, f"{stem}_LQ{ext}", f"{stem}_lq{ext}"]:
            prompts[k] = entry

    filename = os.path.basename(args.input_image)
    if filename not in prompts:
        raise KeyError(f"No prompt for '{filename}'")
    entry = prompts[filename]

    pos_text = strip_preamble(entry["pos"])
    na_text  = strip_preamble(entry["na"])

    print(f"\nImage  : {filename}")
    print(f"POS    : {pos_text[:120]}")
    print(f"NA     : {na_text[:120]}")
    print(f"film_neg_weight = {args.film_neg_weight}\n")

    def encode_text(text):
        ids = tokenizer([text], padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        hidden = text_encoder(ids).last_hidden_state
        eos_pos = (ids == tokenizer.eos_token_id).float().argmax(dim=1)
        eos = hidden[torch.arange(1, device=device), eos_pos]
        return eos.unsqueeze(1).expand(-1, tokenizer.model_max_length, -1).to(weight_dtype)

    pos_emb     = encode_text(pos_text)
    na_emb      = encode_text(na_text)
    # Swapped: what was pos becomes na and vice versa
    pos_emb_sw  = encode_text(na_text)
    na_emb_sw   = encode_text(pos_text)

    lq_img = Image.open(args.input_image).convert("RGB")
    lq = F.to_tensor(lq_img).unsqueeze(0).to(device, dtype=weight_dtype) * 2 - 1
    lq = Fun.interpolate(lq, (512, 512), mode="bilinear", align_corners=True)

    # Generate all 3 outputs
    print("Running inference — no text (baseline)...")
    out_baseline = run_forward(unet, vae, img_encoder, embedding_change, conditioner,
                               lq, alphas_cumprod, weight_dtype, device,
                               pos_emb=None, neg_weight=args.film_neg_weight)

    print("Running inference — correct prompt...")
    out_correct  = run_forward(unet, vae, img_encoder, embedding_change, conditioner,
                               lq, alphas_cumprod, weight_dtype, device,
                               pos_emb=pos_emb, na_emb=na_emb, neg_weight=args.film_neg_weight)

    print("Running inference — swapped prompt (pos<->na)...")
    out_swapped  = run_forward(unet, vae, img_encoder, embedding_change, conditioner,
                               lq, alphas_cumprod, weight_dtype, device,
                               pos_emb=pos_emb_sw, na_emb=na_emb_sw, neg_weight=args.film_neg_weight)

    # --- Save individual images ---
    Image.fromarray(out_baseline).save(os.path.join(args.output_dir, "out_baseline.png"))
    Image.fromarray(out_correct ).save(os.path.join(args.output_dir, "out_correct.png"))
    Image.fromarray(out_swapped ).save(os.path.join(args.output_dir, "out_swapped.png"))

    # --- Assemble panel ---
    panels = [
        (np.array(lq_img.resize((512, 512))), "LQ Input",       "black"),
        (out_baseline,  "OSD Output\n(no text)",                 "gray"),
        (out_correct,   "OSD + Text\n(correct prompt)",          "green"),
        (out_swapped,   "OSD + Text\n(swapped prompt pos↔na)",   "red"),
    ]

    # Add HQ reference if provided
    if args.hq_image and os.path.exists(args.hq_image):
        hq_img = Image.open(args.hq_image).convert("RGB")
        panels.insert(1, (np.array(hq_img.resize((512, 512))), "HQ Reference", "gold"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (img_arr, title, border_color) in zip(axes, panels):
        ax.imshow(img_arr)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    # Add prompt text below
    pos_wrapped = f"POS: {pos_text[:160]}" + ("..." if len(pos_text) > 160 else "")
    na_wrapped  = f"NA:  {na_text[:160]}"  + ("..." if len(na_text)  > 160 else "")
    fig.text(0.5, 0.01, pos_wrapped + "\n" + na_wrapped,
             ha="center", va="bottom", fontsize=7, color="dimgray",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.suptitle(f"Counterfactual Analysis — film_neg_weight={args.film_neg_weight}",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "counterfactual_panel.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPanel saved → {out_path}")

    # Pixel-level difference stats
    correct_arr = out_correct.astype(float)
    swap_arr    = out_swapped.astype(float)
    baseline_arr= out_baseline.astype(float)

    diff_correct_vs_baseline = np.abs(correct_arr - baseline_arr).mean()
    diff_swapped_vs_baseline = np.abs(swap_arr - baseline_arr).mean()
    diff_correct_vs_swapped  = np.abs(correct_arr - swap_arr).mean()

    print("\n--- Pixel difference statistics (mean |A - B| per channel) ---")
    print(f"  correct  vs baseline : {diff_correct_vs_baseline:.2f}")
    print(f"  swapped  vs baseline : {diff_swapped_vs_baseline:.2f}")
    print(f"  correct  vs swapped  : {diff_correct_vs_swapped:.2f}")
    print("\nInterpretation hint:")
    print("  If 'correct vs swapped' >> 'correct vs baseline', the prompt is")
    print("  causing a large shift in output appearance — text IS doing work.")
    print("  If all diffs are similar, the FiLM modulation may be too weak.")

    stats_path = os.path.join(args.output_dir, "diff_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"film_neg_weight: {args.film_neg_weight}\n")
        f.write(f"image: {filename}\n")
        f.write(f"pos: {pos_text}\n")
        f.write(f"na:  {na_text}\n\n")
        f.write(f"correct  vs baseline : {diff_correct_vs_baseline:.4f}\n")
        f.write(f"swapped  vs baseline : {diff_swapped_vs_baseline:.4f}\n")
        f.write(f"correct  vs swapped  : {diff_correct_vs_swapped:.4f}\n")
    print(f"Stats saved → {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image",  "-i", required=True)
    parser.add_argument("--prompts_json", required=True)
    parser.add_argument("--hq_image",    type=str, default=None, help="Optional HQ reference for panel")
    parser.add_argument("--output_dir",   "-o", default="results/interp_counterfactual")
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