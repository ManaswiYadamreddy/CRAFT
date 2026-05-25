"""
interp_pca_diff.py — PCA feature difference maps (Revelio §4.5 approach)

Runs the same image twice through the UNet:
  (A) with text conditioning active  (film_neg_weight=0.1)
  (B) without text conditioning (baseline OSDFace)

Then extracts intermediate hidden states from selected BasicTransformerBlocks
and applies PCA to visualize WHERE in the image the text is having an effect.

The difference PCA map (A - B) highlights the spatial regions that the text
prompt is modulating — if the prompt encodes "no glasses / mouth open", you
expect to see activations in the eye and mouth regions.

Revelio shows up_ft1 blocks carry fine-grained semantics. This script lets
you confirm whether your FiLM modulation is hitting those layers.

Usage:
    python interp_pca_diff.py \
        --input_image /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2/00000037.png \
        --prompts_json /projectnb/cs585/projects/craft/data/test/CelebA/celeba_prompts_output.json \
        --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21\
        --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
        --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
        --conditioner_path checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
        --film_neg_weight 0.1 \
        --output_dir results/interp_pca/37 \
        --gpu_ids 0 \
        --mixed_precision fp16

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
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from PIL import Image
from safetensors import safe_open
import torchvision.transforms.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
 
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
            delta = torch.matmul(W_B.view(orig.shape[0], -1), W_A.view(W_A.shape[0], -1)
                                 ).view(orig.shape) if orig.ndim == 4 else torch.mm(W_B, W_A)
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
# Feature extractor hook
# ---------------------------------------------------------------------------
 
class FeatureExtractor:
    """Registers forward hooks on named BasicTransformerBlocks and stores outputs."""
 
    def __init__(self, unet: torch.nn.Module, target_keys: list):
        self.features: dict = {}
        self._hooks = []
        for name, module in unet.named_modules():
            if isinstance(module, BasicTransformerBlock):
                key = name.replace(".", "_")
                if any(t in key for t in target_keys):
                    hook = module.register_forward_hook(self._make_hook(key))
                    self._hooks.append(hook)
 
    def _make_hook(self, key):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            self.features[key] = h.detach().float().cpu()
        return hook
 
    def clear(self):
        self.features.clear()
 
    def remove(self):
        for h in self._hooks:
            h.remove()
 
 
def pca_to_rgb(feat_map: np.ndarray) -> np.ndarray:
    """
    feat_map: (H*W, C) flattened spatial features.
    Returns (H*W, 3) RGB image with PCA components as channels, normalized to [0,255].
    """
    n_components = min(3, feat_map.shape[1], feat_map.shape[0])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(feat_map)  # (H*W, 3)
    if projected.shape[1] < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])))
    # Normalize each channel to [0, 255]
    for c in range(3):
        col = projected[:, c]
        col = (col - col.min()) / (col.max() - col.min() + 1e-8)
        projected[:, c] = col
    return (projected * 255).astype(np.uint8)
 
 
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
        for key in [name, f"{stem}_LQ{ext}", f"{stem}_lq{ext}"]:
            prompts[key] = entry
 
    filename = os.path.basename(args.input_image)
    if filename not in prompts:
        raise KeyError(f"No prompt found for '{filename}'")
    entry = prompts[filename]
 
    print(f"Image  : {filename}")
    print(f"POS    : {strip_preamble(entry['pos'])[:100]}")
    print(f"NA     : {strip_preamble(entry['na'])[:100]}")
 
    def encode_text(text):
        ids = tokenizer([text], padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        hidden = text_encoder(ids).last_hidden_state
        eos_pos = (ids == tokenizer.eos_token_id).float().argmax(dim=1)
        eos = hidden[torch.arange(1, device=device), eos_pos]
        return eos.unsqueeze(1).expand(-1, tokenizer.model_max_length, -1).to(weight_dtype)
 
    pos_emb = encode_text(strip_preamble(entry["pos"]))
    na_emb  = encode_text(strip_preamble(entry["na"]))
 
    img = Image.open(args.input_image).convert("RGB")
    lq  = F.to_tensor(img).unsqueeze(0).to(device, dtype=weight_dtype) * 2 - 1
    lq  = Fun.interpolate(lq, (512, 512), mode="bilinear", align_corners=True)
 
    # Target blocks to visualize — covering bottleneck, up_ft1, up_ft2
    # These key fragments match the naming in SD 2.1 UNet
    target_keys = ["mid_block", "up_blocks_0", "up_blocks_1", "up_blocks_2"]
 
    timestep = 399
 
    results = {}
 
    for mode in ["with_text", "without_text"]:
        extractor = FeatureExtractor(unet, target_keys)
 
        with torch.no_grad():
            visual_embeds = embedding_change(img_encoder(lq).reshape(1, 77, -1))
            lq_latent = vae.encode(lq).latent_dist.sample() * vae.config.scaling_factor
 
            if mode == "with_text":
                conditioner.set_text_embedding(pos_emb, na_emb, neg_weight=args.film_neg_weight)
 
            _ = unet(lq_latent, timestep, encoder_hidden_states=visual_embeds).sample
 
            if mode == "with_text":
                conditioner.clear_text_embedding()
 
        results[mode] = {k: v for k, v in extractor.features.items()}
        extractor.remove()
        print(f"  [{mode}] extracted {len(extractor.features)} feature maps")
 
    # PCA and diff visualization
    keys_present = sorted(set(results["with_text"].keys()) & set(results["without_text"].keys()))
    print(f"\nBlocks captured: {keys_present}")
 
    n_blocks = len(keys_present)
    fig = plt.figure(figsize=(5 * 4, 4 * n_blocks + 2))
    gs  = gridspec.GridSpec(n_blocks + 1, 4, figure=fig,
                            hspace=0.4, wspace=0.3)
 
    # Input image on top
    ax_img = fig.add_subplot(gs[0, :2])
    ax_img.imshow(img.resize((256, 256)))
    ax_img.set_title(f"Input: {filename}\nPOS: {strip_preamble(entry['pos'])[:80]}\n"
                     f"NA: {strip_preamble(entry['na'])[:80]}", fontsize=7)
    ax_img.axis("off")
 
    ax_legend = fig.add_subplot(gs[0, 2:])
    ax_legend.text(0.05, 0.82, "Column guide:", fontsize=9, fontweight="bold", transform=ax_legend.transAxes)
    ax_legend.text(0.05, 0.65, "[A] PCA of features WITH text cond", fontsize=8, transform=ax_legend.transAxes)
    ax_legend.text(0.05, 0.50, "[B] PCA of features WITHOUT text cond", fontsize=8, transform=ax_legend.transAxes)
    ax_legend.text(0.05, 0.35, "[A-B] Signed diff  |  RED = text stronger  |  BLUE = baseline stronger",
                   fontsize=8, color="purple", transform=ax_legend.transAxes)
    ax_legend.text(0.05, 0.20, "[A-B] Absolute diff heatmap (overall magnitude)", fontsize=8, color="darkorange", transform=ax_legend.transAxes)
    ax_legend.text(0.05, 0.05, f"film_neg_weight={args.film_neg_weight}", fontsize=8, transform=ax_legend.transAxes)
    ax_legend.axis("off")
 
    for row_idx, key in enumerate(keys_present):
        feat_w  = results["with_text"][key]
        feat_wo = results["without_text"][key]
 
        def to_spatial(feat):
            feat = feat.squeeze(0).float().numpy()
            N, C = feat.shape
            side = int(N ** 0.5)
            if side * side == N:
                return feat, side, side
            return feat, N, 1
 
        feat_w_np,  H_w,  W_w  = to_spatial(feat_w)
        feat_wo_np, H_wo, W_wo = to_spatial(feat_wo)
 
        side = int(feat_w_np.shape[0] ** 0.5)
        H = W = side if side * side == feat_w_np.shape[0] else feat_w_np.shape[0]
 
        def norm(x):
            x = x - x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True) + 1e-8
            return x / std
 
        rgb_w  = pca_to_rgb(norm(feat_w_np))
        rgb_wo = pca_to_rgb(norm(feat_wo_np))
 
        # Signed diff: mean over channel dim — positive = text stronger, negative = baseline stronger
        signed_diff = (feat_w_np.astype(float) - feat_wo_np.astype(float)).mean(axis=-1)
        abs_diff    = np.abs(signed_diff)
 
        def reshape_if_square(arr, H, W):
            if H * W == arr.shape[0] or (len(arr.shape) > 1 and H * W == arr.shape[0]):
                if len(arr.shape) == 1:
                    return arr.reshape(H, W)
                return arr.reshape(H, W, -1) if arr.shape[-1] > 1 else arr.reshape(H, W)
            return arr
 
        rgb_w_2d      = reshape_if_square(rgb_w,       H, W)
        rgb_wo_2d     = reshape_if_square(rgb_wo,      H, W)
        signed_diff_2d = reshape_if_square(signed_diff, H, W)
        abs_diff_2d    = reshape_if_square(abs_diff,    H, W)
 
        # Symmetric colorscale for signed diff so zero = white/neutral
        vmax = max(abs(signed_diff_2d.max()), abs(signed_diff_2d.min())) + 1e-8
 
        titles = [
            f"{key}\n[A] w/ text",
            f"{key}\n[B] w/o text",
            f"{key}\n[A-B] signed\nred=text blue=base",
            f"{key}\n[A-B] abs heatmap",
        ]
        row = row_idx + 1
        for col, (data, title, cmap, vmin_vmax) in enumerate([
            (rgb_w_2d,       titles[0], None,      None),
            (rgb_wo_2d,      titles[1], None,      None),
            (signed_diff_2d, titles[2], "RdBu_r",  (-vmax, vmax)),
            (abs_diff_2d,    titles[3], "hot",      None),
        ]):
            ax = fig.add_subplot(gs[row, col])
            if cmap and vmin_vmax:
                im = ax.imshow(data, cmap=cmap, vmin=vmin_vmax[0], vmax=vmin_vmax[1],
                               interpolation="nearest")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            elif cmap:
                ax.imshow(data, cmap=cmap, interpolation="nearest")
            else:
                ax.imshow(data, interpolation="nearest")
            ax.set_title(title, fontsize=7)
            ax.axis("off")
 
    out_path = os.path.join(args.output_dir, "pca_feature_diff.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPCA diff figure saved → {out_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image",  "-i", required=True)
    parser.add_argument("--prompts_json", required=True)
    parser.add_argument("--output_dir",   "-o", default="results/interp_pca")
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