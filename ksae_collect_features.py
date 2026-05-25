"""
ksae_collect_features.py — Stage 1: Collect FiLM-modulated UNet features

Runs images through the OSDFace + TextConditioner pipeline and extracts
hidden states from TWO target blocks AFTER FiLM modulation, in a single
forward pass per image (no redundant computation).

Target blocks:
  up_blocks_1_attentions_1  (gamma=0.618) — highest modulation magnitude;
                             reinforces the visual prior from VRE cross-attention
  up_blocks_2_attentions_1  (gamma=0.208) — where the text effect spatially
                             localizes to the eye/glasses region (per PCA diff)

This two-block design lets you tell a mechanistic story:
  "FiLM modulates the prior at up_blocks_1; the attribute-specific effect
   then localizes spatially at up_blocks_2."

For each image we store per block:
  - Spatially mean-pooled feature vector  (d,) — used for k-SAE training
  - Image filename                         (str) — for attribute lookup
  - Attribute binary vector                (28,) — parsed from prompts_json

Usage:
    # Collect training features (70k images)
    python ksae_collect_features.py \
        --image_dir /projectnb/cs585/projects/craft/data/train/LQ_images_512x512 \
        --prompts_json /projectnb/cs585/projects/craft/prompts_output_final.json \
        --output_dir features/ksae_dual \
        --split train \
        --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \
        --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
        --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
        --conditioner_path checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
        --film_neg_weight 0.1 \
        --batch_size 8 \
        --gpu_ids 0 \
        --mixed_precision fp16

    # Collect test features (3k images) — same command, add --split test
"""

import os
import sys
import json
import glob
import copy
import argparse

import torch
import torch.nn.functional as Fun
import numpy as np
from PIL import Image
from tqdm import tqdm
from safetensors import safe_open
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lq_embed import vqvae_encoder, TwoLayerConv1x1
from text_conditioner import TextConditioner


# ---------------------------------------------------------------------------
# Blocks to extract from — edit here if you want different blocks
# ---------------------------------------------------------------------------

TARGET_BLOCKS = {
    # "up1a1": "up_blocks_1_attentions_1_transformer_blocks_0",  # prior reinforcement
    # "up2a1": "up_blocks_2_attentions_1_transformer_blocks_0",  # spatial localization
    "up2a2": "up_blocks_2_attentions_2_transformer_blocks_0",  # prior reinforcement
    "up3a0": "up_blocks_3_attentions_0_transformer_blocks_0",  # spatial localization

}


# ---------------------------------------------------------------------------
# Attribute vocabulary
# ---------------------------------------------------------------------------

PAPER_28_TO_CELEBA40 = {
    'Black Hair':'Black_Hair', 'Blond Hair':'Blond_Hair', 'Blurry':'Blurry',
    'Brown Hair':'Brown_Hair', 'Eyeglasses':'Eyeglasses', 'Gray Hair':'Gray_Hair',
    'Heavy Makeup':'Heavy_Makeup', 'Mouth Slightly Open':'Mouth_Slightly_Open',
    'Mustache':'Mustache', 'Big Eyes':'Narrow_Eyes',
    'No Beard':'No_Beard', 'Receding Hairline':'Receding_Hairline',
    'Sideburns':'Sideburns', 'Smiling':'Smiling', 'Straight Hair':'Straight_Hair',
    'Wearing Earrings':'Wearing_Earrings', 'Wearing Hat':'Wearing_Hat', 'Male':'Male',
    'Wearing Necklace':'Wearing_Necklace', 'Big Nose':'Big_Nose',
    'Wearing Lipstick':'Wearing_Lipstick', 'Young':'Young', 'Wavy Hair':'Wavy_Hair',
    'Big Lips':'Big_Lips', 'Bald':'Bald', 'Bangs':'Bangs',
    'Chubby':'Chubby', 'Double Chin':'Double_Chin',
}
PAPER_28 = list(PAPER_28_TO_CELEBA40.keys())
N_ATTRS  = len(PAPER_28)


def parse_attrs_from_prompt_entry(entry: dict) -> np.ndarray:
    vec = np.zeros(N_ATTRS, dtype=np.float32)
    if "positive_attrs" in entry:
        for attr in entry["positive_attrs"]:
            if attr in PAPER_28:
                vec[PAPER_28.index(attr)] = 1.0
        return vec
    # Fallback: substring match on pos text
    pos = entry.get("pos", "")
    for marker in ["in the description of ", "not in the description of "]:
        idx = pos.find(marker)
        if idx != -1:
            pos = pos[idx + len(marker):]
            break
    for i, attr in enumerate(PAPER_28):
        if attr.lower() in pos.lower():
            vec[i] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FaceFeatureDataset(Dataset):
    def __init__(self, image_dir, prompts_json, resolution=512):
        exts = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(image_dir, e)))
        all_paths = sorted(set(paths))

        with open(prompts_json) as f:
            raw = json.load(f)

        self.prompts = {}
        for item in raw:
            name = item["image"]
            stem, ext = os.path.splitext(name)
            entry = {k: item.get(k, "") for k in ["pos","na","positive_attrs","negative_attrs"]}
            for key in [name, f"{stem}_LQ{ext}", f"{stem}_lq{ext}"]:
                self.prompts[key] = entry

        self.resolution = resolution

        # --- Coverage check: report and hard-fail on bad coverage ---
        matched   = [p for p in all_paths if os.path.basename(p) in self.prompts]
        unmatched = [p for p in all_paths if os.path.basename(p) not in self.prompts]

        print(f"\nPrompt coverage: {len(matched)}/{len(all_paths)} images matched "
              f"({100*len(matched)/max(len(all_paths),1):.1f}%)")

        if unmatched:
            print(f"  WARNING: {len(unmatched)} images have no prompt entry. "
                  f"First 5: {[os.path.basename(p) for p in unmatched[:5]]}")

        if len(matched) == 0:
            raise ValueError(
                f"No images matched any prompt key in {prompts_json}. "
                f"Check that image filenames correspond to prompt 'image' fields."
            )

        if len(matched) / max(len(all_paths), 1) < 0.95:
            raise ValueError(
                f"Only {len(matched)}/{len(all_paths)} images have prompts "
                f"({100*len(matched)/len(all_paths):.1f}%). Expected >95%. "
                f"Check filename alignment between image_dir and prompts_json."
            )

        # --- Validate prompt content: catch empty pos/na before the long run ---
        empty_pos = [k for k, v in self.prompts.items() if not v.get("pos", "").strip()]
        empty_na  = [k for k, v in self.prompts.items() if not v.get("na",  "").strip()]
        if empty_pos:
            raise ValueError(
                f"{len(empty_pos)} prompt entries have empty 'pos' text. "
                f"First 3: {empty_pos[:3]}. "
                f"Features collected with empty prompts would be invalid."
            )
        if empty_na:
            raise ValueError(
                f"{len(empty_na)} prompt entries have empty 'na' text. "
                f"First 3: {empty_na[:3]}."
            )

        # --- Validate attribute parsing: warn if zero attrs parsed (likely a format mismatch) ---
        sample_entries = list(self.prompts.values())[:20]
        zero_attr_count = sum(
            1 for e in sample_entries if parse_attrs_from_prompt_entry(e).sum() == 0
        )
        if zero_attr_count > len(sample_entries) // 2:
            print(f"\n  WARNING: {zero_attr_count}/20 sampled prompts parsed zero attributes. "
                  f"Attribute labels may be incorrect. Check that prompts_json contains "
                  f"'positive_attrs' lists or that pos text contains recognizable attribute names.")

        self.paths = matched
        print(f"Dataset ready: {len(self.paths)} images\n")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = os.path.basename(path)
        img  = Image.open(path).convert("RGB").resize((self.resolution, self.resolution))
        lq   = (torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0) * 2 - 1

        # Hard fail here too — by this point name is guaranteed to be in self.prompts
        # (filtered at __init__), but be explicit so any future refactor doesn't regress.
        if name not in self.prompts:
            raise KeyError(
                f"Image '{name}' has no prompt entry. This should not happen — "
                f"check that __init__ filtering is working correctly."
            )

        entry = self.prompts[name]
        attrs = parse_attrs_from_prompt_entry(entry)
        return {
            "lq":       lq,
            "filename": name,
            "pos":      entry["pos"],
            "na":       entry["na"],
            "attrs":    torch.from_numpy(attrs),
        }


# ---------------------------------------------------------------------------
# Model helpers
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
                rank = W_A.shape[0]; out_ch = orig.shape[0]
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
# Multi-block feature hook
# ---------------------------------------------------------------------------

class MultiBlockHook:
    """
    Registers forward hooks on multiple BasicTransformerBlocks simultaneously.
    All blocks are captured in a single UNet forward pass — no extra cost.

    features dict: {block_shortname -> (B, feat_dim) tensor}
    """
    def __init__(self):
        self.features: dict = {}   # shortname -> (B, C) cpu tensor
        self._hooks:   list = []

    def register(self, unet: torch.nn.Module, target_blocks: dict):
        """
        target_blocks: {shortname: full_block_key} e.g.
            {"up1a1": "up_blocks_1_attentions_1_transformer_blocks_0"}
        """
        # Build reverse map: full_key -> shortname
        key_to_name = {v: k for k, v in target_blocks.items()}
        hooked = set()

        for name, module in unet.named_modules():
            if not isinstance(module, BasicTransformerBlock):
                continue
            block_key = name.replace(".", "_")
            if block_key in key_to_name:
                shortname = key_to_name[block_key]
                hook = module.register_forward_hook(self._make_hook(shortname))
                self._hooks.append(hook)
                hooked.add(shortname)
                print(f"  Hooked [{shortname}]: {name}")

        missing = set(target_blocks.keys()) - hooked
        if missing:
            raise ValueError(f"Could not find blocks: {missing}. "
                             f"Check TARGET_BLOCKS keys match UNet module names.")

    def _make_hook(self, shortname: str):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # h: (B, seq_len, C) — mean pool spatial tokens → (B, C)
            self.features[shortname] = h.detach().float().mean(dim=1).cpu()
        return hook_fn

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect(args):
    device = torch.device(f"cuda:{args.gpu_ids[0]}")
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    unet_merged = merge_unet(args)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(device, dtype=weight_dtype)

    unet = copy.deepcopy(unet_merged).to(device, dtype=weight_dtype)
    unet.eval().requires_grad_(False)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    ).to(device, dtype=weight_dtype)
    text_encoder.eval().requires_grad_(False)

    img_encoder = vqvae_encoder(args).to(device, dtype=weight_dtype)
    img_encoder.eval()

    embedding_change = TwoLayerConv1x1(512, 1024)
    embedding_change.load_state_dict(
        torch.load(os.path.join(args.ckpt_path, "embedding_change_weights.pth"),
                   weights_only=False)
    )
    embedding_change.to(device, dtype=weight_dtype).eval()

    conditioner = TextConditioner(unet, text_dim=1024)
    conditioner.register_hooks(unet)
    conditioner.load(args.conditioner_path, map_location="cpu")
    conditioner.to(device, dtype=weight_dtype).eval()

    # Register multi-block hooks
    print("Registering feature hooks:")
    hook = MultiBlockHook()
    hook.register(unet, TARGET_BLOCKS)

    # Text encoding helper
    def encode_text_batch(pos_list, na_list):
        def enc(texts):
            ids = tokenizer(
                texts, padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            hidden  = text_encoder(ids).last_hidden_state
            eos_pos = (ids == tokenizer.eos_token_id).float().argmax(dim=1)
            eos     = hidden[torch.arange(len(texts), device=device), eos_pos]
            return eos.unsqueeze(1).expand(-1, tokenizer.model_max_length, -1).to(weight_dtype)
        return (enc([strip_preamble(p) for p in pos_list]),
                enc([strip_preamble(p) for p in na_list]))

    # Dataset + loader
    dataset = FaceFeatureDataset(args.image_dir, args.prompts_json)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    # Dummy forward to determine feature dims per block
    print("\nDetermining feature dimensions...")
    dummy_batch = next(iter(loader))
    dummy_lq  = dummy_batch["lq"].to(device, dtype=weight_dtype)
    dummy_ve  = embedding_change(img_encoder(dummy_lq).reshape(dummy_lq.shape[0], 77, -1))
    dummy_lat = vae.encode(dummy_lq).latent_dist.sample() * vae.config.scaling_factor
    pos_emb, na_emb = encode_text_batch(dummy_batch["pos"], dummy_batch["na"])
    conditioner.set_text_embedding(pos_emb, na_emb, neg_weight=args.film_neg_weight)
    unet(dummy_lat, 399, encoder_hidden_states=dummy_ve)
    conditioner.clear_text_embedding()

    feat_dims = {name: hook.features[name].shape[-1] for name in TARGET_BLOCKS}
    hook.clear()
    for name, dim in feat_dims.items():
        print(f"  [{name}] feature dim = {dim}")

    # Pre-allocate arrays per block
    N = len(dataset)
    print(f"\nCollecting features from {N} images → {args.output_dir}/")

    block_feats = {
        name: np.zeros((N, dim), dtype=np.float32)
        for name, dim in feat_dims.items()
    }
    attrs_arr  = np.zeros((N, N_ATTRS), dtype=np.float32)
    filenames  = []

    idx = 0
    for batch in tqdm(loader, desc=f"Collecting [{args.split}]"):
        B   = batch["lq"].shape[0]
        lq  = batch["lq"].to(device, dtype=weight_dtype)

        pos_emb, na_emb = encode_text_batch(batch["pos"], batch["na"])
        ve  = embedding_change(img_encoder(lq).reshape(B, 77, -1))
        lat = vae.encode(lq).latent_dist.sample() * vae.config.scaling_factor

        hook.clear()
        conditioner.set_text_embedding(pos_emb, na_emb, neg_weight=args.film_neg_weight)
        unet(lat, 399, encoder_hidden_states=ve)   # single forward — both hooks fire
        conditioner.clear_text_embedding()

        for name in TARGET_BLOCKS:
            block_feats[name][idx:idx+B] = hook.features[name].numpy()

        attrs_arr[idx:idx+B] = batch["attrs"].numpy()
        filenames.extend(batch["filename"])
        idx += B

    # Save per block + shared attrs/filenames
    print(f"\nSaving outputs to {args.output_dir}/")
    for name in TARGET_BLOCKS:
        feat_path = os.path.join(args.output_dir, f"features_{name}_{args.split}.npy")
        np.save(feat_path, block_feats[name][:idx])
        print(f"  features [{name}] → {feat_path}  shape={block_feats[name][:idx].shape}")

    attr_path  = os.path.join(args.output_dir, f"attrs_{args.split}.npy")
    names_path = os.path.join(args.output_dir, f"filenames_{args.split}.json")
    np.save(attr_path, attrs_arr[:idx])
    with open(names_path, "w") as f:
        json.dump(filenames[:idx], f)
    print(f"  attrs      → {attr_path}  shape={attrs_arr[:idx].shape}")
    print(f"  filenames  → {names_path}  ({idx} entries)")

    # Attribute prevalence report
    print(f"\nAttribute prevalence in [{args.split}] set:")
    attr_means = attrs_arr[:idx].mean(axis=0)
    for i, name in enumerate(PAPER_28):
        bar = "█" * int(attr_means[i] * 20)
        print(f"  {name:<25} {attr_means[i]:.3f}  {bar}")

    hook.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",    required=True)
    parser.add_argument("--prompts_json", required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--split",        default="train", choices=["train","test"])
    parser.add_argument("--pretrained_model_name_or_path",
                        default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--img_encoder_weight", default="pretrained/associate_2.ckpt")
    parser.add_argument("--ckpt_path",        required=True)
    parser.add_argument("--conditioner_path", required=True)
    parser.add_argument("--film_neg_weight",  type=float, default=0.1)
    parser.add_argument("--batch_size",       type=int,   default=8)
    parser.add_argument("--mixed_precision",  choices=["fp16","fp32"], default="fp16")
    parser.add_argument("--gpu_ids",          nargs="+",  type=int, default=[0])
    parser.add_argument("--lora_rank",        type=int,   default=16)
    parser.add_argument("--lora_alpha",       type=float, default=16)
    parser.add_argument("--cat_prompt_embedding", action="store_true")
    parser.add_argument("--use_pos_embedding",    action="store_true")
    parser.add_argument("--use_att_pool",         action="store_true")
    parser.add_argument("--learnable_pos_emb",    action="store_true")
    args = parser.parse_args()
    collect(args)