"""
generate_heatmaps.py — FiLM modulation heatmaps for trained TextConditioner checkpoints

For each (checkpoint, probe_image, neg_weight), this script:
  1. Loads the trained text_conditioner.pth onto an SD2.1 UNet skeleton
  2. Encodes the probe's pos and na prompts using the SD2.1 CLIP text encoder
     (same EOS-pooling scheme as infer_textcond.py)
  3. Runs each per-block FiLM MLP forward to get (γ_pos, β_pos, γ_na, β_na)
  4. Computes the *net* modulation magnitudes:
        ||γ_net|| = || γ_pos - neg_weight * γ_na ||_2
        ||β_net|| = || β_pos - neg_weight * β_na ||_2
  5. Saves:
       - a CSV of per-block magnitudes
       - a heatmap PNG (blocks × neg_weights)
       - a JSON manifest with metadata

This is *static* analysis — no image inference, no diffusion, no VAE. It only
exercises the FiLM heads, so it runs in seconds on a tiny GPU (or CPU).
The UNet is loaded only to enumerate the BasicTransformerBlock names so the
TextConditioner constructs the same set of FiLM layers as during training.

USAGE
-----

Single checkpoint:
    python heatmaps/generate_heatmaps.py \\
        --checkpoint /projectnb/cs585/projects/craft/checkpoints/textcond_v5/checkpoint-50000/text_conditioner.pth \\
        --output_dir heatmaps/out/textcond_v5_step50000 \\
        --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \\
        --prompts_json /projectnb/cs585/projects/craft/prompts_output_final.json \\
        --probe_image 00020.png \\
        --neg_weights 0.0 0.5 1.0

Multiple checkpoints / runs from config:
    python heatmaps/generate_heatmaps.py --config heatmaps/checkpoints.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from transformers import CLIPTextModel, CLIPTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the project root importable so we can reuse text_conditioner.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from text_conditioner import TextConditioner  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_preamble(text: str) -> str:
    """Mirror infer_textcond.strip_preamble — drops the boilerplate prefix
    that prompts.json entries start with so the CLIP encoder sees only the
    descriptive attributes."""
    markers = [
        "in the description of",
        "not in the description of",
    ]
    for m in markers:
        idx = text.lower().find(m)
        if idx != -1:
            return text[idx + len(m):].strip(" .,")
    return text.strip()


@torch.no_grad()
def encode_eos(
    prompts: list[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """EOS-token pooled embeddings, broadcast to (B, 77, 1024) — matches
    infer_textcond._encode_text exactly."""
    input_ids = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    hidden = text_encoder(input_ids).last_hidden_state          # (B, 77, 1024)
    eos_pos = (input_ids == tokenizer.eos_token_id).float().argmax(dim=1)
    eos = hidden[torch.arange(hidden.shape[0], device=device), eos_pos]  # (B, 1024)
    eos = eos.unsqueeze(1).expand(-1, tokenizer.model_max_length, -1)
    return eos.to(dtype)


def load_prompts(prompts_json: str) -> dict:
    """Index prompts.json by image filename (with and without _LQ suffix)."""
    with open(prompts_json) as f:
        raw = json.load(f)
    out = {}
    for item in raw:
        entry = {"pos": item.get("pos", ""), "na": item.get("na", "")}
        name = item["image"]
        stem, ext = os.path.splitext(name)
        out[name] = entry
        out[f"{stem}_LQ{ext}"] = entry
        out[f"{stem}_lq{ext}"] = entry
    return out


def list_block_keys(unet) -> list[str]:
    """The same key scheme TextConditioner uses internally — name with dots → underscores,
    in the order BasicTransformerBlocks appear in unet.named_modules()."""
    keys = []
    for name, mod in unet.named_modules():
        if isinstance(mod, BasicTransformerBlock):
            keys.append(name.replace(".", "_"))
    return keys


# ---------------------------------------------------------------------------
# Core: compute FiLM magnitudes for one (checkpoint, probe, neg_weight grid)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_film_magnitudes(
    conditioner: TextConditioner,
    block_keys: list[str],
    pos_emb: torch.Tensor,
    na_emb: torch.Tensor,
    neg_weights: list[float],
) -> dict:
    """For each block, compute ||γ_net|| and ||β_net|| at every neg_weight.

    Returns:
      {
        "block_keys":  [str, ...],            # length N
        "neg_weights": [float, ...],          # length W
        "gamma_norm":  ndarray (W, N),
        "beta_norm":   ndarray (W, N),
        "gamma_pos_norm": ndarray (N,),       # for reference (independent of neg_weight)
        "gamma_na_norm":  ndarray (N,),
        "beta_pos_norm":  ndarray (N,),
        "beta_na_norm":   ndarray (N,),
      }
    """
    pos_pooled = pos_emb.mean(dim=1)   # (B=1, 1024)
    na_pooled  = na_emb.mean(dim=1)

    g_pos_list, b_pos_list = [], []
    g_na_list,  b_na_list  = [], []

    for key in block_keys:
        gp, bp = conditioner.film_pos_layers[key](pos_pooled)   # (1, 1, F)
        gn, bn = conditioner.film_na_layers[key](na_pooled)
        g_pos_list.append(gp.squeeze().float().cpu())
        b_pos_list.append(bp.squeeze().float().cpu())
        g_na_list.append(gn.squeeze().float().cpu())
        b_na_list.append(bn.squeeze().float().cpu())

    gamma_norm = np.zeros((len(neg_weights), len(block_keys)), dtype=np.float32)
    beta_norm  = np.zeros_like(gamma_norm)

    for wi, w in enumerate(neg_weights):
        for bi, key in enumerate(block_keys):
            g_net = g_pos_list[bi] - w * g_na_list[bi]
            b_net = b_pos_list[bi] - w * b_na_list[bi]
            gamma_norm[wi, bi] = torch.linalg.norm(g_net).item()
            beta_norm[wi, bi]  = torch.linalg.norm(b_net).item()

    return {
        "block_keys":     block_keys,
        "neg_weights":    list(neg_weights),
        "gamma_norm":     gamma_norm,
        "beta_norm":      beta_norm,
        "gamma_pos_norm": np.array([torch.linalg.norm(g).item() for g in g_pos_list]),
        "gamma_na_norm":  np.array([torch.linalg.norm(g).item() for g in g_na_list]),
        "beta_pos_norm":  np.array([torch.linalg.norm(b).item() for b in b_pos_list]),
        "beta_na_norm":   np.array([torch.linalg.norm(b).item() for b in b_na_list]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_heatmap(
    result: dict,
    out_path: Path,
    title: str,
):
    """Two side-by-side heatmaps: ||γ_net|| and ||β_net|| (blocks × neg_weights)."""
    block_keys  = result["block_keys"]
    neg_weights = result["neg_weights"]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, 0.35 * len(block_keys)), 4.5))

    for ax, key, label in [
        (axes[0], "gamma_norm", r"$\|\gamma_{net}\|$"),
        (axes[1], "beta_norm",  r"$\|\beta_{net}\|$"),
    ]:
        data = result[key]
        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(neg_weights)))
        ax.set_yticklabels([f"{w:g}" for w in neg_weights])
        ax.set_xticks(range(len(block_keys)))
        ax.set_xticklabels(block_keys, rotation=90, fontsize=6)
        ax.set_xlabel("BasicTransformerBlock")
        ax.set_ylabel("neg_weight")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_csv(result: dict, out_path: Path):
    block_keys  = result["block_keys"]
    neg_weights = result["neg_weights"]
    lines = ["metric,neg_weight," + ",".join(block_keys)]
    for wi, w in enumerate(neg_weights):
        lines.append("gamma_net_norm," + f"{w:g}," + ",".join(f"{x:.6f}" for x in result["gamma_norm"][wi]))
        lines.append("beta_net_norm,"  + f"{w:g}," + ",".join(f"{x:.6f}" for x in result["beta_norm"][wi]))
    lines.append("gamma_pos_norm,-," + ",".join(f"{x:.6f}" for x in result["gamma_pos_norm"]))
    lines.append("gamma_na_norm,-,"  + ",".join(f"{x:.6f}" for x in result["gamma_na_norm"]))
    lines.append("beta_pos_norm,-,"  + ",".join(f"{x:.6f}" for x in result["beta_pos_norm"]))
    lines.append("beta_na_norm,-,"   + ",".join(f"{x:.6f}" for x in result["beta_na_norm"]))
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Per-checkpoint driver
# ---------------------------------------------------------------------------

def run_one_checkpoint(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    unet,
    tokenizer,
    text_encoder,
    prompts: dict,
    probe_images: list[str],
    neg_weights: list[float],
    device: torch.device,
    dtype: torch.dtype,
    label: str,
):
    """Generate heatmaps for a single text_conditioner.pth across all probes."""
    print(f"\n>>> {label}")
    print(f"    checkpoint: {checkpoint_path}")
    print(f"    output:     {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build TextConditioner against the UNet skeleton, load weights
    conditioner = TextConditioner(unet, text_dim=1024).to(device).eval()
    conditioner.load(str(checkpoint_path), map_location=device)

    block_keys = list_block_keys(unet)

    manifest = {
        "checkpoint": str(checkpoint_path),
        "label":      label,
        "n_blocks":   len(block_keys),
        "neg_weights": list(neg_weights),
        "probes":     {},
    }

    for probe in probe_images:
        if probe not in prompts:
            print(f"    [skip] '{probe}' not in prompts.json")
            continue

        entry  = prompts[probe]
        pos_em = encode_eos([strip_preamble(entry["pos"])], tokenizer, text_encoder, device, dtype)
        na_em  = encode_eos([strip_preamble(entry["na"])],  tokenizer, text_encoder, device, dtype)

        result = compute_film_magnitudes(conditioner, block_keys, pos_em, na_em, neg_weights)

        stem = Path(probe).stem
        png  = output_dir / f"heatmap__{stem}.png"
        csv  = output_dir / f"heatmap__{stem}.csv"
        save_heatmap(result, png, title=f"{label}  |  probe={probe}")
        save_csv(result, csv)

        manifest["probes"][probe] = {
            "png": png.name,
            "csv": csv.name,
            "gamma_norm_max": float(result["gamma_norm"].max()),
            "beta_norm_max":  float(result["beta_norm"].max()),
        }
        print(f"    [ok]   {probe} → {png.name}")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Top-level: CLI + config dispatch
# ---------------------------------------------------------------------------

def build_unet_and_text_encoder(pretrained_path: str, device: torch.device, dtype: torch.dtype):
    """Loaded once and reused across all checkpoints — the UNet is only used
    as a structural skeleton for TextConditioner, so weights don't matter for
    correctness, but using the real SD2.1 weights guarantees the block names
    line up with whatever was used at training time."""
    print(f"Loading SD2.1 UNet + text encoder from {pretrained_path} ...")
    unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet").to(device, dtype=dtype).eval()
    unet.requires_grad_(False)
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder").to(device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)
    return unet, tokenizer, text_encoder


def parse_args():
    p = argparse.ArgumentParser(description="FiLM modulation heatmaps for TextConditioner checkpoints.")
    # Single-checkpoint mode
    p.add_argument("--checkpoint", type=str, default=None, help="Path to a single text_conditioner.pth")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--probe_image", type=str, action="append", default=None, help="Repeat to add multiple probes")
    p.add_argument("--neg_weights", type=float, nargs="+", default=None)

    # Config-driven mode
    p.add_argument("--config", type=str, default=None, help="Path to checkpoints.yaml")

    # Shared
    p.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    p.add_argument("--prompts_json", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",  type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    return p.parse_args()


DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    dtype  = DTYPE_MAP[args.dtype] if device.type == "cuda" else torch.float32

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        defaults = cfg.get("defaults", {})
        runs     = cfg.get("runs", [])

        pretrained = args.pretrained_model_name_or_path or defaults["pretrained_model_name_or_path"]
        prompts    = load_prompts(args.prompts_json or defaults["prompts_json"])
        unet, tok, txt_enc = build_unet_and_text_encoder(pretrained, device, dtype)

        out_root = Path(defaults.get("output_dir", "heatmaps/out"))

        for run in runs:
            run_name  = run["name"]
            ckpt_dir  = Path(run["checkpoint_dir"])
            steps     = run["steps"]
            probes    = run.get("probe_images", defaults["probe_images"])
            negs      = run.get("neg_weights",  defaults["neg_weights"])

            for step in steps:
                ckpt = ckpt_dir / f"checkpoint-{step}" / "text_conditioner.pth"
                if not ckpt.exists():
                    print(f"[warn] missing checkpoint: {ckpt}")
                    continue
                run_one_checkpoint(
                    checkpoint_path=ckpt,
                    output_dir=out_root / run_name / f"step-{step}",
                    unet=unet, tokenizer=tok, text_encoder=txt_enc,
                    prompts=prompts,
                    probe_images=probes, neg_weights=negs,
                    device=device, dtype=dtype,
                    label=f"{run_name} @ step {step}",
                )

    elif args.checkpoint:
        if not args.pretrained_model_name_or_path or not args.prompts_json:
            sys.exit("--pretrained_model_name_or_path and --prompts_json are required in single-checkpoint mode")
        if not args.output_dir:
            sys.exit("--output_dir is required in single-checkpoint mode")

        unet, tok, txt_enc = build_unet_and_text_encoder(args.pretrained_model_name_or_path, device, dtype)
        prompts = load_prompts(args.prompts_json)
        run_one_checkpoint(
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.output_dir),
            unet=unet, tokenizer=tok, text_encoder=txt_enc,
            prompts=prompts,
            probe_images=args.probe_image or ["00020.png"],
            neg_weights=args.neg_weights or [0.0, 0.5, 1.0],
            device=device, dtype=dtype,
            label=Path(args.checkpoint).parent.name,
        )

    else:
        sys.exit("Pass either --config <yaml> or --checkpoint <path>")

    print("\nDone.")


if __name__ == "__main__":
    main()
