import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from revelio_probe_common import extract_features


class LightProbe(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return train_idx, val_idx


def run_probe(
    x_np: np.ndarray,
    y_np: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    train_idx, val_idx = split_indices(n=len(x_np), val_ratio=val_ratio, seed=seed)
    train_idx_t = torch.from_numpy(train_idx).to(device)
    val_idx_t = torch.from_numpy(val_idx).to(device)

    n_classes = int(y_np.max()) + 1
    model = LightProbe(input_dim=x.shape[1], n_classes=n_classes, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = 0.0
    for _ in range(epochs):
        model.train()
        perm = train_idx_t[torch.randperm(train_idx_t.shape[0], device=device)]
        for i in range(0, perm.shape[0], batch_size):
            b = perm[i : i + batch_size]
            logits = model(x[b])
            loss = F.cross_entropy(logits, y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x[val_idx_t])
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y[val_idx_t]).float().mean().item()
            best_val = max(best_val, val_acc)

    model.eval()
    with torch.no_grad():
        train_logits = model(x[train_idx_t])
        train_acc = (train_logits.argmax(dim=1) == y[train_idx_t]).float().mean().item()
        val_logits = model(x[val_idx_t])
        val_acc = (val_logits.argmax(dim=1) == y[val_idx_t]).float().mean().item()

    return {
        "train_acc": float(train_acc),
        "val_acc_last": float(val_acc),
        "val_acc_best": float(best_val),
        "n_train": int(train_idx_t.shape[0]),
        "n_val": int(val_idx_t.shape[0]),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Diff-C style lightweight probe for text-conditioned features.")
    p.add_argument("--input_image", required=True, help="LQ image file or directory")
    p.add_argument("--prompts_json", default=None)
    p.add_argument("--labels_json", required=True, help="Required image->label mapping")
    p.add_argument("--output_json", required=True)
    p.add_argument("--mode", choices=["no_text", "text_real", "text_shuffled"], required=True)
    p.add_argument("--capture_layer", required=True, help="Exact UNet layer name to hook.")
    p.add_argument("--list_layers", action="store_true")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--timestep", type=int, default=399)

    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.2)

    # Model/config args reused from inference.
    p.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-2-1-base")
    p.add_argument("--img_encoder_weight", default="pretrained/associate_2.ckpt")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--conditioner_path", default=None)
    p.add_argument("--film_neg_weight", type=float, default=0.5)
    p.add_argument("--mixed_precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=16)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # passthrough flags expected by lq_embed/vqvae encoder
    p.add_argument("--cat_prompt_embedding", action="store_true")
    p.add_argument("--use_pos_embedding", action="store_true")
    p.add_argument("--use_att_pool", action="store_true")
    p.add_argument("--learnable_pos_emb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    extracted = extract_features(args)
    if args.list_layers:
        return
    if extracted.features.shape[0] == 0:
        raise RuntimeError("No features extracted.")
    if extracted.labels is None:
        raise RuntimeError("labels_json is required for the downstream probe.")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    metrics = run_probe(
        x_np=extracted.features,
        y_np=extracted.labels,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=device,
    )

    out = {
        "mode": args.mode,
        "capture_layer": extracted.layer_name,
        "n_images": int(extracted.features.shape[0]),
        "feature_dim": int(extracted.features.shape[1]),
        "probe": {
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "val_ratio": float(args.val_ratio),
            **metrics,
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved probe results -> {args.output_json}")


if __name__ == "__main__":
    main()

