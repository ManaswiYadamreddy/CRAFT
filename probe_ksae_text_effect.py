import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from revelio_probe_common import extract_features


class KSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, k: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.k = k

    def topk_masked(self, z: torch.Tensor) -> torch.Tensor:
        # Retain only top-k activations per sample.
        vals, idx = torch.topk(z, k=min(self.k, z.shape[1]), dim=1)
        masked = torch.zeros_like(z)
        masked.scatter_(1, idx, vals)
        return masked

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_k = self.topk_masked(z)
        x_hat = self.decoder(z_k)
        return x_hat, z_k


def train_ksae(
    x_np: np.ndarray,
    hidden_dim: int,
    k: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Tuple[KSparseAutoencoder, List[float], np.ndarray]:
    x = torch.from_numpy(x_np).to(device)
    model = KSparseAutoencoder(input_dim=x.shape[1], hidden_dim=hidden_dim, k=k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses: List[float] = []

    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        running = 0.0
        steps = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x[idx]
            x_hat, _ = model(xb)
            loss = F.mse_loss(x_hat, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            steps += 1
        losses.append(running / max(steps, 1))

    model.eval()
    with torch.no_grad():
        _, z_all = model(x)
    return model, losses, z_all.detach().cpu().numpy()


def neuron_label_purity(z: np.ndarray, labels: np.ndarray, top_m: int) -> Dict[str, float]:
    n_neurons = z.shape[1]
    stds = []
    dominant_frac = []
    for j in range(n_neurons):
        idx = np.argsort(-z[:, j])[:top_m]
        top_labels = labels[idx]
        stds.append(float(np.std(top_labels)))
        vals, counts = np.unique(top_labels, return_counts=True)
        dominant_frac.append(float(counts.max() / counts.sum()))
    return {
        "sigma_label_mean": float(np.mean(stds)),
        "dominant_label_fraction_mean": float(np.mean(dominant_frac)),
    }


def top_neurons_report(
    z: np.ndarray,
    filenames: List[str],
    top_neurons: int,
    top_m: int,
) -> List[Dict]:
    mean_act = z.mean(axis=0)
    top_ids = np.argsort(-mean_act)[:top_neurons]
    report = []
    for j in top_ids:
        idx = np.argsort(-z[:, j])[:top_m]
        report.append(
            {
                "neuron": int(j),
                "mean_activation": float(mean_act[j]),
                "top_images": [filenames[i] for i in idx],
            }
        )
    return report


def parse_args():
    p = argparse.ArgumentParser(description="k-SAE probe for text-conditioning representation effects.")
    p.add_argument("--input_image", required=True, help="LQ image file or directory")
    p.add_argument("--prompts_json", default=None)
    p.add_argument("--labels_json", default=None, help="Optional image->label mapping for purity.")
    p.add_argument("--output_json", required=True)
    p.add_argument("--mode", choices=["no_text", "text_real", "text_shuffled"], required=True)
    p.add_argument("--capture_layer", required=True, help="Exact UNet layer name to hook.")
    p.add_argument("--list_layers", action="store_true")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--timestep", type=int, default=399)

    p.add_argument("--hidden_dim", type=int, default=4096)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--top_m", type=int, default=10)
    p.add_argument("--top_neurons", type=int, default=10)

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

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    _, losses, z = train_ksae(
        x_np=extracted.features,
        hidden_dim=args.hidden_dim,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    summary = {
        "mode": args.mode,
        "capture_layer": extracted.layer_name,
        "n_images": int(extracted.features.shape[0]),
        "feature_dim": int(extracted.features.shape[1]),
        "ksae": {
            "hidden_dim": int(args.hidden_dim),
            "k": int(args.k),
            "epochs": int(args.epochs),
            "loss_first": float(losses[0]),
            "loss_last": float(losses[-1]),
            "loss_curve": losses,
        },
        "top_neurons": top_neurons_report(
            z=z,
            filenames=extracted.filenames,
            top_neurons=args.top_neurons,
            top_m=args.top_m,
        ),
    }

    if extracted.labels is not None:
        summary["purity"] = neuron_label_purity(z=z, labels=extracted.labels, top_m=args.top_m)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved k-SAE probe results -> {args.output_json}")


if __name__ == "__main__":
    main()

