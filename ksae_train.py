"""
ksae_train.py — Stage 2: Train k-sparse autoencoder on collected features

Trains a k-SAE on FiLM-modulated features from up_blocks_1_attentions_1.
Follows Revelio exactly:
  - k=32 active neurons per forward pass (TopK activation)
  - expansion factor=64 → n_latents = feat_dim * 64
  - Unit normalization on decoder weights after each update
  - Adam optimizer, lr=4e-4, 500-step warmup, 10M steps total
  - Pre-encoder bias (Bricken et al. 2023)

At 70k images, 10M steps ≈ ~143 epochs. Revelio trains for 1hr on A6000.
You can stop earlier — the SAE converges fast in the first few epochs.

Usage:
    python ksae_train.py \
        --features_path features/ksae_up1a1/features_train.npy \
        --output_dir checkpoints/ksae_up1a1 \
        --k 32 \
        --expansion_factor 64 \
        --lr 4e-4 \
        --max_steps 2000000 \
        --batch_size 2048 \
        --save_every 100000 \
        --gpu_ids 0

Notes:
  - 2M steps at batch=2048 on 70k features ≈ ~58 epochs, ~30-60min on A6000
  - For a quick check, use --max_steps 200000
  - The k-SAE is pure PyTorch, no diffusion models needed for this stage
"""

import os
import argparse
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# k-SAE model
# ---------------------------------------------------------------------------

class kSAE(nn.Module):
    """
    k-sparse autoencoder following Makhzani & Frey (2013) and Revelio.

    Architecture:
        encoder: Linear(d, n) + TopK
        decoder: Linear(n, d)

    Zero-initialized pre-encoder bias subtracted before encoding
    and added back after decoding (Bricken et al. 2023).

    Unit normalization on decoder columns is enforced after each optimizer step.
    """

    def __init__(self, d: int, n: int, k: int):
        super().__init__()
        self.d = d  # input feature dim
        self.n = n  # SAE hidden dim (d * expansion_factor)
        self.k = k  # number of active neurons

        self.W_enc  = nn.Linear(d, n, bias=True)
        self.W_dec  = nn.Linear(n, d, bias=False)
        self.b_pre  = nn.Parameter(torch.zeros(d))   # pre-encoder bias

        # Initialize encoder weights with decoder transpose (tied init)
        nn.init.kaiming_uniform_(self.W_enc.weight, a=math.sqrt(5))
        self.W_dec.weight.data = self.W_enc.weight.data.T.clone()

        # Unit normalize decoder columns at init
        self._normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d) → z: (B, n) sparse"""
        x_centered = x - self.b_pre
        pre_acts   = self.W_enc(x_centered)    # (B, n)
        # TopK: keep k largest activations, zero the rest
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        z = torch.zeros_like(pre_acts)
        z.scatter_(1, topk_idx, F.relu(topk_vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, n) → x_hat: (B, d)"""
        return self.W_dec(z) + self.b_pre

    def forward(self, x: torch.Tensor):
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def _normalize_decoder(self):
        """Unit-normalize each decoder column (Revelio / Sharkey et al.)"""
        norms = self.W_dec.weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.weight.data = self.W_dec.weight.data / norms

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Normalized MSE reconstruction loss."""
        return F.mse_loss(x_hat, x)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class FeatureDataset(Dataset):
    def __init__(self, features_path: str):
        print(f"Loading features from {features_path}...")
        self.features = torch.from_numpy(np.load(features_path)).float()
        print(f"  Shape: {self.features.shape}  dtype: {self.features.dtype}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load features
    dataset = FeatureDataset(args.features_path)
    d       = dataset.features.shape[1]
    n       = d * args.expansion_factor

    print(f"\nk-SAE config:")
    print(f"  Input dim d        = {d}")
    print(f"  Hidden dim n       = {n}  (expansion={args.expansion_factor}x)")
    print(f"  Active neurons k   = {args.k}")
    print(f"  Batch size         = {args.batch_size}")
    print(f"  Max steps          = {args.max_steps:,}")
    print(f"  Device             = {device}\n")

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=True)

    model = kSAE(d=d, n=n, k=args.k).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"k-SAE parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Cosine warmup schedule
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        # Cosine decay after warmup
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    data_iter   = iter(loader)

    # Resume if checkpoint exists
    ckpt_path = os.path.join(args.output_dir, "ksae_latest.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["step"]
        print(f"Resumed from step {global_step:,}")

    pbar = tqdm(total=args.max_steps, initial=global_step, desc="k-SAE training")
    running_loss = 0.0
    log_every    = 1000

    while global_step < args.max_steps:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)

        x = x.to(device)

        x_hat, z = model(x)
        loss      = model.loss(x, x_hat)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Unit normalize decoder after every step
        model._normalize_decoder()

        running_loss += loss.item()
        global_step  += 1
        pbar.update(1)

        if global_step % log_every == 0:
            avg_loss   = running_loss / log_every
            sparsity   = (z == 0).float().mean().item()
            dead_frac  = (z.sum(dim=0) == 0).float().mean().item()
            pbar.set_postfix({
                "loss":    f"{avg_loss:.5f}",
                "sparse":  f"{sparsity:.3f}",
                "dead":    f"{dead_frac:.3f}",
                "lr":      f"{scheduler.get_last_lr()[0]:.2e}",
            })
            running_loss = 0.0

        if global_step % args.save_every == 0:
            _save(model, optimizer, scheduler, global_step, args)

    pbar.close()
    _save(model, optimizer, scheduler, global_step, args)
    print(f"\nTraining complete. Model saved to {args.output_dir}")


def _save(model, optimizer, scheduler, step, args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Latest (overwritten)
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step":      step,
        "config": {
            "d": model.d, "n": model.n, "k": model.k,
        }
    }, os.path.join(args.output_dir, "ksae_latest.pt"))

    # Checkpoint snapshot
    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "ksae_model.pt"))
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump({"d": model.d, "n": model.n, "k": model.k, "step": step}, f, indent=2)
    print(f"  Saved checkpoint → {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path",   required=True, help="Path to features_train.npy")
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--k",               type=int,   default=32,    help="TopK active neurons")
    parser.add_argument("--expansion_factor",type=int,   default=64,    help="n = d * expansion_factor")
    parser.add_argument("--lr",              type=float, default=4e-4)
    parser.add_argument("--warmup_steps",    type=int,   default=500)
    parser.add_argument("--max_steps",       type=int,   default=2000000)
    parser.add_argument("--batch_size",      type=int,   default=2048)
    parser.add_argument("--save_every",      type=int,   default=100000)
    parser.add_argument("--gpu_ids",         nargs="+",  type=int, default=[0])
    args = parser.parse_args()
    train(args)