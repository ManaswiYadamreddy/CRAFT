"""
region_aware_vq.py — Region-Aware Vector Quantization for CRAFT Stage 1.

Orchestrates the face parser and per-region ResidualVQ instances to produce 
a region-aware visual prompt p_L from encoder features.

Pipeline (during training forward pass):
    1. Encoder produces feature map z: (B, 512, 16, 16)
    2. Face parser segments the input image into 4 regions at 16×16
    3. For each region, gather feature vectors at masked positions → (N_r, 512)
    4. Quantize with region-specific ResidualVQ → (N_r, 512)
    5. Scatter quantized vectors back to original spatial positions
    6. Output: quantized feature map z_q: (B, 512, 16, 16)

The output z_q can be reshaped to (B, 256, 512) to serve as the visual 
prompt p_L for Stage 2's UNet cross-attention.

Dependencies:
    - residual_vq.py (ResidualVQ, build_region_rq_vae)
    - face_parser.py (FaceParser, REGION_NAMES)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_vq import ResidualVQ
from .face_parser import FaceParser, REGION_NAMES


class RegionAwareVQ(nn.Module):
    """
    Region-Aware Vector Quantization module.
    
    Segments the encoder feature map into facial regions using a frozen 
    BiSeNet face parser, then quantizes each region with its own dedicated 
    3-level ResidualVQ codebook.
    
    Codebook sizes are proportional to the number of spatial positions each
    region typically receives (balances expressiveness vs. collapse risk):
        hair: 512 codes  (80-120 positions/image, high texture variety)
        skin: 256 codes  (60-100 positions/image, low texture variety)
        bg:   256 codes  (40-80 positions/image, background/cloth/jewellery —
                         split out from skin so the skin codebook is not
                         contaminated with non-face texture)
        eyes: 128 codes  (10-20 positions/image, high variety but few positions)
        lips:  64 codes  (5-10 positions/image, collapse-prone with larger codebooks)
    
    Args:
        region_n_codes:   Dict mapping region name → codebook size per level.
                          If None, uses the defaults above.
        e_dim:            Embedding dimension (default: 512, matches encoder).
        n_levels:         Residual quantization levels (default: 3).
        beta:             Commitment cost weight (default: 0.25).
        ema_decay:        EMA decay for codebook updates (default: 0.99).
        entropy_weight:   Entropy regularization weight (default: 0.1).
        parser_ckpt:      Path to BiSeNet checkpoint (79999_iter.pth).
                          None = random weights (for testing only).
    """

    # Default per-region codebook sizes (Option B: proportional to position count)
    DEFAULT_REGION_N_CODES = {
        "eyes": 128,
        "skin": 256,
        "hair": 512,
        "lips": 64,
        "bg":   256,
    }

    # Default per-region EMA expiry thresholds.
    # Steady-state ema_count for a uniformly-used code equals positions/batch / n_codes,
    # so small-region codebooks need lower thresholds to avoid flagging healthy codes:
    #   eyes: ~3 pos/img * 32 / 128 codes ≈ 0.75  → threshold 0.3
    #   lips: ~3 pos/img * 32 /  64 codes ≈ 1.5   → threshold 0.3
    # Large regions (skin/hair/bg) are comfortable at 1.0.
    DEFAULT_EXPIRE_THRESHOLDS = {
        "eyes": 0.3,
        "skin": 1.0,
        "hair": 1.0,
        "lips": 0.3,
        "bg":   1.0,
    }

    def __init__(
        self,
        region_n_codes=None,
        e_dim=512,
        n_levels=3,
        beta=0.25,
        ema_decay=0.99,
        entropy_weight=0.1,
        parser_ckpt=None,
        use_magnitude_head=False,
        magnitude_init=1.0,
        expire_thresholds=None,
    ):
        super().__init__()
        self.e_dim = e_dim
        self.n_regions = len(REGION_NAMES)

        if region_n_codes is None:
            region_n_codes = self.DEFAULT_REGION_N_CODES
        self.region_n_codes = region_n_codes

        if expire_thresholds is None:
            expire_thresholds = dict(self.DEFAULT_EXPIRE_THRESHOLDS)
        self.expire_thresholds = expire_thresholds

        # --- Face parser (frozen, no gradients) ---
        self.face_parser = FaceParser(checkpoint_path=parser_ckpt)

        # --- Per-region ResidualVQ codebooks (trainable) ---
        self.region_codebooks = nn.ModuleDict(
            {
                name: ResidualVQ(
                    n_codes=region_n_codes[name],
                    e_dim=e_dim,
                    n_levels=n_levels,
                    beta=beta,
                    ema_decay=ema_decay,
                    entropy_weight=entropy_weight,
                )
                for name in REGION_NAMES
            }
        )

        # --- Per-position magnitude head (optional) ---
        # Each region's ResidualVQ collapses spatial magnitude to a single
        # scalar (`codebook_scale`). The magnitude head predicts a positive
        # multiplicative factor per spatial position from the pre-quantized
        # feature map, restoring the magnitude information the quantizer
        # discards. Init is softplus(bias)=magnitude_init with zero weights,
        # so the layer acts as identity at step 0 and a checkpoint loaded
        # with strict=False behaves identically to the old quantizer.
        self.use_magnitude_head = use_magnitude_head
        if use_magnitude_head:
            self.magnitude_head = nn.Conv2d(e_dim, 1, kernel_size=1)
            nn.init.zeros_(self.magnitude_head.weight)
            # softplus(b) = m  =>  b = log(exp(m) - 1)
            nn.init.constant_(
                self.magnitude_head.bias,
                math.log(math.exp(float(magnitude_init)) - 1.0),
            )

    def forward(self, z, images, masks=None):
        """
        Region-aware quantization of encoder features.
        
        Args:
            z:      (B, C, H, W) encoder feature map (typically B, 512, 16, 16).
            images: (B, 3, H_img, W_img) input images in [0, 1] range,
                    used by the face parser to compute region masks.
                    Ignored if masks is provided.
            masks:  Optional precomputed dict of region masks from 
                    face_parser.get_region_masks(). If None, computed from images.
        
        Returns:
            z_q:        (B, C, H, W) quantized feature map, same shape as input.
                        Has straight-through gradient to z.
            all_losses: dict with aggregated VQ losses across all regions:
                        'commitment', 'entropy', 'total_vq'
            all_info:   dict with per-region, per-level monitoring metrics.
        """
        B, C, H, W = z.shape
        assert C == self.e_dim, f"Expected e_dim={self.e_dim}, got C={C}"

        # --- Step 1: Get region masks ---
        if masks is None:
            masks = self.face_parser.get_region_masks(images, target_h=H, target_w=W)
        # masks[name] is (B, H, W) bool

        # --- Step 2: Flatten spatial dimensions ---
        # (B, C, H, W) → (B, H*W, C)
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # Prepare output buffer
        z_q_flat = torch.zeros_like(z_flat)

        # Flatten masks: (B, H, W) → (B, H*W)
        masks_flat = {
            name: mask.reshape(B, H * W) for name, mask in masks.items()
        }

        # --- Step 3: Per-region quantization ---
        all_losses = {"commitment": 0.0, "entropy": 0.0, "total_vq": 0.0}
        all_info = {}

        for name in REGION_NAMES:
            mask = masks_flat[name]  # (B, H*W) bool
            rq = self.region_codebooks[name]

            # Gather: collect all feature vectors belonging to this region
            # across the entire batch into a single (N_r, C) tensor
            region_features = z_flat[mask]  # (N_r, C)

            if region_features.shape[0] == 0:
                # No positions for this region in this batch (rare but possible)
                all_info[f"{name}/n_positions"] = 0
                continue

            # Quantize
            z_q_region, losses, info = rq(region_features)

            # Scatter: place quantized vectors back into output
            z_q_flat[mask] = z_q_region.to(z_q_flat.dtype)

            # Aggregate losses (weighted equally across regions)
            for key in ("commitment", "entropy", "total_vq"):
                all_losses[key] = all_losses[key] + losses[key]

            # Store per-region info
            all_info[f"{name}/n_positions"] = region_features.shape[0]
            for key, val in info.items():
                if key != "all_indices":
                    all_info[f"{name}/{key}"] = val

        # --- Step 4: Reshape back to spatial ---
        # (B, H*W, C) → (B, C, H, W)
        z_q = z_q_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # --- Step 5: Per-position magnitude scaling (optional) ---
        # Restores per-position magnitude info that the VQ scalar discards.
        if self.use_magnitude_head:
            mag = F.softplus(self.magnitude_head(z))  # (B, 1, H, W)
            z_q = z_q * mag
            mag_d = mag.detach().float()
            all_info["mag/mean"] = mag_d.mean()
            all_info["mag/std"] = mag_d.std()
            all_info["mag/min"] = mag_d.min()
            all_info["mag/max"] = mag_d.max()

        return z_q, all_losses, all_info

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def to_prompt(self, z_q):
        """
        Reshape quantized feature map to visual prompt format for Stage 2.
        
        Args:
            z_q: (B, C, H, W) quantized feature map.
        
        Returns:
            p_L: (B, H*W, C) visual prompt embedding, ready for UNet 
                 cross-attention (K, V).
        """
        B, C, H, W = z_q.shape
        return z_q.permute(0, 2, 3, 1).reshape(B, H * W, C)

    @torch.no_grad()
    def expire_dead_codes(self, z, images, masks=None):
        """
        Replace dead codebook entries across all regions.
        Call periodically during training (e.g., every 1000 steps).
        
        Args:
            z:      (B, C, H, W) encoder features.
            images: (B, 3, H_img, W_img) input images for face parsing.
            masks:  Optional precomputed masks.
        
        Returns:
            expired: dict mapping region name → number of expired codes.
        """
        B, C, H, W = z.shape
        if masks is None:
            masks = self.face_parser.get_region_masks(images, target_h=H, target_w=W)

        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C)
        masks_flat = {
            name: mask.reshape(B, H * W) for name, mask in masks.items()
        }

        expired = {}
        for name in REGION_NAMES:
            mask = masks_flat[name]
            region_features = z_flat[mask]
            if region_features.shape[0] > 0:
                threshold = self.expire_thresholds.get(name, 1.0)
                expired[name] = self.region_codebooks[name].expire_dead_codes(
                    region_features, threshold=threshold,
                )
            else:
                expired[name] = 0

        return expired

    @torch.no_grad()
    def get_detailed_stats(self, z=None, masks=None, images=None):
        """
        Per-region, per-level diagnostic dump.

        For each region × RQ level, report:
            - codebook norm (mean, std)
            - dead-code fraction (EMA count < 1.0)
            - EMA count (min / max / mean)
            - learnable codebook_scale
            - n_positions assigned to this region this batch (if z+masks given)

        If z and masks (or images) are supplied, also returns the encoder
        feature norm (z_norm) so you can compare it against codebook norm.
        """
        stats = {}

        if z is not None:
            B, C, H, W = z.shape
            if masks is None and images is not None:
                masks = self.face_parser.get_region_masks(
                    images, target_h=H, target_w=W
                )
            stats["z_norm_mean"] = z.float().norm(dim=1).mean().item()
            stats["z_norm_std"] = z.float().norm(dim=1).std().item()

        for name in REGION_NAMES:
            rq = self.region_codebooks[name]
            scale = rq.codebook_scale.abs().item()
            stats[f"{name}/codebook_scale"] = scale

            if z is not None and masks is not None:
                mask = masks[name].reshape(B, H * W)
                n_pos = mask.sum().item()
                stats[f"{name}/n_positions"] = n_pos
                stats[f"{name}/n_positions_per_img"] = n_pos / max(B, 1)

            for lvl, vq in enumerate(rq.levels):
                w = vq.embedding.weight
                norms = w.norm(dim=-1)
                ema = vq.ema_count
                stats[f"{name}/L{lvl}/norm_mean"] = norms.mean().item()
                stats[f"{name}/L{lvl}/norm_std"] = norms.std().item()
                stats[f"{name}/L{lvl}/dead_frac"] = (
                    (ema < 1.0).float().mean().item()
                )
                stats[f"{name}/L{lvl}/ema_min"] = ema.min().item()
                stats[f"{name}/L{lvl}/ema_max"] = ema.max().item()
                stats[f"{name}/L{lvl}/ema_mean"] = ema.mean().item()

        return stats

    @torch.no_grad()
    def print_detailed_stats(self, z=None, masks=None, images=None, prefix=""):
        """Pretty-print get_detailed_stats() output."""
        stats = self.get_detailed_stats(z=z, masks=masks, images=images)
        print(f"{prefix}--- RegionAwareVQ diagnostics ---")
        if "z_norm_mean" in stats:
            print(f"{prefix}z_norm: mean={stats['z_norm_mean']:.2f} "
                  f"std={stats['z_norm_std']:.2f}")
        for name in REGION_NAMES:
            scale = stats.get(f"{name}/codebook_scale", float('nan'))
            n_pos = stats.get(f"{name}/n_positions", None)
            n_per = stats.get(f"{name}/n_positions_per_img", None)
            pos_str = (f" n_pos={n_pos} ({n_per:.1f}/img)"
                       if n_pos is not None else "")
            print(f"{prefix}[{name}] scale={scale:.3f}{pos_str}")
            for lvl in range(len(self.region_codebooks[name].levels)):
                nm = stats[f"{name}/L{lvl}/norm_mean"]
                ns = stats[f"{name}/L{lvl}/norm_std"]
                df = stats[f"{name}/L{lvl}/dead_frac"]
                emi = stats[f"{name}/L{lvl}/ema_min"]
                ema = stats[f"{name}/L{lvl}/ema_mean"]
                print(f"{prefix}  L{lvl}: "
                      f"norm={nm:.3f}±{ns:.3f}  "
                      f"dead={df*100:.0f}%  "
                      f"ema[min/mean]={emi:.2f}/{ema:.2f}")
        print(f"{prefix}" + "-" * 34)

    @torch.no_grad()
    def get_codebook_stats(self):
        """
        Collect codebook statistics across all regions for logging.
        
        Returns:
            stats: dict with per-region, per-level stats.
        """
        stats = {}
        for name in REGION_NAMES:
            region_stats = self.region_codebooks[name].get_codebook_stats()
            for key, val in region_stats.items():
                stats[f"{name}/{key}"] = val
        return stats