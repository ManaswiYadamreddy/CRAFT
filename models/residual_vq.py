"""
residual_vq.py — Residual Vector Quantizer (RQ-VAE) with HQ-VAE collapse prevention.

This module implements the core quantization for CRAFT's hierarchical 
region-aware VQ. Each facial region (eyes, skin, hair, lips) gets its own
ResidualVQ instance with 3 codebook levels.

Architecture per region:
    Level 1 (coarse):  quantizes the encoder feature  →  captures broad category
    Level 2 (mid):     quantizes the residual          →  refines structure  
    Level 3 (fine):    quantizes the remaining residual →  recovers texture detail

    Final output = level1_q + level2_q + level3_q

Collapse prevention (HQ-VAE, Takida et al. 2024):
    - Codebook collapse: entropy regularization on average code usage encourages 
      all codes to be selected, preventing dead codes.
    - Layer collapse: per-level entropy regularization ensures later residual 
      levels actively contribute rather than learning identity.
    - EMA codebook updates for stable training.
    - Data-driven codebook initialization from first batch.

References:
    - RQ-VAE: Lee et al., "Autoregressive Image Generation Using Residual 
      Quantization", CVPR 2022
    - HQ-VAE: Takida et al., "HQ-VAE: Hierarchical Discrete Representation 
      Learning with Variational Bayes", 2024
    - VQ-VAE: van den Oord et al., "Neural Discrete Representation Learning",
      NeurIPS 2017
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQLevel(nn.Module):
    """
    Single vector quantization level with EMA codebook updates and 
    entropy regularization for collapse prevention.
    
    During training:
        1. Compute L2 distances from input features to all codebook entries.
        2. Assign each feature to its nearest code (hard assignment).
        3. Apply straight-through estimator for gradient flow.
        4. Update codebook via exponential moving average (more stable than 
           gradient-based updates).
        5. Compute entropy of average code usage to penalize collapse.
    
    Args:
        n_codes:   Number of codebook entries (default: 256).
        e_dim:     Dimension of each codebook entry (default: 512).
        beta:      Commitment cost weight (default: 0.25).
        ema_decay: EMA decay rate for codebook updates (default: 0.99).
        eps:       Laplace smoothing epsilon for EMA counts (default: 1e-5).
    """

    def __init__(self, n_codes=256, e_dim=512, beta=0.25, ema_decay=0.99, eps=1e-5):
        super().__init__()
        self.n_codes = n_codes
        self.e_dim = e_dim
        self.beta = beta
        self.ema_decay = ema_decay
        self.eps = eps

        # Codebook embeddings
        self.embedding = nn.Embedding(n_codes, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)

        # EMA tracking buffers (not model parameters, not saved in optimizer)
        self.register_buffer("ema_count", torch.zeros(n_codes))
        self.register_buffer("ema_weight", self.embedding.weight.data.clone())
        self.register_buffer("inited", torch.tensor(False))

    # ------------------------------------------------------------------
    # Codebook management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _init_codebook(self, z_flat):
        """Initialize codebook from first batch of data (better than random)."""
        if self.inited:
            return
        n = z_flat.shape[0]
        if n >= self.n_codes:
            # Pick n_codes random vectors from the batch
            perm = torch.randperm(n, device=z_flat.device)[:self.n_codes]
            self.embedding.weight.data.copy_(z_flat[perm])
        else:
            # If batch is smaller than codebook, fill what we can
            self.embedding.weight.data[:n].copy_(z_flat)
        self.ema_weight.data.copy_(self.embedding.weight.data)
        self.ema_count.fill_(1.0)  # avoid division by zero on first update
        self.inited.fill_(True)

    @torch.no_grad()
    def _ema_update(self, z_flat, indices):
        """Update codebook entries via exponential moving average."""
        one_hot = F.one_hot(indices, self.n_codes).float()  # (N, n_codes)
        counts = one_hot.sum(0)                              # (n_codes,)
        dw = one_hot.t() @ z_flat                            # (n_codes, e_dim)

        self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

        # Laplace smoothing to prevent division by zero
        n = self.ema_count.sum()
        count_smoothed = (
            (self.ema_count + self.eps)
            / (n + self.n_codes * self.eps)
            * n
        )
        self.embedding.weight.data.copy_(
            self.ema_weight / count_smoothed.unsqueeze(1)
        )

    @torch.no_grad()
    def _expire_dead_codes(self, z_flat, indices, threshold=1.0):
        """
        Replace codebook entries that are rarely used (dead codes) 
        with randomly sampled encoder outputs.
        
        Called periodically during training to recover from partial collapse.
        """
        counts = F.one_hot(indices, self.n_codes).float().sum(0)
        dead_mask = self.ema_count < threshold  # codes with very few assignments
        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            return 0

        # Replace dead codes with random samples from the batch
        n_available = z_flat.shape[0]
        if n_available > 0:
            replace_indices = torch.randint(0, n_available, (n_dead,), device=z_flat.device)
            self.embedding.weight.data[dead_mask] = z_flat[replace_indices]
            self.ema_weight.data[dead_mask] = z_flat[replace_indices]
            self.ema_count[dead_mask] = 1.0

        return n_dead

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, z_flat):
        """
        Quantize input features.
        
        Args:
            z_flat: (N, e_dim) flattened feature vectors.
                    N = number of spatial positions belonging to one region 
                    across the entire batch.
        
        Returns:
            z_q:     (N, e_dim) quantized vectors with straight-through gradient.
            indices: (N,) integer codebook indices.
            losses:  dict with 'commitment' and 'entropy' scalar losses.
            info:    dict with monitoring metrics (perplexity, usage ratio).
        """
        # Initialize codebook from data on first forward pass
        if self.training and not self.inited:
            self._init_codebook(z_flat)

        # --- Distance computation ---
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e^T
        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2.0 * z_flat @ self.embedding.weight.t()
        )  # (N, n_codes)

        # --- Hard assignment ---
        indices = d.argmin(dim=-1)  # (N,)
        z_q = self.embedding(indices)  # (N, e_dim)

        # --- Losses ---
        # Commitment loss: pushes encoder output toward codebook entries
        # (codebook itself is updated via EMA, not gradients)
        commitment_loss = self.beta * F.mse_loss(z_flat, z_q.detach())

        # Entropy regularization (HQ-VAE style collapse prevention)
        # Compute empirical distribution of code assignments
        avg_probs = F.one_hot(indices, self.n_codes).float().mean(dim=0)  # (n_codes,)
        # Entropy of this distribution (higher = more uniform = healthier)
        avg_entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
        max_entropy = math.log(self.n_codes)
        # Entropy loss: minimize (max_entropy - actual_entropy)
        entropy_loss = max_entropy - avg_entropy

        # --- EMA codebook update ---
        if self.training:
            self._ema_update(z_flat, indices)

        # --- Straight-through estimator ---
        # Forward: use z_q (discrete). Backward: gradient flows through z_flat.
        z_q_st = z_flat + (z_q - z_flat).detach()

        # --- Monitoring info ---
        info = {
            "perplexity": torch.exp(avg_entropy).detach(),
            "codebook_usage": (avg_probs > 0).float().sum().detach() / self.n_codes,
            "mean_distance": d.min(dim=-1).values.mean().detach(),
        }

        losses = {
            "commitment": commitment_loss,
            "entropy": entropy_loss,
        }

        return z_q_st, indices, losses, info


class ResidualVQ(nn.Module):
    """
    Multi-level Residual Vector Quantizer (RQ-VAE) with HQ-VAE objective.
    
    Quantization proceeds level by level on the residual error:
        r_0 = z                          (original feature)
        z_q1 = quantize(r_0)             Level 1: coarse
        r_1 = r_0 - z_q1                 (residual)
        z_q2 = quantize(r_1)             Level 2: mid
        r_2 = r_1 - z_q2                 (residual)
        z_q3 = quantize(r_2)             Level 3: fine
        
        output = z_q1 + z_q2 + z_q3      (sum of all levels)
    
    Gradient flow:
        A single straight-through estimator is applied to the final sum
        relative to the original input z. This avoids gradient amplification
        that would occur if per-level straight-through were summed.
    
    Args:
        n_codes:        Number of codebook entries per level (default: 256).
        e_dim:          Embedding dimension (default: 512, matches OSDFace).
        n_levels:       Number of residual quantization levels (default: 3).
        beta:           Commitment cost weight (default: 0.25).
        ema_decay:      EMA decay for codebook updates (default: 0.99).
        entropy_weight: Weight for entropy regularization loss (default: 0.1).
    """

    def __init__(
        self,
        n_codes=256,
        e_dim=512,
        n_levels=3,
        beta=0.25,
        ema_decay=0.99,
        entropy_weight=0.1,
    ):
        super().__init__()
        self.n_codes = n_codes
        self.e_dim = e_dim
        self.n_levels = n_levels
        self.entropy_weight = entropy_weight

        self.levels = nn.ModuleList(
            [VQLevel(n_codes, e_dim, beta, ema_decay) for _ in range(n_levels)]
        )

    def forward(self, z):
        """
        Quantize features through all residual levels.
        
        Args:
            z: (N, e_dim) flattened feature vectors for one region.
               N = total number of spatial positions in this region across 
               the batch. For example, if batch_size=4 and a region covers 
               50 positions per image, N=200.
        
        Returns:
            z_q:        (N, e_dim) quantized vectors with straight-through 
                        gradient to z.
            all_losses: dict with aggregated losses:
                        'commitment' — sum of per-level commitment losses
                        'entropy'    — weighted sum of per-level entropy losses
                        'total_vq'   — commitment + entropy (ready to backprop)
            all_info:   dict with per-level monitoring metrics and indices.
        """
        # Accumulate hard-quantized vectors (no gradient)
        z_q_hard_sum = torch.zeros_like(z)
        residual = z

        all_losses = {"commitment": 0.0, "entropy": 0.0}
        all_indices = []
        all_info = {}

        for lvl, vq_level in enumerate(self.levels):
            # Quantize the current residual
            # Note: vq_level returns z_q with straight-through, but we only
            # use the hard-quantized value for residual computation
            z_q_st, indices, losses, info = vq_level(residual)

            # The actual hard-quantized value (no gradient) for residual calc
            z_q_hard = vq_level.embedding(indices)

            # Accumulate
            z_q_hard_sum = z_q_hard_sum + z_q_hard
            residual = residual - z_q_hard.detach()

            # Aggregate losses
            all_losses["commitment"] = all_losses["commitment"] + losses["commitment"]
            all_losses["entropy"] = all_losses["entropy"] + losses["entropy"]

            # Per-level monitoring
            all_indices.append(indices)
            all_info[f"level_{lvl}/perplexity"] = info["perplexity"]
            all_info[f"level_{lvl}/codebook_usage"] = info["codebook_usage"]
            all_info[f"level_{lvl}/residual_norm"] = (
                residual.detach().norm(dim=-1).mean()
            )

        # --- Single straight-through for the entire RQ ---
        # Forward: z_q_hard_sum (discrete). Backward: gradient to z.
        z_q = z + (z_q_hard_sum - z).detach()

        # Weight the entropy loss
        all_losses["entropy"] = all_losses["entropy"] * self.entropy_weight

        # Combined VQ loss (add to reconstruction loss during training)
        all_losses["total_vq"] = all_losses["commitment"] + all_losses["entropy"]

        # Store indices for potential use in association loss
        all_info["all_indices"] = all_indices

        return z_q, all_losses, all_info

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, z):
        """
        Encode features to multi-level codebook indices (inference only).
        
        Args:
            z: (N, e_dim) feature vectors.
        
        Returns:
            indices_list: list of n_levels tensors, each (N,) with int indices.
        """
        indices_list = []
        residual = z

        for vq_level in self.levels:
            d = (
                residual.pow(2).sum(1, keepdim=True)
                + vq_level.embedding.weight.pow(2).sum(1)
                - 2.0 * residual @ vq_level.embedding.weight.t()
            )
            indices = d.argmin(dim=-1)
            z_q = vq_level.embedding(indices)
            residual = residual - z_q
            indices_list.append(indices)

        return indices_list

    @torch.no_grad()
    def decode(self, indices_list):
        """
        Decode multi-level indices back to quantized feature vectors.
        
        Args:
            indices_list: list of n_levels tensors, each (N,) with int indices.
        
        Returns:
            z_q: (N, e_dim) reconstructed quantized vectors.
        """
        z_q = torch.zeros(
            indices_list[0].shape[0],
            self.e_dim,
            device=indices_list[0].device,
            dtype=self.levels[0].embedding.weight.dtype,
        )
        for lvl, indices in enumerate(indices_list):
            z_q = z_q + self.levels[lvl].embedding(indices)
        return z_q

    @torch.no_grad()
    def expire_dead_codes(self, z):
        """
        Replace dead codebook entries across all levels.
        Call periodically during training (e.g., every N steps).
        
        Args:
            z: (N, e_dim) encoder outputs to sample replacements from.
        
        Returns:
            n_expired: total number of codes replaced across all levels.
        """
        total_expired = 0
        residual = z

        for vq_level in self.levels:
            d = (
                residual.pow(2).sum(1, keepdim=True)
                + vq_level.embedding.weight.pow(2).sum(1)
                - 2.0 * residual @ vq_level.embedding.weight.t()
            )
            indices = d.argmin(dim=-1)
            n_expired = vq_level._expire_dead_codes(residual, indices)
            total_expired += n_expired
            z_q = vq_level.embedding(indices)
            residual = residual - z_q

        return total_expired

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_codebook_stats(self):
        """
        Return per-level codebook statistics for logging.
        
        Returns:
            stats: dict with per-level embedding norms and EMA counts.
        """
        stats = {}
        for lvl, vq_level in enumerate(self.levels):
            weight = vq_level.embedding.weight
            stats[f"level_{lvl}/embedding_norm_mean"] = weight.norm(dim=1).mean()
            stats[f"level_{lvl}/embedding_norm_std"] = weight.norm(dim=1).std()
            stats[f"level_{lvl}/ema_count_min"] = vq_level.ema_count.min()
            stats[f"level_{lvl}/ema_count_max"] = vq_level.ema_count.max()
            stats[f"level_{lvl}/ema_count_mean"] = vq_level.ema_count.mean()
            # Dead code ratio (codes with very few assignments)
            dead_ratio = (vq_level.ema_count < 1.0).float().mean()
            stats[f"level_{lvl}/dead_code_ratio"] = dead_ratio
        return stats


# ======================================================================
# Convenience factory
# ======================================================================

def build_region_rq_vae(
    n_regions=4,
    n_codes=256,
    e_dim=512,
    n_levels=3,
    beta=0.25,
    ema_decay=0.99,
    entropy_weight=0.1,
):
    """
    Build a set of ResidualVQ instances, one per facial region.
    
    Args:
        n_regions: Number of facial regions (default: 4 — eyes, skin, hair, lips).
        (other args forwarded to ResidualVQ)
    
    Returns:
        nn.ModuleDict mapping region name → ResidualVQ.
    """
    region_names = ["eyes", "skin", "hair", "lips"][:n_regions]
    return nn.ModuleDict(
        {
            name: ResidualVQ(
                n_codes=n_codes,
                e_dim=e_dim,
                n_levels=n_levels,
                beta=beta,
                ema_decay=ema_decay,
                entropy_weight=entropy_weight,
            )
            for name in region_names
        }
    )