"""
vqvae.py — VQVAE architecture for CRAFT Stage 1 training.

Encoder and decoder architectures are from OSDFace (Wang et al., 2025).
This file provides:

    Building blocks (from OSDFace):
        - ResnetBlock, MultiHeadAttnBlock, Upsample, Downsample
        - MultiHeadEncoder:  512×512 → 16×16 feature maps (512 channels)
        - MultiHeadDecoder:  16×16 → 512×512 reconstruction

    Quantizers:
        - GlobalVQ:  Standard flat codebook (1024 codes) for the HQ branch.
                     Matches OSDFace's VectorQuantizer.
        - (RegionAwareVQ is imported from region_aware_vq.py for the LQ branch)

    Top-level model:
        - VQVAE:  Wraps encoder + quant_conv + quantizer + post_quant_conv + decoder.
                  Accepts either GlobalVQ or RegionAwareVQ as the quantizer.

Architecture parameters (OSDFace defaults):
    ch=64, ch_mult=(1,2,2,2,4,8), num_res_blocks=2, attn_resolutions=[16]
    resolution=512, z_channels=512, embed_dim=512, n_embed=1024
    latent resolution = 512 / 2^5 = 16  →  256 spatial positions
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint


# ======================================================================
# Building blocks (from OSDFace vqvae.py)
# ======================================================================

def nonlinearity(x):
    """Swish activation."""
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = nonlinearity(self.norm1(x))
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = nonlinearity(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, head_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert in_channels % head_size == 0

        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, y=None):
        h_ = self.norm1(x)
        if y is None:
            y = h_
        else:
            y = self.norm2(y)

        q = self.q(y)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, self.head_size, self.att_size, h * w).permute(0, 3, 1, 2)
        k = k.reshape(b, self.head_size, self.att_size, h * w).permute(0, 3, 1, 2)
        v = v.reshape(b, self.head_size, self.att_size, h * w).permute(0, 3, 1, 2)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        scale = int(self.att_size) ** (-0.5)
        # Compute attention in float32 to prevent overflow under AMP
        q_f32, k_f32 = q.float(), k.float()
        w_ = F.softmax(q_f32.mul(scale) @ k_f32, dim=3).to(v.dtype)
        atten_weight = w_.detach().clone()

        out = (w_ @ v).transpose(1, 2).contiguous().view(b, h, w, -1).permute(0, 3, 1, 2)
        out = self.proj_out(out)

        return x + out, atten_weight


# ======================================================================
# Encoder (from OSDFace)
# ======================================================================

class MultiHeadEncoder(nn.Module):
    """
    OSDFace encoder: 512×512 RGB → 16×16 feature map with 512 channels.
    
    Architecture: 6 resolution levels with ch_mult=(1,2,2,2,4,8),
    each with 2 ResNet blocks. Attention at 16×16 resolution.
    5 downsampling steps: 512→256→128→64→32→16.
    """

    def __init__(self, ch=64, out_ch=3, ch_mult=(1, 2, 2, 2, 4, 8), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=512, double_z=False, enable_mid=True,
                 head_size=1, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.enable_mid = enable_mid

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=self.temb_ch, dropout=dropout,
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(
                in_channels=block_in, out_channels=block_in,
                temb_channels=self.temb_ch, dropout=dropout,
            )
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(
                in_channels=block_in, out_channels=block_in,
                temb_channels=self.temb_ch, dropout=dropout,
            )

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels if double_z else z_channels,
            kernel_size=3, stride=1, padding=1,
        )
        self._use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self._use_gradient_checkpointing = True

    def _ckpt(self, fn, *args):
        if self._use_gradient_checkpointing and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        hs = {}
        temb = None
        h = self.conv_in(x)
        hs["in"] = h

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self._ckpt(self.down[i_level].block[i_block], h, temb)
                if len(self.down[i_level].attn) > 0:
                    h, _ = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                hs[f"block_{i_level}"] = h
                h = self.down[i_level].downsample(h)

        atten_weight = None
        if self.enable_mid:
            h = self._ckpt(self.mid.block_1, h, temb)
            hs[f"block_{i_level}_atten"] = h
            h, atten_weight = self.mid.attn_1(h)
            h = self._ckpt(self.mid.block_2, h, temb)
            hs["mid_atten"] = h

        h = nonlinearity(self.norm_out(h))
        h = self.conv_out(h)
        hs["out"] = h
        return hs, atten_weight


# ======================================================================
# Decoder (from OSDFace)
# ======================================================================

class MultiHeadDecoder(nn.Module):
    """
    OSDFace decoder: 16×16 feature map → 512×512 RGB reconstruction.
    
    Mirror architecture of the encoder with upsampling.
    """

    def __init__(self, ch=64, out_ch=3, ch_mult=(1, 2, 2, 2, 4, 8), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=512, give_pre_end=False, enable_mid=True,
                 head_size=1, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(
                in_channels=block_in, out_channels=block_in,
                temb_channels=self.temb_ch, dropout=dropout,
            )
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(
                in_channels=block_in, out_channels=block_in,
                temb_channels=self.temb_ch, dropout=dropout,
            )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block.append(ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=self.temb_ch, dropout=dropout,
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        self._use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self._use_gradient_checkpointing = True

    def _ckpt(self, fn, *args):
        if self._use_gradient_checkpointing and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, z):
        temb = None
        h = self.conv_in(z)

        if self.enable_mid:
            h = self._ckpt(self.mid.block_1, h, temb)
            h, _ = self.mid.attn_1(h)
            h = self._ckpt(self.mid.block_2, h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self._ckpt(self.up[i_level].block[i_block], h, temb)
                if len(self.up[i_level].attn) > 0:
                    h, _ = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = nonlinearity(self.norm_out(h))
        h = self.conv_out(h)
        return h


# ======================================================================
# Global VQ (for HQ branch — matches OSDFace's flat codebook)
# ======================================================================

class GlobalVQ(nn.Module):
    """
    Standard flat vector quantizer for the HQ branch.
    
    Same interface as RegionAwareVQ so both can be plugged into VQVAE 
    interchangeably. Uses EMA codebook updates and entropy regularization.
    
    Args:
        n_codes:        Number of codebook entries (default: 1024).
        e_dim:          Embedding dimension (default: 512).
        beta:           Commitment cost weight (default: 0.25).
        ema_decay:      EMA decay for codebook updates (default: 0.99).
        entropy_weight: Entropy regularization weight (default: 0.1).
    """

    def __init__(self, n_codes=1024, e_dim=512, beta=0.25, ema_decay=0.95,
                 entropy_weight=0.1):
        super().__init__()
        self.n_codes = n_codes
        self.e_dim = e_dim
        self.beta = beta
        self.ema_decay = ema_decay
        self.entropy_weight = entropy_weight

        self.embedding = nn.Embedding(n_codes, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        # Normalize initial codebook to unit sphere
        with torch.no_grad():
            self.embedding.weight.data.copy_(
                F.normalize(self.embedding.weight.data, dim=1)
            )

        self.register_buffer("ema_count", torch.zeros(n_codes))
        self.register_buffer("ema_weight", self.embedding.weight.data.clone())
        self.register_buffer("inited", torch.tensor(False))

        # Learnable scale: decoder can recover magnitude via this scalar
        # Initialized to 8.0 (typical z_norm / sqrt(e_dim) baseline)
        self.codebook_scale = nn.Parameter(torch.tensor(8.0))

    @torch.no_grad()
    def _init_codebook(self, z_flat):
        """Initialize codebook from first batch (L2-normalized)."""
        if self.inited:
            return
        n = z_flat.shape[0]
        if n >= self.n_codes:
            perm = torch.randperm(n, device=z_flat.device)[:self.n_codes]
            self.embedding.weight.data.copy_(F.normalize(z_flat[perm], dim=1))
        else:
            self.embedding.weight.data[:n].copy_(F.normalize(z_flat[:n], dim=1))
        self.ema_weight.data.copy_(self.embedding.weight.data)
        self.ema_count.fill_(1.0)
        self.inited.fill_(True)

    @torch.no_grad()
    def _ema_update(self, z_flat, indices):
        one_hot = F.one_hot(indices, self.n_codes).float()
        counts = one_hot.sum(0)
        dw = one_hot.t() @ z_flat

        self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

        n = self.ema_count.sum()
        count_smoothed = (self.ema_count + 1e-5) / (n + self.n_codes * 1e-5) * n
        updated = self.ema_weight / count_smoothed.unsqueeze(1)
        # Re-normalize codebook to unit sphere after EMA update
        self.embedding.weight.data.copy_(F.normalize(updated, dim=1))

    def forward(self, z, images=None, masks=None):
        """
        Quantize feature map using a flat codebook.

        Args:
            z:      (B, C, H, W) encoder feature map.
            images: Unused (present for interface compatibility with RegionAwareVQ).
            masks:  Unused.

        Returns:
            z_q:        (B, C, H, W) quantized feature map with straight-through gradient.
            all_losses: dict with 'commitment', 'entropy', 'total_vq'.
            all_info:   dict with monitoring metrics.
        """
        B, C, H, W = z.shape

        # (B, C, H, W) → (B*H*W, C)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)

        # Force float32 for VQ ops to prevent overflow under AMP
        z_flat_f32 = z_flat.float()

        # L2-normalize encoder outputs before quantization
        # This constrains z and codebook to the unit hypersphere,
        # preventing the norm drift that causes NaN
        z_norm_raw = z_flat_f32.detach().norm(dim=-1).mean()  # for logging
        z_flat_f32_normed = F.normalize(z_flat_f32, dim=1)

        embed_f32 = self.embedding.weight.float()
        # Codebook is already normalized via _ema_update / _init_codebook

        # Data-driven init (from normalized vectors)
        if self.training and not self.inited:
            self._init_codebook(z_flat_f32_normed)

        # Distance on unit sphere: ||a-b||^2 = 2 - 2*a·b (since ||a||=||b||=1)
        # Using dot product is more numerically stable
        sim = z_flat_f32_normed @ embed_f32.t()  # (N, n_codes)
        indices = sim.argmax(dim=-1)  # closest = highest cosine similarity
        z_q_flat = self.embedding(indices)  # already normalized

        # Commitment loss on normalized vectors — bounded by construction
        # since both lie on unit sphere, MSE ∈ [0, 4]
        commitment_loss = self.beta * F.mse_loss(z_flat_f32_normed, z_q_flat.float().detach())

        # Entropy regularization
        avg_probs = F.one_hot(indices, self.n_codes).float().mean(0)
        avg_entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
        max_entropy = math.log(self.n_codes)
        entropy_loss = (max_entropy - avg_entropy) * self.entropy_weight

        # EMA update on normalized vectors
        if self.training:
            self._ema_update(z_flat_f32_normed, indices)

        # Apply learnable scale so decoder can recover magnitude
        scale = self.codebook_scale.abs()  # ensure positive
        z_q_scaled = z_q_flat * scale

        # Straight-through: gradient flows to z_flat (pre-normalization)
        # We use the normalized z as the "input" side of straight-through
        z_flat_normed = z_flat_f32_normed.to(z_flat.dtype)
        z_q_st = z_flat_normed * scale + (z_q_scaled - z_flat_normed * scale).detach()

        # Reshape back
        z_q = z_q_st.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        all_losses = {
            "commitment": commitment_loss,
            "entropy": entropy_loss,
            "total_vq": commitment_loss + entropy_loss,
        }
        all_info = {
            "perplexity": torch.exp(avg_entropy).detach(),
            "codebook_usage": (avg_probs > 0).float().sum().detach() / self.n_codes,
            "z_norm": z_norm_raw,
            "codebook_norm": embed_f32.detach().norm(dim=-1).mean(),
            "codebook_scale": scale.detach(),
        }

        return z_q, all_losses, all_info


# ======================================================================
# VQVAE top-level model
# ======================================================================

class VQVAE(nn.Module):
    """
    Complete VQVAE model: encoder → quant_conv → quantizer → post_quant_conv → decoder.
    
    Works with either GlobalVQ (HQ branch) or RegionAwareVQ (LQ branch).
    
    Args:
        ch:               Base channel count (default: 64).
        ch_mult:          Channel multipliers per resolution level.
        num_res_blocks:   ResNet blocks per level (default: 2).
        attn_resolutions: Resolutions where attention is applied (default: [16]).
        dropout:          Dropout rate (default: 0.0).
        in_channels:      Input image channels (default: 3).
        out_channels:     Output image channels (default: 3).
        resolution:       Input image resolution (default: 512).
        z_channels:       Latent feature channels (default: 512).
        embed_dim:        VQ embedding dimension (default: 512).
        quantizer:        Quantization module — either GlobalVQ or RegionAwareVQ.
                          If None, no quantization is applied (pass-through).
    """

    def __init__(
        self,
        ch=64,
        ch_mult=(1, 2, 2, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        in_channels=3,
        out_channels=3,
        resolution=512,
        z_channels=512,
        embed_dim=512,
        quantizer=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.z_channels = z_channels

        # Encoder
        self.encoder = MultiHeadEncoder(
            ch=ch, out_ch=out_channels, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions,
            dropout=dropout, in_channels=in_channels, resolution=resolution,
            z_channels=z_channels, double_z=False, enable_mid=True, head_size=1,
        )

        # Pre-quantization projection
        self.quant_conv = nn.Conv2d(z_channels, embed_dim, kernel_size=1)

        # Quantizer (GlobalVQ for HQ, RegionAwareVQ for LQ, or None)
        self.quantizer = quantizer

        # Post-quantization projection
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, kernel_size=1)

        # Decoder
        self.decoder = MultiHeadDecoder(
            ch=ch, out_ch=out_channels, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions,
            dropout=dropout, in_channels=in_channels, resolution=resolution,
            z_channels=z_channels, enable_mid=True, head_size=1,
        )

    def encode(self, x):
        """
        Encode image to pre-quantization features.
        
        Args:
            x: (B, 3, 512, 512) input image in [-1, 1].
        
        Returns:
            z: (B, embed_dim, 16, 16) pre-quantization feature map.
        """
        hs, _ = self.encoder(x)
        z = self.quant_conv(hs["out"])
        return z

    def decode(self, z_q):
        """
        Decode quantized features to image.
        
        Args:
            z_q: (B, embed_dim, 16, 16) quantized feature map.
        
        Returns:
            x_rec: (B, 3, 512, 512) reconstructed image.
        """
        h = self.post_quant_conv(z_q)
        return self.decoder(h)

    def forward(self, x, images_01=None, masks=None):
        """
        Full forward: encode → quantize → decode.

        Args:
            x:          (B, 3, 512, 512) input image in [-1, 1] (goes through encoder).
            images_01:  (B, 3, 512, 512) same image in [0, 1] (for face parser in
                        RegionAwareVQ). Only needed when quantizer is RegionAwareVQ
                        and masks is not provided.
            masks:      Optional pre-computed region masks dict. If provided, the face
                        parser is skipped entirely. Keys: region names, values: (B, H, W) bool.

        Returns:
            x_rec:      (B, 3, 512, 512) reconstructed image.
            z:          (B, embed_dim, 16, 16) pre-quantization features (for association loss).
            z_q:        (B, embed_dim, 16, 16) post-quantization features.
            vq_losses:  dict with VQ loss components ('commitment', 'entropy', 'total_vq').
            vq_info:    dict with monitoring metrics.
        """
        # Encode
        z = self.encode(x)

        # Quantize
        if self.quantizer is not None:
            z_q, vq_losses, vq_info = self.quantizer(z, images=images_01, masks=masks)
        else:
            z_q = z
            vq_losses = {"commitment": 0.0, "entropy": 0.0, "total_vq": 0.0}
            vq_info = {}

        # Decode
        x_rec = self.decode(z_q)

        return x_rec, z, z_q, vq_losses, vq_info

    def get_features_flat(self, x):
        """
        Get encoder features in flattened format (B, K, D) for association loss.
        
        Args:
            x: (B, 3, 512, 512) input image in [-1, 1].
        
        Returns:
            z_flat: (B, H*W, embed_dim) = (B, 256, 512).
        """
        z = self.encode(x)  # (B, C, H, W)
        B, C, H, W = z.shape
        return z.permute(0, 2, 3, 1).reshape(B, H * W, C)


# ======================================================================
# Factory functions
# ======================================================================

def build_hq_vqvae(n_codes=1024, embed_dim=512, **kwargs):
    """Build VQVAE with global flat codebook for HQ branch training."""
    quantizer = GlobalVQ(n_codes=n_codes, e_dim=embed_dim)
    return VQVAE(embed_dim=embed_dim, quantizer=quantizer, **kwargs)


def build_lq_vqvae(region_aware_vq, embed_dim=512, **kwargs):
    """Build VQVAE with region-aware RQ for LQ branch training."""
    return VQVAE(embed_dim=embed_dim, quantizer=region_aware_vq, **kwargs)