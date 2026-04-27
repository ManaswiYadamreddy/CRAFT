"""
text_conditioner.py — FiLM-based text conditioning for OSDFace

Two separate FiLM MLPs per BasicTransformerBlock:
    film_pos: learns how to modulate features given positive attributes
    film_na:  learns how to suppress features given negative attributes

Each MLP is trained independently so they can learn genuinely different
modulation directions. The net modulation is:

    h = h * (1 + γ_pos - neg_weight * γ_na) + (β_pos - neg_weight * β_na)

Both MLPs are zero-initialized so training starts from a stable OSDFace
baseline (identity modulation at step 0).

v1 bug: a single FiLM MLP was shared for both pos and na. Because pos and
na share similar language, the MLP produced similar (γ, β) for both, so
γ_pos - γ_na ≈ 0 — effectively no text influence regardless of film_neg_weight.
"""

import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock


# ---------------------------------------------------------------------------
# Per-block FiLM layer
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    Two-layer MLP that maps a pooled text embedding → (γ, β).
    Zero-initialized so it starts as identity modulation.
    """
    def __init__(self, text_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.SiLU(),
            nn.Linear(text_dim // 2, 2 * feature_dim),
        )
        # Zero-init: γ=0, β=0 at start → h * (1+0) + 0 = h (identity)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, text_emb: torch.Tensor):
        """
        Args:
            text_emb: (B, text_dim) — mean-pooled text embedding
        Returns:
            gamma: (B, 1, feature_dim)
            beta:  (B, 1, feature_dim)
        """
        out = self.net(text_emb)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma.unsqueeze(1), beta.unsqueeze(1)


# ---------------------------------------------------------------------------
# TextConditioner
# ---------------------------------------------------------------------------

class TextConditioner(nn.Module):
    """
    Attaches two FiLMLayers per BasicTransformerBlock in the UNet:
        film_pos_layers: one MLP per block for positive attribute guidance
        film_na_layers:  one MLP per block for negative attribute suppression

    Having separate MLPs means each one can learn a genuinely different
    mapping from text → feature modulation, solving the cancellation problem
    that occurs when pos and na share a single MLP.

    Usage:
        conditioner = TextConditioner(unet, text_dim=1024)
        conditioner.register_hooks(unet)

        conditioner.set_text_embedding(pos_embeds, na_embeds, neg_weight=0.5)
        output = unet(...)
        conditioner.clear_text_embedding()
    """

    def __init__(self, unet: nn.Module, text_dim: int = 1024):
        super().__init__()

        self._pos_emb:    torch.Tensor | None = None
        self._na_emb:     torch.Tensor | None = None
        self._neg_weight: float = 0.5
        self._hooks:      list  = []

        # Two separate ModuleDicts — one for pos, one for na
        self.film_pos_layers = nn.ModuleDict()
        self.film_na_layers  = nn.ModuleDict()

        for name, module in unet.named_modules():
            if isinstance(module, BasicTransformerBlock):
                feature_dim = module.norm1.normalized_shape[0]
                key = name.replace(".", "_")
                self.film_pos_layers[key] = FiLMLayer(text_dim, feature_dim)
                self.film_na_layers[key]  = FiLMLayer(text_dim, feature_dim)

        n     = len(self.film_pos_layers)
        total = sum(p.numel() for p in self.parameters())
        print(f"TextConditioner: {n} blocks x 2 FiLM MLPs = {total:,} trainable params")

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hooks(self, unet: nn.Module):
        self._remove_hooks()
        for name, module in unet.named_modules():
            if isinstance(module, BasicTransformerBlock):
                key = name.replace(".", "_")
                if key in self.film_pos_layers:
                    hook = module.register_forward_hook(self._make_hook(key))
                    self._hooks.append(hook)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, key: str):
        def hook(module, input, output):
            if self._pos_emb is None:
                return output

            if isinstance(output, tuple):
                h    = output[0]
                rest = output[1:]
            else:
                h    = output
                rest = None

            device, dtype = h.device, h.dtype
            film_dtype = next(self.film_pos_layers[key].parameters()).dtype

            # Positive guidance — dedicated pos MLP
            pos_pooled = self._pos_emb.mean(dim=1).to(device, dtype=film_dtype)
            gamma_pos, beta_pos = self.film_pos_layers[key](pos_pooled)
            gamma_pos = gamma_pos.to(dtype=dtype)
            beta_pos = beta_pos.to(dtype=dtype)

            if self._na_emb is not None and self._neg_weight > 0:
                # Negative suppression — dedicated na MLP (learns independently)
                na_pooled = self._na_emb.mean(dim=1).to(device, dtype=film_dtype)
                gamma_na, beta_na = self.film_na_layers[key](na_pooled)
                gamma_na = gamma_na.to(dtype=dtype)
                beta_na = beta_na.to(dtype=dtype)

                gamma_net = gamma_pos - self._neg_weight * gamma_na
                beta_net  = beta_pos  - self._neg_weight * beta_na
            else:
                gamma_net = gamma_pos
                beta_net  = beta_pos

            h = h * (1 + gamma_net) + beta_net

            if rest is not None:
                return (h,) + rest
            return h

        return hook

    # ------------------------------------------------------------------
    # Embedding management
    # ------------------------------------------------------------------

    def set_text_embedding(
        self,
        pos_embeds:  torch.Tensor,
        na_embeds:   torch.Tensor | None = None,
        neg_weight:  float = 0.5,
    ):
        self._pos_emb    = pos_embeds
        self._na_emb     = na_embeds
        self._neg_weight = neg_weight

    def clear_text_embedding(self):
        self._pos_emb    = None
        self._na_emb     = None
        self._neg_weight = 0.5

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "film_pos_layers": self.film_pos_layers.state_dict(),
            "film_na_layers":  self.film_na_layers.state_dict(),
        }, path)

    def load(self, path: str, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        if "film_pos_layers" in ckpt and "film_na_layers" in ckpt:
            self.film_pos_layers.load_state_dict(ckpt["film_pos_layers"])
            self.film_na_layers.load_state_dict(ckpt["film_na_layers"])
        else:
            raise ValueError(
                "Checkpoint appears to be from the old single-MLP architecture "
                "and is not compatible with this version. Please retrain."
            )