"""
losses.py — Loss functions for CRAFT Stage 1 VQVAE training.

Implements all losses needed for the hierarchical region-aware VQVAE:

    Reconstruction losses:
        - L1Loss:              Absolute pixel difference
        - VGGPerceptualLoss:   Feature-level similarity using VGG19

    Adversarial losses:
        - PatchDiscriminator:  PatchGAN discriminator network (70×70 receptive field)
        - gan_g_loss:          Generator adversarial loss
        - gan_d_loss:          Discriminator adversarial loss

    Association loss:
        - AssociationLoss:     HQ-LQ feature alignment (OSDFace Eq. 9-10)

    Combined:
        - Stage1VQLoss:        Aggregates all losses with configurable weights

Loss weights from OSDFace (Eq. 8, 11):
    λ_per   = 1.0   (perceptual)
    λ_dis   = 0.8   (adversarial)
    λ_assoc = 0→1   (association, enabled after initial training)
    β       = 0.25  (VQ commitment, handled in residual_vq.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp
from torch.nn.utils import spectral_norm
import torchvision


# ======================================================================
# Reconstruction losses
# ======================================================================

class L1Loss(nn.Module):
    """Simple L1 reconstruction loss."""

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 3, H, W) reconstructed image.
            target: (B, 3, H, W) ground truth image.
        Returns:
            Scalar L1 loss.
        """
        return F.l1_loss(pred, target)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features (Johnson et al., 2016).
    
    Computes L2 distance between feature maps extracted at multiple 
    layers of a pretrained VGG19 network.
    
    Layers used: relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
    (indices 2, 7, 12, 21, 30 in VGG19 sequential model).
    
    Args:
        pretrained: Whether to load ImageNet-pretrained weights.
                    Set to False only for testing without internet.
        weights:    Optional list of per-layer weights. 
                    Default: equal weighting (1.0 each).
    """

    # VGG19 layer indices for feature extraction
    _LAYER_INDICES = [2, 7, 12, 21, 30]

    def __init__(self, pretrained=True, weights=None):
        super().__init__()

        vgg = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        )
        features = vgg.features

        # Split VGG into blocks up to each extraction point
        self.blocks = nn.ModuleList()
        prev_idx = 0
        for idx in self._LAYER_INDICES:
            self.blocks.append(nn.Sequential(*list(features.children())[prev_idx:idx + 1]))
            prev_idx = idx + 1

        # Disable inplace ReLU (required for gradient computation through pred)
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        # Freeze all VGG parameters
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        # Per-layer loss weights
        if weights is None:
            weights = [1.0] * len(self._LAYER_INDICES)
        self.register_buffer(
            "layer_weights", torch.tensor(weights, dtype=torch.float32)
        )

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode=True):
        """Keep VGG always in eval mode."""
        return super().train(False)

    def _normalize(self, x):
        """Normalize from [0,1] or [-1,1] to ImageNet stats."""
        # If input is in [-1, 1], rescale to [0, 1] first
        if x.min() < -0.1:
            x = x * 0.5 + 0.5
        return (x - self.mean) / self.std

    @torch.no_grad()
    def extract_features(self, x):
        """Extract multi-layer VGG features."""
        x = self._normalize(x)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 3, H, W) reconstructed image.
            target: (B, 3, H, W) ground truth image.
        Returns:
            Scalar perceptual loss (weighted sum of per-layer L2 distances).
        """
        # Run entire VGG forward in fp32 to prevent overflow in deep layers.
        # fp16 max is 65504; VGG relu4_1/relu5_1 features regularly exceed this,
        # causing NaN that propagates through backward.
        with torch.amp.autocast("cuda", enabled=False):
            pred_f32 = pred if pred.dtype == torch.float32 else pred.float()
            target_f32 = target if target.dtype == torch.float32 else target.float()
            pred_normalized = self._normalize(pred_f32)
            with torch.no_grad():
                target_normalized = self._normalize(target_f32)

            loss = torch.zeros(1, device=pred.device, dtype=torch.float32)
            x_pred = pred_normalized
            x_target = target_normalized

            for i, block in enumerate(self.blocks):
                x_pred = block(x_pred)
                with torch.no_grad():
                    x_target = block(x_target)
                loss = loss + self.layer_weights[i] * F.mse_loss(x_pred, x_target)

        return loss


# ======================================================================
# Adversarial losses (PatchGAN)
# ======================================================================

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator (Isola et al., 2017).
    
    Classifies 70×70 overlapping patches as real or fake,
    producing a spatial map of discriminator predictions rather 
    than a single scalar. This encourages high-frequency detail.
    
    Architecture: C64 → C128 → C256 → C512 → 1
    (4 conv layers with stride 2, then 1×1 output)
    
    Args:
        in_channels: Input channels (default: 3 for RGB images).
        n_layers:    Number of downsampling layers (default: 3).
        ndf:         Base number of discriminator filters (default: 64).
    """

    def __init__(self, in_channels=3, n_layers=3, ndf=64):
        super().__init__()

        layers = [
            spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final 1-channel output (real/fake prediction per patch)
        layers += [
            spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image.
        Returns:
            (B, 1, H', W') patch-level real/fake predictions.
        """
        return self.model(x)


def gan_g_loss(discriminator, fake):
    """
    Generator adversarial loss (non-saturating).
    
    The generator wants the discriminator to classify its output as real.
    
    Args:
        discriminator: PatchDiscriminator network.
        fake:          (B, 3, H, W) generated image.
    
    Returns:
        Scalar generator loss.
    """
    pred_fake = discriminator(fake)
    # Non-saturating loss: -log(D(fake))
    return F.binary_cross_entropy_with_logits(
        pred_fake, torch.ones_like(pred_fake)
    )


def gan_d_loss(discriminator, real, fake):
    """
    Discriminator adversarial loss.
    
    The discriminator wants to classify real images as real 
    and generated images as fake.
    
    Args:
        discriminator: PatchDiscriminator network.
        real:          (B, 3, H, W) ground truth image.
        fake:          (B, 3, H, W) generated image (detached from generator).
    
    Returns:
        Scalar discriminator loss.
    """
    pred_real = discriminator(real)
    pred_fake = discriminator(fake.detach())

    loss_real = F.binary_cross_entropy_with_logits(
        pred_real, torch.ones_like(pred_real)
    )
    loss_fake = F.binary_cross_entropy_with_logits(
        pred_fake, torch.zeros_like(pred_fake)
    )

    return (loss_real + loss_fake) * 0.5


# ======================================================================
# HQ-LQ Association loss (OSDFace Eq. 9-10)
# ======================================================================

class AssociationLoss(nn.Module):
    """
    HQ-LQ feature association loss (DAEFR / OSDFace).
    
    Aligns the LQ encoder's feature space with the HQ encoder's 
    feature space using symmetric cross-entropy on a cosine 
    similarity matrix. This guides the LQ encoder to attend to 
    the same semantic categories as the HQ encoder.
    
    For each (HQ, LQ) image pair, we compute a K×K similarity matrix
    where K is the number of spatial positions (256 for 16×16 features).
    The target is the identity: position i in HQ should match position i 
    in LQ (same spatial location, same face).
    
    L_assoc = (CE(M, I) + CE(M^T, I)) / 2
    
    where M_{i,j} = cosine_sim(z_H[i], z_L[j]) / temperature
    
    Args:
        temperature: Softmax temperature (default: 0.07, following CLIP).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_H, z_L):
        """
        Args:
            z_H: (B, K, D) HQ encoder features (pre-quantization).
            z_L: (B, K, D) LQ encoder features (pre-quantization).
                 K = number of spatial positions (typically 256).
                 D = embedding dimension (typically 512).
        
        Returns:
            Scalar association loss averaged over the batch.
        """
        B, K, D = z_H.shape

        # L2 normalize for cosine similarity
        z_H = F.normalize(z_H, dim=-1)
        z_L = F.normalize(z_L, dim=-1)

        # Compute similarity matrices: (B, K, K)
        sim = torch.bmm(z_H, z_L.transpose(1, 2)) / self.temperature

        # Target: each position should match itself (diagonal)
        targets = torch.arange(K, device=sim.device).unsqueeze(0).expand(B, -1)

        # Cross-entropy along HQ axis (each HQ position → which LQ position?)
        loss_h = F.cross_entropy(sim.reshape(B * K, K), targets.reshape(B * K))

        # Cross-entropy along LQ axis (each LQ position → which HQ position?)
        loss_l = F.cross_entropy(
            sim.transpose(1, 2).reshape(B * K, K), targets.reshape(B * K)
        )

        return (loss_h + loss_l) / 2.0


# ======================================================================
# Combined Stage 1 loss
# ======================================================================

class Stage1VQLoss(nn.Module):
    """
    Aggregates all Stage 1 training losses with configurable weights.
    
    Total generator loss:
        L_gen = L1 + λ_per * L_per + λ_dis * L_G + L_vq + λ_assoc * L_assoc
    
    Where L_vq (commitment + entropy) comes from the ResidualVQ module,
    not computed here.
    
    The discriminator loss L_D is computed separately via gan_d_loss().
    
    Args:
        lambda_per:     Perceptual loss weight (default: 1.0).
        lambda_dis:     Adversarial loss weight (default: 0.8).
        lambda_assoc:   Association loss weight (default: 0.0, enabled later).
        vgg_pretrained: Whether to load pretrained VGG (default: True).
        disc_in_channels: Discriminator input channels (default: 3).
    """

    def __init__(
        self,
        lambda_per=1.0,
        lambda_dis=0.8,
        lambda_assoc=0.0,
        vgg_pretrained=True,
        disc_in_channels=3,
    ):
        super().__init__()
        self.lambda_per = lambda_per
        self.lambda_dis = lambda_dis
        self.lambda_assoc = lambda_assoc

        self.l1_loss = L1Loss()
        self.perceptual_loss = VGGPerceptualLoss(pretrained=vgg_pretrained)
        self.discriminator = PatchDiscriminator(in_channels=disc_in_channels)
        self.association_loss = AssociationLoss()

    def generator_loss(self, pred, target, vq_loss, z_H=None, z_L=None):
        """
        Compute the full generator loss for one training step.
        
        Args:
            pred:    (B, 3, H, W) reconstructed image.
            target:  (B, 3, H, W) ground truth image.
            vq_loss: Scalar VQ loss from ResidualVQ (commitment + entropy).
            z_H:     (B, K, D) HQ features for association loss (optional).
            z_L:     (B, K, D) LQ features for association loss (optional).
        
        Returns:
            total_loss: Scalar total generator loss.
            loss_dict:  Dict of individual loss components for logging.
        """
        # Reconstruction
        l1 = self.l1_loss(pred, target)

        # Perceptual — clamp to prevent single-batch spikes from
        # wiping out all other gradient signals after grad clipping
        l_per = self.perceptual_loss(pred, target)
        l_per = torch.clamp(l_per, max=150.0)

        # Adversarial (generator wants discriminator to say "real")
        l_gan_g = gan_g_loss(self.discriminator, pred)
        l_gan_g = torch.clamp(l_gan_g, max=50.0)

        # Start with base losses
        total = l1 + self.lambda_per * l_per + self.lambda_dis * l_gan_g + vq_loss

        loss_dict = {
            "l1": l1.detach(),
            "perceptual": l_per.detach(),
            "gan_g": l_gan_g.detach(),
            "vq": vq_loss.detach() if torch.is_tensor(vq_loss) else vq_loss,
        }

        # Association loss (only when z_H and z_L are provided and λ > 0)
        if self.lambda_assoc > 0 and z_H is not None and z_L is not None:
            l_assoc = self.association_loss(z_H, z_L)
            total = total + self.lambda_assoc * l_assoc
            loss_dict["association"] = l_assoc.detach()

        loss_dict["total_gen"] = total.detach()
        return total, loss_dict

    def discriminator_loss(self, real, fake):
        """
        Compute the discriminator loss for one training step.
        
        Args:
            real: (B, 3, H, W) ground truth image.
            fake: (B, 3, H, W) reconstructed image (will be detached).
        
        Returns:
            d_loss:    Scalar discriminator loss.
            loss_dict: Dict with loss components for logging.
        """
        d_loss = gan_d_loss(self.discriminator, real, fake)
        return d_loss, {"gan_d": d_loss.detach()}

    def set_lambda_assoc(self, value):
        """Update association loss weight (e.g., 0→1 during training)."""
        self.lambda_assoc = value