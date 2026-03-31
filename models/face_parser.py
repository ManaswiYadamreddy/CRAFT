"""
face_parser.py — Lightweight face parsing for CRAFT Stage 1.

Wraps a pretrained BiSeNet (trained on CelebAMask-HQ) to produce 4-region 
semantic masks at any target resolution. Used to partition the encoder 
feature map into facial regions for region-aware residual VQ.

The BiSeNet architecture is included inline so there's no external dependency 
on the face-parsing repo. You only need to download the pretrained checkpoint:


CelebAMask-HQ 19-class label indices:
    0:  background      1:  skin          2:  l_brow        3:  r_brow
    4:  l_eye           5:  r_eye         6:  eye_g         7:  l_ear
    8:  r_ear           9:  nose         10:  mouth        11:  u_lip
    12: l_lip          13:  hair         14:  hat          15:  ear_r
    16: neck_l         17:  neck         18:  cloth

CRAFT 4-region mapping:
    eyes:  l_eye(4), r_eye(5), l_brow(2), r_brow(3), eye_g(6)
    skin:  skin(1), nose(9), l_ear(7), r_ear(8), neck(17),
           ear_r(15), neck_l(16), background(0), cloth(18)
    hair:  hair(13), hat(14)
    lips:  u_lip(11), l_lip(12), mouth(10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo


# ======================================================================
# BiSeNet architecture (from zllrunning/face-parsing.PyTorch)
# Inlined here for self-containedness.
# ======================================================================

_RESNET18_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class _BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = _conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        shortcut = self.downsample(x) if self.downsample is not None else x
        return self.relu(shortcut + residual)


def _create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [_BasicBlock(in_chan, out_chan, stride=stride)]
    for _ in range(bnum - 1):
        layers.append(_BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class _Resnet18(nn.Module):
    def __init__(self, pretrained_backbone=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = _create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = _create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = _create_layer_basic(256, 512, bnum=2, stride=2)
        if pretrained_backbone:
            self._load_pretrained()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32

    def _load_pretrained(self):
        state_dict = modelzoo.load_url(_RESNET18_URL)
        self_state = self.state_dict()
        for k, v in state_dict.items():
            if "fc" in k:
                continue
            self_state[k] = v
        self.load_state_dict(self_state)


class _ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class _BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = _ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))


class _AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = _ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.sigmoid(self.bn_atten(self.conv_atten(F.avg_pool2d(feat, feat.size()[2:]))))
        return feat * atten


class _ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = _Resnet18()
        self.arm16 = _AttentionRefinementModule(256, 128)
        self.arm32 = _AttentionRefinementModule(512, 128)
        self.conv_head32 = _ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = _ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = _ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        H16, W16 = feat16.size()[2:]
        H8, W8 = feat8.size()[2:]

        avg = self.conv_avg(F.avg_pool2d(feat32, feat32.size()[2:]))
        avg_up = F.interpolate(avg, feat32.size()[2:], mode="nearest")

        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = self.conv_head32(F.interpolate(feat32_sum, (H16, W16), mode="nearest"))

        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = self.conv_head16(F.interpolate(feat16_sum, (H8, W8), mode="nearest"))

        return feat8, feat16_up, feat32_up


class _FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = _ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)

    def forward(self, fsp, fcp):
        feat = self.convblk(torch.cat([fsp, fcp], dim=1))
        atten = torch.sigmoid(self.conv2(F.relu(self.conv1(F.avg_pool2d(feat, feat.size()[2:])))))
        return feat * atten + feat


class BiSeNet(nn.Module):
    """BiSeNet for face parsing (19 classes, CelebAMask-HQ)."""

    def __init__(self, n_classes=19):
        super().__init__()
        self.cp = _ContextPath()
        self.ffm = _FeatureFusionModule(256, 256)
        self.conv_out = _BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = _BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = _BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        feat_out = F.interpolate(
            self.conv_out(feat_fuse), (H, W), mode="bilinear", align_corners=True
        )
        feat_out16 = F.interpolate(
            self.conv_out16(feat_cp8), (H, W), mode="bilinear", align_corners=True
        )
        feat_out32 = F.interpolate(
            self.conv_out32(feat_cp16), (H, W), mode="bilinear", align_corners=True
        )
        return feat_out, feat_out16, feat_out32


# ======================================================================
# Region mapping constants
# ======================================================================

# CelebAMask-HQ label indices → CRAFT region assignment
REGION_MAP = {
    "eyes": [2, 3, 4, 5, 6],      # l_brow, r_brow, l_eye, r_eye, eye_g
    "skin": [0, 1, 7, 8, 9, 15, 16, 17, 18],  # bg, skin, ears, nose, neck, earring, necklace, cloth
    "hair": [13, 14],               # hair, hat
    "lips": [10, 11, 12],           # mouth, u_lip, l_lip
}

# Ordered region names (consistent with residual_vq.py)
REGION_NAMES = ["eyes", "skin", "hair", "lips"]

# Number of CelebAMask-HQ classes
N_CLASSES = 19


# ======================================================================
# FaceParser wrapper
# ======================================================================

class FaceParser(nn.Module):
    """
    Frozen BiSeNet wrapper that produces 4-region masks for CRAFT.
    
    Usage:
        parser = FaceParser(checkpoint_path="pretrained/79999_iter.pth")
        parser = parser.to(device)
        
        # During training, with encoder features at 16×16:
        masks = parser.get_region_masks(images, target_h=16, target_w=16)
        # masks["eyes"]  → (B, 16, 16) bool tensor
        # masks["skin"]  → (B, 16, 16) bool tensor
        # masks["hair"]  → (B, 16, 16) bool tensor
        # masks["lips"]  → (B, 16, 16) bool tensor
    
    Args:
        checkpoint_path: Path to 79999_iter.pth (BiSeNet pretrained on CelebAMask-HQ).
                         If None, uses randomly initialized weights (for testing only).
        device:          Device to load the model on (auto-detected if None).
    """

    def __init__(self, checkpoint_path=None, device=None):
        super().__init__()
        self.net = BiSeNet(n_classes=N_CLASSES)

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.net.load_state_dict(state_dict)

        # Freeze everything — no gradients, no training
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        # Precompute label→region index lookup table for fast mapping
        # region_lut[label_id] = region_index (0=eyes, 1=skin, 2=hair, 3=lips)
        lut = torch.zeros(N_CLASSES, dtype=torch.long)
        for region_idx, name in enumerate(REGION_NAMES):
            for label_id in REGION_MAP[name]:
                lut[label_id] = region_idx
        self.register_buffer("region_lut", lut)

        # ImageNet normalization (BiSeNet expects this)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode=True):
        """Override train() to keep BiSeNet always in eval mode."""
        return super().train(False)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def parse(self, images):
        """
        Run BiSeNet on images to get per-pixel class labels.
        
        Args:
            images: (B, 3, H, W) tensor in [0, 1] range.
        
        Returns:
            labels: (B, H, W) int64 tensor with CelebAMask-HQ class indices (0-18).
        """
        # Resize to 512×512 if needed (BiSeNet trained at this resolution)
        B, C, H, W = images.shape
        if H != 512 or W != 512:
            x = F.interpolate(images, size=(512, 512), mode="bilinear", align_corners=False)
        else:
            x = images

        # ImageNet normalization
        x = (x - self.mean) / self.std

        # Forward pass (only use the main output head)
        logits, _, _ = self.net(x)  # (B, 19, 512, 512)
        labels = logits.argmax(dim=1)  # (B, 512, 512)

        return labels

    @torch.no_grad()
    def get_region_masks(self, images, target_h=16, target_w=16):
        """
        Produce 4-region binary masks at the target spatial resolution.
        
        This is the primary method used during CRAFT Stage 1 training.
        It parses the input images and produces downsampled region masks 
        aligned with the encoder feature map resolution.
        
        Args:
            images:   (B, 3, H, W) tensor in [0, 1] range.
            target_h: Target mask height (default: 16, matching encoder output).
            target_w: Target mask width  (default: 16, matching encoder output).
        
        Returns:
            masks: dict mapping region name → (B, target_h, target_w) bool tensor.
                   Every spatial position is assigned to exactly one region 
                   (masks are mutually exclusive and collectively exhaustive).
        """
        # Get full-resolution labels
        labels = self.parse(images)  # (B, 512, 512)

        # Map labels to region indices using the lookup table
        region_indices = self.region_lut[labels]  # (B, 512, 512) values in {0,1,2,3}

        # Downsample to target resolution using nearest-neighbor
        # (preserves discrete labels, no interpolation artifacts)
        region_indices = F.interpolate(
            region_indices.unsqueeze(1).float(),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(1).long()  # (B, target_h, target_w)

        # Create per-region binary masks
        masks = {}
        for region_idx, name in enumerate(REGION_NAMES):
            masks[name] = region_indices == region_idx

        return masks

    @torch.no_grad()
    def get_region_indices(self, images, target_h=16, target_w=16):
        """
        Like get_region_masks but returns a single index tensor.
        
        Args:
            images:   (B, 3, H, W) tensor in [0, 1] range.
            target_h: Target height.
            target_w: Target width.
        
        Returns:
            region_indices: (B, target_h, target_w) int64 tensor.
                            Values: 0=eyes, 1=skin, 2=hair, 3=lips.
        """
        labels = self.parse(images)
        region_indices = self.region_lut[labels]
        region_indices = F.interpolate(
            region_indices.unsqueeze(1).float(),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(1).long()
        return region_indices

    # ------------------------------------------------------------------
    # Diagnostic / visualization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_full_segmentation(self, images):
        """
        Get the full 19-class segmentation at input resolution (for visualization).
        
        Args:
            images: (B, 3, H, W) tensor in [0, 1] range.
        
        Returns:
            labels: (B, H, W) int64 tensor with class indices.
        """
        labels = self.parse(images)  # (B, 512, 512)

        # Resize back to original input resolution if needed
        H, W = images.shape[2:]
        if H != 512 or W != 512:
            labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=(H, W),
                mode="nearest",
            ).squeeze(1).long()

        return labels

    @torch.no_grad()
    def count_region_positions(self, images, target_h=16, target_w=16):
        """
        Count how many spatial positions each region gets per image.
        Useful for verifying region balance and debugging.
        
        Args:
            images: (B, 3, H, W) tensor in [0, 1] range.
        
        Returns:
            counts: dict mapping region name → (B,) int tensor with 
                    per-image position counts.
        """
        masks = self.get_region_masks(images, target_h, target_w)
        counts = {}
        for name, mask in masks.items():
            counts[name] = mask.sum(dim=(1, 2))  # (B,)
        return counts