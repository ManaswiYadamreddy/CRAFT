# CRAFT 

## Stage 1

This repository contains:

1. **CRAFT** — our hierarchical region-aware VQ-VAE face-restoration model.
2. **OSDFace Stage 1 (VRE)** — a reimplementation of OSDFace's Visual
   Representation Embedder, wired into this same codebase as an
   apples-to-apples baseline.

Both pipelines share the same encoder/decoder architecture, the same dataset
loader, the same losses, and — critically — the **same unit-sphere codebook
with cosine similarity** (CRAFT's `GlobalVQ`). This differs from the raw-space
L2 quantizer in the original OSDFace paper, but it's what lets the two models
be compared on the same footing.

<!-- 
## Repository layout

```
configs/
  train.yaml              # CRAFT Stage 1 config
  train_osdface.yaml      # OSDFace Stage 1 config  (new)
data/dataset.py           # FFHQPairedDataset         (shared)
losses/losses.py          # Stage1VQLoss, AssociationLoss, PatchDiscriminator  (shared)
models/vqvae.py           # MultiHeadEncoder/Decoder, GlobalVQ, build_hq_vqvae (shared)
models/region_aware_vq.py # CRAFT-only region-aware RQ
models/face_parser.py     # CRAFT-only BiSeNet wrapper

train_stage1.py           # CRAFT Stage 1 trainer (Phases A / B / C)
train_osdface_stage1.py   # OSDFace Stage 1 trainer  (new)
``` -->


## Data layout

Both trainers expect the same directory structure:

```
<data_root>/
  images512x512/       # HQ FFHQ images (PNG)
  LQ_images_512x512/   # degraded versions, named <stem>_LQ.png
  masks_16x16/         # (optional) pre-computed face-parsing masks for CRAFT Phase B/C
```

## Training CRAFT

CRAFT Stage 1 has three phases:

| Phase | What it trains | Config key |
|-------|----------------|------------|
| A | HQ encoder + global codebook + HQ decoder (self-recon on HQ) | `hq_epochs` |
| B | LQ encoder + region-aware RQ + LQ decoder (self-recon on LQ), HQ frozen | `lq_epochs` |
| C | Continues B with HQ↔LQ association loss enabled | `assoc_epochs` |

```bash
# Phase A only (HQ pretraining)
python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True train_stage1.py --config configs/train.yaml --phase A

# Phase B only (needs Phase A checkpoint)
python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True train_stage1.py --config configs/train.yaml --phase B \
    --hq_ckpt checkpoints/phase_a/final.pt

# Phase C only (needs both checkpoints)
python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True train_stage1.py --config configs/train.yaml --phase C \
    --hq_ckpt checkpoints/phase_a/final.pt \
    --lq_ckpt checkpoints/phase_b/final.pt

# All three phases sequentially
python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True train_stage1.py --config configs/train.yaml --phase all
```

## Training OSDFace Stage 1 (VRE)

OSDFace Stage 1 trains **two flat VQ-VAEs jointly** — one HQ branch and one
LQ branch — aligned by an HQ↔LQ feature association loss (OSDFace Eq. 9–10).
Both codebooks live on the unit sphere and quantize via cosine similarity, so
the quantizer geometry matches CRAFT exactly.

**You do not need to retrain Phase A for OSDFace.** CRAFT Phase A is already
"HQ `MultiHeadEncoder`/`MultiHeadDecoder` + `GlobalVQ` (unit sphere, cosine)",
which is exactly what OSDFace Stage 1's HQ branch is in this codebase. The
state-dict keys are identical because both are built by the same
`build_hq_vqvae()` factory.

By default, the OSDFace trainer **warm-starts the HQ branch from CRAFT's
Phase A checkpoint and freezes it**, so only the LQ branch + association
loss train. This is the cleanest apples-to-apples comparison against CRAFT
Phase B/C.

### Default run (recommended)

```bash
python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True train_osdface_stage1.py --config configs/train_osdface.yaml
```

This picks up `hq_ckpt` and `freeze_hq: true` from the config file.

### Override the HQ checkpoint path

```bash
python train_osdface_stage1.py --config configs/train_osdface.yaml \
    --hq_ckpt /path/to/your/phase_a/final.pt
```

### Joint HQ+LQ training (closer to the original OSDFace paper)

```bash
python train_osdface_stage1.py --config configs/train_osdface.yaml --no_freeze_hq
```

### Key config knobs

| Key | Default | Meaning |
|---|---|---|
| `epochs` | 50 | Total Stage 1 epochs |
| `hq_n_codes` / `lq_n_codes` | 1024 | Codebook sizes |
| `lambda_per` | 1.0 | Perceptual weight |
| `lambda_dis` | 0.8 | Discriminator weight |
| `lambda_assoc` | 1.0 | Association-loss weight after warmup |
| `assoc_warmup_epochs` | 5 | Keep λ_assoc=0 for this many epochs |
| `hq_ckpt` | CRAFT Phase A path | HQ branch warm-start checkpoint (defaults to CRAFT Phase A `final.pt`) |
| `freeze_hq` | true | Freeze HQ branch entirely (pass `--no_freeze_hq` to train it jointly) |

Checkpoints are written to `ckpt_dir` (default `checkpoints_osdface/`) as
`latest.pt`, `epoch_XXX.pt`, and `final.pt`. Each file contains both
`hq_model` and `lq_model` state dicts plus both discriminators, so you can
resume with just the config.

## Notes on comparability

- **Quantizer:** both models use `GlobalVQ` (unit-sphere L2-normalized
  codebook, cosine-similarity argmax, EMA updates, entropy regularization).
  Raw-space L2 is **not** used anywhere.
- **Architecture:** both use OSDFace's `MultiHeadEncoder`/`MultiHeadDecoder`
  at 512×512, latent 16×16×512.
- **Losses:** L1 + VGG perceptual + PatchGAN + VQ commitment/entropy, plus
  HQ↔LQ `AssociationLoss` (OSDFace Eq. 9–10) on pre-quant features.
- The only axis that varies between the two pipelines is what CRAFT adds on
  top of the shared baseline (region-aware RQ, face-parsing conditioning,
  etc.).
