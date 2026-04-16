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
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_stage1.py --config configs/train.yaml --phase A

# Phase B only (needs Phase A checkpoint)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_stage1.py --config configs/train.yaml --phase B \
    --hq_ckpt checkpoints/phase_a/final.pt

# Phase C only (needs both checkpoints)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_stage1.py --config configs/train.yaml --phase C \
    --hq_ckpt checkpoints/phase_a/final.pt \
    --lq_ckpt checkpoints/phase_b/final.pt

# All three phases sequentially
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_stage1.py --config configs/train.yaml --phase all
```

## Training OSDFace Stage 1 (VRE)

OSDFace Stage 1 (paper §3.2) trains a flat HQ VQ-VAE for 50 epochs, then a
flat LQ VQ-VAE for 10 epochs with `λ_assoc=0`, and finally an additional
10 epochs with `λ_assoc=1`. The training therefore splits into the same
A / B / C structure as CRAFT, with the same HQ encoder + global codebook
acting as the frozen "tokenizer" during the LQ phases.

| Phase | What it trains | λ_assoc | Epochs |
|-------|----------------|---------|--------|
| A (HQ) | HQ encoder + HQ codebook + HQ decoder | n/a | 50 |
| B (LQ) | LQ encoder + LQ codebook + LQ decoder, HQ frozen | 0 | 10 |
| C (LQ + assoc) | continues B with HQ↔LQ association loss | 1 | 10 |

**You do not need to retrain Phase A for OSDFace.** CRAFT Phase A is already
"HQ `MultiHeadEncoder`/`MultiHeadDecoder` + `GlobalVQ` (unit sphere, cosine)",
which is exactly what OSDFace Stage 1's HQ branch is in this codebase. The
state-dict keys are identical because both are built by the same
`build_hq_vqvae()` factory.

A standalone Phase A runner is provided for completeness (so the OSDFace
baseline can be trained end-to-end without any CRAFT artifacts), but
**`--phase all` skips Phase A by default** and reuses `hq_ckpt` (CRAFT's
Phase A `final.pt`) as the frozen HQ branch. To explicitly retrain the HQ
branch under the OSDFace pipeline, use `--phase A`.

### Phases B and C sequentially (recommended, paper-equivalent)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_osdface_stage1.py --config configs/train_osdface.yaml --phase all
```

(`--phase all` reuses `hq_ckpt` and runs only Phases B + C — 10 + 10 epochs.)

### Phase A only (retrain the HQ branch from scratch under OSDFace)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_osdface_stage1.py --config configs/train_osdface.yaml --phase A
```

This trains the HQ VQ-VAE on HQ self-reconstruction for `hq_epochs` (paper:
50). The resulting `checkpoints_osdface/phase_a/final.pt` can then be used
as the `hq_ckpt` for Phases B and C. Skip this command unless you
specifically want a Phase A trained without any CRAFT involvement — by
default `--phase all` already reuses CRAFT's Phase A.

### Phase B only

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_osdface_stage1.py --config configs/train_osdface.yaml --phase B
```

### Phase C only (continues from Phase B checkpoint)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_osdface_stage1.py --config configs/train_osdface.yaml --phase C \
    --lq_ckpt checkpoints_osdface/phase_b/final.pt
```

### Key config knobs

| Key | Default | Meaning |
|---|---|---|
| `hq_epochs` | 50 | Phase A epochs (paper: 50, only used by `--phase A`) |
| `lq_epochs` | 10 | Phase B epochs (paper: 10) |
| `assoc_epochs` | 10 | Phase C epochs (paper: 10) |
| `hq_n_codes` / `lq_n_codes` | 1024 | Codebook sizes (paper: 1024 each) |
| `lambda_per` | 1.0 | Perceptual weight |
| `lambda_dis` | 0.8 | Discriminator weight |
| `lambda_assoc` | 1.0 | Association-loss weight (Phase C) |
| `hq_ckpt` | CRAFT Phase A path | Frozen HQ branch checkpoint |
| `lq_ckpt` | "" | Phase B checkpoint, required for `--phase C` |

Checkpoints are written under `ckpt_dir` (default `checkpoints_osdface/`):

```
checkpoints_osdface/
  phase_a/{latest.pt, epoch_XXX.pt, final.pt}   # only if you ran --phase A
  phase_b/{latest.pt, epoch_XXX.pt, final.pt}
  phase_c/{latest.pt, epoch_XXX.pt, final.pt}
```

Each checkpoint contains the LQ branch state dict, the discriminator, and
the optimizer/scheduler/scaler state, so any phase can be resumed from
`latest.pt` automatically.

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
