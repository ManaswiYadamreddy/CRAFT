# Code-swap reconstruction experiment

A causal interpretability test for CRAFT's region-aware codebook. Both
scripts in this folder are **standalone** — they do not import from any
other interpretability tool (CLIP / atlas / panel). They depend only on
`models/` in the repo root.

## What it does

For each face:

1. Encode → quantize → decode normally → **baseline reconstruction**.
2. For each region, replace **all** indices at one RQ level with one
   *donor code id* (a different code from the same region's codebook).
   The other regions stay untouched.
3. Decode again → **swap reconstruction**.
4. Save side-by-side panels: `[HQ | LQ |] baseline | donor_1 | … | donor_N`.

If a single code id genuinely encodes a visual property, only that region of
the reconstruction should change after a swap.

## Files

- `code_swap.py` — single-image, Stage-1 reconstruction.
- `code_swap_batch.py` — batch over a folder of LQ images, Stage-1 reconstruction.
- `code_swap_batch_stage2.py` — batch with the **Stage-2 diffusion generator**
  doing the reconstruction (LQ encoder + region-aware VQ + swap → SD-1.5/2.x
  one-step diffusion → I_hat). Uses your trained Stage-2 LoRA + projector.

All three are self-contained — pick whichever matches what you want to show.

## How to read the output

| Outcome | Interpretation |
|---|---|
| Targeted region clearly changes; others don't | The code id carries that property. **Codebook is doing real work.** |
| Reconstruction identical to baseline | Code id was redundant — pre-quantization features already encoded the answer. |
| Reconstruction garbled / distorted | Codebook is not smoothly swappable; codes only "work" in their original feature context. |

Any of those is a publishable result.

## Output layout (per image, under `<out_dir>/<stem>/`)

```
original_lq.png             input  (resized to 512×512)
original_hq.png             ground truth (only if --hq_dir is given)
recon_baseline.png          baseline reconstruction (no swap)
swap_panel_eyes.png         HQ | LQ | baseline | donor_1 | … panel
swap_panel_lips.png         (same)
swap_panel_hair.png         (same)
swap_panel_skin.png         (same)
swap_indices.json           donors used + original code-id grid per region
```

(For single-image `code_swap.py` the layout is identical, just at the level
of `--out_dir` directly instead of `<out_dir>/<stem>/`.)

## Single image

```bash
python code_swap/code_swap.py \
    --stage1_ckpt /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
    --parser_ckpt /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
    --image       /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2/00219.png \
    --regions     eyes lips hair \
    --level       0 \
    --n_donors    4 \
    --out_dir     /projectnb/cs585/projects/craft/code_swap_outputs/face_00219
```

## Batch (multiple images)

LQ ↔ HQ filenames are assumed identical (no prefix/suffix mangling).

```bash
python code_swap/code_swap_batch.py \
    --stage1_ckpt /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
    --parser_ckpt /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
    --input_dir   /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2/ \
    --hq_dir      /projectnb/cs585/projects/craft/data/test/CelebA/CelebA_Validation/celeba_512_validation/ \
    --out_dir     /projectnb/cs585/projects/craft/code_swap_outputs/celeba_lq \
    --regions     eyes lips hair \
    --level       0 \
    --n_donors    4 \
    --max_images  20 \
    --skip_existing
```

`--hq_dir` is optional — drop it if you only have LQ. The panel just
collapses to `LQ | baseline | donor_1 | …` when HQ isn't found.

## Batch with the Stage-2 diffusion generator

Same swap logic, but each cell of the panel is now a one-step SD
reconstruction conditioned on the (possibly-swapped) visual prompt — i.e.
"what does the diffusion model paint when the codebook routes are nudged?"

```bash
python code_swap/code_swap_batch_stage2.py \
    --stage1_ckpt   /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
    --stage2_ckpt   /projectnb/cs585/projects/craft/checkpoints_stage2/final.pt \
    --parser_ckpt   /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
    --input_dir     /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2/ \
    --hq_dir        /projectnb/cs585/projects/craft/data/test/CelebA/CelebA_Validation/celeba_512_validation/ \
    --out_dir       /projectnb/cs585/projects/craft/code_swap_outputs/celeba_lq_stage2 \
    --regions       eyes lips hair \
    --level         0 \
    --n_donors      4 \
    --max_images    20 \
    --use_hq_for_parser \
    --skip_existing
```

If your Stage-2 training used SD 2.1 instead of SD 1.5, override the SD
defaults to match:

```bash
    --pretrained_model stabilityai/stable-diffusion-2-1-base \
    --context_dim      1024 \
```

The Stage-2 swap pipeline is the *interesting* one: it asks "does swapping
a code id at position p change what the diffusion model paints in region p,
or does the LoRA-tuned UNet wash the swap out?" Both possible outcomes are
informative.

## Knobs

- `--regions eyes lips hair skin` — which regions to swap. `bg` is omitted
  by default (it's a residual codebook with no clean semantic role).
- `--level {0, 1, 2}` — which RQ level to swap. Level 0 (coarsest) carries
  the most semantic content; level 2 is texture detail. Try 0 first.
- `--n_donors 4` — how many donor code ids to test per region. By default
  donors are the **top-N most-used codes** at that level (high EMA count =
  stable, frequently-selected codes).
- `--donor_codes "17,42,108"` — explicit donor ids (overrides `--n_donors`).
- `--max_images 20` — cap how many images to process (good for a quick
  sanity check before running on the whole set).
- `--skip_existing` — skip images whose output folder already has
  `swap_indices.json`. Useful when re-running.

## Suggested experiment for the report

1. Run the batch script with default donors on ~20 LQ faces.
2. Skim the per-region panels — pick 2–3 faces where the eye / lip swaps
   visibly change the targeted region. Those are the figures.
3. Caption: *"Replacing a single discrete code id induces a localised,
   region-correct change in the reconstruction — evidence that the
   codebook itself, not just the routing, carries visual semantics."*

This is what turns the interpretability story from correlational ("patches
under code 17 look like blue eyes") into causal ("setting the code id
changes the reconstruction in the corresponding region").
