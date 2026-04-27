# Code-swap reconstruction experiment

A causal interpretability test for CRAFT's region-aware codebook.

## What it does

For a single face:

1. Encode → quantize → decode normally → **baseline reconstruction**.
2. For each region, replace **all** indices at one RQ level with one *donor code id* (a different code from the same region's codebook). Decode again → **swap reconstruction**.
3. Save side-by-side panels: `baseline | swap_donor_1 | swap_donor_2 | …` per region.

The other regions stay untouched — only the chosen region's chosen level is swapped. So if a single code id genuinely encodes a visual property, only that region of the reconstruction should change.

## How to read the output

| Outcome | Interpretation |
|---|---|
| The targeted region clearly changes (e.g. eye color, lip shape), other regions don't | The code id carries that property. **Codebook is doing real work.** |
| Reconstruction is identical to baseline | The code id was redundant — meaning was already baked into pre-quantization features. |
| Reconstruction is garbled / distorted | The codebook is not smoothly swappable; codes only "work" in their original feature context. |

Any of those is a publishable result.

## Files written under `--out_dir`

```
original.png                  Input face (resized to 512×512)
recon_baseline.png            Original reconstruction (no swap)
swap_panel_eyes.png           baseline | donor_1 | donor_2 | … panel for eyes
swap_panel_lips.png           same for lips
swap_panel_hair.png           same for hair
swap_panel_skin.png           same for skin
swap_indices.json             Donor ids tried, top-K most-used codes per region,
                              and original code-id grid per region.
```

## Example command

```bash
python code_swap/code_swap.py \
    --stage1_ckpt /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
    --parser_ckpt /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
    --image       /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2/00219_LQ.png \
    --regions     eyes lips hair \
    --level       0 \
    --n_donors    4 \
    --out_dir     /projectnb/cs585/projects/craft/code_swap_outputs/face_00219
```

## Knobs

- `--regions eyes lips hair skin` — which regions to swap (skips `bg` by default; bg is a residual codebook).
- `--level {0,1,2}` — which RQ level to swap. Level 0 (coarsest) carries the most semantic content. Level 2 (finest) is texture detail. Try 0 first.
- `--n_donors 4` — how many donor code ids to test per region. Donors default to the **top-N most-used codes** at that level (high EMA count = stable/used codes that matter visually).
- `--donor_codes "17,42,108"` — explicit donor ids (overrides `--n_donors`). Useful if you've used `code_face_atlas.py` to find labelled codes (e.g. "blue-eye-code") and want to swap to a known-meaning donor.

## Suggested experiment for the report

1. Pick 3 representative faces.
2. Run with `--level 0`, default donors (top-4 most-used codes per region).
3. Run a second time on the same faces with `--level 0 --regions eyes` and explicit `--donor_codes` from your atlas (e.g. one "blue-eye" code id and one "dark-eye" code id, taken from `code_atlas_labels.json`).
4. Add the resulting panels (3 faces × 1–2 regions) as a single figure. Caption: *"Replacing a single discrete code id induces a localised, region-correct change in the reconstruction — evidence that the codebook itself, not just the routing, carries visual semantics."*

This is the figure that turns the interpretability story from correlational ("patches under code 17 look like blue eyes") into causal ("setting code id to 17 produces blue eyes; setting it to 42 produces dark eyes").

## Quick sanity check before running

If your `code_face_atlas.py` outputs already exist at e.g.
`/projectnb/cs585/projects/craft/clip_interpretability/code_face_atlas/code_atlas_labels.json`,
you can grep that JSON for atlas-labelled donor ids:

```bash
python -c "
import json
labels = json.load(open('/projectnb/cs585/projects/craft/clip_interpretability/code_face_atlas/code_atlas_labels.json'))
for row in labels['eyes'][:5]:
    print(row)"
```

Pick two donors with semantically different labels (e.g. one "wide-set eyes",
one "dark brown eyes") and pass them as `--donor_codes` for the cleanest figure.
