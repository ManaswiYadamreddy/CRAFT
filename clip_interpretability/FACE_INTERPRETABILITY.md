# Face-level CLIP interpretability

Two new scripts produce per-face interpretability images on the **test set**,
complementing the existing aggregate heatmaps in this folder
(`clip_codebook_*.{png,json,md}`) and the cross-region matrices from
`visualize_clip_cross_region.py`.

Neither script modifies any existing file. Both reuse the same code paths as
`cross_region_clip_eval.py` / `text_interpretability.py` (encode -> region
masks -> native residual VQ replay -> CLIP centroids), but produce
*per-face*, not aggregate, outputs.

| Script | Question it answers | Output |
|---|---|---|
| `face_codebook_panel.py` | "On a real test face, does each region's codebook fire on its own region and stay quiet elsewhere?" | One panel per face: original + segmentation + reconstruction + 5 spatial CLIP heatmaps (one per codebook) overlaid on the face with the parser region outlined, plus top-N text labels per codebook from `data.json`. |
| `code_face_atlas.py` | "What does each individual code in each region's codebook actually represent on faces?" | One atlas per region (rows = top-K most-used codes, cols = top activating face crops) plus a combined overview, with each row labelled by its best CLIP text match from `data.json`. |

Both scripts:

- Run on the **test split** (pass `--test_root`).
- Optionally read the existing aggregate `clip_codebook_summary.json`
  (`face_codebook_panel.py` only — it prints the precomputed
  diagonal-dominance numbers in each panel caption for context).
- Pull text labels from `data.json` (the same vocabulary used by
  `text_interpretability.py`).
- Save a small JSON describing the run (`face_panel_run.json` /
  `code_atlas_labels.json`) so labels can be reused without rerunning.

## `face_codebook_panel.py`

```bash
python face_codebook_panel.py \
    --craft_ckpt        /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
    --parser_ckpt       /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
    --test_root         /projectnb/cs585/projects/craft/data/test \
    --vocab_json        /projectnb/cs585/projects/craft/data/data.json \
    --existing_summary  clip_interpretability/clip_codebook_summary.json \
    --n_faces           16 \
    --bank_images       200 \
    --top_k_codes       16 \
    --top_k_texts       3 \
    --out_dir           /projectnb/cs585/projects/craft/clip_face_panels
```

Each `face_panel_<idx>_<stem>.png` contains:

```
+--------------+--------------+--------------+
|   Original   | BiSeNet regs | RAVQ recon   |
+--------------+--------------+--------------+
| eyes heat | skin heat | hair heat | lips heat | bg heat |
+-----------+-----------+-----------+-----------+---------+

Top CLIP text matches per codebook (from data.json):
  eyes:  "..." [src, sim] ; ...
  skin:  ...
  ...
```

Each row-2 heatmap is the spatial CLIP cosine similarity from the cell's face
patch to the **best code in that codebook**. The face-parser outline for the
panel's region is drawn on top in the matching color so you can immediately
see whether high-similarity cells fall inside the native region.

`--bank_images` controls how many test images go into building the per-code
CLIP centroids. The `--n_faces` rendered are taken from the **end** of the
test set so they don't overlap the bank.

## `code_face_atlas.py`

```bash
python code_face_atlas.py \
    --craft_ckpt    /projectnb/cs585/projects/craft/checkpoints/phase_d/final.pt \
    --parser_ckpt   /projectnb/cs585/projects/craft/pretrained/79999_iter.pth \
    --test_root     /projectnb/cs585/projects/craft/data/test \
    --vocab_json    /projectnb/cs585/projects/craft/data/data.json \
    --bank_images   300 \
    --top_k_codes   10 \
    --top_n_patches 8 \
    --context       80 \
    --out_dir       /projectnb/cs585/projects/craft/clip_code_atlas
```

Outputs:

- `atlas_<region>.png` for each region — top-K most-used codes (rows) with
  their top activating face patches (cols). Leftmost column shows
  `L<level> c<id>`, usage count, and the best matching text phrase
  (`"thick dark lashes" [eyes, sim=0.31]`).
- `atlas_overview.png` — three top codes per region in one figure
  (handy for slides / the project report).
- `code_atlas_labels.json` — machine-readable label dump for downstream use.

## Reading the figures

If region-aware codebooks are working you should see, in
`face_codebook_panel.py`:

- **Diagonal panel bright inside its outline.** The `eyes` heatmap should
  be saturated where the eyes outline is drawn and dim everywhere else;
  same for `skin`, `hair`, `lips`. `bg` should be brightest on
  background/clothing/jewellery.
- **Cross panels dim across the whole face.** A high value of, say, the
  `lips` heatmap on the eye region would mean the lips codebook is
  representing eye-like content — i.e., region specificity has broken.
- **Text labels diagonal-consistent.** The top text phrase for each
  codebook should come from that region's own vocab in `data.json`
  (the `[src, sim]` tag will say `[eyes, ...]` for the eyes codebook etc.)

In `code_face_atlas.py`:

- **Per-row visual coherence.** All eight crops in a row should look like
  the same kind of patch (e.g., a row of "thick dark eyebrows" crops,
  or a row of "freckled skin" crops).
- **Labels match the visuals.** The text label printed on the left of
  each row should describe what you see in the crops.

If a row is visually heterogeneous or the label is from another region,
that code is poorly specialised and is a place to investigate.
