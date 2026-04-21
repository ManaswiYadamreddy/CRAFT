"""
interpret_codebook.py - CLIP-based labelling of CRAFT's region-specific codebooks.

For every (region, RQ-level, code_id) in the trained RegionAwareVQ, we:
  1. gather image patches that activated the code,
  2. embed them with CLIP,
  3. pick the top-scoring natural-language description from the per-region
     vocabulary loaded from --vocab_json (e.g., data.json).

Outputs: code_labels.json, code_labels.md, and optional per-code patch grids.

Usage:
    python interpret_codebook.py \
        --craft_ckpt  /path/to/checkpoints/phase_d/final.pt \
        --parser_ckpt /path/to/pretrained/79999_iter.pth \
        --data_root   /path/to/data/train \
        --vocab_json  /path/to/data.json \
        --n_images    500 \
        --top_k       16 \
        --out_dir     eval_results/interpretability \
        --save_grids
"""
import argparse, json, os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

from transformers import CLIPModel, CLIPProcessor

from models.vqvae          import build_lq_vqvae
from models.region_aware_vq import RegionAwareVQ
from models.face_parser     import REGION_NAMES
from data.dataset           import FFHQPairedDataset


# ---------------------------------------------------------------------------
# Vocabulary loader - reads data.json and prepends a short prompt so
# CLIP treats each entry as an image caption.
# ---------------------------------------------------------------------------
def load_vocab(vocab_json_path, prompt_prefix="a close-up photo of "):
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab = {}
    for region, phrases in raw.items():
        if not isinstance(phrases, list) or not phrases:
            continue
        vocab[region] = [prompt_prefix + p for p in phrases]
        print(f"  vocab[{region}]: {len(vocab[region])} descriptions")
    return vocab


# ---------------------------------------------------------------------------
# Pass 1 - record code assignments.
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_code_assignments(model, loader, device, active_regions):
    ravq = model.quantizer
    assignments = {r: defaultdict(lambda: defaultdict(list)) for r in active_regions}
    images_kept = []
    global_idx = 0

    for batch in loader:
        hq   = batch["hq"].to(device)          # [-1, 1]
        hq01 = batch["hq_01"].to(device)       # [ 0, 1]
        B = hq.shape[0]

        z = model.encode(hq)                   # (B, 512, 16, 16)
        _, _, H, W = z.shape
        masks  = ravq.face_parser.get_region_masks(hq01, target_h=H, target_w=W)
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, ravq.e_dim)

        for name in active_regions:
            if name not in ravq.region_codebooks:
                continue
            mask_flat = masks[name].reshape(B, H * W)
            if mask_flat.sum() == 0:
                continue
            feats = z_flat[mask_flat]                                  # (N_r, 512)
            b_idx, p_idx = mask_flat.nonzero(as_tuple=True)
            rows = (p_idx // W).cpu().tolist()
            cols = (p_idx %  W).cpu().tolist()
            imgs = (b_idx + global_idx).cpu().tolist()

            # Replay ResidualVQ level-by-level on the unit sphere.
            rq       = ravq.region_codebooks[name]
            residual = F.normalize(feats.float(), dim=1)
            for lvl, vq in enumerate(rq.levels):
                emb  = F.normalize(vq.embedding.weight.float(), dim=1)
                sims = F.normalize(residual, dim=1) @ emb.t()
                max_sim, idx = sims.max(dim=-1)
                idx_l = idx.cpu().tolist()
                sim_l = max_sim.cpu().tolist()
                for i, cid in enumerate(idx_l):
                    assignments[name][lvl][cid].append(
                        (imgs[i], rows[i], cols[i], sim_l[i])
                    )
                residual = residual - vq.embedding(idx).float()

        images_kept.append(hq01.cpu())
        global_idx += B

    return assignments, torch.cat(images_kept, dim=0)


def crop_patch(image_01, row, col, cell=32, context=64):
    _, H, W = image_01.shape
    cy, cx = row * cell + cell // 2, col * cell + cell // 2
    y0 = max(0, cy - context // 2); y0 = min(y0, H - context)
    x0 = max(0, cx - context // 2); x0 = min(x0, W - context)
    return image_01[:, y0:y0 + context, x0:x0 + context]


# ---------------------------------------------------------------------------
# Pass 2 - CLIP scoring against the per-region vocabulary.
# ---------------------------------------------------------------------------
@torch.no_grad()
def label_codes_with_clip(
    assignments, images_01, vocab, device,
    clip_model_id="openai/clip-vit-large-patch14",
    top_k=16, context=64, min_hits=4,
):
    clip = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(clip_model_id)

    text_emb_per_region = {}
    for r, phrases in vocab.items():
        txt = proc(text=phrases, return_tensors="pt", padding=True).to(device)
        emb = F.normalize(clip.get_text_features(**txt), dim=-1).float()
        text_emb_per_region[r] = emb

    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                              device=device).view(1, 3, 1, 1)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                              device=device).view(1, 3, 1, 1)

    labels = {}
    for region, region_levels in assignments.items():
        if region not in vocab:
            print(f"  [skip] no vocab for region '{region}'")
            continue
        phrases  = vocab[region]
        text_emb = text_emb_per_region[region]
        labels[region] = {}
        for lvl, code_dict in region_levels.items():
            labels[region][lvl] = {}
            for code_id, hits in code_dict.items():
                if len(hits) < min_hits:
                    labels[region][lvl][code_id] = {
                        "label": "<insufficient-data>",
                        "score": 0.0, "n_hits": len(hits),
                    }
                    continue

                hits.sort(key=lambda x: -x[3])
                top = hits[:top_k]
                patches = torch.stack([
                    crop_patch(images_01[i], r, c, context=context)
                    for (i, r, c, _) in top
                ]).to(device)
                patches = F.interpolate(patches, size=(224, 224),
                                         mode="bilinear", align_corners=False)
                patches = (patches - clip_mean) / clip_std

                img_emb  = F.normalize(
                    clip.get_image_features(pixel_values=patches), dim=-1
                ).float()
                code_emb = F.normalize(img_emb.mean(dim=0, keepdim=True), dim=-1)
                scores   = (code_emb @ text_emb.t()).squeeze(0)
                best     = scores.argmax().item()
                topn     = scores.topk(min(5, scores.numel()))
                labels[region][lvl][code_id] = {
                    "label":  phrases[best],
                    "score":  float(scores[best].item()),
                    "n_hits": len(hits),
                    "top_k":  int(patches.shape[0]),
                    "top5":   [(phrases[i.item()], float(s.item()))
                               for s, i in zip(topn.values, topn.indices)],
                }
    return labels


def save_code_grids(assignments, images_01, out_dir, top_k=16, context=64):
    for region, region_levels in assignments.items():
        for lvl, code_dict in region_levels.items():
            region_dir = os.path.join(out_dir, "patch_grids", region, f"level_{lvl}")
            os.makedirs(region_dir, exist_ok=True)
            for code_id, hits in code_dict.items():
                if not hits: continue
                hits.sort(key=lambda x: -x[3])
                top     = hits[:top_k]
                patches = torch.stack([
                    crop_patch(images_01[i], r, c, context=context)
                    for (i, r, c, _) in top
                ])
                grid = make_grid(patches,
                                 nrow=int(np.ceil(np.sqrt(len(top)))),
                                 padding=1, pad_value=1.0)
                save_image(grid,
                    os.path.join(region_dir, f"code_{code_id:04d}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft_ckpt",  required=True)
    ap.add_argument("--parser_ckpt", required=True)
    ap.add_argument("--data_root",   required=True)
    ap.add_argument("--vocab_json",  required=True,
                    help="Path to data.json with {region: [descriptions]}.")
    ap.add_argument("--n_images",   type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--top_k",      type=int, default=16)
    ap.add_argument("--context",    type=int, default=64)
    ap.add_argument("--min_hits",   type=int, default=4)
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out_dir",    default="eval_results/interpretability")
    ap.add_argument("--save_grids", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Vocabulary (from data.json) ---
    print(f"Loading vocabulary from {args.vocab_json}")
    vocab = load_vocab(args.vocab_json)
    active_regions = [r for r in REGION_NAMES if r in vocab]
    print(f"Active regions: {active_regions}")

    # --- Load CRAFT LQ model ---
    ckpt  = torch.load(args.craft_ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("lq_model"))
    has_mag = any("magnitude_head" in k for k in state.keys())
    ravq  = RegionAwareVQ(
        e_dim=512, n_levels=3,
        parser_ckpt=args.parser_ckpt,
        use_magnitude_head=has_mag,
    )
    model = build_lq_vqvae(ravq, embed_dim=512)
    model.load_state_dict(state)
    model = model.to(device).eval()

    # --- Data ---
    ds = FFHQPairedDataset(args.data_root, hq_only=True, masks_folder="")
    if args.n_images < len(ds):
        ds = Subset(ds, range(args.n_images))
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    # --- Pass 1: assignments ---
    print("[1/2] Collecting codebook assignments...")
    assignments, images_01 = collect_code_assignments(
        model, loader, device, active_regions
    )
    assignments = {r: {l: dict(d) for l, d in lvls.items()}
                   for r, lvls in assignments.items()}

    # --- Pass 2: CLIP labels ---
    print("[2/2] Scoring codes with CLIP...")
    labels = label_codes_with_clip(
        assignments, images_01, vocab, device,
        clip_model_id=args.clip_model,
        top_k=args.top_k, context=args.context,
        min_hits=args.min_hits,
    )

    # --- Export ---
    with open(os.path.join(args.out_dir, "code_labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    with open(os.path.join(args.out_dir, "code_labels.md"), "w", encoding="utf-8") as f:
        for region in active_regions:
            if region not in labels: continue
            f.write(f"## {region}\n\n")
            for lvl in sorted(labels[region].keys()):
                f.write(f"### Level {lvl}\n\n")
                items = sorted(labels[region][lvl].items(),
                               key=lambda kv: -kv[1]["n_hits"])
                for code_id, info in items:
                    f.write(f"- code **{code_id:4d}** "
                            f"({info['n_hits']:>5d} hits, "
                            f"CLIP={info['score']:.3f}) -> *{info['label']}*\n")
                f.write("\n")

    if args.save_grids:
        print("Saving per-code patch grids...")
        save_code_grids(assignments, images_01, args.out_dir,
                        top_k=args.top_k, context=args.context)
    print(f"Done -> {args.out_dir}")


if __name__ == "__main__":
    main()
