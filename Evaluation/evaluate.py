"""
evaluate.py — OSDFace-paper metrics for CRAFT and OSDFace restorations.

Computes every metric reported in the OSDFace paper (Tables 1 & 2):

    Full-reference (require paired HQ ground truth):
        LPIPS ↓, DISTS ↓, PSNR ↑, SSIM ↑,
        Deg. ↓  (ArcFace angular distance),
        LMD  ↓  (68-point landmark mean L2 distance, at 512×512 scale).

    No-reference (restored image only):
        MUSIQ ↑, NIQE ↓, CLIPIQA ↑, MANIQA ↑

    Distribution-level:
        FID(FFHQ) ↓   (restored vs FFHQ HQ set)
        FID(HQ)   ↓   (restored vs paired-HQ set; only if --hq_dir given)

Inputs:
    --restored_dir   folder of restored 512×512 PNGs (outputs of a model)
    --hq_dir         (optional) folder of HQ GT images, matched by filename stem
    --ffhq_dir       (optional) folder of FFHQ HQ images for FID reference
    --out_json       where to write the aggregate + per-image metrics

Typical usage:
    # CelebA-Test (paired)
    python -m Evaluation.evaluate \
        --restored_dir results/craft/celeba \
        --hq_dir       /projectnb/cs585/projects/craft/data/test/CelebA/CelebA_Validation/celeba_512_validation \
        --ffhq_dir     /projectnb/cs585/projects/craft/data/train/images512x512 \
        --out_json     results/craft/celeba/metrics.json

    # LFW-Test (no GT)
    python -m Evaluation.evaluate \
        --restored_dir results/craft/lfw \
        --ffhq_dir     /projectnb/cs585/projects/craft/data/train/images512x512 \
        --out_json     results/craft/lfw/metrics.json

Matching LQ filenames (e.g. "00001_LQ.png") to HQ filenames (e.g. "00001.png")
is handled automatically: common suffixes (_LQ, _lq, _bicubic) are stripped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from Evaluation.metrics.iqa      import IQAMetrics
from Evaluation.metrics.identity import IdentityDistance
from Evaluation.metrics.landmark import LandmarkDistance
from Evaluation.metrics.fid      import FIDScorer


IMG_EXTS = (".png", ".jpg", ".jpeg")
_LQ_SUFFIX_RE = re.compile(r"(_LQ|_lq|_bicubic|_BICUBIC|_deg|_DEG)$")


# ──────────────────────────────────────────────────────────────────────────
# Filename matching: restored stem  ↔  HQ stem
# ──────────────────────────────────────────────────────────────────────────

def _norm_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    return _LQ_SUFFIX_RE.sub("", stem)


def _index_dir(d: str) -> dict[str, str]:
    """Map normalised stem → absolute path for all images in `d`."""
    out: dict[str, str] = {}
    if not d or not os.path.isdir(d):
        return out
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        if os.path.splitext(p)[1].lower() in IMG_EXTS:
            out[_norm_stem(p)] = p
    return out


# ──────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────

def _load_img_01(path: str, resolution: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0            # (H, W, 3)
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    if t.shape[-2:] != (resolution, resolution):
        t = F.interpolate(t, size=(resolution, resolution),
                          mode="bilinear", align_corners=False)
    return t.clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── list restored images ────────────────────────────────────────────
    restored_map = _index_dir(args.restored_dir)
    if not restored_map:
        raise FileNotFoundError(f"No images in --restored_dir {args.restored_dir}")
    hq_map = _index_dir(args.hq_dir) if args.hq_dir else {}

    have_hq = bool(hq_map)
    if args.hq_dir and not have_hq:
        print(f"[WARN] --hq_dir {args.hq_dir} has no images; full-ref skipped.")

    # Build aligned pairs (or solo list)
    stems = sorted(restored_map.keys())
    if have_hq:
        matched = [(s, restored_map[s], hq_map[s]) for s in stems if s in hq_map]
        missing = [s for s in stems if s not in hq_map]
        print(f"Matched {len(matched)}/{len(stems)} restored images to HQ.")
        if missing:
            print(f"  {len(missing)} restored images had no HQ match "
                  f"(e.g. {missing[:3]}) — they'll get no-ref metrics only.")
        extra_nohq = [(s, restored_map[s], None) for s in missing]
        items = matched + extra_nohq
    else:
        items = [(s, restored_map[s], None) for s in stems]

    # ── instantiate metric modules ──────────────────────────────────────
    print("\nLoading metric models...")
    iqa = IQAMetrics(
        device=device,
        full_ref=have_hq,
        no_ref=True,
        metrics=args.metrics or None,
    )
    print(f"  IQA loaded: {list(iqa.metrics.keys())}")
    id_metric = IdentityDistance(device=device) if have_hq else None
    if id_metric is not None:
        print("  ArcFace identity (Deg.) loaded")
    lmk_metric = LandmarkDistance(device=device) if have_hq else None
    if lmk_metric is not None:
        print("  FAN 68-point landmark (LMD) loaded")

    # ── per-image loop ─────────────────────────────────────────────────
    per_image: list[dict] = []
    print(f"\nScoring {len(items)} images...")
    for stem, r_path, h_path in tqdm(items, desc="metrics"):
        row: dict = {"name": stem, "restored": r_path, "hq": h_path}
        r_01 = _load_img_01(r_path, args.resolution).to(device)
        h_01 = _load_img_01(h_path, args.resolution).to(device) if h_path else None

        # IQA metrics (both full-ref and no-ref applicable)
        row.update(iqa.compute(r_01, h_01))

        # Identity
        if id_metric is not None and h_01 is not None:
            row.update(id_metric.compute(r_01, h_01))

        # Landmarks
        if lmk_metric is not None and h_01 is not None:
            lm = lmk_metric.compute(r_01, h_01)
            row["lmd"] = lm["lmd"]

        per_image.append(row)

    # ── dataset-level FID ──────────────────────────────────────────────
    fid_scores: dict[str, float] = {}
    if args.ffhq_dir or args.hq_dir:
        print("\nComputing FID (dataset-level)...")
        fid_scorer = FIDScorer(device=device)
        if args.ffhq_dir:
            fid_scores.update(
                fid_scorer.compute(args.restored_dir, args.ffhq_dir, "fid_ffhq")
            )
            print(f"  FID(FFHQ) = {fid_scores.get('fid_ffhq', float('nan')):.4f}")
        if args.hq_dir:
            fid_scores.update(
                fid_scorer.compute(args.restored_dir, args.hq_dir, "fid_hq")
            )
            print(f"  FID(HQ)   = {fid_scores.get('fid_hq', float('nan')):.4f}")

    # ── aggregate ──────────────────────────────────────────────────────
    def _mean(key: str) -> Optional[float]:
        vals = [r[key] for r in per_image
                if key in r and r[key] is not None
                and isinstance(r[key], (int, float))
                and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else None

    def _std(key: str) -> Optional[float]:
        vals = [r[key] for r in per_image
                if key in r and r[key] is not None
                and isinstance(r[key], (int, float))
                and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.std(vals)) if vals else None

    # Collect every scalar key that appeared in per_image
    all_keys: set[str] = set()
    for r in per_image:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                all_keys.add(k)

    summary: dict[str, dict] = {}
    for k in sorted(all_keys):
        mu = _mean(k)
        if mu is None:
            continue
        summary[k] = {"mean": round(mu, 6), "std": round(_std(k) or 0.0, 6),
                      "n": sum(1 for r in per_image if k in r)}
    summary.update({k: {"value": round(v, 6)} for k, v in fid_scores.items()})

    # ── write outputs ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    out = {
        "restored_dir": args.restored_dir,
        "hq_dir":       args.hq_dir,
        "ffhq_dir":     args.ffhq_dir,
        "n_images":     len(per_image),
        "summary":      summary,
        "per_image":    per_image,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics → {args.out_json}")

    # CSV: one row per image, columns = all scalar keys
    csv_path = os.path.splitext(args.out_json)[0] + ".csv"
    try:
        import pandas as pd
        df = pd.DataFrame([{k: v for k, v in r.items()
                            if isinstance(v, (int, float, str))}
                           for r in per_image])
        df.to_csv(csv_path, index=False)
        print(f"Saved per-image CSV → {csv_path}")
    except Exception as e:
        print(f"[WARN] CSV write skipped: {e}")

    # ── pretty summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    for k, v in summary.items():
        if "mean" in v:
            print(f"  {k:<15s}  mean = {v['mean']:10.4f}  ± {v['std']:.4f}  (n={v['n']})")
        else:
            print(f"  {k:<15s}  = {v['value']:10.4f}")
    print("=" * 70)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restored_dir", required=True,
                    help="Folder of restored images from a single model/dataset")
    ap.add_argument("--hq_dir", default="",
                    help="(optional) Paired HQ ground truth folder")
    ap.add_argument("--ffhq_dir", default="",
                    help="(optional) FFHQ HQ reference folder for FID(FFHQ)")
    ap.add_argument("--out_json", required=True,
                    help="Where to write aggregated + per-image metrics (JSON)")

    ap.add_argument("--resolution", type=int, default=512,
                    help="Images will be resized to this before metrics are run")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="Override IQA metric list (e.g. lpips dists niqe)")

    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
