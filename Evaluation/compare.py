"""
compare.py — Side-by-side CRAFT vs OSDFace table for one dataset.

Reads the two metrics.json files produced by `evaluate.py` (one per model) and
writes a single JSON + pretty markdown comparison.

Usage:
    python -m Evaluation.compare \
        --craft_json   results/craft/celeba/metrics.json \
        --osdface_json results/osdface/celeba/metrics.json \
        --out_json     results/compare_celeba.json \
        --dataset      celeba
"""
from __future__ import annotations

import argparse
import json
import math
import os


# Direction (True = higher is better) for each OSDFace-paper metric
DIRECTION = {
    "lpips":   False,
    "dists":   False,
    "psnr":    True,
    "ssim":    True,
    "musiq":   True,
    "niqe":    False,
    "clipiqa": True,
    "maniqa":  True,
    "deg":     False,
    "cos_sim": True,
    "lmd":     False,
    "fid_ffhq": False,
    "fid_hq":   False,
}

ARROW = {True: "↑", False: "↓"}


def _get_val(summary: dict, key: str):
    v = summary.get(key)
    if v is None:
        return None
    return v.get("mean", v.get("value"))


def _fmt(v):
    if v is None:
        return "   —   "
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "  NaN  "
    return f"{v:8.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft_json",   required=True)
    ap.add_argument("--osdface_json", required=True)
    ap.add_argument("--out_json",     required=True)
    ap.add_argument("--dataset",      default="")
    args = ap.parse_args()

    with open(args.craft_json)   as f: craft_data   = json.load(f)
    with open(args.osdface_json) as f: osdface_data = json.load(f)

    craft_s   = craft_data["summary"]
    osdface_s = osdface_data["summary"]

    # union of keys, ordered by the paper's natural order
    order = ["lpips", "dists", "psnr", "ssim",
             "musiq", "niqe", "clipiqa", "maniqa",
             "deg", "cos_sim", "lmd",
             "fid_ffhq", "fid_hq"]
    seen = set()
    keys = [k for k in order if k in craft_s or k in osdface_s]
    for k in list(craft_s) + list(osdface_s):
        if k not in seen and k not in keys:
            keys.append(k); seen.add(k)

    table = []
    for k in keys:
        c = _get_val(craft_s, k)
        o = _get_val(osdface_s, k)
        higher_better = DIRECTION.get(k, None)
        winner = None
        if c is not None and o is not None and \
           not (isinstance(c, float) and math.isnan(c)) and \
           not (isinstance(o, float) and math.isnan(o)):
            if higher_better is True:
                winner = "craft" if c > o else ("osdface" if o > c else "tie")
            elif higher_better is False:
                winner = "craft" if c < o else ("osdface" if o < c else "tie")
        table.append({
            "metric":        k,
            "arrow":         ARROW.get(higher_better, ""),
            "craft":         c,
            "osdface":       o,
            "winner":        winner,
            "higher_better": higher_better,
        })

    # ── Markdown pretty print ───────────────────────────────────────────
    md_lines = []
    md_lines.append(f"# {args.dataset or 'dataset'} — CRAFT vs OSDFace\n")
    md_lines.append(f"- restored_craft   : `{craft_data['restored_dir']}`")
    md_lines.append(f"- restored_osdface : `{osdface_data['restored_dir']}`")
    md_lines.append(f"- hq_dir           : `{craft_data.get('hq_dir')}`")
    md_lines.append(f"- ffhq_dir         : `{craft_data.get('ffhq_dir')}`")
    md_lines.append(f"- n_images         : CRAFT={craft_data['n_images']}  "
                    f"OSDFace={osdface_data['n_images']}\n")

    md_lines.append("| Metric | ↑/↓ | CRAFT | OSDFace | Winner |")
    md_lines.append("|---|---|---|---|---|")
    for row in table:
        md_lines.append(
            f"| {row['metric']} | {row['arrow']} | {_fmt(row['craft']).strip()} | "
            f"{_fmt(row['osdface']).strip()} | {row['winner'] or '—'} |"
        )
    md = "\n".join(md_lines)

    # ── outputs ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "dataset":       args.dataset,
            "craft_json":    args.craft_json,
            "osdface_json":  args.osdface_json,
            "table":         table,
        }, f, indent=2)
    md_path = os.path.splitext(args.out_json)[0] + ".md"
    with open(md_path, "w") as f:
        f.write(md)

    # ── stdout ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {args.dataset or 'dataset'}  —  CRAFT vs OSDFace")
    print("=" * 70)
    print(f"{'metric':<10}  {'':2}  {'CRAFT':>10}  {'OSDFace':>10}  winner")
    print("-" * 70)
    for row in table:
        print(f"{row['metric']:<10}  {row['arrow']:>2}  "
              f"{_fmt(row['craft'])}  {_fmt(row['osdface'])}  {row['winner'] or '—'}")
    print("=" * 70)
    print(f"Saved → {args.out_json}")
    print(f"Saved → {md_path}")


if __name__ == "__main__":
    main()
