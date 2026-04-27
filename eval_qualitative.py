


import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def find_images(directory: Path) -> List[Path]:
    files = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def build_hq_lookup(hq_dir: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for p in find_images(hq_dir):
        name = p.name
        stem, ext = os.path.splitext(name)
        lookup[name] = p
        lookup[f"{stem}_LQ{ext}"] = p
        lookup[f"{stem}_lq{ext}"] = p
    return lookup


def stem_variants(name: str) -> List[str]:
    stem, ext = os.path.splitext(name)
    variants = {name}
    variants.add(f"{stem}_LQ{ext}")
    variants.add(f"{stem}_lq{ext}")
    variants.add(f"{stem}_HQ{ext}")
    if stem.endswith("_LQ") or stem.endswith("_lq"):
        base = stem[:-3]
        variants.add(f"{base}{ext}")
    return list(variants)


def build_prompt_lookup(prompts_json: Path) -> Dict[str, Dict[str, str]]:
    with open(prompts_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    lookup: Dict[str, Dict[str, str]] = {}
    for item in raw:
        if "image" not in item:
            continue
        name = str(item["image"])
        entry = {"pos": item.get("pos", ""), "na": item.get("na", "")}
        for v in stem_variants(name):
            lookup[v] = entry
    return lookup


def validate_required_inputs(lq_dir: Path, hq_dir: Path, prompts_json: Path) -> None:
    lq_images = find_images(lq_dir)
    if not lq_images:
        raise RuntimeError(f"No images found in lq_dir: {lq_dir}")

    hq_lookup = build_hq_lookup(hq_dir)
    prompt_lookup = build_prompt_lookup(prompts_json)
    missing_hq: List[str] = []
    missing_prompt: List[str] = []

    for lq_img in lq_images:
        if lq_img.name not in hq_lookup:
            missing_hq.append(lq_img.name)
        if lq_img.name not in prompt_lookup:
            missing_prompt.append(lq_img.name)

    if missing_hq:
        preview = ", ".join(missing_hq[:10])
        raise RuntimeError(
            f"HQ image not found for {len(missing_hq)} LQ files. First missing: {preview}"
        )
    if missing_prompt:
        preview = ", ".join(missing_prompt[:10])
        raise RuntimeError(
            f"Prompt not found for {len(missing_prompt)} LQ files in {prompts_json}. "
            f"First missing: {preview}"
        )


def collect_pairs(lq_dir: Path, hq_dir: Path) -> List[Tuple[Path, Path]]:
    hq_lookup = build_hq_lookup(hq_dir)
    pairs: List[Tuple[Path, Path]] = []
    for lq_img in find_images(lq_dir):
        hq_match = hq_lookup.get(lq_img.name)
        if hq_match is not None:
            pairs.append((lq_img, hq_match))
    return pairs


def run_osd_single(args: argparse.Namespace, lq_img: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "infer.py",
        "--input_image",
        str(lq_img),
        "--output_dir",
        str(out_dir),
        "--pretrained_model_name_or_path",
        args.pretrained_model_name_or_path,
        "--img_encoder_weight",
        args.img_encoder_weight,
        "--ckpt_path",
        args.ckpt_path,
        "--mixed_precision",
        args.mixed_precision,
        "--gpu_ids",
        *[str(i) for i in args.gpu_ids],
    ]
    if args.merge_lora:
        cmd.append("--merge_lora")
    subprocess.run(cmd, check=True)
    out_path = out_dir / lq_img.name
    if not out_path.exists():
        raise RuntimeError(f"Expected OSD output image not found: {out_path}")
    return out_path


def run_text_single(args: argparse.Namespace, lq_img: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "infer_textcond.py",
        "--input_image",
        str(lq_img),
        "--output_dir",
        str(out_dir),
        "--prompts_json",
        args.prompts_json,
        "--pretrained_model_name_or_path",
        args.pretrained_model_name_or_path,
        "--img_encoder_weight",
        args.img_encoder_weight,
        "--ckpt_path",
        args.ckpt_path,
        "--conditioner_path",
        args.conditioner_path,
        "--film_neg_weight",
        str(args.film_neg_weight),
        "--mixed_precision",
        args.mixed_precision,
        "--gpu_ids",
        *[str(i) for i in args.gpu_ids],
    ]
    subprocess.run(cmd, check=True)
    out_path = out_dir / lq_img.name
    if not out_path.exists():
        raise RuntimeError(f"Expected text-conditioned output image not found: {out_path}")
    return out_path


def add_label(image: Image.Image, text: str) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 20], fill=(0, 0, 0))
    draw.text((5, 4), text, fill=(255, 255, 255), font=font)
    return img


def make_panel(lq_path: Path, hq_path: Path, osd_path: Path, text_path: Path, save_path: Path) -> None:
    lq = add_label(Image.open(lq_path).convert("RGB"), "LQ")
    hq = add_label(Image.open(hq_path).convert("RGB"), "HQ")
    osd = add_label(Image.open(osd_path).convert("RGB"), "OSD")
    text = add_label(Image.open(text_path).convert("RGB"), "OSD + Text")

    w, h = lq.size
    grid = Image.new("RGB", (w * 2, h * 2), color=(255, 255, 255))
    grid.paste(lq, (0, 0))
    grid.paste(hq, (w, 0))
    grid.paste(osd, (0, h))
    grid.paste(text, (w, h))
    grid.save(save_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qualitative comparison for OSDFace baseline vs text-conditioned variant."
    )
    p.add_argument("--lq_dir", required=True, type=Path)
    p.add_argument("--hq_dir", required=True, type=Path)
    p.add_argument("--prompts_json", required=True)
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--img_encoder_weight", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--conditioner_path", required=True)
    p.add_argument("--film_neg_weight", type=float, default=0.5)
    p.add_argument("--mixed_precision", choices=["fp16", "fp32", "bf16"], default="fp16")
    p.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_root", type=Path, default=Path("eval_outputs/qualitative"))
    p.add_argument("--merge_lora", action="store_true", help="Pass-through flag for infer.py baseline.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    validate_required_inputs(args.lq_dir, args.hq_dir, Path(args.prompts_json))

    pairs = collect_pairs(args.lq_dir, args.hq_dir)
    if len(pairs) == 0:
        raise RuntimeError("No matched LQ/HQ pairs found. Check filename alignment between lq_dir and hq_dir.")

    n = min(args.num_samples, len(pairs))
    sampled = random.sample(pairs, k=n)

    osd_dir = args.output_root / "osd"
    text_dir = args.output_root / "osd_text"
    lq_copy_dir = args.output_root / "lq"
    hq_copy_dir = args.output_root / "hq"
    panel_dir = args.output_root / "panels"
    for d in [osd_dir, text_dir, lq_copy_dir, hq_copy_dir, panel_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Generating qualitative set with {n} random samples...")
    for idx, (lq_img, hq_img) in enumerate(sampled):
        print(f"[{idx + 1}/{n}] {lq_img.name}")
        osd_out = run_osd_single(args, lq_img, osd_dir)
        text_out = run_text_single(args, lq_img, text_dir)

        lq_save = lq_copy_dir / lq_img.name
        hq_save = hq_copy_dir / hq_img.name
        if not lq_save.exists():
            Image.open(lq_img).convert("RGB").save(lq_save)
        if not hq_save.exists():
            Image.open(hq_img).convert("RGB").save(hq_save)

        panel_name = f"{idx:02d}_{lq_img.stem}_comparison.png"
        make_panel(lq_img, hq_img, osd_out, text_out, panel_dir / panel_name)

    print("\nSaved outputs to:")
    print(f"- LQ copies:       {lq_copy_dir}")
    print(f"- HQ copies:       {hq_copy_dir}")
    print(f"- OSD outputs:     {osd_dir}")
    print(f"- OSD+Text outputs:{text_dir}")
    print(f"- 2x2 panels:      {panel_dir}")


if __name__ == "__main__":
    main()
