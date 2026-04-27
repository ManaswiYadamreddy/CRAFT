
'''


python eval_quantitative.py \
  --lq_dir /projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2 \
  --hq_dir /projectnb/cs585/projects/craft/data/test/CelebA/CelebA_Validation/celeba_512_validation \
  --prompts_json /projectnb/cs585/projects/craft/data/test/CelebA/celeba_prompts_output.json \
  --pretrained_model_name_or_path /projectnb/cs585/projects/craft/osdface/pretrained/sd21 \
  --img_encoder_weight /projectnb/cs585/projects/craft/osdface/pretrained/associate_2.ckpt \
  --ckpt_path /projectnb/cs585/projects/craft/osdface/pretrained \
  --conditioner_path checkpoints/textcond_final/checkpoint-50000/text_conditioner.pth \
  --film_neg_weight 0.1 \
  --mixed_precision fp16 \
  --output_root eval_outputs_final/quantitative \
  --require_deg_lmd

'''
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class EvalResult:
    method: str
    metrics: Dict[str, float]
    n_images: int


def find_images(directory: Path) -> List[Path]:
    files = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


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


def build_hq_lookup(hq_dir: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for p in find_images(hq_dir):
        name = p.name
        lookup[name] = p
        stem, ext = os.path.splitext(name)
        lookup[f"{stem}_LQ{ext}"] = p
        lookup[f"{stem}_lq{ext}"] = p
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


def pair_output_with_hq(output_dir: Path, hq_dir: Path, limit: Optional[int]) -> List[Tuple[Path, Path]]:
    hq_lookup = build_hq_lookup(hq_dir)
    pairs: List[Tuple[Path, Path]] = []
    missing_hq: List[str] = []
    for out_img in find_images(output_dir):
        match = hq_lookup.get(out_img.name)
        if match is None:
            missing_hq.append(out_img.name)
            continue
        pairs.append((out_img, match))
    if missing_hq:
        preview = ", ".join(missing_hq[:10])
        raise RuntimeError(
            f"HQ image not found for {len(missing_hq)} generated outputs in {output_dir}. "
            f"First missing: {preview}"
        )
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def _safe_read_rgb_np(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def compute_deg_lmd_metrics(
    pairs: List[Tuple[Path, Path]],
    device: torch.device,
    require_deg_lmd: bool,
) -> Dict[str, float]:
    """
    Compute:
      - DEG: identity embedding angle (lower is better)
      - LMD: landmark distance normalized by inter-ocular distance (lower is better)

    Uses:
      - facenet-pytorch (MTCNN + InceptionResnetV1) for DEG
      - face-alignment for 68-point landmarks for LMD
    """
    try:
        import face_alignment
        from facenet_pytorch import InceptionResnetV1, MTCNN
    except ImportError as exc:
        msg = (
            "Deg/LMD dependencies missing. Install with: "
            "pip install facenet-pytorch face-alignment"
        )
        if require_deg_lmd:
            raise RuntimeError(msg) from exc
        print(f"[WARN] {msg}")
        return {}

    fa_device = "cuda" if device.type == "cuda" else "cpu"
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=fa_device,
        flip_input=False,
    )
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    id_model = InceptionResnetV1(pretrained="vggface2").to(device).eval()

    deg_vals: List[float] = []
    lmd_vals: List[float] = []
    deg_valid = 0
    lmd_valid = 0

    for pred_path, ref_path in pairs:
        pred_pil = Image.open(pred_path).convert("RGB")
        ref_pil = Image.open(ref_path).convert("RGB")

        # ---- DEG ----
        pred_face = mtcnn(pred_pil)
        ref_face = mtcnn(ref_pil)
        if pred_face is not None and ref_face is not None:
            with torch.no_grad():
                e_pred = id_model(pred_face.unsqueeze(0).to(device))
                e_ref = id_model(ref_face.unsqueeze(0).to(device))
                cos_sim = F.cosine_similarity(e_pred, e_ref).item()
            cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
            deg_vals.append(float(np.degrees(np.arccos(cos_sim))))
            deg_valid += 1

        # ---- LMD ----
        pred_lm_all = fa.get_landmarks(_safe_read_rgb_np(pred_path))
        ref_lm_all = fa.get_landmarks(_safe_read_rgb_np(ref_path))
        if pred_lm_all and ref_lm_all:
            pred_lm = pred_lm_all[0]  # (68, 2)
            ref_lm = ref_lm_all[0]
            if pred_lm.shape == ref_lm.shape and pred_lm.shape[0] >= 46:
                inter_ocular = float(np.linalg.norm(ref_lm[36] - ref_lm[45]))
                if inter_ocular > 1e-6:
                    point_dist = np.linalg.norm(pred_lm - ref_lm, axis=1)
                    lmd_vals.append(float(point_dist.mean() / inter_ocular))
                    lmd_valid += 1

    results: Dict[str, float] = {}
    if deg_vals:
        results["DEG"] = float(np.mean(deg_vals))
        results["DEG_VALID_PAIRS"] = float(deg_valid)
    elif require_deg_lmd:
        raise RuntimeError("Deg could not be computed for any image pair (face detection/embedding failed).")

    if lmd_vals:
        results["LMD"] = float(np.mean(lmd_vals))
        results["LMD_VALID_PAIRS"] = float(lmd_valid)
    elif require_deg_lmd:
        raise RuntimeError("LMD could not be computed for any image pair (landmark detection failed).")

    if not lmd_vals:
        print("[WARN] LMD unavailable for all pairs; check face-alignment detection quality.")
    if not deg_vals:
        print("[WARN] DEG unavailable for all pairs; check facenet face detection quality.")
    return results


def run_inference_osd(args: argparse.Namespace, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "infer.py",
        "--input_image",
        str(args.lq_dir),
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


def run_inference_text(args: argparse.Namespace, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "infer_textcond.py",
        "--input_image",
        str(args.lq_dir),
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


def compute_metrics_for_dir(
    method_name: str,
    output_dir: Path,
    hq_dir: Path,
    ffhq_dir: Optional[Path],
    limit: Optional[int],
    device: torch.device,
    require_deg_lmd: bool,
) -> EvalResult:
    try:
        import pyiqa
    except ImportError as exc:
        raise RuntimeError(
            "pyiqa is required for quantitative evaluation. Install with: pip install pyiqa"
        ) from exc

    pairs = pair_output_with_hq(output_dir, hq_dir, limit)
    if not pairs:
        raise RuntimeError(f"No paired outputs found in {output_dir} against HQ dir {hq_dir}.")

    ref_metric_names = ["lpips", "dists"]
    nr_metric_names = ["musiq", "niqe"]
    metrics: Dict[str, float] = {}

    ref_metrics = {name: pyiqa.create_metric(name, device=device) for name in ref_metric_names}
    nr_metrics = {name: pyiqa.create_metric(name, device=device) for name in nr_metric_names}

    for name, metric in ref_metrics.items():
        vals = []
        for out_img, hq_img in pairs:
            pred = load_image_tensor(out_img, device)
            ref = load_image_tensor(hq_img, device)
            vals.append(float(metric(pred, ref).item()))
        metrics[name.upper()] = float(np.mean(vals))

    for name, metric in nr_metrics.items():
        vals = []
        for out_img, _ in pairs:
            pred = load_image_tensor(out_img, device)
            vals.append(float(metric(pred).item()))
        metrics[name.upper()] = float(np.mean(vals))

    # Paper-style identity/fidelity metrics.
    metrics.update(
        compute_deg_lmd_metrics(
            pairs=pairs,
            device=device,
            require_deg_lmd=require_deg_lmd,
        )
    )

    # FID is dataset-level and optional.
    try:
        fid_metric = pyiqa.create_metric("fid", device=device)
        metrics["FID_HQ"] = float(fid_metric(str(output_dir), str(hq_dir)).item())
        if ffhq_dir is not None:
            metrics["FID_FFHQ"] = float(fid_metric(str(output_dir), str(ffhq_dir)).item())
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] FID metric unavailable or failed: {exc}")

    return EvalResult(method=method_name, metrics=metrics, n_images=len(pairs))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantitative evaluation for OSDFace baseline vs text-conditioned variant."
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
    p.add_argument("--output_root", type=Path, default=Path("eval_outputs/quantitative"))
    p.add_argument("--max_images", type=int, default=None, help="Optional cap on paired images for faster eval.")
    p.add_argument("--fid_ffhq_dir", type=Path, default=None)
    p.add_argument("--skip_inference", action="store_true", help="Assume outputs already exist in output_root.")
    p.add_argument("--merge_lora", action="store_true", help="Pass-through flag for infer.py baseline.")
    p.add_argument(
        "--require_deg_lmd",
        action="store_true",
        help="Fail if Deg/LMD cannot be computed (missing deps or no detected faces).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_required_inputs(args.lq_dir, args.hq_dir, Path(args.prompts_json))

    osd_dir = args.output_root / "osd"
    text_dir = args.output_root / "osd_text"
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_inference:
        osd_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        print("Running OSD inference (without text conditioning)...")
        run_inference_osd(args, osd_dir)
        print("Running OSD inference (with text conditioning)...")
        run_inference_text(args, text_dir)

    print("Computing quantitative metrics...")
    osd_result = compute_metrics_for_dir(
        "OSDFace (no text)",
        osd_dir,
        args.hq_dir,
        args.fid_ffhq_dir,
        args.max_images,
        device,
        args.require_deg_lmd,
    )
    text_result = compute_metrics_for_dir(
        "OSDFace + TextCond",
        text_dir,
        args.hq_dir,
        args.fid_ffhq_dir,
        args.max_images,
        device,
        args.require_deg_lmd,
    )

    summary = {
        "paper_metrics_reference": [
            "LPIPS",
            "DISTS",
            "MUSIQ",
            "NIQE",
            "DEG",
            "LMD",
            "FID_HQ (optional)",
            "FID_FFHQ (optional)",
        ],
        "results": [
            {"method": osd_result.method, "n_images": osd_result.n_images, **osd_result.metrics},
            {"method": text_result.method, "n_images": text_result.n_images, **text_result.metrics},
        ],
    }

    out_json = args.output_root / "quantitative_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Quantitative Results ===")
    for row in summary["results"]:
        print(json.dumps(row, indent=2))
    print(f"\nSaved results to: {out_json}")


if __name__ == "__main__":
    main()
