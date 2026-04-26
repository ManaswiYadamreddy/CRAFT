"""
Concatenates PNG images with the same filename from 6 directories side by side,
adds titles above each image, and saves results to an output directory.

Usage:
    python concat_images.py --out_dir /path/to/output
    python concat_images.py --out_dir /path/to/output --img_height 512
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

# ── Directory config ──────────────────────────────────────────────────────────
SOURCES = [
    {
        "title": "HQ Image",
        "path": "/projectnb/cs585/projects/craft/data/test/CelebA/CelebA_Validation/celeba_512_validation",
    },
    {
        "title": "LQ Image",
        "path": "/projectnb/cs585/projects/craft/data/test/CelebA/Self_CelebA_Validation_v2/self_celeba_512_v2",
    },
    {
        "title": "OSDFace (Ours)",
        "path": "/projectnb/cs585/projects/craft/Evaluation_Results/osdface_s2/celeba/restored",
    },
    {
        "title": "CRAFT (Ours)",
        "path": "/projectnb/cs585/projects/craft/Evaluation_Results/craft_s2/celeba/restored",
    },
    {
        "title": "OSDFace (Original)",
        "path": "/projectnb/cs585/projects/craft/text_cond/eval_outputs/quantitative/osd",
    },
    {
        "title": "OSDFace + Text Inputs",
        "path": "/projectnb/cs585/projects/craft/text_cond/eval_outputs/quantitative/osd_text",
    },
]

# ── Layout constants ──────────────────────────────────────────────────────────
TITLE_HEIGHT   = 36      # px reserved above each image for the label
TITLE_FONT_SIZE = 22
PADDING        = 8       # px gap between images
BG_COLOR       = (20, 20, 20)      # dark background
TEXT_COLOR     = (240, 240, 240)   # near-white text
BORDER_COLOR   = (60, 60, 60)      # subtle border around each cell


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a truetype font; fall back to the built-in bitmap font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def find_common_images(sources: list[dict]) -> list[str]:
    """Return sorted list of filenames present in ALL source directories."""
    sets = []
    for src in sources:
        p = Path(src["path"])
        if not p.is_dir():
            print(f"[WARN] Directory not found: {p}", file=sys.stderr)
            sets.append(set())
        else:
            sets.append({f.name for f in p.glob("*.png")})

    common = sets[0]
    for s in sets[1:]:
        common &= s

    if not common:
        print("[ERROR] No common PNG filenames found across all directories.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(common)} common images.")
    return sorted(common)


def make_row(img_name: str, sources: list[dict], target_h: int, font) -> Image.Image:
    """
    Build one wide image: all 6 panels side by side.
    Each panel = title bar + image (resized to target_h).
    """
    panels = []

    for src in sources:
        img_path = Path(src["path"]) / img_name
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open {img_path}: {e}", file=sys.stderr)
            # Create a placeholder panel
            img = Image.new("RGB", (target_h, target_h), (80, 80, 80))

        # Resize preserving aspect ratio to target height
        orig_w, orig_h = img.size
        new_w = int(orig_w * target_h / orig_h)
        img = img.resize((new_w, target_h), Image.LANCZOS)

        # Panel canvas: title + image
        panel_h = TITLE_HEIGHT + target_h
        panel = Image.new("RGB", (new_w, panel_h), BG_COLOR)

        # Draw title bar
        draw = ImageDraw.Draw(panel)
        title = src["title"]

        # Center title text
        bbox = font.getbbox(title)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = max((new_w - tw) // 2, 4)
        ty = (TITLE_HEIGHT - th) // 2

        draw.text((tx, ty), title, fill=TEXT_COLOR, font=font)

        # Paste image below title
        panel.paste(img, (0, TITLE_HEIGHT))

        # Draw subtle border
        draw.rectangle([0, 0, new_w - 1, panel_h - 1], outline=BORDER_COLOR)

        panels.append(panel)

    # Combine panels horizontally with padding
    total_w = sum(p.width for p in panels) + PADDING * (len(panels) - 1)
    total_h = panels[0].height  # all same height
    row = Image.new("RGB", (total_w, total_h), BG_COLOR)

    x = 0
    for panel in panels:
        row.paste(panel, (x, 0))
        x += panel.width + PADDING

    return row


def main():
    parser = argparse.ArgumentParser(description="Concatenate images side-by-side with titles.")
    parser.add_argument("--out_dir",    default="/projectnb/cs585/projects/craft/images_concat", help="Output directory for concatenated images.")
    parser.add_argument("--img_height", type=int, default=512, help="Height each image is resized to (default: 512).")
    parser.add_argument("--font_size",  type=int, default=TITLE_FONT_SIZE, help="Title font size (default: 22).")
    parser.add_argument("--limit",      type=int, default=None, help="Process only the first N images (for quick testing).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    font = load_font(args.font_size)
    common = find_common_images(SOURCES)

    if args.limit:
        common = common[: args.limit]
        print(f"[INFO] Limited to first {args.limit} images.")

    import errno as errno_mod, time

    for i, img_name in enumerate(common, 1):
        row = make_row(img_name, SOURCES, args.img_height, font)
        out_path = out_dir / img_name
        tmp_path = out_dir / f".tmp_{img_name}"

        # Retry loop — guards against NFS stale file handle (errno 116)
        max_retries = 5
        saved = False
        for attempt in range(1, max_retries + 1):
            try:
                row.save(tmp_path)          # write to temp first
                tmp_path.rename(out_path)   # atomic move → no partial files
                saved = True
                break
            except OSError as e:
                if e.errno in (errno_mod.ESTALE, errno_mod.EIO, errno_mod.EBUSY) and attempt < max_retries:
                    wait = 2 ** attempt     # exponential back-off: 2, 4, 8, 16 s
                    print(f"[WARN] NFS error on {img_name} (attempt {attempt}/{max_retries}),"
                          f" retrying in {wait}s… ({e})", file=sys.stderr)
                    time.sleep(wait)
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                else:
                    print(f"[ERROR] Failed to save {img_name} after {attempt} attempt(s): {e}",
                          file=sys.stderr)
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    break

        if i % 50 == 0 or i == len(common):
            status = "Saved" if saved else "SKIPPED"
            print(f"[INFO] {status} {i}/{len(common)}: {img_name}")

    print(f"\n✓ Done. {len(common)} images saved to: {out_dir}")


if __name__ == "__main__":
    main()