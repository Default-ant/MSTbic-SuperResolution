"""
MSTbic Dataset Preparation (prepare_mstbic_dataset.py)
=======================================================
Generates pseudo-HR/LR pairs from LR-only images using multi-scale bicubic
downsampling — the core idea of the MSTbic training paradigm.

Workflow:
  1. Take a folder of LR source images (your only available resolution).
  2. Downsample each image by scale factor s (e.g. x4) with bicubic -> pseudo-LR.
  3. The original image is the pseudo-HR target.
  4. Save LR/HR crop pairs for training the SwinIR-lw model.

Usage (uv):
  uv run prepare_mstbic_dataset.py --src data/raw_lr --out data/MSTbic --scale 4 --patch 256 --splits 0.85 0.10 0.05
"""

# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "Pillow>=9.0",
#   "numpy>=1.21",
#   "tqdm>=4.64",
# ]
# ///

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def bicubic_downsample(img: Image.Image, scale: int) -> Image.Image:
    """Downsample a PIL image by `scale` using bicubic interpolation."""
    w, h = img.size
    lw, lh = w // scale, h // scale
    return img.resize((lw, lh), Image.BICUBIC)


def extract_crops(img: np.ndarray, hr_size: int, stride: int):
    """Slide a window over `img` and yield (y, x, crop) tuples."""
    H, W = img.shape[:2]
    for y in range(0, H - hr_size + 1, stride):
        for x in range(0, W - hr_size + 1, stride):
            yield y, x, img[y:y + hr_size, x:x + hr_size]


def prepare_split(
    src_files: list,
    hr_dir: Path,
    lr_dir: Path,
    scale: int,
    hr_patch: int,
    stride: int,
):
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for fpath in tqdm(src_files, desc=f"  → {hr_dir.parent.name}"):
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as e:
            print(f"  [skip] {fpath.name}: {e}")
            continue

        w, h = img.size
        if w < hr_patch or h < hr_patch:
            # Upscale small images to at least hr_patch size
            ratio = max(hr_patch / w, hr_patch / h) + 0.01
            img = img.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC)

        img_np = np.array(img)

        for _, _, hr_crop in extract_crops(img_np, hr_patch, stride):
            hr_pil = Image.fromarray(hr_crop)
            lr_pil = bicubic_downsample(hr_pil, scale)

            stem = f"{fpath.stem}_crop{idx:06d}"
            hr_pil.save(hr_dir / f"{stem}.png")
            lr_pil.save(lr_dir / f"{stem}.png")
            idx += 1

    print(f"  Saved {idx} LR/HR pairs to {hr_dir.parent}")


def main():
    parser = argparse.ArgumentParser(description="MSTbic Dataset Preparation")
    parser.add_argument("--src",   required=True, help="Folder of LR source images (your only data)")
    parser.add_argument("--out",   required=True, help="Output root folder")
    parser.add_argument("--scale", type=int, default=4, help="SR scale factor (default: 4)")
    parser.add_argument("--patch", type=int, default=256, help="HR patch size (default: 256)")
    parser.add_argument("--stride", type=int, default=None, help="Sliding stride (default: patch//2)")
    parser.add_argument("--splits", nargs=3, type=float, default=[0.85, 0.10, 0.05],
                        metavar=("TRAIN", "VAL", "TEST"), help="Train/Val/Test split ratios")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    stride = args.stride or args.patch // 2
    src = Path(args.src)
    out = Path(args.out)

    all_files = sorted([f for f in src.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"}])
    random.shuffle(all_files)

    n = len(all_files)
    n_train = int(args.splits[0] * n)
    n_val   = int(args.splits[1] * n)

    splits = {
        "train": all_files[:n_train],
        "val":   all_files[n_train:n_train + n_val],
        "test":  all_files[n_train + n_val:],
    }

    for split_name, files in splits.items():
        if not files:
            print(f"  [warning] No files for split '{split_name}' — skipping.")
            continue
        print(f"\n[{split_name.upper()}] {len(files)} source images")
        prepare_split(
            src_files=files,
            hr_dir=out / split_name / "HR",
            lr_dir=out / split_name / "LR",
            scale=args.scale,
            hr_patch=args.patch,
            stride=stride,
        )

    print("\n✓ MSTbic dataset preparation complete.")
    print(f"  Output: {out.resolve()}")
    print(f"  Config dataroot_H: {(out / 'train' / 'HR').resolve()}")
    print(f"  Config dataroot_L: {(out / 'train' / 'LR').resolve()}")


if __name__ == "__main__":
    main()
