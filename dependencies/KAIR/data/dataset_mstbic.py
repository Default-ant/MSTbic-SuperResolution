"""
Gradient-Weighted Patch Selection Dataset (dataset_mstbic.py)
=============================================================
A PyTorch Dataset that replaces random cropping with a gradient-aware
sampling strategy. Patches with higher edge density are sampled more often,
forcing the model to focus on complex, high-frequency textures.

How it works:
  1. At dataset init, for every LR/HR pair compute a saliency score
     per valid crop position using the Sobel gradient of the HR image.
  2. Build a probability distribution from these scores.
  3. At __getitem__, use np.random.choice with these weights.

This file integrates with the KAIR `define_Dataset` interface.
"""

import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def _get_files(directory: str, exts=(".png", ".jpg", ".jpeg")):
    d = Path(directory)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in exts)


def _compute_gradient_scores(img_np: np.ndarray, patch_size: int, stride: int) -> list:
    if img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = img_np.astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    H, W = gray.shape
    scores = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch_score = grad_mag[y:y + patch_size, x:x + patch_size].mean()
            scores.append((y, x, float(patch_score)))

    return scores


class MSTbicGradientDataset(Dataset):
    """
    Dataset for MSTbic training with gradient-weighted patch selection.
    """
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        hr_patch_size: int = 256,
        scale: int = 4,
        top_k_fraction: float = 0.5,
        augment: bool = True,
        precompute: bool = True,
    ):
        super().__init__()
        self.hr_patch = hr_patch_size
        self.lr_patch = hr_patch_size // scale
        self.scale = scale
        self.augment = augment
        self.top_k = top_k_fraction

        hr_files = _get_files(hr_dir)
        lr_files = _get_files(lr_dir)
        assert len(hr_files) == len(lr_files), f"HR/LR count mismatch: {len(hr_files)} vs {len(lr_files)}"
        self.pairs = list(zip(hr_files, lr_files))

        self.patch_pool = []
        self.weights = np.array([])

        if precompute:
            self._build_pool()

    def _build_pool(self):
        print("[MSTbicGradientDataset] Pre-computing gradient scores...")
        all_entries = []
        all_scores = []
        stride = self.hr_patch // 2  # 50% overlap

        for hr_path, lr_path in self.pairs:
            hr_np = cv2.imread(str(hr_path))
            if hr_np is None:
                continue
            hr_np = cv2.cvtColor(hr_np, cv2.COLOR_BGR2RGB)

            scored = _compute_gradient_scores(hr_np, self.hr_patch, stride)
            if not scored:
                continue

            scored.sort(key=lambda t: t[2], reverse=True)
            keep = max(1, int(len(scored) * self.top_k))
            scored = scored[:keep]

            for y, x, score in scored:
                all_entries.append((str(hr_path), str(lr_path), y, x))
                all_scores.append(score)

        total = sum(all_scores)
        self.patch_pool = all_entries
        self.weights = np.array(all_scores) / total
        print(f"  Pool size: {len(self.patch_pool)} patches across {len(self.pairs)} images")

    def __len__(self):
        return len(self.patch_pool)

    def __getitem__(self, _):
        idx = np.random.choice(len(self.patch_pool), p=self.weights)
        hr_path, lr_path, y_hr, x_hr = self.patch_pool[idx]

        hr_img = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)
        lr_img = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)

        y_lr = y_hr // self.scale
        x_lr = x_hr // self.scale

        hr_crop = hr_img[y_hr:y_hr + self.hr_patch, x_hr:x_hr + self.hr_patch]
        lr_crop = lr_img[y_lr:y_lr + self.lr_patch, x_lr:x_lr + self.lr_patch]

        if self.augment:
            if random.random() > 0.5:
                hr_crop = np.fliplr(hr_crop).copy()
                lr_crop = np.fliplr(lr_crop).copy()
            if random.random() > 0.5:
                hr_crop = np.flipud(hr_crop).copy()
                lr_crop = np.flipud(lr_crop).copy()
            k = random.randint(0, 3)
            if k:
                hr_crop = np.rot90(hr_crop, k).copy()
                lr_crop = np.rot90(lr_crop, k).copy()

        hr_t = torch.from_numpy(hr_crop.transpose(2, 0, 1)).float() / 255.0
        lr_t = torch.from_numpy(lr_crop.transpose(2, 0, 1)).float() / 255.0

        return {"L": lr_t, "H": hr_t, "L_path": lr_path, "H_path": hr_path}
