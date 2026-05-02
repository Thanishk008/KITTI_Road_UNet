from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset

from .utils import load_json

IGNORE_INDEX = 255


def scenario_from_name(name: str) -> str:
    prefix = name.split("_", 1)[0].lower()
    return prefix if prefix in {"um", "umm", "uu"} else "unknown"


def mask_to_binary(mask: Image.Image) -> np.ndarray:
    """Convert KITTI road GT image to road=1, background=0, ignore=255.

    KITTI Road ground truth is distributed as color-coded PNGs. In the common
    road masks, magenta/white-like pixels denote road, red pixels denote valid
    background, and black pixels denote ignore/void. This converter also
    supports grayscale masks so small teaching subsets stay compatible.
    """
    arr = np.asarray(mask)
    if arr.ndim == 2:
        out = np.zeros(arr.shape, dtype=np.uint8)
        out[arr > 0] = 1
        return out

    rgb = arr[:, :, :3].astype(np.uint8)
    magenta_road = (rgb[:, :, 0] > 128) & (rgb[:, :, 2] > 128) & (rgb[:, :, 1] < 128)
    bright_road = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] > 180) & (rgb[:, :, 2] > 180)
    road = magenta_road | bright_road
    ignore = rgb.sum(axis=2) == 0

    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    out[road] = 1
    out[ignore] = IGNORE_INDEX
    return out


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1)


def resize_pair(image: Image.Image, mask: Image.Image, image_size: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
    height, width = image_size
    return (
        image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR),
        mask.resize((width, height), Image.Resampling.NEAREST),
    )


class KITTIRoadDataset(Dataset):
    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        image_size: Tuple[int, int],
        augment: bool = False,
        seed: int = 6681,
    ):
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)
        split_path = self.processed_dir / "split.json"
        split_data: Dict[str, List[Dict[str, Any]]] = load_json(split_path)
        if split not in split_data:
            raise ValueError(f"Split {split!r} not found in {split_path}")
        self.samples = split_data[split]
        if not self.samples:
            raise ValueError(f"No samples found for split={split}")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.rng.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
        image = ImageEnhance.Brightness(image).enhance(self.rng.uniform(0.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(self.rng.uniform(0.85, 1.15))
        image = ImageEnhance.Color(image).enhance(self.rng.uniform(0.9, 1.1))
        return image, mask

    def __getitem__(self, index: int):
        item = self.samples[index]
        image = Image.open(item["image"]).convert("RGB")
        mask = Image.open(item["mask"])
        image, mask = resize_pair(image, mask, self.image_size)
        if self.augment:
            image, mask = self._augment(image, mask)
        target = mask_to_binary(mask)
        target_tensor = torch.from_numpy(target.astype(np.int64))
        return image_to_tensor(image), target_tensor, item["scenario"], item["id"]
