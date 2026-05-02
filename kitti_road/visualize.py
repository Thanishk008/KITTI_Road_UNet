from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def road_overlay(image: Image.Image, prob: np.ndarray, threshold: float = 0.5, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB")
    mask = (prob >= threshold).astype(np.uint8)
    overlay = np.asarray(base).copy()
    color = np.array([20, 220, 120], dtype=np.uint8)
    overlay[mask == 1] = (overlay[mask == 1] * (1 - alpha) + color * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def save_probability_mask(prob: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8)).save(path)


def save_binary_mask(prob: np.ndarray, path: str | Path, threshold: float = 0.5) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(((prob >= threshold).astype(np.uint8) * 255)).save(path)


def plot_curves(rows: list[Dict], report_dir: str | Path, experiment_name: str) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    prefix = f"{experiment_name}_" if experiment_name else ""
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in rows], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in rows], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / f"{prefix}loss_curves.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["iou"] for row in rows], label="IoU")
    plt.plot(epochs, [row["dice"] for row in rows], label="Dice/F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / f"{prefix}iou_curves.png", dpi=160)
    plt.close()


def plot_pr_curve(threshold_rows: Iterable[Dict], report_dir: str | Path, experiment_name: str) -> None:
    rows = list(threshold_rows)
    if not rows:
        return
    report_dir = Path(report_dir)
    plt.figure(figsize=(6, 6))
    plt.plot([row["recall"] for row in rows], [row["precision"] for row in rows], marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{experiment_name} precision-recall sweep")
    plt.tight_layout()
    plt.savefig(report_dir / f"{experiment_name}_pr_curve.png", dpi=160)
    plt.close()


def save_overlay_grid(items: list[tuple[Image.Image, Image.Image]], path: str | Path, cols: int = 2) -> None:
    if not items:
        return
    thumbs = []
    for original, overlay in items:
        pair = Image.new("RGB", (original.width * 2, original.height))
        pair.paste(original.convert("RGB"), (0, 0))
        pair.paste(overlay.convert("RGB"), (original.width, 0))
        thumbs.append(pair)
    rows = int(np.ceil(len(thumbs) / cols))
    width = max(t.width for t in thumbs)
    height = max(t.height for t in thumbs)
    canvas = Image.new("RGB", (cols * width, rows * height), "white")
    for idx, thumb in enumerate(thumbs):
        canvas.paste(thumb, ((idx % cols) * width, (idx // cols) * height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
