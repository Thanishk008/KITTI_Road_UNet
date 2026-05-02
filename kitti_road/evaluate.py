from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .datasets import KITTIRoadDataset, image_to_tensor, mask_to_binary, resize_pair
from .losses import BCEDiceLoss
from .metrics import MetricAccumulator, threshold_sweep
from .models import build_model
from .utils import append_csv_row
from .utils import save_json
from .visualize import plot_pr_curve, road_overlay


def save_error_map(prob: np.ndarray, target: np.ndarray, path: Path, threshold: float) -> None:
    pred = prob >= threshold
    truth = target == 1
    valid = target != 255
    canvas = np.zeros((*target.shape, 3), dtype=np.uint8)
    canvas[(pred & truth) & valid] = [20, 220, 120]
    canvas[(pred & ~truth) & valid] = [40, 120, 255]
    canvas[(~pred & truth) & valid] = [255, 70, 70]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def write_qualitative_examples(model, dataset: KITTIRoadDataset, config, report_dir: Path, experiment: str, device, threshold: float, limit: int = 12) -> None:
    image_size = tuple(config["data"].get("image_size", [384, 1216]))
    overlay_dir = report_dir / "qualitative_overlays" / experiment
    error_dir = report_dir / "error_examples" / experiment
    overlay_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for item in dataset.samples[:limit]:
            original = Image.open(item["image"]).convert("RGB")
            gt_mask = Image.open(item["mask"])
            resized_img, _ = resize_pair(original, gt_mask, image_size)
            tensor = image_to_tensor(resized_img).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            prob_original = np.asarray(
                Image.fromarray((prob * 255).astype(np.uint8)).resize(original.size, Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            target_original = mask_to_binary(gt_mask)
            road_overlay(original, prob_original, threshold).save(overlay_dir / f"{item['id']}_overlay.png")
            save_error_map(prob_original, target_original, error_dir / f"{item['id']}_errors.png", threshold)


def evaluate(checkpoint: Path, split: str = "val", cpu: bool = False) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload["config"]
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model = build_model(config["model"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    data_cfg = config["data"]
    image_size = tuple(data_cfg.get("image_size", [384, 1216]))
    dataset = KITTIRoadDataset(data_cfg["processed"], split, image_size, augment=False, seed=data_cfg.get("seed", 6681))
    loader = DataLoader(dataset, batch_size=int(config["training"].get("batch_size", 4)), shuffle=False, num_workers=int(config["training"].get("num_workers", 4)))
    threshold = float(payload.get("threshold", config["training"].get("threshold", 0.5)))
    criterion = BCEDiceLoss()
    accumulator = MetricAccumulator(threshold=threshold)
    total_loss = 0.0

    with torch.no_grad():
        for images, masks, scenarios, _ids in tqdm(loader, desc=f"evaluate {split}"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, masks).item())
            accumulator.update(logits, masks, list(scenarios))

    metrics = accumulator.compute()
    metrics["loss"] = total_loss / max(1, len(loader))
    metrics["per_scenario"] = accumulator.per_scenario()
    report_dir = Path(config["outputs"]["report_dir"])
    experiment = config["experiment_name"]
    save_json(metrics, report_dir / f"{experiment}_{split}_evaluation.json")

    if accumulator.scores:
        scores = np.concatenate(accumulator.scores)
        labels = np.concatenate(accumulator.labels)
        rows = threshold_sweep(scores, labels)
        sweep_path = report_dir / f"{experiment}_{split}_threshold_sweep.csv"
        with sweep_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        plot_pr_curve(rows, report_dir, f"{experiment}_{split}")

    summary_path = report_dir / "experiment_summary.csv"
    summary_row = {
        "experiment": experiment,
        "split": split,
        "checkpoint": str(checkpoint),
        "iou": round(metrics["iou"], 6),
        "dice": round(metrics["dice"], 6),
        "precision": round(metrics["precision"], 6),
        "recall": round(metrics["recall"], 6),
        "pixel_accuracy": round(metrics["pixel_accuracy"], 6),
        "ap": round(metrics["ap"], 6),
        "max_f": round(metrics["max_f"], 6),
        "loss": round(metrics["loss"], 6),
    }
    append_csv_row(summary_path, summary_row, list(summary_row.keys()))
    write_qualitative_examples(model, dataset, config, report_dir, experiment, device, threshold)

    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, args.cpu)


if __name__ == "__main__":
    main()
