from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .datasets import KITTIRoadDataset
from .losses import BCEDiceLoss
from .metrics import MetricAccumulator
from .models import build_model
from .utils import append_csv_row, load_config, load_json, save_json, set_seed, stable_hash
from .visualize import plot_curves


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    training_cfg = config["training"]
    data_cfg = config["data"]
    outputs_cfg = config["outputs"]
    if args.epochs is not None:
        training_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        training_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        training_cfg["num_workers"] = int(args.num_workers)
    if args.lr is not None:
        training_cfg["lr"] = float(args.lr)
    if args.weight_decay is not None:
        training_cfg["weight_decay"] = float(args.weight_decay)
    if args.save_every is not None:
        training_cfg["save_every"] = int(args.save_every)
    if args.threshold is not None:
        training_cfg["threshold"] = float(args.threshold)
    if args.amp:
        training_cfg["amp"] = True
    if args.no_amp:
        training_cfg["amp"] = False
    if args.image_size is not None:
        data_cfg["image_size"] = [int(args.image_size[0]), int(args.image_size[1])]
    if args.processed is not None:
        data_cfg["processed"] = str(args.processed)
    if args.checkpoint_dir is not None:
        outputs_cfg["checkpoint_dir"] = str(args.checkpoint_dir)
    if args.report_dir is not None:
        outputs_cfg["report_dir"] = str(args.report_dir)
    return config


def make_loaders(config: Dict[str, Any], train: bool = True):
    data_cfg = config["data"]
    training_cfg = config["training"]
    image_size = tuple(data_cfg.get("image_size", [384, 1216]))
    train_ds = KITTIRoadDataset(data_cfg["processed"], "train", image_size, augment=train, seed=data_cfg.get("seed", 6681))
    val_ds = KITTIRoadDataset(data_cfg["processed"], "val", image_size, augment=False, seed=data_cfg.get("seed", 6681))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(training_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def evaluate_epoch(model, loader, criterion, device, threshold: float, use_amp: bool, desc: str = "val") -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    accumulator = MetricAccumulator(threshold=threshold)
    with torch.no_grad():
        for images, masks, scenarios, _ids in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            total_loss += float(loss.item())
            accumulator.update(logits, masks, list(scenarios))
    metrics = accumulator.compute()
    metrics["loss"] = total_loss / max(1, len(loader))
    metrics["per_scenario"] = accumulator.per_scenario()
    return metrics


def checkpoint_payload(model, optimizer, scheduler, config, epoch, best_metric, current_metrics, split_hash) -> Dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_name": config["model"]["name"],
        "epoch": epoch,
        "architecture": config["model"],
        "best_metric": best_metric,
        "current_metrics": current_metrics,
        "config": config,
        "split_hash": split_hash,
        "seed": config["data"].get("seed", 6681),
        "amp": bool(config["training"].get("amp", True)),
        "threshold": float(config["training"].get("threshold", 0.5)),
    }


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train(config_path: Path, resume: Path | None = None, cpu: bool = False, overrides: argparse.Namespace | None = None) -> None:
    config = load_config(config_path)
    if overrides is not None:
        config = apply_overrides(config, overrides)
    set_seed(int(config["data"].get("seed", 6681)))
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    use_amp = bool(config["training"].get("amp", True) and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader = make_loaders(config)
    model = build_model(config["model"]).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config["training"]["epochs"])))
    scaler = GradScaler(device="cuda", enabled=use_amp)

    start_epoch = 1
    best_iou = -1.0
    experiment_name = config["experiment_name"]
    checkpoint_dir = Path(config["outputs"]["checkpoint_dir"])
    report_dir = Path(config["outputs"]["report_dir"])
    metrics_path = report_dir / f"{experiment_name}_metrics.csv"
    split_data = load_json(Path(config["data"]["processed"]) / "split.json")
    split_hash = stable_hash(split_data)

    if resume:
        payload = torch.load(resume, map_location="cpu")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_iou = float(payload.get("best_metric", -1.0))

    print(
        {
            "experiment": experiment_name,
            "model": config["model"],
            "device": str(device),
            "amp": use_amp,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "split_hash": split_hash,
        }
    )

    rows = []
    fieldnames = ["epoch", "train_loss", "val_loss", "iou", "dice", "precision", "recall", "pixel_accuracy", "ap", "max_f", "seconds"]
    total_epochs = int(config["training"]["epochs"])
    for epoch in range(start_epoch, total_epochs + 1):
        started = time.perf_counter()
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{total_epochs} train", leave=True)
        for images, masks, _scenarios, _ids in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{running_loss / max(1, progress.n):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        scheduler.step()

        val = evaluate_epoch(model, val_loader, criterion, device, float(config["training"].get("threshold", 0.5)), use_amp, desc=f"epoch {epoch}/{total_epochs} val")
        row = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, len(train_loader)), 6),
            "val_loss": round(val["loss"], 6),
            "iou": round(val["iou"], 6),
            "dice": round(val["dice"], 6),
            "precision": round(val["precision"], 6),
            "recall": round(val["recall"], 6),
            "pixel_accuracy": round(val["pixel_accuracy"], 6),
            "ap": round(val["ap"], 6),
            "max_f": round(val["max_f"], 6),
            "seconds": round(time.perf_counter() - started, 2),
        }
        append_csv_row(metrics_path, row, fieldnames)
        rows.append(row)
        print(row)

        payload = checkpoint_payload(model, optimizer, scheduler, config, epoch, best_iou, val, split_hash)
        save_checkpoint(checkpoint_dir / f"{experiment_name}_last.pt", payload)
        if val["iou"] > best_iou:
            best_iou = float(val["iou"])
            payload["best_metric"] = best_iou
            save_checkpoint(checkpoint_dir / f"{experiment_name}_best.pt", payload)
            print(f"saved best checkpoint: {checkpoint_dir / f'{experiment_name}_best.pt'} iou={best_iou:.4f}")
        if int(config["training"].get("save_every", 10)) > 0 and epoch % int(config["training"].get("save_every", 10)) == 0:
            save_checkpoint(checkpoint_dir / f"{experiment_name}_epoch_{epoch:03d}.pt", payload)

    plot_curves(rows, report_dir, experiment_name)
    save_json({"experiment": experiment_name, "best_iou": best_iou, "metrics_csv": str(metrics_path)}, report_dir / f"{experiment_name}_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--amp", action="store_true", help="Force AMP on when CUDA is available.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even when config enables it.")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--processed", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    train(args.config, args.resume, args.cpu, args)


if __name__ == "__main__":
    main()
