from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIGS = [
    "configs/road_unet.yaml",
    "configs/plain_unet.yaml",
    "configs/no_skip_unet.yaml",
]

EXPERIMENTS = ["road_unet", "plain_unet", "no_skip_unet"]


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def check_runtime(allow_cpu: bool) -> None:
    import torch

    print(
        {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        flush=True,
    )
    if not torch.cuda.is_available() and not allow_cpu:
        raise SystemExit("CUDA GPU is required for the full Colab A100 pipeline. Reconnect with a GPU runtime or pass --allow-cpu for debugging only.")


def train_command(config: str, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-m", "kitti_road.train", "--config", config]
    command.extend(["--processed", str(args.processed)])
    command.extend(["--checkpoint-dir", str(args.checkpoint_dir)])
    command.extend(["--report-dir", str(args.reports)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    if args.lr is not None:
        command.extend(["--lr", str(args.lr)])
    if args.weight_decay is not None:
        command.extend(["--weight-decay", str(args.weight_decay)])
    if args.no_amp:
        command.append("--no-amp")
    elif args.amp:
        command.append("--amp")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Colab A100 experiment pipeline.")
    parser.add_argument("--data", type=Path, default=Path("/content/data/data_road"), help="Path to unzipped KITTI data_road directory.")
    parser.add_argument("--processed", type=Path, default=Path("data/processed/kitti_road"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--skip-train", action="store_true", help="Only run audit, figure generation, evaluation, analysis, and artifact checks.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution for debugging; not recommended for full training.")
    parser.add_argument("--epochs", type=int, help="Override training epochs for all configs.")
    parser.add_argument("--batch-size", type=int, help="Override training batch size for all configs.")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader workers for all configs.")
    parser.add_argument("--lr", type=float, help="Override learning rate for all configs.")
    parser.add_argument("--weight-decay", type=float, help="Override weight decay for all configs.")
    parser.add_argument("--amp", action="store_true", help="Force AMP on when CUDA is available.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even when configs enable it.")
    args = parser.parse_args()

    check_runtime(args.allow_cpu)
    run([sys.executable, "scripts/audit_kitti_masks.py", "--data", str(args.data), "--out", str(args.reports / "mask_audit")])
    run([sys.executable, "scripts/prepare_kitti_road.py", "--data", str(args.data), "--out", str(args.processed)])
    run([sys.executable, "scripts/make_model_figure.py", "--out", str(args.reports / "model_figure.png")])

    if not args.skip_train:
        for config in CONFIGS:
            run(train_command(config, args))

    for experiment in EXPERIMENTS:
        checkpoint = args.checkpoint_dir / experiment / f"{experiment}_best.pt"
        run(
            [
                sys.executable,
                "-m",
                "kitti_road.evaluate",
                "--checkpoint",
                str(checkpoint),
                "--split",
                "val",
                "--processed",
                str(args.processed),
                "--report-dir",
                str(args.reports),
            ]
        )

    run([sys.executable, "scripts/summarize_analysis.py", "--report-dir", str(args.reports), "--processed-dir", str(args.processed)])
    run([sys.executable, "scripts/verify_artifacts.py", "--checkpoint-dir", str(args.checkpoint_dir), "--report-dir", str(args.reports)])


if __name__ == "__main__":
    main()
