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

CHECKPOINTS = [
    "checkpoints/road_unet/best.pt",
    "checkpoints/plain_unet/best.pt",
    "checkpoints/no_skip_unet/best.pt",
]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Colab A100 experiment pipeline.")
    parser.add_argument("--data", type=Path, default=Path("/content/data/data_road"), help="Path to unzipped KITTI data_road directory.")
    parser.add_argument("--processed", type=Path, default=Path("data/processed/kitti_road"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--skip-train", action="store_true", help="Only run audit, figure generation, evaluation, analysis, and artifact checks.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution for debugging; not recommended for full training.")
    args = parser.parse_args()

    check_runtime(args.allow_cpu)
    run([sys.executable, "scripts/audit_kitti_masks.py", "--data", str(args.data), "--out", str(args.reports / "mask_audit")])
    run([sys.executable, "scripts/prepare_kitti_road.py", "--data", str(args.data), "--out", str(args.processed)])
    run([sys.executable, "scripts/make_model_figure.py"])

    if not args.skip_train:
        for config in CONFIGS:
            run([sys.executable, "-m", "kitti_road.train", "--config", config])

    for checkpoint in CHECKPOINTS:
        run([sys.executable, "-m", "kitti_road.evaluate", "--checkpoint", checkpoint, "--split", "val"])

    run([sys.executable, "scripts/summarize_analysis.py", "--report-dir", str(args.reports), "--processed-dir", str(args.processed)])
    run([sys.executable, "scripts/verify_artifacts.py"])


if __name__ == "__main__":
    main()
