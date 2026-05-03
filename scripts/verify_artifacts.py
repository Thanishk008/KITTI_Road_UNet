from __future__ import annotations

import argparse
from pathlib import Path


EXPERIMENTS = ["road_unet", "plain_unet", "no_skip_unet"]


def any_exists(paths: list[Path]) -> bool:
    return any(path.exists() for path in paths)


def required_paths(checkpoint_dir: Path, report_dir: Path) -> list[Path]:
    paths: list[Path] = [
        report_dir / "mask_audit" / "mask_audit.json",
        report_dir / "mask_audit" / "mask_audit_preview.png",
        report_dir / "model_figure.png",
        report_dir / "experiment_summary.csv",
        report_dir / "analysis" / "model_comparison.csv",
        report_dir / "analysis" / "model_comparison.md",
        report_dir / "analysis" / "per_scenario_metrics.csv",
        report_dir / "analysis" / "report_numbers.json",
        report_dir / "analysis" / "combined_train_loss.png",
        report_dir / "analysis" / "combined_val_loss.png",
        report_dir / "analysis" / "combined_iou.png",
        report_dir / "analysis" / "combined_dice.png",
        report_dir / "analysis" / "road_unet_overlay_contact_sheet.png",
        report_dir / "analysis" / "road_unet_error_contact_sheet.png",
    ]
    for experiment in EXPERIMENTS:
        model_report_dir = report_dir / experiment
        paths.extend(
            [
                model_report_dir / f"{experiment}_metrics.csv",
                model_report_dir / f"{experiment}_summary.json",
                model_report_dir / f"{experiment}_val_evaluation.json",
                model_report_dir / f"{experiment}_val_threshold_sweep.csv",
                model_report_dir / f"{experiment}_loss_curves.png",
                model_report_dir / f"{experiment}_iou_curves.png",
                model_report_dir / f"{experiment}_val_pr_curve.png",
                report_dir / "qualitative_overlays" / experiment,
                report_dir / "error_examples" / experiment,
            ]
        )
    return paths


def missing_checkpoint_groups(checkpoint_dir: Path) -> list[str]:
    missing = []
    for experiment in EXPERIMENTS:
        candidates = {
            "best": [
                checkpoint_dir / experiment / "best.pt",
                checkpoint_dir / experiment / f"{experiment}_best.pt",
                checkpoint_dir / f"{experiment}_best.pt",
            ],
            "last": [
                checkpoint_dir / experiment / "last.pt",
                checkpoint_dir / experiment / f"{experiment}_last.pt",
                checkpoint_dir / f"{experiment}_last.pt",
            ],
        }
        for name, paths in candidates.items():
            if not any_exists(paths):
                missing.append(f"{experiment} {name} checkpoint. Checked: {', '.join(str(path) for path in paths)}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that all A-level submission artifacts exist.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    missing = [path for path in required_paths(args.checkpoint_dir, args.report_dir) if not path.exists()]
    missing_checkpoints = missing_checkpoint_groups(args.checkpoint_dir)
    if missing:
        print("Missing required artifacts:")
        for path in missing:
            print(f"- {path}")
    if missing_checkpoints:
        print("Missing required checkpoints:")
        for item in missing_checkpoints:
            print(f"- {item}")
    if missing or missing_checkpoints:
        raise SystemExit(1)
    print("All required experiment artifacts are present.")


if __name__ == "__main__":
    main()
