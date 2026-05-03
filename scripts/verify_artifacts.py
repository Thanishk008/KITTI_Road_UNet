from __future__ import annotations

import argparse
from pathlib import Path


EXPERIMENTS = ["road_unet", "plain_unet", "no_skip_unet"]


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
    ]
    for experiment in EXPERIMENTS:
        model_report_dir = report_dir / experiment
        model_analysis_dir = report_dir / "analysis" / experiment
        model_checkpoint_dir = checkpoint_dir / experiment
        paths.extend(
            [
                model_checkpoint_dir / f"{experiment}_best.pt",
                model_checkpoint_dir / f"{experiment}_last.pt",
                model_report_dir / f"{experiment}_metrics.csv",
                model_report_dir / f"{experiment}_summary.json",
                model_report_dir / f"{experiment}_loss_curves.png",
                model_report_dir / f"{experiment}_iou_curves.png",
                model_analysis_dir / f"{experiment}_val_evaluation.json",
                model_analysis_dir / f"{experiment}_val_threshold_sweep.csv",
                model_analysis_dir / f"{experiment}_val_pr_curve.png",
                model_analysis_dir / f"{experiment}_overlay_contact_sheet.png",
                model_analysis_dir / f"{experiment}_error_contact_sheet.png",
                report_dir / "qualitative_overlays" / experiment,
                report_dir / "error_examples" / experiment,
            ]
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that all A-level submission artifacts exist.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    missing = [path for path in required_paths(args.checkpoint_dir, args.report_dir) if not path.exists()]
    if missing:
        print("Missing required artifacts:")
        for path in missing:
            print(f"- {path}")
        raise SystemExit(1)
    print("All required experiment artifacts are present.")


if __name__ == "__main__":
    main()
