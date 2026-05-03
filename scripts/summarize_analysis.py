from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image


EXPERIMENTS = ["road_unet", "plain_unet", "no_skip_unet"]
DISPLAY_NAMES = {
    "road_unet": "Residual U-Net",
    "plain_unet": "Plain U-Net",
    "no_skip_unet": "No-skip U-Net",
}
METRIC_COLUMNS = ["iou", "dice", "precision", "recall", "pixel_accuracy", "ap", "max_f", "loss"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "TBD"


def build_model_comparison(report_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(report_dir / "experiment_summary.csv")
    latest_by_experiment = {row["experiment"]: row for row in rows if row.get("split") == "val"}
    comparison = []
    for experiment in EXPERIMENTS:
        row = latest_by_experiment.get(experiment, {})
        comparison.append(
            {
                "experiment": experiment,
                "model": DISPLAY_NAMES[experiment],
                **{metric: format_float(row.get(metric)) for metric in METRIC_COLUMNS},
                "checkpoint": row.get("checkpoint", ""),
            }
        )
    write_csv(comparison, out_dir / "model_comparison.csv", ["experiment", "model", *METRIC_COLUMNS, "checkpoint"])

    md_lines = [
        "| Model | IoU | Dice/F1 | Precision | Recall | Pixel Acc. | AP | MaxF | Loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        md_lines.append(
            f"| {row['model']} | {row['iou']} | {row['dice']} | {row['precision']} | {row['recall']} | "
            f"{row['pixel_accuracy']} | {row['ap']} | {row['max_f']} | {row['loss']} |"
        )
    (out_dir / "model_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return comparison


def build_per_scenario_table(report_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for experiment in EXPERIMENTS:
        metrics = read_json(report_dir / "by_model" / experiment / "val_evaluation.json") or {}
        for scenario, values in (metrics.get("per_scenario") or {}).items():
            row = {"experiment": experiment, "model": DISPLAY_NAMES[experiment], "scenario": scenario}
            row.update({metric: format_float(values.get(metric)) for metric in ["iou", "dice", "precision", "recall", "pixel_accuracy"]})
            rows.append(row)
    write_csv(rows, out_dir / "per_scenario_metrics.csv", ["experiment", "model", "scenario", "iou", "dice", "precision", "recall", "pixel_accuracy"])
    return rows


def plot_combined_curves(report_dir: Path, out_dir: Path) -> None:
    curve_specs = [
        ("train_loss", "Training loss", "combined_train_loss.png"),
        ("val_loss", "Validation loss", "combined_val_loss.png"),
        ("iou", "Validation IoU", "combined_iou.png"),
        ("dice", "Validation Dice/F1", "combined_dice.png"),
    ]
    for column, title, filename in curve_specs:
        plt.figure(figsize=(8, 5))
        wrote_any = False
        for experiment in EXPERIMENTS:
            rows = read_csv_rows(report_dir / "by_model" / experiment / "metrics.csv")
            if not rows:
                continue
            epochs = [int(row["epoch"]) for row in rows]
            values = [float(row[column]) for row in rows]
            plt.plot(epochs, values, label=DISPLAY_NAMES[experiment])
            wrote_any = True
        if wrote_any:
            plt.xlabel("Epoch")
            plt.ylabel(title)
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / filename, dpi=180)
        plt.close()


def make_contact_sheet(paths: list[Path], out_path: Path, thumb_width: int = 360, max_images: int = 8) -> None:
    selected = paths[:max_images]
    if not selected:
        return
    thumbs = []
    for path in selected:
        image = Image.open(path).convert("RGB")
        scale = thumb_width / image.width
        thumb = image.resize((thumb_width, max(1, int(image.height * scale))), Image.Resampling.BILINEAR)
        thumbs.append((path.name, thumb))
    label_height = 24
    width = thumb_width * 2
    rows = (len(thumbs) + 1) // 2
    height = rows * (max(thumb.height for _, thumb in thumbs) + label_height)
    canvas = Image.new("RGB", (width, height), "white")
    for idx, (name, thumb) in enumerate(thumbs):
        x = (idx % 2) * thumb_width
        y = (idx // 2) * (thumb.height + label_height)
        canvas.paste(thumb, (x, y + label_height))
        # PIL default font is enough; filename helps trace examples in the report.
        from PIL import ImageDraw

        ImageDraw.Draw(canvas).text((x + 4, y + 4), name, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_contact_sheets(report_dir: Path, out_dir: Path) -> None:
    overlay_paths = sorted((report_dir / "by_model" / "road_unet" / "qualitative_overlays").glob("*.png"))
    error_paths = sorted((report_dir / "by_model" / "road_unet" / "error_examples").glob("*.png"))
    make_contact_sheet(overlay_paths, out_dir / "road_unet_overlay_contact_sheet.png")
    make_contact_sheet(error_paths, out_dir / "road_unet_error_contact_sheet.png")


def summarize(report_dir: Path, processed_dir: Path) -> None:
    out_dir = report_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_model_comparison(report_dir, out_dir)
    per_scenario = build_per_scenario_table(report_dir, out_dir)
    plot_combined_curves(report_dir, out_dir)
    build_contact_sheets(report_dir, out_dir)

    report_numbers = {
        "model_comparison": comparison,
        "per_scenario_metrics": per_scenario,
        "dataset_stats": read_json(processed_dir / "stats.json"),
        "mask_audit": read_json(report_dir / "mask_audit" / "mask_audit.json"),
        "analysis_outputs": {
            "model_comparison_csv": str(out_dir / "model_comparison.csv"),
            "model_comparison_markdown": str(out_dir / "model_comparison.md"),
            "per_scenario_metrics_csv": str(out_dir / "per_scenario_metrics.csv"),
            "combined_plots_dir": str(out_dir),
        },
    }
    write_json(report_numbers, out_dir / "report_numbers.json")
    print(f"wrote analysis outputs under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create report-ready analysis tables and combined plots.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/kitti_road"))
    args = parser.parse_args()
    summarize(args.report_dir, args.processed_dir)


if __name__ == "__main__":
    main()
