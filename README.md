# KITTI Road U-Net

Deep learning project for **ESI6681 Image Segmentation with U-Net**. The project trains from-scratch U-Net-family models for binary road/background segmentation on the **KITTI Road/Lane Detection 2013** dataset.

## What This Implements

- Binary road segmentation from RGB KITTI driving images.
- Main model: `ResidualRoadUNet`.
- Baseline: `PlainUNet`.
- Ablation: `NoSkipUNet`.
- KITTI mask audit, deterministic train/validation split, checkpoints, metrics, curves, PR sweeps, overlays, error maps, and final analysis artifacts.

## Required Experiments

The final experiment set is:

```text
road_unet
plain_unet
no_skip_unet
```

These three experiments provide the main model, a baseline, and a skip-connection ablation under matched training conditions.

## Repository Tree

Current high-level repo layout:

```text
KITTI_Road_UNet/
├── checkpoints/
│   ├── no_skip_unet/
│   ├── plain_unet/
│   ├── road_unet/
│   └── .gitkeep
├── configs/
│   ├── no_skip_unet.yaml
│   ├── plain_unet.yaml
│   └── road_unet.yaml
├── data/
│   ├── data_road/
│   ├── processed/
│   └── README.md
├── kitti_road/
│   ├── __init__.py
│   ├── datasets.py
│   ├── evaluate.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   ├── predict.py
│   ├── train.py
│   ├── utils.py
│   └── visualize.py
├── reports/
│   ├── analysis/
│   ├── error_examples/
│   ├── mask_audit/
│   ├── no_skip_unet/
│   ├── plain_unet/
│   ├── qualitative_overlays/
│   ├── road_unet/
│   ├── .gitkeep
│   ├── experiment_summary.csv
│   ├── Final_Report.pdf
│   └── model_figure.png
├── scripts/
│   ├── audit_kitti_masks.py
│   ├── make_model_figure.py
│   ├── prepare_kitti_road.py
│   ├── run_everything.py
│   ├── summarize_analysis.py
│   └── verify_artifacts.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── kitti_road_dataset.zip
```

## Data

Use KITTI Road/Lane Detection 2013:

https://www.cvlibs.net/datasets/kitti/eval_road.php

Expected layout:

```text
data_road/
  training/
    image_2/*.png
    gt_image_2/*.png
  testing/
    image_2/*.png
```

The public KITTI test split has no labels, so this project evaluates on a deterministic validation split from the labeled training set.

## Colab GPU Workflow

Select a CUDA GPU runtime first:

```bash
!nvidia-smi
```

Install dependencies:

```bash
%cd /content/KITTI_Road_UNet
!pip install -r requirements.txt
```

Run manually to manage compute:

```bash
!python scripts/audit_kitti_masks.py --data "{DATA_ROAD}" --out reports/mask_audit
!python scripts/prepare_kitti_road.py --data "{DATA_ROAD}" --out data/processed/kitti_road
!python scripts/make_model_figure.py
```

Train and evaluate each model:

```bash
!python -m kitti_road.train --config configs/road_unet.yaml --batch-size 32 --num-workers 8 --lr 0.0005 --amp
!python -m kitti_road.evaluate --checkpoint checkpoints/road_unet/road_unet_best.pt --split val

!python -m kitti_road.train --config configs/plain_unet.yaml --batch-size 32 --num-workers 8 --lr 0.0005 --amp
!python -m kitti_road.evaluate --checkpoint checkpoints/plain_unet/plain_unet_best.pt --split val

!python -m kitti_road.train --config configs/no_skip_unet.yaml --batch-size 32 --num-workers 8 --lr 0.0005 --amp
!python -m kitti_road.evaluate --checkpoint checkpoints/no_skip_unet/no_skip_unet_best.pt --split val
```

Generate final analysis:

```bash
!python scripts/summarize_analysis.py --report-dir reports --processed-dir data/processed/kitti_road
!python scripts/verify_artifacts.py
```

You can also run the full pipeline with one command:

```bash
!python scripts/run_everything.py --data "{DATA_ROAD}" --processed data/processed/kitti_road --reports reports --checkpoint-dir checkpoints --batch-size 32 --num-workers 8 --lr 0.0005 --amp
```

This runs the full long-running pipeline: CUDA check, mask audit, split preparation, figure generation, all three model trainings, evaluation, analysis generation, and artifact verification.

## Output Layout

Per-model outputs are organized directly under `reports/`:

```text
checkpoints/<experiment>/
reports/<experiment>/
```

Example:

```text
checkpoints/road_unet/road_unet_best.pt
reports/road_unet/road_unet_metrics.csv
reports/road_unet/road_unet_loss_curves.png
reports/analysis/road_unet/road_unet_val_evaluation.json
reports/analysis/road_unet/road_unet_val_threshold_sweep.csv
reports/analysis/road_unet/road_unet_val_pr_curve.png
reports/qualitative_overlays/road_unet/
reports/error_examples/road_unet/
```

Combined report outputs:

```text
reports/analysis/model_comparison.md
reports/analysis/model_comparison.csv
reports/analysis/per_scenario_metrics.csv
reports/analysis/report_numbers.json
reports/analysis/combined_train_loss.png
reports/analysis/combined_val_loss.png
reports/analysis/combined_iou.png
reports/analysis/combined_dice.png
reports/analysis/<experiment>/<experiment>_val_evaluation.json
reports/analysis/<experiment>/<experiment>_val_threshold_sweep.csv
reports/analysis/<experiment>/<experiment>_val_pr_curve.png
reports/analysis/<experiment>/<experiment>_overlay_contact_sheet.png
reports/analysis/<experiment>/<experiment>_error_contact_sheet.png
```

## Scripts

- `audit_kitti_masks.py`: verifies KITTI label colors before training.
- `prepare_kitti_road.py`: creates deterministic train/validation split.
- `make_model_figure.py`: generates the original method figure.
- `run_everything.py`: optional one-command experiment pipeline.
- `summarize_analysis.py`: creates comparison tables, combined curves, and qualitative contact sheets.
- `verify_artifacts.py`: checks final required outputs.
