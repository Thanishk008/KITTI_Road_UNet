# KITTI Road U-Net

Deep learning project for **ESI6681 Image Segmentation with U-Net**. The core model is a from-scratch residual U-Net for binary road/background segmentation on the **KITTI Road/Lane Detection 2013** dataset. The required A-level experiment set also includes a pretrained ResNet34 U-Net transfer-learning comparison, clearly disclosed as ImageNet-pretrained and fine-tuned on KITTI.

## What This Implements

- Binary road segmentation from RGB KITTI driving images.
- Main from-scratch model: `ResidualRoadUNet`.
- From-scratch baselines/ablations: `PlainUNet` and `NoSkipUNet`.
- Advanced comparison: `PretrainedResNet34UNet` with a torchvision ImageNet-pretrained encoder.
- KITTI mask audit, deterministic train/validation split, checkpoints, metrics, curves, PR sweeps, overlays, and error maps.
- A report source-of-truth: `reports/a_plus_report_blueprint.md`.

## Required Experiments

The default Colab pipeline trains and evaluates:

```text
road_unet
plain_unet
no_skip_unet
pretrained_resnet34_unet
```

`fcn` was intentionally dropped from the required run to keep Colab A100 runtime reasonable while preserving the rubric: real-world baseline comparison, ablation, transfer-learning comparison, quantitative analysis, and qualitative analysis.

## Data

Download KITTI Road/Lane Detection 2013 manually:

https://www.cvlibs.net/datasets/kitti/eval_road.php

Expected layout after unzip:

```text
data_road/
  training/
    image_2/*.png
    gt_image_2/*.png
  testing/
    image_2/*.png
```

The public KITTI test split has no labels, so this project evaluates on a deterministic validation split from the labeled training set.

## Environment Macros

Use `.env.example` as the uniform macro reference. In Colab, the defaults assume:

```text
PROJECT_DIR=/content/KITTI_Road_UNet
KITTI_ROAD_ROOT=/content/data/data_road
KITTI_PROCESSED_DIR=data/processed/kitti_road
REPORT_DIR=reports
CHECKPOINT_DIR=checkpoints
```

The Python pipeline still accepts CLI args, so `.env` is mainly for reproducible notebooks/shell cells and project documentation.

## Google Colab A100 Workflow

Select a CUDA GPU runtime first:

```bash
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

Install dependencies, including CUDA-enabled PyTorch/torchvision from Colab:

```bash
cd /content/KITTI_Road_UNet
pip install -r requirements.txt
```

Unzip KITTI:

```bash
mkdir -p /content/data
unzip -q "/content/drive/MyDrive/data_road.zip" -d /content/data
```

Run the complete long-running experiment pipeline:

```bash
python scripts/run_everything.py --data /content/data/data_road
```

This is the full training/evaluation workflow, not just a verifier. It runs mask audit, split preparation, figure generation, all four required model trainings, evaluation for all best checkpoints, report-ready analysis generation, and final artifact verification. Expect it to take a long time on Colab A100 because it trains every required experiment.

## Manual Commands

Prepare data and figures:

```bash
python scripts/audit_kitti_masks.py --data /content/data/data_road --out reports/mask_audit
python scripts/prepare_kitti_road.py --data /content/data/data_road --out data/processed/kitti_road
python scripts/make_model_figure.py
```

Train:

```bash
python -m kitti_road.train --config configs/road_unet.yaml
python -m kitti_road.train --config configs/plain_unet.yaml
python -m kitti_road.train --config configs/no_skip_unet.yaml
python -m kitti_road.train --config configs/pretrained_resnet34_unet.yaml
```

The YAML defaults are set for a normal Colab A100 run with `batch_size: 8` and `num_workers: 8`. Optional tuning overrides can still be passed without editing YAML:

```bash
python -m kitti_road.train --config configs/road_unet.yaml --batch-size 12
python -m kitti_road.train --config configs/pretrained_resnet34_unet.yaml --batch-size 8 --lr 0.0001
```

Common override flags:

```text
--epochs
--batch-size
--num-workers
--lr
--weight-decay
--save-every
--threshold
--amp / --no-amp
--image-size HEIGHT WIDTH
--processed
--checkpoint-dir
--report-dir
```

Evaluate:

```bash
python -m kitti_road.evaluate --checkpoint checkpoints/road_unet/best.pt --split val
python -m kitti_road.evaluate --checkpoint checkpoints/plain_unet/best.pt --split val
python -m kitti_road.evaluate --checkpoint checkpoints/no_skip_unet/best.pt --split val
python -m kitti_road.evaluate --checkpoint checkpoints/pretrained_resnet34_unet/best.pt --split val
python scripts/verify_artifacts.py
```

Generate extra prediction overlays:

```bash
python -m kitti_road.predict --checkpoint checkpoints/road_unet/best.pt --images /content/data/data_road/training/image_2 --out reports/predictions
```

## Scripts

These are not disposable internal files; they make the project reproducible:

- `audit_kitti_masks.py`: verifies KITTI label colors before training.
- `prepare_kitti_road.py`: creates deterministic train/validation split.
- `make_model_figure.py`: generates the original method figure.
- `run_everything.py`: one-command Colab A100 pipeline for the full long-running experiment workflow.
- `summarize_analysis.py`: creates report-ready comparison tables, combined curves, and qualitative contact sheets after evaluation.
- `verify_artifacts.py`: checks final required outputs.

## Generated Analysis

After `run_everything.py` completes, use these files for the report:

- `reports/analysis/model_comparison.md`: paste-ready model comparison table.
- `reports/analysis/model_comparison.csv`: numeric table for spreadsheets.
- `reports/analysis/per_scenario_metrics.csv`: scenario-level metrics.
- `reports/analysis/report_numbers.json`: dataset stats, mask audit, model metrics, and report pointers.
- `reports/analysis/combined_train_loss.png`
- `reports/analysis/combined_val_loss.png`
- `reports/analysis/combined_iou.png`
- `reports/analysis/combined_dice.png`
- `reports/analysis/road_unet_overlay_contact_sheet.png`
- `reports/analysis/road_unet_error_contact_sheet.png`

Per-model outputs are organized under:

```text
reports/by_model/<experiment>/
checkpoints/<experiment>/
```

For example:

```text
reports/by_model/road_unet/metrics.csv
reports/by_model/road_unet/val_evaluation.json
reports/by_model/road_unet/loss_curves.png
reports/by_model/road_unet/qualitative_overlays/
checkpoints/road_unet/best.pt
```

## Report Source

Use only this master file for report planning:

```text
reports/a_plus_report_blueprint.md
```

It contains the rubric crosswalk, codebase details, tables, figures, presentation checklist, and final submission checklist. The pretrained ResNet34 model must be presented as a transfer-learning comparison; the from-scratch residual U-Net remains the course-designed model.

## Submission Safety

Large files are ignored by `.gitignore`: raw/processed data, checkpoints, logs, zips, generated CSV/JSON/PNG report artifacts, notebook checkpoints, and cache folders. Keep source code, configs, README files, and the A+ blueprint tracked.
