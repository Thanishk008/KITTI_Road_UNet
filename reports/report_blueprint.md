# A+ Report Blueprint And Rubric Crosswalk

Use this file as the single source of truth while writing the final PDF and preparing the presentation. It correlates the course guideline PDF with the current codebase, generated artifacts, and report content needed for a strong graduate-level submission.

This replaces the older short report scaffold; use this file as the baseline for the final report.

## Target Framing

Working title:

```text
Low-Resource RGB Road Segmentation on KITTI with From-Scratch U-Net Variants
```

Core novelty/problem claim:

```text
This project proposes a practical low-resource autonomous-driving perception problem:
can a from-scratch RGB-only U-Net learn reliable drivable-road segmentation from a small
labeled KITTI split without pretrained weights, stereo, LiDAR, or public test labels?
```

Method hypothesis:

```text
Residual convolution blocks with GroupNorm should improve small-batch optimization and
validation quality compared with a plain U-Net and a no-skip U-Net under the same split,
loss, optimizer, image size, and evaluation protocol.
```

Why this is impactful:

- Road segmentation supports free-space perception for autonomous driving and driver assistance.
- RGB-only segmentation is lower-cost than LiDAR/stereo-heavy approaches.
- From-scratch small-data training is challenging and realistic for class-scale constraints.
- KITTI has only 289 labeled road training images and no public labels for the 290 test images, so rigorous validation splitting matters.

## PDF Rubric Crosswalk

| PDF item                   | Weight | What the final report must contain                                                                                                                            | Code/artifacts that support it                                                                                           | Status before Colab                   |
| -------------------------- | -----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| Novelty                    |    20% | Frame a new, interesting, non-trivial problem: low-resource RGB-only road segmentation on KITTI. Explain why from-scratch learning is useful and challenging. | `kitti_road/models.py`, `configs/*.yaml`, baseline/ablation experiments.                                                 | Framing present; needs final results. |
| Length                     |     5% | Final PDF should be at least 4 pages; 6-8 pages is ideal. Bibliography/appendix do not count.                                                                 | This blueprint.                                                                                                          | Needs final PDF.                      |
| Code and appendix          |    25% | Submit compressed PDF plus code. Include commands, configs, checkpoints, metrics, plots, and enough appendix detail to reproduce.                             | `README.md`, `scripts/run_everything.py`, `scripts/verify_artifacts.py`, `configs/*.yaml`, `checkpoints/*`, `reports/*`. | Code ready; results pending.          |
| Abstract                   |     5% | One paragraph summarizing problem, method, dataset, baselines, and headline result.                                                                           | `reports/experiment_summary.csv` after evaluation.                                                                       | Draft placeholder exists.             |
| Introduction               |     5% | State road segmentation problem, why it matters, why low-resource RGB-only setting is challenging.                                                            | Dataset audit and model setup.                                                                                           | Draft present; refine after results.  |
| Related Work               |     5% | Discuss U-Net, KITTI Road, FCN/encoder-decoder segmentation context, and road segmentation methods using stereo/LiDAR/pretraining as contrast or future work. | README dataset link and final citations.                                                                                 | Needs citations in final PDF.         |
| Model/Method               |    15% | Include original computation graph figure, formal model description, loss, preprocessing, ignore handling, and differences from baselines.                    | `reports/model_figure.png`, `kitti_road/models.py`, `kitti_road/losses.py`, `kitti_road/datasets.py`.                    | Figure generated; write details.      |
| Experiments                |    15% | Include baseline comparison, ablation study, quantitative results, qualitative overlays, dataset details, metrics, training protocol, tricks.                 | Training/evaluation scripts, metrics, plots, overlays, mask audit.                                                       | Pipeline ready; run Colab.            |
| Conclusion and future work |     5% | Summarize takeaways, limitations, and future directions. Must honestly state whether residual GroupNorm helped.                                               | Final metrics and qualitative errors.                                                                                    | Pending results.                      |

## Codebase Details To Mention In The Report

### KITTI dataset discovery and model-design lesson

Include this as a short subsection in the Introduction, Dataset section, or Discussion:

```text
A key discovery during project setup was that KITTI is not a single dataset but a family of benchmarks with different annotation types. Some Kaggle-hosted KITTI variants are designed for object detection and provide bounding boxes, calibration files, and sometimes LiDAR data. Those annotations are not directly compatible with a U-Net road-segmentation pipeline, which requires dense pixel-wise masks.

This clarified an important model-design lesson: better KITTI models require aligning the architecture, supervision, and evaluation protocol with the specific KITTI benchmark. Object-detection KITTI variants require bounding-box detectors, road/lane segmentation requires dense segmentation networks, and stereo/LiDAR KITTI variants can support multimodal models. For this project, the correct alignment is KITTI Road/Lane Detection 2013 with `gt_image_2` pixel-level masks, a binary U-Net output head, pixel-wise BCE/Dice loss, and IoU/Dice-style segmentation metrics.

After correcting the setup to use KITTI Road/Lane Detection 2013 and auditing the color-coded masks, the training objective became aligned with the U-Net output: one road/background prediction per pixel. This dataset-model alignment is part of how stronger models can be built on KITTI datasets: first select the correct KITTI task and annotation type, then design the model and loss for that supervision.
```

Short version for presentation:

```text
The main dataset lesson is that KITTI contains multiple benchmarks. Better KITTI models come from matching the model to the annotation type: object detectors for bounding boxes, U-Nets for dense road masks, and multimodal models for stereo/LiDAR tasks.
```

### Dataset and split

- Dataset: KITTI Road/Lane Detection 2013.
- Task: binary road/background segmentation.
- Input: RGB left camera images from `training/image_2`.
- Ground truth: color masks from `training/gt_image_2`.
- Public test labels are unavailable; evaluation uses deterministic validation split from the labeled training set.
- Split implementation: `scripts/prepare_kitti_road.py`.
- Split strategy: stratified by KITTI scenario prefix: `um`, `umm`, `uu`.
- Default validation fraction: `0.2`.
- Default seed: `6681`.
- Required report artifact: `data/processed/kitti_road/stats.json`.
- Required values to paste: total labeled count, train count, validation count, scenario counts, split hash.

### Mask conversion and label audit

- Conversion code: `kitti_road/datasets.py`.
- Expected color rule:
  - magenta/white-like pixels -> road `1`
  - red pixels -> valid background `0`
  - black pixels -> ignore/void `255`
- Ignore pixels are excluded from loss and metrics.
- Audit command:

```bash
python scripts/audit_kitti_masks.py --data /content/data/data_road --out reports/mask_audit
```

- Required report artifacts:
  - `reports/mask_audit/mask_audit.json`
  - `reports/mask_audit/mask_audit_preview.png`
- Required text: "Before training, I audited raw KITTI mask colors and verified the conversion used for training/evaluation."

### Preprocessing and augmentation

- Image size: `384 x 1216`, from `configs/*.yaml`.
- Resize method:
  - images: bilinear
  - masks: nearest-neighbor
- Normalization: ImageNet mean/std in `image_to_tensor`.
- Training augmentations:
  - random horizontal flip
  - mild brightness jitter
  - mild contrast jitter
  - mild color jitter
- Validation/test augmentation: none.

### Models

- Final model: `ResidualRoadUNet`.
- Baseline 1: `PlainUNet`.
- Baseline 2/ablation: `NoSkipUNet`.
- Model code: `kitti_road/models.py`.
- Shared U-Net details:
  - depth fixed at 4 for a clear computation graph
  - base channels default to 32
  - encoder uses max pooling
  - decoder uses transposed convolutions
  - skip connections concatenate encoder features into decoder features
  - output head is a 1x1 convolution producing one logit per pixel
- Residual model details:
  - residual two-convolution blocks
  - GroupNorm for small-batch training
  - ReLU activations
- Figure command:

```bash
python scripts/make_model_figure.py
```

- Required report artifact: `reports/model_figure.png`.

### Loss, optimizer, and training protocol

- Loss code: `kitti_road/losses.py`.
- Loss: `0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss`.
- Valid/ignore handling: ignore index `255` is masked out.
- Optimizer: AdamW.
- Learning rate: `0.0003`.
- Weight decay: `0.0001`.
- Scheduler: CosineAnnealingLR.
- Epochs: `80`.
- Batch size: `8`.
- Mixed precision: AMP enabled on CUDA.
- Checkpoint cadence: best, last, and every 10 epochs.
- Device logic: CUDA if available unless `--cpu`.
- Configs:
  - `configs/road_unet.yaml`
  - `configs/plain_unet.yaml`
  - `configs/no_skip_unet.yaml`

`configs/fcn.yaml` and the FCN model were removed from the required experiment set. The A+ report should focus on the lean three-model package.

### Metrics and evaluation

- Metric code: `kitti_road/metrics.py`.
- Evaluation code: `kitti_road/evaluate.py`.
- Threshold default: `0.5`.
- Metrics:
  - IoU
  - Dice/F1
  - precision
  - recall
  - pixel accuracy
  - AP
  - MaxF from threshold sweep
- Scenario-level metrics are computed by `um`, `umm`, `uu`.
- Required report artifacts:
  - `reports/experiment_summary.csv`
  - `reports/analysis/model_comparison.md`
  - `reports/analysis/model_comparison.csv`
  - `reports/analysis/<experiment>/<experiment>_val_evaluation.json`
  - `reports/analysis/<experiment>/<experiment>_val_threshold_sweep.csv`
  - `reports/analysis/<experiment>/<experiment>_val_pr_curve.png`
  - `reports/analysis/per_scenario_metrics.csv`

### Qualitative analysis

- Overlay/error-map code: `kitti_road/evaluate.py`, `kitti_road/predict.py`, `kitti_road/visualize.py`.
- Required artifacts:
  - `reports/qualitative_overlays/road_unet/*.png`
  - `reports/error_examples/road_unet/*.png`
  - optionally `reports/predictions/*_overlay.png`
- Error-map legend:
  - green: true positive
  - blue: false positive
  - red: false negative
- Required writing:
  - one success case
  - one false positive case
  - one false negative case
  - what these reveal about shadows, road boundaries, occlusions, or unusual surfaces

## Experiment Rubric Coverage

The code should satisfy multiple experiment options from the PDF:

| PDF experiment option                                                        | How this project covers it                                                                                       |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Compare your model/method with baselines on at least one real-world dataset. | ResidualRoadUNet vs PlainUNet and NoSkipUNet on KITTI Road.                                                      |
| Ablation study on specific design choices.                                   | Plain U-Net vs NoSkipUNet tests skip connections; ResidualRoadUNet vs PlainUNet tests residual blocks/GroupNorm. |
| Quantitative and/or qualitative analysis.                                    | Metrics CSV/JSON, PR curves, threshold sweeps, loss/IoU curves, overlays, error maps.                            |
| Dataset description, metrics, training details, tricks.                      | Dataset audit, split stats, mask conversion, configs, optimizer/loss/scheduler, AMP, augmentation.               |

Do not add a synthetic experiment unless time remains after the real KITTI experiments. The real-world comparison plus ablation plus quantitative/qualitative analysis is enough and cleaner.

## Required Tables For The Final PDF

### Table 1: Dataset and split

Fill from `data/processed/kitti_road/stats.json`.

| Dataset    | Total labeled | Train | Val | Scenarios | Val fraction | Seed | Split hash |
| ---------- | ------------: | ----: | --: | --------- | -----------: | ---: | ---------- |
| KITTI Road |           TBD |   TBD | TBD | TBD       |          0.2 | 6681 | TBD        |

### Table 2: Model comparison

Fill from `reports/experiment_summary.csv`.

| Model          |    IoU | Dice/F1 | Precision | Recall | Pixel Acc. |     AP |   MaxF |
| -------------- | -----: | ------: | --------: | -----: | ---------: | -----: | -----: |
| Plain U-Net    | 0.9063 |  0.9509 |    0.9447 | 0.9571 |     0.9824 | 0.9794 | 0.9511 |
| No-skip U-Net  |    TBD |     TBD |       TBD |    TBD |        TBD |    TBD |    TBD |
| Residual U-Net | 0.9073 |  0.9514 |    0.9487 | 0.9541 |     0.9827 | 0.9841 | 0.9514 |

Current main-model validation result:

```text
ResidualRoadUNet achieved 0.9073 validation IoU and 0.9514 Dice/F1 on the deterministic KITTI validation split. Precision and recall were balanced at 0.9487 and 0.9541, respectively, with AP=0.9841 and MaxF=0.9514.
```

Important caveat:

```text
These are validation-split results from labeled KITTI training images, not official hidden KITTI test-server results.
```

Per-scenario ResidualRoadUNet metrics:

| Scenario |    IoU | Dice/F1 | Precision | Recall | Pixel Acc. | Interpretation                                                            |
| -------- | -----: | ------: | --------: | -----: | ---------: | ------------------------------------------------------------------------- |
| `um`     | 0.9053 |  0.9503 |    0.9405 | 0.9603 |     0.9840 | Strong marked-road performance.                                           |
| `umm`    | 0.9378 |  0.9679 |    0.9767 | 0.9592 |     0.9848 | Best scenario; multi-lane markings likely provide clearer road structure. |
| `uu`     | 0.8634 |  0.9267 |    0.9144 | 0.9393 |     0.9794 | Hardest scenario; unmarked roads make boundaries less explicit.           |

Current Plain U-Net validation result:

```text
Plain U-Net achieved 0.9063 validation IoU and 0.9509 Dice/F1, nearly matching the residual U-Net. Its precision was 0.9447, recall was 0.9571, AP was 0.9794, and MaxF was 0.9511.
```

Plain U-Net vs ResidualRoadUNet interpretation:

```text
The residual U-Net and plain U-Net are effectively tied overall: 0.9073 IoU for ResidualRoadUNet versus 0.9063 IoU for Plain U-Net. This suggests that on the deterministic KITTI validation split, the skip-connected U-Net encoder-decoder structure is the dominant factor, while residual blocks and GroupNorm provide only marginal aggregate improvement. The scenario-level results are mixed: ResidualRoadUNet performs better on urban multiple marked-lane scenes, while Plain U-Net slightly outperforms it on unmarked roads.
```

Per-scenario Plain U-Net metrics:

| Scenario |    IoU | Dice/F1 | Precision | Recall | Pixel Acc. | Interpretation                                                  |
| -------- | -----: | ------: | --------: | -----: | ---------: | --------------------------------------------------------------- |
| `um`     | 0.9063 |  0.9508 |    0.9382 | 0.9638 |     0.9842 | Slightly stronger than ResidualRoadUNet on marked urban roads.  |
| `umm`    | 0.9273 |  0.9623 |    0.9622 | 0.9624 |     0.9820 | Strong, but below ResidualRoadUNet on multi-marked-lane scenes. |
| `uu`     | 0.8738 |  0.9326 |    0.9239 | 0.9415 |     0.9812 | Slightly better than ResidualRoadUNet on unmarked roads.        |

### Table 3: Ablation interpretation

| Comparison                    | Design choice tested        | Expected interpretation                                               | Observed result                                                                                                                     |
| ----------------------------- | --------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Plain U-Net vs No-skip U-Net  | Skip connections            | Skip connections should improve spatial detail and boundary recovery. | TBD                                                                                                                                 |
| Residual U-Net vs Plain U-Net | Residual blocks + GroupNorm | Residual/GroupNorm should improve optimization under small batches.   | Near-tie overall: ResidualRoadUNet IoU 0.9073 vs Plain U-Net IoU 0.9063. Residual helps on `umm`; Plain is slightly better on `uu`. |

### Table 4: Training setup

| Setting      | Value                                                                 |
| ------------ | --------------------------------------------------------------------- |
| Image size   | 384 x 1216                                                            |
| Epochs       | 80                                                                    |
| Batch size   | 8 default; Colab A100 runs used `--batch-size 16` when memory allowed |
| Optimizer    | AdamW                                                                 |
| LR           | 0.0003                                                                |
| Weight decay | 0.0001                                                                |
| Scheduler    | CosineAnnealingLR                                                     |
| Loss         | BCEWithLogits + Dice                                                  |
| AMP          | enabled on CUDA                                                       |
| Threshold    | 0.5                                                                   |
| Seed         | 6681                                                                  |

## Required Figures

- Figure 1: Problem example with RGB image and ground-truth mask.
- Figure 2: Original model computation graph: `reports/model_figure.png`.
- Figure 3: Mask audit preview: `reports/mask_audit/mask_audit_preview.png`.
- Figure 4: Training/validation loss curves for all models or the final model plus baselines.
- Figure 5: IoU/Dice curves for all models or the final model plus baselines.
- Figure 6: PR curve or threshold sweep for the final model.
- Figure 7: Qualitative overlay success examples.
- Figure 8: Error maps showing false positives and false negatives.

Report-ready generated analysis files:

- `reports/analysis/model_comparison.md`: paste-ready model comparison table.
- `reports/analysis/report_numbers.json`: dataset stats, mask audit, model comparison, per-scenario metrics.
- `reports/analysis/combined_train_loss.png`
- `reports/analysis/combined_val_loss.png`
- `reports/analysis/combined_iou.png`
- `reports/analysis/combined_dice.png`
- `reports/analysis/<experiment>/<experiment>_val_evaluation.json`
- `reports/analysis/<experiment>/<experiment>_val_threshold_sweep.csv`
- `reports/analysis/<experiment>/<experiment>_val_pr_curve.png`
- `reports/analysis/<experiment>/<experiment>_overlay_contact_sheet.png`
- `reports/analysis/<experiment>/<experiment>_error_contact_sheet.png`

## Suggested Final Report Structure

Aim for 6-8 pages excluding bibliography/appendix.

1. Abstract
   - 150-200 words.
   - Include problem, method, dataset, baselines, metrics, best result, and main takeaway.

2. Introduction
   - Define road segmentation.
   - Explain low-resource RGB-only setting.
   - Explain impact and challenge.
   - State contributions:
     - problem framing
     - from-scratch residual U-Net
     - baseline/ablation study
     - quantitative and qualitative KITTI analysis

3. Related Work
   - U-Net.
   - FCN.
   - KITTI Road.
   - Road segmentation with extra sensors/pretraining as contrast.

4. Model/Method
   - Include computation graph.
   - Define preprocessing, mask conversion, model, loss, optimizer.
   - Clearly distinguish final model from baselines.

5. Experiments
   - Dataset and split.
   - Mask audit.
   - Training setup table.
   - Model comparison table.
   - Ablation table.
   - Curves and qualitative examples.

6. Discussion
   - Did residual GroupNorm help?
   - Did skip connections help?
   - Which scenes failed and why?
   - Discuss AP/MaxF and threshold sensitivity if relevant.

7. Conclusion and Future Work
   - Main takeaways.
   - Limitations.
   - Future directions: stereo/LiDAR, carefully designed pretrained fine-tuning, larger datasets, CRF/boundary refinement, threshold tuning.

8. Appendix
   - Commands.
   - Full configs.
   - Extra qualitative examples.
   - Extra per-scenario metrics.

## Presentation Checklist

The PDF also includes a presentation rubric. For a 10-minute individual talk or 15-minute group talk:

- Motivation/problem importance: explain low-resource RGB road segmentation.
- Related work: U-Net, FCN, KITTI Road, modern sensor-heavy road segmentation.
- Main idea: residual GroupNorm U-Net and controlled baselines.
- Main figure: use `reports/model_figure.png`.
- Experiments/results: model comparison, ablation, qualitative overlays.
- Finish under time: rehearse and keep 1-2 backup slides only.

## Colab Execution Checklist

Use Google Colab A100 or another CUDA GPU runtime. Confirm CUDA before training:

```bash
python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

Run from `/content/KITTI_Road_UNet` after installing requirements and unzipping KITTI. This is the full long-running training/evaluation pipeline, not a quick verification command:

```bash
python scripts/run_everything.py --data /content/data/data_road
```

`run_everything.py` executes CUDA checks, mask audit, split preparation, figure generation, all three required model trainings, evaluation for all best checkpoints, report-ready analysis generation, and final artifact verification.

If running manually:

```bash
python scripts/audit_kitti_masks.py --data /content/data/data_road --out reports/mask_audit
python scripts/prepare_kitti_road.py --data /content/data/data_road --out data/processed/kitti_road
python scripts/make_model_figure.py
python -m kitti_road.train --config configs/road_unet.yaml
python -m kitti_road.train --config configs/plain_unet.yaml
python -m kitti_road.train --config configs/no_skip_unet.yaml
python -m kitti_road.evaluate --checkpoint checkpoints/road_unet/road_unet_best.pt --split val
python -m kitti_road.evaluate --checkpoint checkpoints/plain_unet/plain_unet_best.pt --split val
python -m kitti_road.evaluate --checkpoint checkpoints/no_skip_unet/no_skip_unet_best.pt --split val
python scripts/summarize_analysis.py
python scripts/verify_artifacts.py
```

For Colab A100 tuning, training configs can be overridden from the command line. YAML values remain the defaults, currently using `batch_size: 8` and `num_workers: 8`.

```bash
python -m kitti_road.train --config configs/road_unet.yaml --batch-size 12
```

Useful override flags: `--epochs`, `--batch-size`, `--num-workers`, `--lr`, `--weight-decay`, `--save-every`, `--threshold`, `--amp`, `--no-amp`, `--image-size HEIGHT WIDTH`, `--processed`, `--checkpoint-dir`, `--report-dir`.

## Scripts Policy

Keep all scripts in the repo. They are reproducibility tools, not hidden internal scratch files:

- `audit_kitti_masks.py`: required label sanity check.
- `prepare_kitti_road.py`: required deterministic split creation.
- `make_model_figure.py`: required original model figure.
- `run_everything.py`: required one-command A100 workflow for the complete long-running experiment pipeline.
- `summarize_analysis.py`: required post-evaluation report analysis tables, combined curves, and contact sheets.
- `verify_artifacts.py`: required final artifact checklist.

## Environment Macros

Use `.env.example` as the single macro reference for notebooks and shell cells. The Python code primarily uses CLI arguments and YAML configs, but these macros keep paths consistent across Colab:

- `PROJECT_DIR=/content/KITTI_Road_UNet`
- `KITTI_ROAD_ROOT=/content/data/data_road`
- `KITTI_PROCESSED_DIR=data/processed/kitti_road`
- `REPORT_DIR=reports`
- `CHECKPOINT_DIR=checkpoints`
- `REQUIRE_CUDA=true`

## Gitignore Policy

Do not commit raw data, processed data, checkpoints, logs, archives, generated report CSV/JSON/PNG/PDF artifacts, notebook checkpoints, or experiment tracker folders. Keep source code, configs, README files, `.env.example`, and `reports/report_blueprint.md`.

## Final Submission Package Checklist

Include:

- Final PDF report.
- Source code.
- `README.md`.
- `configs/*.yaml`.
- `reports/report_blueprint.md`.
- `reports/mask_audit/`.
- `reports/model_figure.png`.
- `reports/<experiment>/<experiment>_metrics.csv`.
- `reports/*_summary.json`.
- `reports/analysis/<experiment>/<experiment>_val_evaluation.json`.
- `reports/analysis/<experiment>/<experiment>_val_threshold_sweep.csv`.
- `reports/<experiment>/<experiment>_loss_curves.png`.
- `reports/<experiment>/<experiment>_iou_curves.png`.
- `reports/analysis/<experiment>/<experiment>_val_pr_curve.png`.
- `reports/experiment_summary.csv`.
- `reports/analysis/`.
- `reports/qualitative_overlays/<experiment>/`.
- `reports/error_examples/<experiment>/`.
- `checkpoints/<experiment>/<experiment>_best.pt`.
- `checkpoints/<experiment>/<experiment>_last.pt`.

Optional appendix artifacts:

- Extra checkpoints every 10 epochs.
- Prediction overlays on unlabeled/public test images, clearly labeled as qualitative-only.
- Per-scenario metric table from each `reports/analysis/<experiment>/<experiment>_val_evaluation.json`.

## Common A+ Risks To Avoid

- Do not claim official KITTI test performance unless submitting to the hidden benchmark; this project uses validation split performance.
- Do not let the novelty sound like "I used U-Net." The novelty is the low-resource problem framing plus controlled empirical study.
- Do not omit baselines; baseline comparison is central to the experiment grade.
- Do not omit ablation; it directly supports the method claim.
- Do not report metrics without qualitative examples; segmentation needs visual evidence.
- Do not include borrowed architecture figures; use the generated original figure.
- Do not ignore mask conversion; include the audit preview and explain the road/background/ignore rule.
- Do not overclaim if residual U-Net does not win; interpret results honestly and discuss limitations.

## Exact Post-Training Fill-Ins

Replace every `TBD` in this blueprint and the final PDF with:

- Best model by IoU.
- Best model by MaxF.
- Whether residual U-Net beats plain U-Net.
- Whether plain U-Net beats no-skip U-Net.
- Dataset split counts and scenario counts.
- Mask audit top colors and converted pixel ratios.
- One success example filename.
- One false positive example filename.
- One false negative example filename.
- Main limitation observed in qualitative examples.
