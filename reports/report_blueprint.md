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
labeled KITTI split without external model initialization, stereo, LiDAR, or public test labels?
```

Novelty target:

```text
Aim for the lowest credible A+ novelty band in the PDF: propose an interesting/impactful
and challenging/non-trivial problem, then support it with a careful empirical study.
Do not sell this as a fundamentally new segmentation architecture.
```

Method hypothesis:

```text
Residual convolution blocks should improve small-batch optimization and validation
quality compared with a plain GroupNorm U-Net and a no-skip GroupNorm U-Net under
the same split, loss, optimizer, image size, and evaluation protocol.
```

Why this is impactful:

- Road segmentation supports free-space perception for autonomous driving and driver assistance.
- RGB-only segmentation is lower-cost than LiDAR/stereo-heavy approaches.
- From-scratch small-data training is challenging and realistic for class-scale constraints.
- KITTI has only 289 labeled road training images and no public labels for the 290 test images, so rigorous validation splitting matters.

## Full PDF Context And Non-Negotiables

This section captures the course guideline PDF details that should shape the final report, appendix, and presentation. Treat the PDF as the authority; this section is a faithful working checklist for this project.

### General course-project policy

- Students may work individually or in groups of up to three.
- Larger groups are expected to do more than smaller groups or individual projects.
- Every student in a group receives an individual score.
- A project may relate to work or another external project, but the work submitted for this course must be substantial additional work and cannot be submitted for grades in another course.
- The grade depends on the proposed architectures/ideas, the quality of the report presentation, how clearly the work is positioned relative to prior literature, how illuminating or convincing the experiments are, and how well-supported the conclusions are.
- Full marks require a novel contribution.
- Students give a short presentation toward the end of the course: roughly 10 minutes for an individual and 15 minutes for a larger group.
- Every group must submit a project report at the end of the class.
- Reports, including the proposal and final report, are recommended to use NeurIPS conference format.
- If LaTeX is not used, Microsoft Word is acceptable as long as the report is exported as PDF.
- All submissions should be PDF.
- Late work is automatically penalized by the formula shown in the PDF: `min(100, e^d + 10)%`, where `d` is the number of days after the deadline.

### Final presentation rubric from the PDF

The presentation is about explaining the ideas clearly and concisely. The PDF calls it an online video and states 15 minutes, with 10 minutes for an individual. If results are limited or did not match expectations, the PDF explicitly says that is acceptable; explain what was planned, what was expected, and why the expected result was not obtained.

| Presentation item | Weight | Exact expectation to satisfy |
| ----------------- | -----: | ---------------------------- |
| Motivate and define the problem | 20% | Explain why the problem is important and what new thing can be learned if the project succeeds. |
| Briefly mention related work | 20% | Mention related work; the PDF says this can be left until the end if preferred. |
| Explain at least one main idea clearly | 20% | Teach the audience the main idea; clarity and educational value matter. |
| Show a draft of the main figure | 10% | Show the main figure or some visualization of the main idea. |
| Explain planned experiments or show results | 20% | Explain what can be learned from the experiments. |
| Finish under time | 10% | If the talk runs long, re-record with abridged wording. |

### Project-report purpose from the PDF

The report should demonstrate the ability to:

- Identify a problem where a deep learning model can be applied or designed to resolve it.
- Apply/design a deep learning architecture, with or without combinations of advanced state-of-the-art techniques, to solve the problem or answer the question.
- Analyze the results.
- Present findings and conclusions.

The PDF says the report is not expected to be a completed research paper, but the work should be as innovative as possible. It also frames the report as a manageable project whose contents would still be needed if the work later became a research paper. If a project does not fit the criteria exactly, the PDF encourages discussing it with the instructor because there are many ways to contribute to a field.

### Novelty scale from the PDF

| Approximate level | PDF description | How this project should position itself |
| ----------------- | --------------- | --------------------------------------- |
| `~B` | Literature review with no new content. | Avoid sounding like a survey or a generic U-Net reproduction. |
| `~A` | Novel combination of existing methods, empirical evidence that it works, and explanation of why it works. | The controlled residual-block, plain U-Net, and no-skip comparison can support this, but only with honest evidence. |
| `~A` | Existing method applied to a new dataset or problem domain with convincing empirical results. | This is partially covered by applying from-scratch U-Net variants to KITTI Road. |
| `~A+` | Propose a new problem that is interesting/impactful and challenging/non-trivial. | This is the target novelty tier: low-resource RGB-only KITTI Road segmentation without external model initialization, stereo/LiDAR, public test labels, or a large labeled dataset. |
| `~A+` | Analyze new properties of existing methods. | The ablation should discuss what the near-tie says about residual blocks and skip connections in this small-data setting, with GroupNorm held constant across variants. |
| `~A+` | Novel methods with evidence-supported motivation, not a random tweak. | Do not claim residual blocks won strongly; instead test the original hypothesis and explain the mixed evidence. |
| `~A+` | Rigorous theoretical analysis of existing phenomena. | Not the focus of this project. |

For full novelty marks, the PDF specifically warns that if the report claims a reason something should work better, it must check whether that reason is actually why it worked better. For this project, that means the final report must explicitly test and discuss whether residual block structure and skip connections actually helped under the observed metrics.

Recommended novelty stance for the final report:

```text
The contribution is not a new architecture in the strongest research-paper sense.
Instead, the A+ novelty claim is the problem formulation and analysis: a low-resource,
RGB-only, from-scratch KITTI Road segmentation setting, paired with controlled U-Net
variants that test which architectural assumptions still matter when labels and
supervision are limited.
```

Do not write:

```text
We propose a novel U-Net architecture that significantly outperforms baselines.
```

Write instead:

```text
We study a challenging low-resource autonomous-driving segmentation setting and
evaluate from-scratch U-Net variants under a controlled protocol. The results show
that residual blocks provide only a marginal aggregate IoU gain, while
scenario-level behavior differs across marked, multi-lane, and unmarked roads.
```

### Final report grading details from the PDF

- Novelty: 20%.
- Length: 5%. The report should be at least 4 pages; 6-8 pages is described as perfect because it matches standard conference-paper length. Appendices and bibliography do not count toward page count. The PDF says not to be afraid to keep the text short and to the point and to include large illustrative figures.
- Code and appendix: 25%. Submit a compressed file, such as a zip, containing the PDF and code unless doing a pure theoretical project. For pure theory, the appendix must include all proof. Appendices may include as many proofs, extra details, experiments, and related material as desired.
- Abstract: 5%. It should summarize the main idea and contributions.
- Introduction: 5%. It should clearly state the problem being addressed and why it is important.
- Related Work: 5%. It should clearly distinguish the proposed contribution from previous literature.
- Model/Method: 15%. It should make the paper accessible, especially to skimming readers.
- Experiments: 15%. It should include one or more of the listed experiment types.
- Conclusion and Future Work: 5%. It should state main takeaways, limitations, and future directions.

### Model/method writing requirements from the PDF

- Include a figure illustrating the main computation graph of the model.
- A nice figure can receive bonus credit.
- The figure must be newly created for this project; do not use someone else's figure, even with attribution.
- Equations are helpful when notation is rigorous and concise.
- An algorithm box is useful when the proposed method is hard to parse from text alone.
- Give a formal description of models, loss functions, conjectures, problem domains, theorems, propositions, or other formal elements when relevant.
- Highlight how the model differs from other approaches, for example with figures or tables.

### Experiment options from the PDF

The experiments section should include one or more of these:

- Compare the model/method with baselines on at least one real-world dataset.
- Run an ablation study on specific design choices.
- Use a synthetic dataset to demonstrate a property the model has that a baseline does not.
- Provide quantitative and/or qualitative analysis of experimental results.
- If doing a review, include a table comparing properties of different approaches.
- If selecting your own dataset, include detailed dataset description: how it was collected, key statistics, properties, evaluation metrics, training procedure, and tricks used to make it work.

For this project, the strongest alignment is the real-world KITTI Road baseline comparison, the residual-block and skip-connection ablations, quantitative metrics, qualitative overlays/error maps, and detailed dataset/mask/split documentation.

### Topic-list context from the PDF

The PDF gives suggested topics and says groups are encouraged to have their own topic, but new topics outside the list should be discussed with the instructor to determine problem details.

- Topic 1: LeNet-like handwriting recognition on MNIST, at most 2 students. The PDF notes LeNet was introduced by Yann LeCun in 1998, MNIST has 60,000 training images and 10,000 test images, the dataset can be obtained from PyTorch, the original LeNet structure cannot be used directly, course CNN techniques may be used to enhance performance, and expected accuracy is beyond 98% or comparable to the original paper.
- Topic 2: Image Segmentation with U-Net. This is the selected topic. It requires applying U-Net to image segmentation, where segmentation divides an image into segments to make representation simpler or more meaningful for analysis. The student creates their own U-Net, a CNN that can segment images quickly and accurately, assigning a label to every pixel. It uses an automotive drive dataset. The PDF says U-Net is beyond the lecture scope and instruction slides are provided.
- Topic 3: Emojify. Uses word-vector representations to build an emojifier that maps sentences to appropriate emoji and permits any sequence model.
- Topic 4: Neural Machine Translation. Uses Seq2Seq or Transformer architectures. The PDF describes encoder/decoder structure, attention, Transformer attention-only structure, and evaluation with BLEU, ROUGE, METEOR, and optionally human fluency/adequacy/fidelity evaluation. It points to manythings.org/anki for language sentence pairs.
- Topic 5: Trigger Word Detection. Constructs a speech dataset and trigger-word algorithm, also called keyword or wake-word detection. A successful project should record a clip and trigger a chime when the trigger word is detected. It can be extended to launch apps or trigger network-connected devices. The PDF says a training voice dataset is provided with trigger word "activate", but students may choose any voice data and trigger word.

## PDF Rubric Crosswalk

| PDF item                   | Weight | What the final report must contain                                                                                                                            | Code/artifacts that support it                                                                                           | Current status                        |
| -------------------------- | -----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| Novelty                    |    20% | Aim for the lowest credible A+ tier: frame a new, interesting, non-trivial problem around low-resource RGB-only road segmentation on KITTI, then support it with controlled evidence. | `kitti_road/models.py`, `configs/*.yaml`, baseline/ablation experiments.                                                 | Framing and final results present.    |
| Length                     |     5% | Final PDF should be at least 4 pages; 6-8 pages is ideal. Bibliography/appendix do not count.                                                                 | This blueprint.                                                                                                          | Needs final PDF.                      |
| Code and appendix          |    25% | Submit compressed PDF plus code. Include commands, configs, checkpoints, metrics, plots, and enough appendix detail to reproduce.                             | `README.md`, `scripts/run_everything.py`, `scripts/verify_artifacts.py`, `configs/*.yaml`, `checkpoints/*`, `reports/*`. | Required artifacts verified.          |
| Abstract                   |     5% | One paragraph summarizing problem, method, dataset, baselines, and headline result.                                                                           | `reports/experiment_summary.csv`.                                                                                        | Ready to write from final metrics.    |
| Introduction               |     5% | State road segmentation problem, why it matters, why low-resource RGB-only setting is challenging.                                                            | Dataset audit and model setup.                                                                                           | Ready to write from final evidence.   |
| Related Work               |     5% | Discuss U-Net, KITTI Road, encoder-decoder segmentation context, and road segmentation methods using stereo/LiDAR or larger-supervision settings as contrast or future work. | README dataset link and final citations.                                                                                 | Needs citations in final PDF.         |
| Model/Method               |    15% | Include original computation graph figure, formal model description, loss, preprocessing, ignore handling, and differences from baselines.                    | `reports/model_figure.png`, `kitti_road/models.py`, `kitti_road/losses.py`, `kitti_road/datasets.py`.                    | Figure generated; write details.      |
| Experiments                |    15% | Include baseline comparison, ablation study, quantitative results, qualitative overlays, dataset details, metrics, training protocol, tricks.                 | Training/evaluation scripts, metrics, plots, overlays, mask audit.                                                       | Final artifacts verified.             |
| Conclusion and future work |     5% | Summarize takeaways, limitations, and future directions. Must honestly state whether residual block structure helped, with GroupNorm held constant.           | Final metrics and qualitative errors.                                                                                    | Ready to write from final evidence.   |

## Codebase Details To Mention In The Report

### Codebase audit notes

- The generated validation results are tied to checkpoint-embedded training overrides: `--batch-size 32 --num-workers 8 --lr 0.0005 --amp`, with 80 configured epochs. The YAML files remain conservative defaults at `batch_size: 8` and `lr: 0.0003`.
- Best-checkpoint epochs: Residual U-Net epoch 71, Plain U-Net epoch 65, No-skip U-Net epoch 75. Final evaluation metrics come from evaluating these best checkpoints, not from the last epoch.
- GroupNorm is not unique to the residual model. `PlainUNet`, `NoSkipUNet`, and `ResidualRoadUNet` all use GroupNorm in their convolution blocks; the residual-vs-plain comparison mainly isolates residual block structure.
- The no-skip model removes decoder skip concatenation but keeps the same encoder/decoder depth, channel schedule, GroupNorm style, loss, optimizer, image size, split, and evaluation protocol.
- `scripts/summarize_analysis.py`, `scripts/make_model_figure.py`, and `kitti_road/visualize.py` use Matplotlib's `Agg` backend so report artifacts can be regenerated in headless Colab or local terminal environments.

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
- Local raw layout note: `training/image_2` contains 289 RGB images, while `gt_image_2` contains 384 PNGs because it includes 289 `*_road_*.png` masks plus 95 `*_lane_*.png` masks. The supervised dataset index pairs the 289 RGB images with road masks only.
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
  - red and any other non-road, non-black pixels -> valid background `0`
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
  - all three variants use GroupNorm with up to 8 groups in convolution blocks
  - encoder uses max pooling
  - decoder uses transposed convolutions
  - skip connections concatenate encoder features into decoder features
  - output head is a 1x1 convolution producing one logit per pixel
- Residual model details:
  - residual two-convolution blocks
  - GroupNorm is shared with the baselines, so the ResidualRoadUNet vs PlainUNet comparison mainly isolates residual block structure
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
- Learning rate: config default `0.0003`; final reported runs used `0.0005`.
- Weight decay: `0.0001`.
- Scheduler: CosineAnnealingLR.
- Epochs: `80`.
- Batch size: config default `8`; final reported runs used `32`.
- Mixed precision: AMP enabled on CUDA.
- Checkpoint cadence: best, last, and every 10 epochs.
- Device logic: CUDA if available unless `--cpu`.
- Configs:
  - `configs/road_unet.yaml`
  - `configs/plain_unet.yaml`
  - `configs/no_skip_unet.yaml`

The A+ report should focus on the lean three-model U-Net package only.

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
| Ablation study on specific design choices.                                   | Plain U-Net vs NoSkipUNet tests skip connections; ResidualRoadUNet vs PlainUNet tests residual blocks while GroupNorm is held constant. |
| Quantitative and/or qualitative analysis.                                    | Metrics CSV/JSON, PR curves, threshold sweeps, loss/IoU curves, overlays, error maps.                            |
| Dataset description, metrics, training details, tricks.                      | Dataset audit, split stats, mask conversion, configs, optimizer/loss/scheduler, AMP, augmentation.               |

Do not add a synthetic experiment unless time remains after the real KITTI experiments. The real-world comparison plus ablation plus quantitative/qualitative analysis is enough and cleaner.

## Required Tables For The Final PDF

### Table 1: Dataset and split

Fill from `data/processed/kitti_road/stats.json`.

| Dataset    | Total labeled | Train | Val | Scenarios | Val fraction | Seed | Split hash |
| ---------- | ------------: | ----: | --: | --------- | -----------: | ---: | ---------- |
| KITTI Road |           289 |   231 |  58 | `um`: 95; `umm`: 96; `uu`: 98 |          0.2 | 6681 | `08a759ddafc37723` |

### Table 2: Model comparison

Fill from `reports/experiment_summary.csv`.

| Model          |    IoU | Dice/F1 | Precision | Recall | Pixel Acc. |     AP |   MaxF |
| -------------- | -----: | ------: | --------: | -----: | ---------: | -----: | -----: |
| Residual U-Net | 0.8896 |  0.9416 |    0.9385 | 0.9447 |     0.9792 | 0.9795 | 0.9416 |
| Plain U-Net    | 0.8891 |  0.9413 |    0.9363 | 0.9464 |     0.9790 | 0.9737 | 0.9414 |
| No-skip U-Net  | 0.8886 |  0.9410 |    0.9329 | 0.9494 |     0.9788 | 0.9695 | 0.9419 |

Current main-model validation result:

```text
ResidualRoadUNet achieved 0.8896 validation IoU and 0.9416 Dice/F1 on the deterministic KITTI validation split. Precision and recall were balanced at 0.9385 and 0.9447, respectively, with AP=0.9795 and MaxF=0.9416.
```

Important caveat:

```text
These are validation-split results from labeled KITTI training images, not official hidden KITTI test-server results.
```

Per-scenario ResidualRoadUNet metrics:

| Scenario |    IoU | Dice/F1 | Precision | Recall | Pixel Acc. | Interpretation                                                            |
| -------- | -----: | ------: | --------: | -----: | ---------: | ------------------------------------------------------------------------- |
| `um`     | 0.8986 |  0.9466 |    0.9389 | 0.9545 |     0.9829 | Strong marked-road performance.                                           |
| `umm`    | 0.9273 |  0.9623 |    0.9713 | 0.9534 |     0.9821 | Best scenario; multi-lane markings likely provide clearer road structure. |
| `uu`     | 0.8245 |  0.9038 |    0.8879 | 0.9203 |     0.9729 | Hardest scenario; unmarked roads make boundaries less explicit.           |

Current Plain U-Net validation result:

```text
Plain U-Net achieved 0.8891 validation IoU and 0.9413 Dice/F1, nearly matching the residual U-Net. Its precision was 0.9363, recall was 0.9464, AP was 0.9737, and MaxF was 0.9414.
```

Plain U-Net vs ResidualRoadUNet interpretation:

```text
The residual U-Net and plain U-Net are effectively tied overall: 0.8896 IoU for ResidualRoadUNet versus 0.8891 IoU for Plain U-Net. Because GroupNorm is used in both variants, this suggests that residual block structure provides only a marginal aggregate improvement on the deterministic KITTI validation split. The scenario-level results are mixed: ResidualRoadUNet performs better on marked urban and multiple marked-lane scenes, while Plain U-Net performs better on unmarked roads.
```

Per-scenario Plain U-Net metrics:

| Scenario |    IoU | Dice/F1 | Precision | Recall | Pixel Acc. | Interpretation                                                  |
| -------- | -----: | ------: | --------: | -----: | ---------: | --------------------------------------------------------------- |
| `um`     | 0.8745 |  0.9330 |    0.9072 | 0.9604 |     0.9781 | Below ResidualRoadUNet on marked urban roads.                   |
| `umm`    | 0.9187 |  0.9576 |    0.9727 | 0.9430 |     0.9801 | Strong, but below ResidualRoadUNet on multi-marked-lane scenes. |
| `uu`     | 0.8601 |  0.9248 |    0.9132 | 0.9367 |     0.9789 | Better than ResidualRoadUNet on unmarked roads.                 |

Per-scenario No-skip U-Net metrics:

| Scenario |    IoU | Dice/F1 | Precision | Recall | Pixel Acc. | Interpretation                                                  |
| -------- | -----: | ------: | --------: | -----: | ---------: | --------------------------------------------------------------- |
| `um`     | 0.8781 |  0.9351 |    0.9083 | 0.9634 |     0.9788 | Below ResidualRoadUNet on marked urban roads.                   |
| `umm`    | 0.9257 |  0.9614 |    0.9750 | 0.9482 |     0.9818 | Close to ResidualRoadUNet and above Plain U-Net on this split.  |
| `uu`     | 0.8447 |  0.9158 |    0.8963 | 0.9362 |     0.9762 | Better than ResidualRoadUNet, but below Plain U-Net.            |

### Table 3: Ablation interpretation

| Comparison                    | Design choice tested        | Expected interpretation                                               | Observed result                                                                                                                     |
| ----------------------------- | --------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Plain U-Net vs No-skip U-Net  | Skip connections            | Skip connections should improve spatial detail and boundary recovery. | Near-tie by IoU: Plain U-Net 0.8891 vs No-skip U-Net 0.8886. No-skip has slightly higher MaxF, so the report should avoid claiming a decisive skip-connection win from aggregate metrics alone. |
| Residual U-Net vs Plain U-Net | Residual block structure    | Residual blocks should improve optimization under small batches.      | Near-tie overall: ResidualRoadUNet IoU 0.8896 vs Plain U-Net IoU 0.8891. GroupNorm is shared by both models; residual blocks help on `um`/`umm`, while Plain is better on `uu`. |

### Table 4: Training setup

| Setting      | Value                                                                 |
| ------------ | --------------------------------------------------------------------- |
| Image size   | 384 x 1216                                                            |
| Epochs       | 80                                                                    |
| Batch size   | Config default 8; final reported checkpoint runs used `--batch-size 32` |
| Optimizer    | AdamW                                                                 |
| LR           | Config default 0.0003; final reported checkpoint runs used `--lr 0.0005` |
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
     - low-resource RGB-only KITTI Road problem framing
     - controlled from-scratch U-Net variant study
     - residual-block and skip-connection ablation
     - quantitative and qualitative KITTI analysis

3. Related Work
   - U-Net.
   - KITTI Road.
   - Road segmentation with extra sensors or larger-supervision settings as contrast.

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
   - Did residual block structure help?
   - Did skip connections help?
   - Which scenes failed and why?
   - Discuss AP/MaxF and threshold sensitivity if relevant.

7. Conclusion and Future Work
   - Main takeaways.
   - Limitations.
   - Future directions: stereo/LiDAR, larger datasets, CRF/boundary refinement, threshold tuning.

8. Appendix
   - Commands.
   - Full configs.
   - Extra qualitative examples.
   - Extra per-scenario metrics.

## Presentation Checklist

The PDF also includes a presentation rubric. For a 10-minute individual talk or 15-minute group talk:

- Motivation/problem importance: explain low-resource RGB road segmentation.
- Related work: U-Net, KITTI Road, modern sensor-heavy road segmentation.
- Main idea: residual-block U-Net and controlled GroupNorm baselines.
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
python scripts/run_everything.py --data /content/data/data_road --processed data/processed/kitti_road --reports reports --checkpoint-dir checkpoints --batch-size 32 --num-workers 8 --lr 0.0005 --amp
```

`run_everything.py` executes CUDA checks, mask audit, split preparation, figure generation, all three required model trainings, evaluation for all best checkpoints, report-ready analysis generation, and final artifact verification.

If running manually:

```bash
python scripts/audit_kitti_masks.py --data /content/data/data_road --out reports/mask_audit
python scripts/prepare_kitti_road.py --data /content/data/data_road --out data/processed/kitti_road
python scripts/make_model_figure.py
python -m kitti_road.train --config configs/road_unet.yaml --batch-size 32 --num-workers 8 --lr 0.0005 --amp
python -m kitti_road.train --config configs/plain_unet.yaml --batch-size 32 --num-workers 8 --lr 0.0005 --amp
python -m kitti_road.train --config configs/no_skip_unet.yaml --batch-size 32 --num-workers 8 --lr 0.0005 --amp
python -m kitti_road.evaluate --checkpoint checkpoints/road_unet/road_unet_best.pt --split val
python -m kitti_road.evaluate --checkpoint checkpoints/plain_unet/plain_unet_best.pt --split val
python -m kitti_road.evaluate --checkpoint checkpoints/no_skip_unet/no_skip_unet_best.pt --split val
python scripts/summarize_analysis.py
python scripts/verify_artifacts.py
```

For Colab A100 tuning, training configs can be overridden from the command line. YAML values remain the defaults, currently using `batch_size: 8`, `num_workers: 8`, and `lr: 0.0003`; the final reported checkpoints used `--batch-size 32 --num-workers 8 --lr 0.0005 --amp`.

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
- `SEED=6681`
- `IMAGE_HEIGHT=384`
- `IMAGE_WIDTH=1216`
- `EPOCHS=80`
- `BATCH_SIZE=8`
- `NUM_WORKERS=8`
- `LR=0.0003`
- `WEIGHT_DECAY=0.0001`
- `SAVE_EVERY_EPOCHS=10`
- `THRESHOLD=0.5`
- `AMP=true`
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
- Do not overstate novelty as a fundamentally new U-Net architecture; aim for the PDF's lowest A+ route: an impactful and non-trivial problem formulation with evidence.
- Do not omit baselines; baseline comparison is central to the experiment grade.
- Do not omit ablation; it directly supports the method claim.
- Do not report metrics without qualitative examples; segmentation needs visual evidence.
- Do not include borrowed architecture figures; use the generated original figure.
- Do not ignore mask conversion; include the audit preview and explain the road/background/ignore rule.
- Do not overclaim if residual U-Net does not win; interpret results honestly and discuss limitations.

## Exact Post-Training Fill-Ins

Use these values in the final PDF:

- Best model by IoU: Residual U-Net, 0.8896.
- Best model by MaxF: No-skip U-Net, 0.9419.
- Residual U-Net vs Plain U-Net: Residual is higher by 0.0005 IoU; treat as a near-tie, not a strong win.
- Plain U-Net vs No-skip U-Net: Plain is higher by 0.0005 IoU, while No-skip has slightly higher MaxF; treat as inconclusive without overclaiming skip connections.
- Dataset split: 289 labeled images, 231 train, 58 validation; scenarios `um`: 95, `umm`: 96, `uu`: 98; split hash `08a759ddafc37723`.
- Mask audit: 384 raw GT mask files in `gt_image_2`; top colors red `255,0,0`, magenta `255,0,255`, black `0,0,0`, blue `0,0,255`; converted ratios road 0.153149, background 0.833033, ignore 0.013818.
- Example qualitative filenames available for success/error discussion: `um_000000_overlay.png`, `um_000000_errors.png`, plus the additional validation examples under `reports/qualitative_overlays/<experiment>/` and `reports/error_examples/<experiment>/`.
- Main limitation to inspect visually: unmarked `uu` scenes are hardest for the residual model, with the lowest per-scenario IoU at 0.8245.
