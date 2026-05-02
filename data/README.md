# Data Directory

Do not commit KITTI images, prepared splits, or generated masks. The project expects the raw KITTI Road/Lane Detection 2013 archive to be stored outside git and prepared at runtime.

## Expected Raw Layout

```text
data_road/
  training/
    image_2/*.png
    gt_image_2/*.png
  testing/
    image_2/*.png
```

## Colab Defaults

The README and `.env.example` assume:

```text
KITTI_ROAD_ROOT=/content/data/data_road
KITTI_PROCESSED_DIR=data/processed/kitti_road
```

## Required Preparation

Audit KITTI masks first:

```bash
python scripts/audit_kitti_masks.py --data /content/data/data_road --out reports/mask_audit
```

Create the deterministic train/validation split:

```bash
python scripts/prepare_kitti_road.py --data /content/data/data_road --out data/processed/kitti_road
```

Generated `data/processed/` contents are ignored by git and should be recreated in Colab.
