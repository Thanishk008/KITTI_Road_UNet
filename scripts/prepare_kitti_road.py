from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitti_road.datasets import mask_to_binary, scenario_from_name
from kitti_road.utils import save_json, stable_hash
from PIL import Image


def find_mask(mask_dir: Path, image_path: Path) -> Path | None:
    stem = image_path.stem
    scenario, rest = stem.split("_", 1)
    candidates = [
        mask_dir / f"{scenario}_road_{rest}.png",
        mask_dir / f"{stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(mask_dir.glob(f"*road*{rest}.png"))
    return matches[0] if matches else None


def build_index(data_root: Path) -> list[dict]:
    image_dir = data_root / "training" / "image_2"
    mask_dir = data_root / "training" / "gt_image_2"
    if not image_dir.exists() or not mask_dir.exists():
        raise SystemExit(f"Expected KITTI Road folders under {data_root}: training/image_2 and training/gt_image_2")
    items = []
    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = find_mask(mask_dir, image_path)
        if mask_path is None:
            continue
        scenario = scenario_from_name(image_path.name)
        with Image.open(mask_path) as mask:
            binary = mask_to_binary(mask)
        road_pixels = int((binary == 1).sum())
        valid_pixels = int((binary != 255).sum())
        items.append(
            {
                "id": image_path.stem,
                "scenario": scenario,
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "road_ratio": round(road_pixels / max(1, valid_pixels), 6),
            }
        )
    if not items:
        raise SystemExit(f"No image/mask pairs found under {data_root}")
    return items


def stratified_split(items: list[dict], val_fraction: float, seed: int) -> dict:
    rng = random.Random(seed)
    by_scenario = defaultdict(list)
    for item in items:
        by_scenario[item["scenario"]].append(item)
    train, val = [], []
    for scenario, group in sorted(by_scenario.items()):
        group = list(group)
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return {"train": sorted(train, key=lambda x: x["id"]), "val": sorted(val, key=lambda x: x["id"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to KITTI data_road directory.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=6681)
    args = parser.parse_args()

    items = build_index(args.data)
    split = stratified_split(items, args.val_fraction, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {
        "dataset": "KITTI Road/Lane Detection 2013",
        "root": str(args.data.resolve()),
        "total_labeled": len(items),
        "train": len(split["train"]),
        "val": len(split["val"]),
        "scenario_counts": dict(Counter(item["scenario"] for item in items)),
        "split_hash": stable_hash(split),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
    }
    save_json(items, args.out / "dataset_index.json")
    save_json(split, args.out / "split.json")
    save_json(stats, args.out / "stats.json")
    print(stats)


if __name__ == "__main__":
    main()
