from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kitti_road.datasets import IGNORE_INDEX, mask_to_binary
from kitti_road.utils import save_json


def resolve_mask_dir(path: Path) -> Path:
    if path.name == "gt_image_2":
        return path
    candidate = path / "training" / "gt_image_2"
    if candidate.exists():
        return candidate
    raise SystemExit(f"Could not find gt_image_2 under {path}")


def color_key(color: tuple[int, int, int]) -> str:
    return f"{color[0]},{color[1]},{color[2]}"


def color_counts(mask_path: Path) -> Counter[str]:
    with Image.open(mask_path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
    colors, counts = np.unique(arr, axis=0, return_counts=True)
    return Counter({color_key(tuple(map(int, color))): int(count) for color, count in zip(colors, counts)})


def binary_preview(binary: np.ndarray) -> Image.Image:
    preview = np.zeros((*binary.shape, 3), dtype=np.uint8)
    preview[binary == 1] = [255, 0, 255]
    preview[binary == 0] = [255, 0, 0]
    preview[binary == IGNORE_INDEX] = [0, 0, 0]
    return Image.fromarray(preview)


def make_contact_sheet(mask_paths: list[Path], out_path: Path, thumb_width: int = 360) -> None:
    rows = []
    for mask_path in mask_paths:
        with Image.open(mask_path) as image:
            original = image.convert("RGB")
            binary = mask_to_binary(original)
        scale = thumb_width / original.width
        thumb_size = (thumb_width, max(1, int(original.height * scale)))
        left = original.resize(thumb_size, Image.Resampling.NEAREST)
        right = binary_preview(binary).resize(thumb_size, Image.Resampling.NEAREST)
        row = Image.new("RGB", (thumb_size[0] * 2, thumb_size[1] + 28), "white")
        row.paste(left, (0, 28))
        row.paste(right, (thumb_size[0], 28))
        draw = ImageDraw.Draw(row)
        draw.text((4, 6), f"{mask_path.name} | raw GT", fill=(0, 0, 0))
        draw.text((thumb_size[0] + 4, 6), "converted: road=magenta bg=red ignore=black", fill=(0, 0, 0))
        rows.append(row)

    if not rows:
        return
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def audit_masks(data: Path, out: Path, limit: int) -> dict:
    mask_dir = resolve_mask_dir(data)
    mask_paths = sorted(mask_dir.glob("*.png"))
    if not mask_paths:
        raise SystemExit(f"No PNG masks found under {mask_dir}")

    selected = mask_paths[: max(1, limit)]
    global_counts: Counter[str] = Counter()
    per_file = []
    total_pixels = road_pixels = background_pixels = ignore_pixels = 0

    for mask_path in mask_paths:
        counts = color_counts(mask_path)
        global_counts.update(counts)
        with Image.open(mask_path) as image:
            binary = mask_to_binary(image)
        total_pixels += int(binary.size)
        road_pixels += int((binary == 1).sum())
        background_pixels += int((binary == 0).sum())
        ignore_pixels += int((binary == IGNORE_INDEX).sum())
        if len(per_file) < limit:
            per_file.append(
                {
                    "file": str(mask_path),
                    "colors": dict(counts.most_common(10)),
                    "road_pixels": int((binary == 1).sum()),
                    "background_pixels": int((binary == 0).sum()),
                    "ignore_pixels": int((binary == IGNORE_INDEX).sum()),
                }
            )

    report = {
        "mask_dir": str(mask_dir.resolve()),
        "num_masks": len(mask_paths),
        "top_colors": dict(global_counts.most_common(20)),
        "converted_pixel_counts": {
            "road": road_pixels,
            "background": background_pixels,
            "ignore": ignore_pixels,
            "total": total_pixels,
        },
        "converted_pixel_ratios": {
            "road": round(road_pixels / max(1, total_pixels), 6),
            "background": round(background_pixels / max(1, total_pixels), 6),
            "ignore": round(ignore_pixels / max(1, total_pixels), 6),
        },
        "sample_files": per_file,
        "expected_rule": "magenta/white-like=road, red=background, black=ignore",
    }

    out.mkdir(parents=True, exist_ok=True)
    save_json(report, out / "mask_audit.json")
    make_contact_sheet(selected, out / "mask_audit_preview.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="KITTI data_road directory or training/gt_image_2 directory.")
    parser.add_argument("--out", type=Path, default=Path("reports/mask_audit"))
    parser.add_argument("--limit", type=int, default=8, help="Number of masks to include in the visual preview.")
    args = parser.parse_args()
    report = audit_masks(args.data, args.out, args.limit)
    print(json.dumps(report["converted_pixel_counts"], indent=2))
    print(f"wrote {args.out / 'mask_audit.json'}")
    print(f"wrote {args.out / 'mask_audit_preview.png'}")


if __name__ == "__main__":
    main()
