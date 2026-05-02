from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from .datasets import image_to_tensor
from .models import build_model
from .visualize import road_overlay, save_binary_mask, save_probability_mask


def iter_images(path: Path):
    if path.is_file():
        yield path
        return
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        yield from sorted(path.glob(ext))


def predict(checkpoint: Path, images: Path, out: Path, cpu: bool = False) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload["config"]
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model = build_model(config["model"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    threshold = float(payload.get("threshold", config["training"].get("threshold", 0.5)))
    height, width = tuple(config["data"].get("image_size", [384, 1216]))
    out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for path in tqdm(list(iter_images(images)), desc="predict"):
            image = Image.open(path).convert("RGB")
            original_size = image.size
            resized = image.resize((width, height), Image.Resampling.BILINEAR)
            tensor = image_to_tensor(resized).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            prob_original = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize(original_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            save_probability_mask(prob_original, out / f"{path.stem}_prob.png")
            save_binary_mask(prob_original, out / f"{path.stem}_mask.png", threshold)
            road_overlay(image, prob_original, threshold).save(out / f"{path.stem}_overlay.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    predict(args.checkpoint, args.images, args.out, args.cpu)


if __name__ == "__main__":
    main()
