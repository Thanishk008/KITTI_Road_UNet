from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
except ImportError:  # pragma: no cover - exercised only in minimal environments
    plt = None
    FancyArrowPatch = None
    FancyBboxPatch = None

from PIL import Image, ImageDraw


def add_box(ax, x: float, y: float, text: str, color: str, width: float = 1.8, height: float = 0.72) -> None:
    if FancyBboxPatch is None:
        raise RuntimeError("matplotlib is not available")
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.4,
        edgecolor="#24313f",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9, color="#111827")


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#334155", style: str = "-|>") -> None:
    if FancyArrowPatch is None:
        raise RuntimeError("matplotlib is not available")
    arrow = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, linewidth=1.35, color=color)
    ax.add_patch(arrow)


def make_figure(out: Path) -> None:
    if plt is None:
        make_pillow_figure(out)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    encoder_y = [5.5, 4.35, 3.2, 2.05]
    decoder_y = [2.05, 3.2, 4.35, 5.5]
    encoder_labels = ["Input\nRGB", "Residual\nBlock 32", "Down + Res\nBlock 64", "Down + Res\nBlock 128", "Down + Res\nBlock 256"]
    bottleneck = "Bottleneck\nRes Block 512"
    decoder_labels = ["Up + Skip\nBlock 256", "Up + Skip\nBlock 128", "Up + Skip\nBlock 64", "Up + Skip\nBlock 32", "1x1 Conv\nRoad Logits"]

    add_box(ax, 0.4, 5.5, encoder_labels[0], "#e0f2fe", width=1.35)
    add_box(ax, 2.0, encoder_y[0], encoder_labels[1], "#dbeafe")
    add_box(ax, 3.8, encoder_y[1], encoder_labels[2], "#bfdbfe")
    add_box(ax, 5.6, encoder_y[2], encoder_labels[3], "#93c5fd")
    add_box(ax, 7.4, encoder_y[3], encoder_labels[4], "#60a5fa")
    add_box(ax, 7.4, 0.85, bottleneck, "#fbbf24")

    add_box(ax, 9.2, decoder_y[0], decoder_labels[0], "#bbf7d0")
    add_box(ax, 9.2, decoder_y[1], decoder_labels[1], "#86efac")
    add_box(ax, 9.2, decoder_y[2], decoder_labels[2], "#4ade80")
    add_box(ax, 9.2, decoder_y[3], decoder_labels[3], "#22c55e")
    add_box(ax, 11.6, 5.5, decoder_labels[4], "#fecaca", width=1.55)

    centers = [(1.75, 5.86), (3.8, 5.86), (5.6, 4.71), (7.4, 3.56), (8.3, 2.41), (8.3, 1.57)]
    for start, end in zip(centers[:-1], centers[1:]):
        add_arrow(ax, start, end)
    add_arrow(ax, (8.3, 1.57), (9.2, 2.41))
    add_arrow(ax, (10.1, 2.77), (10.1, 3.2))
    add_arrow(ax, (10.1, 3.92), (10.1, 4.35))
    add_arrow(ax, (10.1, 5.07), (10.1, 5.5))
    add_arrow(ax, (11.0, 5.86), (11.6, 5.86))

    skip_color = "#db2777"
    add_arrow(ax, (2.9, 5.5), (9.2, 5.5), skip_color)
    add_arrow(ax, (4.7, 4.35), (9.2, 4.35), skip_color)
    add_arrow(ax, (6.5, 3.2), (9.2, 3.2), skip_color)
    add_arrow(ax, (8.3, 2.05), (9.2, 2.05), skip_color)

    ax.text(7.0, 6.55, "ResidualRoadUNet computation graph", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(7.0, 6.18, "Encoder downsamples context, decoder upsamples masks, skip paths restore spatial detail", ha="center", fontsize=10)
    ax.text(5.95, 5.72, "skip concatenations", color=skip_color, fontsize=9)
    ax.text(12.35, 4.82, "BCEWithLogits\n+ Dice loss", ha="center", va="center", fontsize=9, color="#7f1d1d")

    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"wrote {out}")


def make_pillow_figure(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (2200, 1060), "white")
    draw = ImageDraw.Draw(canvas)

    def box(x: int, y: int, w: int, h: int, text: str, fill: tuple[int, int, int]) -> tuple[int, int]:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=24, fill=fill, outline=(36, 49, 63), width=4)
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            draw.text((x + w / 2, y + h / 2 - 18 * len(lines) + idx * 36), line, fill=(17, 24, 39), anchor="mm")
        return x + w // 2, y + h // 2

    def arrow(start: tuple[int, int], end: tuple[int, int], fill: tuple[int, int, int] = (51, 65, 85)) -> None:
        draw.line((start, end), fill=fill, width=5)
        ex, ey = end
        sx, sy = start
        if ex >= sx:
            draw.polygon([(ex, ey), (ex - 22, ey - 12), (ex - 22, ey + 12)], fill=fill)
        else:
            draw.polygon([(ex, ey), (ex + 22, ey - 12), (ex + 22, ey + 12)], fill=fill)

    draw.text((1100, 65), "ResidualRoadUNet computation graph", fill=(17, 24, 39), anchor="mm")
    draw.text((1100, 115), "Encoder downsamples context, decoder upsamples masks, skip paths restore spatial detail", fill=(51, 65, 85), anchor="mm")

    centers = [
        box(80, 240, 220, 115, "Input\nRGB", (224, 242, 254)),
        box(360, 240, 290, 115, "Residual\nBlock 32", (219, 234, 254)),
        box(690, 380, 290, 115, "Down + Res\nBlock 64", (191, 219, 254)),
        box(1020, 520, 290, 115, "Down + Res\nBlock 128", (147, 197, 253)),
        box(1350, 660, 290, 115, "Down + Res\nBlock 256", (96, 165, 250)),
        box(1350, 825, 290, 115, "Bottleneck\nRes Block 512", (251, 191, 36)),
        box(1680, 660, 290, 115, "Up + Skip\nBlock 256", (187, 247, 208)),
        box(1680, 520, 290, 115, "Up + Skip\nBlock 128", (134, 239, 172)),
        box(1680, 380, 290, 115, "Up + Skip\nBlock 64", (74, 222, 128)),
        box(1680, 240, 290, 115, "Up + Skip\nBlock 32", (34, 197, 94)),
        box(2020, 240, 145, 115, "1x1 Conv\nLogits", (254, 202, 202)),
    ]

    for start, end in zip(centers[:6], centers[1:6]):
        arrow(start, end)
    for start, end in zip(centers[5:10], centers[6:10]):
        arrow(start, end)
    arrow(centers[9], centers[10])

    skip = (219, 39, 119)
    for source, target in [(centers[1], centers[9]), (centers[2], centers[8]), (centers[3], centers[7]), (centers[4], centers[6])]:
        arrow(source, target, skip)
    draw.text((965, 235), "skip concatenations", fill=skip, anchor="mm")
    draw.text((2092, 440), "BCEWithLogits\n+ Dice loss", fill=(127, 29, 29), anchor="mm")

    canvas.save(out)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/model_figure.png"))
    args = parser.parse_args()
    make_figure(args.out)


if __name__ == "__main__":
    main()
