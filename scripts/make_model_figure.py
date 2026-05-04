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


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#334155",
    style: str = "-|>",
    connectionstyle: str | None = None,
) -> None:
    if FancyArrowPatch is None:
        raise RuntimeError("matplotlib is not available")
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=1.35,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def add_l_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#334155",
    bend_y: float | None = None,
) -> None:
    sx, sy = start
    ex, ey = end
    corner_y = bend_y if bend_y is not None else ey
    ax.plot([sx, sx], [sy, corner_y], color=color, linewidth=1.35)
    ax.plot([sx, ex], [corner_y, corner_y], color=color, linewidth=1.35)
    add_arrow(ax, (sx, corner_y), end, color=color)


def make_figure(out: Path) -> None:
    if plt is None:
        make_pillow_figure(out)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    encoder_y = [5.15, 4.0, 2.85, 1.7]
    decoder_y = [1.7, 2.85, 4.0, 5.15]
    encoder_labels = ["Input\nRGB", "Residual\nBlock 32", "Down + Res\nBlock 64", "Down + Res\nBlock 128", "Down + Res\nBlock 256"]
    bottleneck = "Bottleneck\nRes Block 512"
    decoder_labels = ["Up + Skip\nBlock 256", "Up + Skip\nBlock 128", "Up + Skip\nBlock 64", "Up + Skip\nBlock 32", "1x1 Conv\nRoad Logits"]

    input_x = 0.2
    enc1_x = 1.85
    enc2_x = 3.65
    enc3_x = 5.45
    enc4_x = 7.25
    bottleneck_x = 7.25

    add_box(ax, input_x, 5.15, encoder_labels[0], "#e0f2fe", width=1.35)
    add_box(ax, enc1_x, encoder_y[0], encoder_labels[1], "#dbeafe")
    add_box(ax, enc2_x, encoder_y[1], encoder_labels[2], "#bfdbfe")
    add_box(ax, enc3_x, encoder_y[2], encoder_labels[3], "#93c5fd")
    add_box(ax, enc4_x, encoder_y[3], encoder_labels[4], "#60a5fa")
    add_box(ax, bottleneck_x, 0.5, bottleneck, "#fbbf24")

    decoder_x = 9.55
    add_box(ax, decoder_x, decoder_y[0], decoder_labels[0], "#bbf7d0")
    add_box(ax, decoder_x, decoder_y[1], decoder_labels[1], "#86efac")
    add_box(ax, decoder_x, decoder_y[2], decoder_labels[2], "#4ade80")
    add_box(ax, decoder_x, decoder_y[3], decoder_labels[3], "#22c55e")
    add_box(ax, 12.0, 5.15, decoder_labels[4], "#fecaca", width=1.55)

    add_arrow(ax, (input_x + 1.35, 5.51), (enc1_x, 5.51))
    add_l_arrow(ax, (enc1_x + 0.9, 5.15), (enc2_x, 4.36))
    add_l_arrow(ax, (enc2_x + 0.9, 4.0), (enc3_x, 3.21))
    add_l_arrow(ax, (enc3_x + 0.9, 2.85), (enc4_x, 2.06))
    add_arrow(ax, (enc4_x + 0.9, 1.7), (bottleneck_x + 0.9, 1.22))
    add_arrow(ax, (bottleneck_x + 1.8, 1.22), (decoder_x, 2.06))
    add_arrow(ax, (decoder_x + 0.9, 2.42), (decoder_x + 0.9, 2.85))
    add_arrow(ax, (decoder_x + 0.9, 3.57), (decoder_x + 0.9, 4.0))
    add_arrow(ax, (decoder_x + 0.9, 4.72), (decoder_x + 0.9, 5.15))
    add_arrow(ax, (decoder_x + 1.8, 5.51), (12.0, 5.51))

    skip_color = "#db2777"
    add_arrow(ax, (enc1_x + 1.8, 5.51), (decoder_x, 5.51), skip_color)
    add_arrow(ax, (enc2_x + 1.8, 4.36), (decoder_x, 4.36), skip_color)
    add_arrow(ax, (enc3_x + 1.8, 3.21), (decoder_x, 3.21), skip_color)
    add_arrow(ax, (enc4_x + 1.8, 2.06), (decoder_x, 2.06), skip_color)

    ax.text(7.0, 7.0, "ResidualRoadUNet computation graph", ha="center", va="center", fontsize=15, weight="bold")
    ax.text(
        7.0,
        6.6,
        "Encoder downsamples context, decoder upsamples masks,\nskip paths restore spatial detail",
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.2,
    )
    ax.text(6.05, 5.82, "skip concatenations", color=skip_color, fontsize=9)
    ax.text(12.8, 4.47, "BCEWithLogits\n+ Dice loss", ha="center", va="center", fontsize=9, color="#7f1d1d")

    plt.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"wrote {out}")


def make_pillow_figure(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (2200, 1160), "white")
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

    def l_arrow(start: tuple[int, int], end: tuple[int, int], fill: tuple[int, int, int] = (51, 65, 85), bend_y: int | None = None) -> None:
        sx, sy = start
        ex, ey = end
        corner_y = bend_y if bend_y is not None else ey
        mid1 = (sx, corner_y)
        mid2 = (ex, corner_y)
        draw.line((start, mid1), fill=fill, width=5)
        draw.line((mid1, mid2), fill=fill, width=5)
        draw.line((mid2, end), fill=fill, width=5)
        draw.polygon([(ex, ey), (ex - 22, ey - 12), (ex - 22, ey + 12)], fill=fill)

    draw.text((1100, 70), "ResidualRoadUNet computation graph", fill=(17, 24, 39), anchor="mm")
    draw.text(
        (1100, 125),
        "Encoder downsamples context, decoder upsamples masks,\nskip paths restore spatial detail",
        fill=(51, 65, 85),
        anchor="mm",
        spacing=8,
        align="center",
    )

    centers = [
        box(40, 290, 220, 115, "Input\nRGB", (224, 242, 254)),
        box(310, 290, 290, 115, "Residual\nBlock 32", (219, 234, 254)),
        box(640, 430, 290, 115, "Down + Res\nBlock 64", (191, 219, 254)),
        box(970, 570, 290, 115, "Down + Res\nBlock 128", (147, 197, 253)),
        box(1300, 710, 290, 115, "Down + Res\nBlock 256", (96, 165, 250)),
        box(1300, 875, 290, 115, "Bottleneck\nRes Block 512", (251, 191, 36)),
        box(1750, 710, 290, 115, "Up + Skip\nBlock 256", (187, 247, 208)),
        box(1750, 570, 290, 115, "Up + Skip\nBlock 128", (134, 239, 172)),
        box(1750, 430, 290, 115, "Up + Skip\nBlock 64", (74, 222, 128)),
        box(1750, 290, 290, 115, "Up + Skip\nBlock 32", (34, 197, 94)),
        box(2085, 290, 115, 115, "1x1 Conv\nLogits", (254, 202, 202)),
    ]

    arrow(centers[0], centers[1])
    l_arrow((455, 405), (640, 487))
    l_arrow((785, 545), (970, 627))
    l_arrow((1115, 685), (1300, 767))
    arrow((centers[4][0], centers[4][1] + 57), centers[5])
    for start, end in zip(centers[5:10], centers[6:10]):
        arrow(start, end)
    arrow(centers[9], centers[10])

    skip = (219, 39, 119)
    for source, target in [(centers[1], centers[9]), (centers[2], centers[8]), (centers[3], centers[7]), (centers[4], centers[6])]:
        arrow(source, target, skip)
    draw.text((1040, 255), "skip concatenations", fill=skip, anchor="mm")
    draw.text((2140, 490), "BCEWithLogits\n+ Dice loss", fill=(127, 29, 29), anchor="mm")

    canvas.save(out)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/model_figure.png"))
    args = parser.parse_args()
    make_figure(args.out)


if __name__ == "__main__":
    main()
