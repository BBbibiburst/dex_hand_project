"""Create compact figures from the P10-Vision JSON evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = args.root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    static = json.loads((args.root / "static_audit/summary.json").read_text())
    direct = json.loads((args.root / "paired/direct_160/summary.json").read_text())
    high = json.loads((args.root / "paired/render_1920_to_160/summary.json").read_text())
    direct_quality = json.loads(
        (args.root / "paired/direct_160/grasp_quality/summary.json").read_text()
    )
    high_quality = json.loads(
        (args.root / "paired/render_1920_to_160/grasp_quality/summary.json").read_text()
    )
    benchmark = json.loads((args.root / "model_resolution_benchmark/summary.json").read_text())

    labels = [f"{row['resolution_wh'][0]}×{row['resolution_wh'][1]}" for row in static["resolutions"]]
    pixels = [row["object_pixels"] for row in static["resolutions"]]
    p95 = [row["inference_p95_ms"] for row in benchmark["rows"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].bar(labels, pixels, color="#3478b8")
    axes[0].set(title="Visible object pixels (Close)", ylabel="segmented pixels")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, p95, color="#d97932")
    axes[1].axhline(50, color="black", linestyle="--", linewidth=1, label="20 Hz budget")
    axes[1].set(title="Native-resolution BC inference", ylabel="P95 latency (ms)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "detail_vs_inference_cost.png", dpi=180)
    plt.close(fig)

    metrics = {
        "Contact (%)": [100 * direct["bc"]["contact_fraction"], 100 * high["bc"]["contact_fraction"]],
        "Multi-finger (%)": [
            100 * direct_quality["multi_finger_fraction"],
            100 * high_quality["multi_finger_fraction"],
        ],
        "Thumb opp. (%)": [
            100 * direct_quality["thumb_opposition_fraction"],
            100 * high_quality["thumb_opposition_fraction"],
        ],
        "Max lift (mm)": [
            1000 * direct["bc"]["maximum_lift_m"],
            1000 * high["bc"]["maximum_lift_m"],
        ],
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    x = range(3)
    names = ("Contact", "Multi-finger", "Thumb opposition")
    keys = ("Contact (%)", "Multi-finger (%)", "Thumb opp. (%)")
    axes[0].bar(
        [value - 0.2 for value in x],
        [metrics[key][0] for key in keys],
        0.4,
        label="160 direct",
    )
    axes[0].bar(
        [value + 0.2 for value in x],
        [metrics[key][1] for key in keys],
        0.4,
        label="1080p→160",
    )
    axes[0].set_xticks(list(x), names, rotation=15)
    axes[0].set_ylabel("Fraction (%)")
    axes[0].set_title("Paired contact metrics")
    axes[0].legend(frameon=False)
    axes[1].bar(("160 direct", "1080p→160"), metrics["Max lift (mm)"], color=("#3478b8", "#d97932"))
    axes[1].set(title="Paired maximum lift", ylabel="mm")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "paired_frozen_bc_comparison.png", dpi=180)
    plt.close(fig)

    previews = sorted((args.root / "roi_independent_roots/previews").glob("*_close.jpg"))
    thumbs = [Image.open(path).convert("RGB").resize((480, 270)) for path in previews]
    canvas = Image.new("RGB", (480 * 2, 270 * 5), "white")
    draw = ImageDraw.Draw(canvas)
    for index, thumb in enumerate(thumbs):
        x0 = (index % 2) * 480
        y0 = (index // 2) * 270
        canvas.paste(thumb, (x0, y0))
        draw.text((x0 + 10, y0 + 10), f"root {index:02d}", fill=(255, 255, 255))
    canvas.save(output / "fixed_roi_independent_roots.jpg", quality=92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
