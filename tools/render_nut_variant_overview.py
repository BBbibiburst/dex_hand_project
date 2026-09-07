"""Render a deterministic overview of the 100 procedural NutAssembly variants."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np

from source.envs.manipulation.nut_assembly import procedural_nut_variant


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "nut_assembly_top100_final" / "nut_variant_overview.png"
SHAPES = (0, 4, 3, 5, 6, 7, 8, 10, 12, 16)
NAMES = (
    "圆形",
    "方形",
    "三角形",
    "五边形",
    "六边形",
    "七边形",
    "八边形",
    "十边形",
    "十二边形",
    "十六边形",
)


def vertices(sides: int, radius: float) -> np.ndarray:
    phase = np.pi / 4.0 if sides == 4 else 0.0
    angle = phase + 2.0 * np.pi * np.arange(sides) / sides
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))


def add_section(ax, variant):
    sides = variant.polygon_sides
    wall_outer_apothem = variant.hole_apothem + 0.016
    if sides == 0:
        ax.add_patch(Circle((0, 0), wall_outer_apothem, fc="#c8ced8", ec="#525b6b", lw=0.7))
        ax.add_patch(Circle((0, 0), variant.hole_apothem, fc="white", ec="#173a8a", lw=0.8))
        ax.add_patch(Circle((0, 0), variant.peg_radius, fc="#65b96e", ec="#1b6f2a", lw=0.7))
    else:
        cosine = np.cos(np.pi / sides)
        ax.add_patch(
            Polygon(
                vertices(sides, wall_outer_apothem / cosine),
                closed=True,
                fc="#c8ced8",
                ec="#525b6b",
                lw=0.7,
            )
        )
        ax.add_patch(
            Polygon(
                vertices(sides, variant.hole_apothem / cosine),
                closed=True,
                fc="white",
                ec="#173a8a",
                lw=0.8,
            )
        )
        ax.add_patch(
            Polygon(
                vertices(sides, variant.peg_radius), closed=True, fc="#65b96e", ec="#1b6f2a", lw=0.7
            )
        )
    ax.set_xlim(-0.072, 0.072)
    ax.set_ylim(-0.072, 0.072)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0, -0.068, f"#{variant.index:02d}", ha="center", va="bottom", fontsize=6.5, color="#555b66"
    )


def main():
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(18, 15), facecolor="white")
    grid = fig.add_gridspec(
        12, 10, height_ratios=[0.7, *([1] * 10), 1.55], hspace=0.08, wspace=0.05
    )
    fig.suptitle("NutAssembly 100 个程序化变体总览", fontsize=25, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.957,
        "列 = 10 种配合截面；行 = 10 个尺寸族；灰色为固定孔座，绿色为自由插销轴截面",
        ha="center",
        fontsize=13,
        color="#4e5665",
    )
    for col, name in enumerate(NAMES):
        ax = fig.add_subplot(grid[0, col])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            name + f"\n({SHAPES[col] if SHAPES[col] else 24} 边建模)",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#173a8a",
        )
    for family in range(10):
        first = procedural_nut_variant(10 * family)
        for col in range(10):
            ax = fig.add_subplot(grid[family + 1, col])
            add_section(ax, procedural_nut_variant(10 * family + col))
        fig.text(
            0.012,
            0.895 - family * 0.0689,
            f"族 {family}\n孔内切径 {2 * first.hole_apothem * 1000:.0f} mm\n标称间隙 {first.clearance * 1000:.0f} mm",
            fontsize=7.5,
            va="center",
            color="#343942",
        )

    ax = fig.add_subplot(grid[11, :])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.text(0.15, 0.83, "结构关系（侧视）", fontsize=13, fontweight="bold", color="#173a8a")
    ax.add_patch(plt.Rectangle((0.35, 0.48), 1.25, 0.30, fc="#4d5668", ec="none"))
    ax.text(0.975, 0.63, "统一圆柱抓持把手", color="white", ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((0.82, 0.18), 0.32, 0.30, fc="#65b96e", ec="#1b6f2a"))
    ax.text(0.98, 0.31, "插销轴", ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((0.48, 0.04), 0.30, 0.22, fc="#c8ced8", ec="#525b6b"))
    ax.add_patch(plt.Rectangle((1.18, 0.04), 0.30, 0.22, fc="#c8ced8", ec="#525b6b"))
    ax.text(0.98, 0.09, "固定孔座 / ‘螺母’", ha="center", va="center", fontsize=9)
    ax.annotate(
        "竖直插入",
        xy=(0.98, 0.23),
        xytext=(0.98, 0.95),
        ha="center",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#b56815", lw=2),
        color="#b56815",
    )
    ax.text(2.1, 0.74, "不是随机的两种不同形状", fontsize=12, fontweight="bold", color="#a93e3e")
    ax.text(
        2.1,
        0.49,
        "每个编号中：插销轴与孔座使用相同的圆形/正多边形截面；插销更小，保留真实间隙。",
        fontsize=11,
        color="#25282e",
    )
    ax.text(
        2.1,
        0.27,
        "100 = 10 种截面 × 10 个尺寸族。把手、肩部和轴长保持一致，变化的是配合截面及尺寸。",
        fontsize=11,
        color="#25282e",
    )
    ax.text(
        2.1,
        0.06,
        "尺寸范围：孔内切直径 68–76 mm；标称径向间隙 14–15 mm；三角插销外接半径上限 40 mm。",
        fontsize=10.5,
        color="#4e5665",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
