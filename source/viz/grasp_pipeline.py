"""Render presentation-ready diagrams of the grasp-generation pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

DEFAULT_OUTPUT_DIR: Final = Path("artifacts/grasp_pipeline")
NAVY: Final = "#183B64"
BLUE: Final = "#2F75B5"
TEAL: Final = "#2E8B89"
ORANGE: Final = "#E67E22"
LIGHT_TEAL: Final = "#E5F4F1"
LIGHT_ORANGE: Final = "#FDEFE0"
DARK: Final = "#2D343C"
GRAY: Final = "#96A0AA"
PLOT_STYLE: Final[dict[str, Any]] = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK HK",
        "Droid Sans Fallback",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ],
    "axes.unicode_minus": False,
}


def _canvas() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(12, 5.1), dpi=200)
    fig.patch.set_alpha(0)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.1)
    ax.axis("off")
    return fig, ax


def _box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    fontsize: int = 15,
    text_color: str = "white",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.12",
        linewidth=1.6,
        edgecolor=color,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=text_color,
    )


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = NAVY,
    style: str = "-|>",
    width: float = 2.2,
    curve: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=16,
            linewidth=width,
            color=color,
            connectionstyle=f"arc3,rad={curve}",
        )
    )


def _save(fig: Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / name
    fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output


def _stage_flow(
    ax: Axes,
    labels: Sequence[str],
    colors: Sequence[str],
    *,
    y: float,
    height: float,
) -> list[float]:
    """Draw the shared five-stage pipeline and return its x coordinates."""

    if len(labels) != len(colors):
        raise ValueError("Pipeline labels and colors must have equal lengths.")
    x_positions = [0.25 + 2.45 * index for index in range(len(labels))]
    center_y = y + height / 2
    for x, label, color in zip(x_positions, labels, colors, strict=True):
        _box(ax, x, y, 1.7, height, label, color, fontsize=14)
    for left, right in pairwise(x_positions):
        _arrow(ax, (left + 1.72, center_y), (right - 0.03, center_y))
    return x_positions


def _render_dexevolve(output_dir: Path) -> Path:
    fig, ax = _canvas()
    labels = [
        "解析抓取种子",
        "Isaac Sim\n物理仿真",
        "无梯度\n进化搜索",
        "稳定性 + 多样性",
        "Diffusion\n蒸馏",
    ]
    colors = [BLUE, TEAL, ORANGE, NAVY, BLUE]
    _stage_flow(ax, labels, colors, y=2.05, height=1.05)
    ax.text(
        6,
        4.25,
        "DexEvolve：模拟器不只验证抓取，还直接优化抓取",
        ha="center",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        6,
        0.85,
        "保留不完美种子，在高保真接触动力学中持续改进",
        ha="center",
        fontsize=16,
        color=DARK,
    )
    return _save(fig, output_dir, "01_dexevolve_pipeline.png")


def _render_migration(output_dir: Path) -> Path:
    fig, ax = _canvas()
    ax.text(
        6,
        4.55,
        "Generate-and-Refine 方法迁移",
        ha="center",
        fontsize=23,
        fontweight="bold",
        color=NAVY,
    )
    _box(ax, 0.65, 2.8, 2.15, 0.85, "解析抓取", BLUE)
    _box(ax, 3.35, 2.8, 2.15, 0.85, "Isaac Sim", TEAL)
    _arrow(ax, (2.82, 3.22), (3.3, 3.22))
    ax.text(0.6, 3.95, "DexEvolve", fontsize=17, fontweight="bold", color=BLUE)
    _box(ax, 0.65, 1.15, 2.15, 0.85, "GraspQP", ORANGE)
    _box(ax, 3.35, 1.15, 2.15, 0.85, "MuJoCo", NAVY)
    _arrow(ax, (2.82, 1.57), (3.3, 1.57))
    ax.text(0.6, 2.3, "本项目", fontsize=17, fontweight="bold", color=ORANGE)
    _arrow(ax, (5.8, 3.22), (7.05, 3.22), GRAY)
    _arrow(ax, (5.8, 1.57), (7.05, 1.57), GRAY)
    panel = FancyBboxPatch(
        (7.15, 0.85),
        4.15,
        3.15,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=LIGHT_TEAL,
        edgecolor=TEAL,
        linewidth=1.8,
    )
    ax.add_patch(panel)
    ax.text(9.23, 3.45, "迁移后的目标", ha="center", fontsize=18, fontweight="bold", color=TEAL)
    ax.text(9.23, 2.65, "欠驱动 Dex Hand", ha="center", fontsize=17, color=DARK)
    ax.text(9.23, 2.05, "真实接触动力学", ha="center", fontsize=17, color=DARK)
    ax.text(9.23, 1.45, "127种物体统一测试", ha="center", fontsize=17, color=DARK)
    return _save(fig, output_dir, "02_method_migration.png")


def _render_graspqp_adapter(output_dir: Path) -> Path:
    fig, ax = _canvas()
    ax.text(
        6,
        4.55,
        "GraspQP 与欠驱动 Dex Hand 适配",
        ha="center",
        fontsize=23,
        fontweight="bold",
        color=NAVY,
    )
    obj = Circle((2.0, 2.5), 0.72, facecolor=LIGHT_ORANGE, edgecolor=ORANGE, linewidth=2)
    ax.add_patch(obj)
    ax.text(
        2.0, 2.5, "物体\n表面", ha="center", va="center", fontsize=17, fontweight="bold", color=DARK
    )
    for angle_start, angle_end in [((0.85, 3.45), (1.5, 2.95)), ((0.82, 1.55), (1.5, 2.05))]:
        _arrow(ax, angle_start, angle_end, BLUE)
    ax.text(2.0, 1.05, "接触点 + 法向 + 摩擦锥", ha="center", fontsize=14, color=BLUE)
    _box(ax, 4.15, 2.05, 2.15, 1.0, "6个执行器", NAVY)
    _arrow(ax, (3.05, 2.5), (4.05, 2.5), GRAY)
    _box(ax, 7.05, 2.05, 2.15, 1.0, "MuJoCo\n肌腱平衡", TEAL)
    _arrow(ax, (6.32, 2.5), (6.95, 2.5))
    _box(ax, 9.85, 2.05, 1.8, 1.0, "位置与\nJacobian", ORANGE)
    _arrow(ax, (9.22, 2.5), (9.75, 2.5))
    ax.text(7.05, 1.15, "被动关节由物理模型求解", fontsize=14, color=TEAL)
    ax.text(
        6,
        3.65,
        "GraspQP：用QP力闭合指标联合优化手腕位姿与手型",
        ha="center",
        fontsize=17,
        color=DARK,
        fontweight="bold",
    )
    return _save(fig, output_dir, "03_graspqp_closed_chain.png")


def _render_evolution(output_dir: Path) -> Path:
    fig, ax = _canvas()
    ax.text(
        6,
        4.55,
        "MuJoCo 进化优化与保持评价",
        ha="center",
        fontsize=23,
        fontweight="bold",
        color=NAVY,
    )
    labels = ["种群初始化", "变异 / 交叉", "定姿释放", "物理保持", "存档与筛选"]
    colors = [BLUE, ORANGE, TEAL, NAVY, BLUE]
    _stage_flow(ax, labels, colors, y=2.45, height=0.92)
    _arrow(ax, (10.85, 2.35), (1.05, 2.12), GRAY, width=1.8, curve=-0.26)
    ax.text(
        6,
        1.62,
        "fitness：稳定性、接触数、漂移、掉落与旋转偏差",
        ha="center",
        fontsize=17,
        fontweight="bold",
        color=DARK,
    )
    ax.text(6, 0.88, "Genome = wrist pose + actuator[6]", ha="center", fontsize=16, color=ORANGE)
    return _save(fig, output_dir, "04_mujoco_evolution.png")


def render_grasp_pipeline_figures(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, ...]:
    """Render all grasp-pipeline diagrams and return their output paths."""

    with plt.rc_context(PLOT_STYLE):
        return (
            _render_dexevolve(output_dir),
            _render_migration(output_dir),
            _render_graspqp_adapter(output_dir),
            _render_evolution(output_dir),
        )
