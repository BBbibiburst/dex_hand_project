"""Render editable-source PNG diagrams for the 2026-08-11 group meeting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUTPUT = Path("artifacts/ppt_20260811")
NAVY = "#183B64"
BLUE = "#2F75B5"
TEAL = "#2E8B89"
ORANGE = "#E67E22"
LIGHT_BLUE = "#E7F1FA"
LIGHT_TEAL = "#E5F4F1"
LIGHT_ORANGE = "#FDEFE0"
DARK = "#2D343C"
GRAY = "#96A0AA"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Micro Hei"],
        "axes.unicode_minus": False,
    }
)


def canvas():
    fig, ax = plt.subplots(figsize=(12, 5.1), dpi=200)
    fig.patch.set_alpha(0)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color, *, fontsize=15, text_color="white"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.12",
        linewidth=1.6,
        edgecolor=color,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=text_color,
    )


def arrow(ax, start, end, color=NAVY, style="-|>", width=2.2, curve=0.0):
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


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_dexevolve():
    fig, ax = canvas()
    labels = ["解析抓取种子", "Isaac Sim\n物理仿真", "无梯度\n进化搜索", "稳定性 + 多样性", "Diffusion\n蒸馏"]
    colors = [BLUE, TEAL, ORANGE, NAVY, BLUE]
    xs = [0.25, 2.7, 5.15, 7.6, 10.05]
    for x, label, color in zip(xs, labels, colors, strict=True):
        box(ax, x, 2.05, 1.7, 1.05, label, color, fontsize=14)
    for x in xs[:-1]:
        arrow(ax, (x + 1.72, 2.58), (x + 2.42, 2.58))
    ax.text(6, 4.25, "DexEvolve：模拟器不只验证抓取，还直接优化抓取", ha="center",
            fontsize=22, fontweight="bold", color=NAVY)
    ax.text(6, 0.85, "保留不完美种子，在高保真接触动力学中持续改进",
            ha="center", fontsize=16, color=DARK)
    save(fig, "01_dexevolve_pipeline.png")


def render_migration():
    fig, ax = canvas()
    ax.text(6, 4.55, "Generate-and-Refine 方法迁移", ha="center", fontsize=23,
            fontweight="bold", color=NAVY)
    box(ax, 0.65, 2.8, 2.15, 0.85, "解析抓取", BLUE)
    box(ax, 3.35, 2.8, 2.15, 0.85, "Isaac Sim", TEAL)
    arrow(ax, (2.82, 3.22), (3.3, 3.22))
    ax.text(0.6, 3.95, "DexEvolve", fontsize=17, fontweight="bold", color=BLUE)
    box(ax, 0.65, 1.15, 2.15, 0.85, "GraspQP", ORANGE)
    box(ax, 3.35, 1.15, 2.15, 0.85, "MuJoCo", NAVY)
    arrow(ax, (2.82, 1.57), (3.3, 1.57))
    ax.text(0.6, 2.3, "本项目", fontsize=17, fontweight="bold", color=ORANGE)
    arrow(ax, (5.8, 3.22), (7.05, 3.22), GRAY)
    arrow(ax, (5.8, 1.57), (7.05, 1.57), GRAY)
    panel = FancyBboxPatch((7.15, 0.85), 4.15, 3.15,
                          boxstyle="round,pad=0.05,rounding_size=0.15",
                          facecolor=LIGHT_TEAL, edgecolor=TEAL, linewidth=1.8)
    ax.add_patch(panel)
    ax.text(9.23, 3.45, "迁移后的目标", ha="center", fontsize=18,
            fontweight="bold", color=TEAL)
    ax.text(9.23, 2.65, "闭链 Dex Hand", ha="center", fontsize=17, color=DARK)
    ax.text(9.23, 2.05, "真实接触动力学", ha="center", fontsize=17, color=DARK)
    ax.text(9.23, 1.45, "127种物体统一测试", ha="center", fontsize=17, color=DARK)
    save(fig, "02_method_migration.png")


def render_graspqp_adapter():
    fig, ax = canvas()
    ax.text(6, 4.55, "GraspQP 与闭链 Dex Hand 适配", ha="center", fontsize=23,
            fontweight="bold", color=NAVY)
    obj = Circle((2.0, 2.5), 0.72, facecolor=LIGHT_ORANGE, edgecolor=ORANGE, linewidth=2)
    ax.add_patch(obj)
    ax.text(2.0, 2.5, "物体\n表面", ha="center", va="center", fontsize=17,
            fontweight="bold", color=DARK)
    for angle_start, angle_end in [((0.85, 3.45), (1.5, 2.95)),
                                   ((0.82, 1.55), (1.5, 2.05))]:
        arrow(ax, angle_start, angle_end, BLUE)
    ax.text(2.0, 1.05, "接触点 + 法向 + 摩擦锥", ha="center", fontsize=14, color=BLUE)
    box(ax, 4.15, 2.05, 2.15, 1.0, "6个执行器", NAVY)
    arrow(ax, (3.05, 2.5), (4.05, 2.5), GRAY)
    box(ax, 7.05, 2.05, 2.15, 1.0, "MuJoCo\n闭链求解", TEAL)
    arrow(ax, (6.32, 2.5), (6.95, 2.5))
    box(ax, 9.85, 2.05, 1.8, 1.0, "位置与\nJacobian", ORANGE)
    arrow(ax, (9.22, 2.5), (9.75, 2.5))
    ax.text(7.05, 1.15, "被动关节由物理模型求解", fontsize=14, color=TEAL)
    ax.text(6, 3.65, "GraspQP：用QP力闭合指标联合优化手腕位姿与手型",
            ha="center", fontsize=17, color=DARK, fontweight="bold")
    save(fig, "03_graspqp_closed_chain.png")


def render_evolution():
    fig, ax = canvas()
    ax.text(6, 4.55, "MuJoCo 进化优化与动态评价", ha="center", fontsize=23,
            fontweight="bold", color=NAVY)
    labels = ["种群初始化", "变异 / 交叉", "动态闭合", "六向扰动", "精英长时复评"]
    colors = [BLUE, ORANGE, TEAL, NAVY, BLUE]
    xs = [0.25, 2.7, 5.15, 7.6, 10.05]
    for x, label, color in zip(xs, labels, colors, strict=True):
        box(ax, x, 2.45, 1.7, 0.92, label, color, fontsize=14)
    for x in xs[:-1]:
        arrow(ax, (x + 1.72, 2.91), (x + 2.42, 2.91))
    arrow(ax, (10.85, 2.35), (1.05, 2.12), GRAY, width=1.8, curve=-0.26)
    ax.text(6, 1.62, "fitness：扰动生存时间为主，漂移/掉落/穿透为辅",
            ha="center", fontsize=17, fontweight="bold", color=DARK)
    ax.text(6, 0.88, "Genome = wrist pose + actuator[6] + preload",
            ha="center", fontsize=16, color=ORANGE)
    save(fig, "04_mujoco_evolution.png")


def main():
    render_dexevolve()
    render_migration()
    render_graspqp_adapter()
    render_evolution()
    for path in sorted(OUTPUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
