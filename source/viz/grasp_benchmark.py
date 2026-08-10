"""Visualization API for grasp catalogue benchmark reports."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Patch


STATUS_ORDER = (
    "stable",
    "unstable",
    "validation_error",
    "search_error",
)

STATUS_LABELS = {
    "stable": "Stable",
    "unstable": "Unstable",
    "validation_error": "Validation error",
    "search_error": "Search error",
}

STATUS_COLORS = {
    "stable": "tab:green",
    "unstable": "tab:orange",
    "validation_error": "tab:red",
    "search_error": "tab:gray",
}


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Report file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError("The benchmark report root must be a JSON object.")

    objects = report.get("objects")
    if not isinstance(objects, list):
        raise ValueError(
            "The benchmark report must contain an 'objects' list."
        )

    return report


def normalize_status(row: Mapping[str, Any]) -> str:
    raw_status = row.get("status")

    if isinstance(raw_status, str):
        status = raw_status.strip().lower()
        status = status.replace("-", "_").replace(" ", "_")

        aliases = {
            "success": "stable",
            "passed": "stable",
            "pass": "stable",
            "failed": "unstable",
            "failure": "unstable",
            "error": "validation_error",
        }
        status = aliases.get(status, status)

        if status in STATUS_ORDER:
            return status

    stable = row.get("stable")

    if stable is True:
        return "stable"

    if stable is False:
        return "unstable"

    return "validation_error"


def object_id(row: Mapping[str, Any]) -> str:
    for key in ("object_id", "name", "id", "object"):
        value = row.get(key)
        if value is not None:
            return str(value)

    return "unknown"


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(number):
        return default

    return number


def metric_array(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Return one value per row without removing missing entries.

    Missing or invalid values are represented as NaN. This is important:
    every metric array always preserves the exact object ordering and length.
    """

    values = [
        as_float(row.get(key)) * scale
        if not math.isnan(as_float(row.get(key)))
        else math.nan
        for row in rows
    ]
    return np.asarray(values, dtype=float)


def finite_or_zero(value: Any) -> float:
    number = as_float(value)

    if math.isnan(number):
        return 0.0

    return number


def failure_score(row: Mapping[str, Any]) -> float:
    """Diagnostic score used only for ranking difficult failed objects."""

    status = normalize_status(row)

    position_drift_mm = (
        finite_or_zero(row.get("position_drift")) * 1000.0
    )
    vertical_drop_mm = (
        max(0.0, finite_or_zero(row.get("vertical_drop"))) * 1000.0
    )
    initial_displacement_mm = (
        finite_or_zero(row.get("initial_displacement")) * 1000.0
    )
    rotation_drift_deg = math.degrees(
        finite_or_zero(row.get("rotation_drift"))
    )

    initial_contacts = finite_or_zero(row.get("initial_contacts"))
    final_contacts = finite_or_zero(row.get("final_contacts"))
    lost_contacts = max(0.0, initial_contacts - final_contacts)

    score = (
        position_drift_mm
        + vertical_drop_mm * 1.5
        + initial_displacement_mm * 0.25
        + rotation_drift_deg * 0.2
        + lost_contacts * 2.0
    )

    if status == "validation_error":
        score += 100.0
    elif status == "search_error":
        score += 150.0
    elif status == "unstable":
        score += 20.0

    return score


def sort_rows(
    rows: Sequence[Mapping[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]

    if mode == "catalog":
        return copied

    if mode == "name":
        return sorted(copied, key=lambda row: object_id(row).lower())

    if mode == "status":
        rank = {status: index for index, status in enumerate(STATUS_ORDER)}
        return sorted(
            copied,
            key=lambda row: (
                rank.get(normalize_status(row), len(rank)),
                object_id(row).lower(),
            ),
        )

    if mode == "drift":
        return sorted(
            copied,
            key=lambda row: failure_score(row),
            reverse=True,
        )

    if mode == "time":
        return sorted(
            copied,
            key=lambda row: finite_or_zero(row.get("elapsed_seconds")),
            reverse=True,
        )

    raise ValueError(f"Unsupported sort mode: {mode}")


def status_color(row: Mapping[str, Any]) -> str:
    return STATUS_COLORS[normalize_status(row)]


def status_legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=STATUS_COLORS[status],
            label=STATUS_LABELS[status],
        )
        for status in STATUS_ORDER
    ]


def set_no_data(ax: Axes, message: str = "No data") -> None:
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def configure_object_axis(
    ax: Axes,
    x: np.ndarray,
    labels: Sequence[str],
    *,
    max_labels: int,
    show_labels: bool = True,
) -> None:
    count = len(labels)

    if count == 0:
        return

    ax.set_xlim(-0.65, count - 0.35)

    if not show_labels:
        ax.set_xticks([])
        return

    max_labels = max(1, max_labels)
    step = max(1, math.ceil(count / max_labels))
    indices = np.arange(0, count, step, dtype=int)

    # Always include the last object when it is not already selected.
    if len(indices) == 0 or indices[-1] != count - 1:
        indices = np.append(indices, count - 1)

    ax.set_xticks(x[indices])
    ax.set_xticklabels(
        [labels[index] for index in indices],
        rotation=60,
        ha="right",
        fontsize=8,
    )


def plot_summary(
    ax: Axes,
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    counts = Counter(normalize_status(row) for row in rows)

    statuses = [
        status
        for status in STATUS_ORDER
        if counts.get(status, 0) > 0
    ]

    if not statuses:
        set_no_data(ax)
        return

    values = [counts[status] for status in statuses]
    labels = [STATUS_LABELS[status] for status in statuses]
    colors = [STATUS_COLORS[status] for status in statuses]

    bars = ax.bar(labels, values, color=colors, width=0.65)

    total = len(rows)
    stable_count = counts.get("stable", 0)
    stable_rate = stable_count / total * 100.0 if total else 0.0

    ax.set_title(
        f"Result summary — stable {stable_count}/{total} "
        f"({stable_rate:.1f}%)"
    )
    ax.set_ylabel("Object count")
    ax.grid(axis="y", alpha=0.25)

    upper = max(values) if values else 1
    ax.set_ylim(0, upper * 1.22 + 0.5)

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )

    summary = report.get("summary")
    if isinstance(summary, Mapping):
        selected = summary.get("selected")
        completed = summary.get("completed")

        text_parts = []

        if selected is not None:
            text_parts.append(f"selected: {selected}")

        if completed is not None:
            text_parts.append(f"completed: {completed}")

        if text_parts:
            ax.text(
                0.98,
                0.96,
                "\n".join(text_parts),
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
            )


def plot_failure_ranking(
    ax: Axes,
    rows: Sequence[Mapping[str, Any]],
    *,
    top_n: int,
) -> None:
    failed_rows = [
        row for row in rows if normalize_status(row) != "stable"
    ]

    failed_rows.sort(key=failure_score, reverse=True)
    failed_rows = failed_rows[: max(1, top_n)]

    if not failed_rows:
        set_no_data(ax, "No failed objects")
        ax.set_title("Worst non-stable objects")
        return

    # Reverse so that the worst object appears at the top of barh.
    failed_rows = list(reversed(failed_rows))

    labels = [object_id(row) for row in failed_rows]
    scores = [failure_score(row) for row in failed_rows]
    colors = [status_color(row) for row in failed_rows]
    y = np.arange(len(failed_rows), dtype=float)

    ax.barh(y, scores, color=colors, height=0.68, align="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Diagnostic failure score")
    ax.set_title("Worst non-stable objects")
    ax.grid(axis="x", alpha=0.25)


def plot_status_strip(
    ax: Axes,
    rows: Sequence[Mapping[str, Any]],
    x: np.ndarray,
) -> None:
    if not rows:
        set_no_data(ax)
        return

    colors = [status_color(row) for row in rows]

    # Each status cell is centered exactly on its integer object coordinate.
    ax.bar(
        x,
        np.ones(len(rows), dtype=float),
        width=0.94,
        color=colors,
        align="center",
        linewidth=0,
    )

    ax.set_title("Per-object benchmark status")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.legend(
        handles=status_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        fontsize=8,
    )


def plot_translation_metrics(
    ax: Axes,
    rows: Sequence[Mapping[str, Any]],
    x: np.ndarray,
) -> None:
    if not rows:
        set_no_data(ax)
        return

    initial = metric_array(
        rows,
        "initial_displacement",
        scale=1000.0,
    )
    drift = metric_array(
        rows,
        "position_drift",
        scale=1000.0,
    )
    drop = metric_array(
        rows,
        "vertical_drop",
        scale=1000.0,
    )

    width = 0.24

    # All three series use the same base x. Missing entries stay as NaN.
    ax.bar(
        x - width,
        initial,
        width=width,
        align="center",
        label="Initial displacement",
    )
    ax.bar(
        x,
        drift,
        width=width,
        align="center",
        label="Position drift",
    )
    ax.bar(
        x + width,
        drop,
        width=width,
        align="center",
        label="Vertical drop",
    )

    ax.axhline(0.0, linewidth=0.8)
    ax.set_title("Translation stability")
    ax.set_ylabel("Distance (mm)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=3)


def plot_rotation_metric(
    ax: Axes,
    rows: Sequence[Mapping[str, Any]],
    x: np.ndarray,
) -> None:
    if not rows:
        set_no_data(ax)
        return

    rotation_deg = metric_array(
        rows,
        "rotation_drift",
        scale=180.0 / math.pi,
    )

    colors = [status_color(row) for row in rows]

    ax.bar(
        x,
        rotation_deg,
        width=0.72,
        align="center",
        color=colors,
    )

    ax.set_title("Rotation drift")
    ax.set_ylabel("Rotation (degrees)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=status_legend_handles(),
        fontsize=8,
        ncol=2,
    )


def plot_contacts(
    ax: Axes,
    rows: Sequence[Mapping[str, Any]],
    x: np.ndarray,
) -> None:
    if not rows:
        set_no_data(ax)
        return

    initial = metric_array(rows, "initial_contacts")
    final = metric_array(rows, "final_contacts")

    width = 0.36

    # Pair center:
    # (x - width / 2 + x + width / 2) / 2 == x
    ax.bar(
        x - width / 2.0,
        initial,
        width=width,
        align="center",
        label="Initial contacts",
    )
    ax.bar(
        x + width / 2.0,
        final,
        width=width,
        align="center",
        label="Final contacts",
    )

    ax.set_title("Object contact retention")
    ax.set_ylabel("Contact count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)


def plot_runtime(
    ax: Axes,
    rows: Sequence[Mapping[str, Any]],
    x: np.ndarray,
) -> Axes | None:
    if not rows:
        set_no_data(ax)
        return None

    elapsed = metric_array(rows, "elapsed_seconds")
    search = metric_array(rows, "search_seconds")
    attempts = metric_array(rows, "attempts_used")

    width = 0.36

    ax.bar(
        x - width / 2.0,
        elapsed,
        width=width,
        align="center",
        label="Total elapsed",
    )
    ax.bar(
        x + width / 2.0,
        search,
        width=width,
        align="center",
        label="Search time",
    )

    ax.set_title("Runtime and search attempts")
    ax.set_ylabel("Time (s)")
    ax.grid(axis="y", alpha=0.25)

    attempts_ax = ax.twinx()
    attempts_ax.plot(
        x,
        attempts,
        marker="o",
        linewidth=1.2,
        markersize=3.5,
        label="Attempts used",
    )
    attempts_ax.set_ylabel("Attempts")

    finite_attempts = attempts[np.isfinite(attempts)]
    if finite_attempts.size:
        attempts_ax.set_ylim(
            0,
            max(1.0, float(np.max(finite_attempts))) + 0.6,
        )

    left_handles, left_labels = ax.get_legend_handles_labels()
    right_handles, right_labels = (
        attempts_ax.get_legend_handles_labels()
    )

    ax.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        fontsize=8,
        ncol=3,
    )

    return attempts_ax


def add_failure_background(
    ax: Axes,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Add subtle bands behind non-stable object positions."""

    for index, row in enumerate(rows):
        if normalize_status(row) == "stable":
            continue

        ax.axvspan(
            index - 0.48,
            index + 0.48,
            alpha=0.045,
            color=status_color(row),
            linewidth=0,
            zorder=0,
        )


def _kpi_card(
    ax: Axes,
    x: float,
    title: str,
    value: str,
    subtitle: str,
    color: str,
) -> None:
    card = FancyBboxPatch(
        (x, 0.08),
        0.22,
        0.82,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=color,
        edgecolor=color,
        alpha=0.95,
    )
    ax.add_patch(card)
    ax.text(x + 0.11, 0.72, title, transform=ax.transAxes, ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    ax.text(x + 0.11, 0.44, value, transform=ax.transAxes, ha="center", va="center",
            color="white", fontsize=24, fontweight="bold")
    ax.text(x + 0.11, 0.20, subtitle, transform=ax.transAxes, ha="center", va="center",
            color="white", fontsize=9)


def plot_algorithm_kpis(
    ax: Axes,
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    summary = report.get("summary", {})
    selected = int(summary.get("selected", len(rows)))
    completed = len(rows)
    generated = sum(normalize_status(row) != "search_error" for row in rows)
    stable = sum(normalize_status(row) == "stable" for row in rows)
    recovered = sum(
        normalize_status(row) == "stable" and row.get("seed_stable") is not True
        for row in rows
    )
    _kpi_card(ax, 0.015, "TEST PROGRESS", f"{completed}/{selected}", "completed", "#1f4e79")
    _kpi_card(
        ax,
        0.265,
        "GENERATION",
        f"{generated / completed:.1%}" if completed else "0%",
        f"{generated}/{completed} generated",
        "#2f75b5",
    )
    _kpi_card(
        ax,
        0.515,
        "FINAL STABLE",
        f"{stable / completed:.1%}" if completed else "0%",
        f"{stable}/{completed} stable",
        "#2e8b89",
    )
    _kpi_card(
        ax,
        0.765,
        "SEED RECOVERY",
        str(recovered),
        "stable from non-stable/unverified seed",
        "#e67e22",
    )
    ax.axis("off")


def plot_evolution_yield(ax: Axes, rows: Sequence[Mapping[str, Any]]) -> None:
    points = []
    for row in rows:
        evolution = row.get("evolution")
        if not isinstance(evolution, Mapping):
            continue
        archive = as_float(evolution.get("archive"))
        stable_candidates = as_float(evolution.get("stable_candidates"))
        if archive > 0 and not math.isnan(stable_candidates):
            points.append((100.0 * stable_candidates / archive, row))
    if not points:
        set_no_data(ax, "No evolution data")
        return
    stable_x = [value for value, row in points if normalize_status(row) == "stable"]
    failed_x = [value for value, row in points if normalize_status(row) != "stable"]
    ax.scatter(stable_x, np.ones(len(stable_x)), color=STATUS_COLORS["stable"], s=38,
               alpha=0.75, label="Final stable")
    ax.scatter(failed_x, np.zeros(len(failed_x)), color=STATUS_COLORS["unstable"], s=38,
               alpha=0.75, label="Final unstable")
    ax.set_yticks((0, 1), ("Unstable", "Stable"))
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Stable candidates in evolution archive (%)")
    ax.set_title("Does short-rollout evolution predict final stability?")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def plot_evolution_convergence(ax: Axes, rows: Sequence[Mapping[str, Any]]) -> Axes | None:
    histories = []
    for row in rows:
        evolution = row.get("evolution")
        history = evolution.get("history") if isinstance(evolution, Mapping) else None
        if isinstance(history, list) and history:
            histories.append(history)
    if not histories:
        set_no_data(ax, "No evolution history")
        return None
    generations = sorted({int(item["generation"]) for history in histories for item in history})
    yield_median, fitness_median = [], []
    for generation in generations:
        generation_rows = [
            item for history in histories for item in history
            if int(item.get("generation", -1)) == generation
        ]
        yields = [100.0 * as_float(item.get("stable")) / as_float(item.get("archive"))
                  for item in generation_rows if as_float(item.get("archive")) > 0]
        fitness = [as_float(item.get("best_fitness")) for item in generation_rows]
        yield_median.append(float(np.nanmedian(yields)))
        fitness_median.append(float(np.nanmedian(fitness)))
    ax.plot(generations, yield_median, color="#2e8b89", marker="o", markersize=3,
            linewidth=2, label="Median stable-candidate ratio")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Archive stable candidates (%)", color="#2e8b89")
    ax.set_title("Evolution convergence across objects")
    ax.grid(alpha=0.25)
    twin = ax.twinx()
    twin.plot(generations, fitness_median, color="#e67e22", linewidth=1.8,
              label="Median best fitness")
    twin.set_ylabel("Best fitness (median)", color="#e67e22")
    handles, labels = ax.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    ax.legend(handles + twin_handles, labels + twin_labels, frameon=False, fontsize=8)
    return twin


def classify_failure(row: Mapping[str, Any]) -> str:
    if normalize_status(row) == "stable":
        return "Stable"
    if bool(row.get("numerical_failure")):
        return "Numerical failure"
    phase = row.get("failure_phase")
    if phase:
        return str(phase).replace("_", " ").title()
    if normalize_status(row) in ("search_error", "validation_error"):
        return STATUS_LABELS[normalize_status(row)]
    if finite_or_zero(row.get("final_contacts")) < 2:
        return "Lost contact"
    if finite_or_zero(row.get("vertical_drop")) > 0.015:
        return "Dropped"
    if finite_or_zero(row.get("position_drift")) > 0.01:
        return "Position drift"
    if finite_or_zero(row.get("rotation_drift")) > 0.35:
        return "Rotation drift"
    return "Other unstable"


def plot_failure_modes(ax: Axes, rows: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(classify_failure(row) for row in rows if normalize_status(row) != "stable")
    if not counts:
        set_no_data(ax, "No failures")
        return
    labels, values = zip(*counts.most_common(), strict=True)
    y = np.arange(len(labels))
    ax.barh(y, values, color="#e67e22", alpha=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Object count")
    ax.set_title("Final failure modes")
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(values):
        ax.text(value + 0.15, index, str(value), va="center", fontsize=9)


def plot_robustness_scatter(ax: Axes, rows: Sequence[Mapping[str, Any]]) -> None:
    position = metric_array(rows, "position_drift", scale=1000.0)
    rotation = metric_array(rows, "rotation_drift", scale=180.0 / math.pi)
    position_cap, rotation_cap = 50.0, 90.0
    clipped = int(np.sum((position > position_cap) | (rotation > rotation_cap)))
    colors = [status_color(row) for row in rows]
    ax.scatter(np.clip(position, 0, position_cap), np.clip(rotation, 0, rotation_cap),
               c=colors, s=30, alpha=0.72)
    ax.axvline(10.0, color="#2f75b5", linestyle="--", linewidth=1.3, label="10 mm limit")
    ax.axhline(math.degrees(0.35), color="#e67e22", linestyle="--", linewidth=1.3,
               label="20.1° limit")
    ax.set_xlabel("Position drift (mm, clipped at 50)")
    ax.set_ylabel("Rotation drift (deg, clipped at 90)")
    ax.set_title("Final physical robustness")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    if clipped:
        ax.text(0.98, 0.04, f"{clipped} outliers clipped", transform=ax.transAxes,
                ha="right", fontsize=8, color="#666666")


def build_figure(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    top_failures: int,
    max_labels: int,
    report_name: str,
    sort_mode: str,
) -> Figure:
    x = np.arange(len(rows), dtype=float)
    labels = [object_id(row) for row in rows]
    figure = plt.figure(
        figsize=(16, 9),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=(0.9, 0.38, 1.35, 1.35),
        width_ratios=(1.0, 1.0),
    )
    ax_kpis = figure.add_subplot(grid[0, :])
    ax_status = figure.add_subplot(grid[1, :])
    ax_evolution_yield = figure.add_subplot(grid[2, 0])
    ax_convergence = figure.add_subplot(grid[2, 1])
    ax_robustness = figure.add_subplot(grid[3, 0])
    ax_failures = figure.add_subplot(grid[3, 1])

    plot_algorithm_kpis(ax_kpis, report, rows)
    plot_status_strip(ax_status, rows, x)
    configure_object_axis(ax_status, x, labels, max_labels=max_labels, show_labels=True)
    plot_evolution_yield(ax_evolution_yield, rows)
    plot_evolution_convergence(ax_convergence, rows)
    plot_robustness_scatter(ax_robustness, rows)
    plot_failure_modes(ax_failures, rows)

    figure.suptitle(
        f"GraspQP → MuJoCo Evolution Benchmark  |  {report_name}\n"
        f"in-progress objects={len(rows)}, sort={sort_mode}",
        fontsize=15,
    )

    return figure


def render_grasp_benchmark_report(
    report_path: Path,
    *,
    output: Path | None = None,
    sort_mode: str = "catalog",
    top_failures: int = 15,
    max_labels: int = 24,
    dpi: int = 180,
    show: bool = False,
) -> Path:
    """Render a benchmark JSON report and return the written image path."""
    report_path = Path(report_path)
    report = load_report(report_path)
    rows = sort_rows(report["objects"], sort_mode)
    output = output or report_path.with_name(f"{report_path.stem}_visualization.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(
        report,
        rows,
        top_failures=top_failures,
        max_labels=max_labels,
        report_name=report_path.name,
        sort_mode=sort_mode,
    )
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return output
