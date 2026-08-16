"""Point-cloud visualization for modular grasp-search results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from source.grasping.search.planning import approach
from source.grasping.search.types import Candidate, Cloud

def draw(cloud: Cloud, candidate: Candidate, *, output: Path | None, show: bool) -> None:
    """Visualize the object and end effector as lightweight point clouds.

    The search still uses the original sampled geometry.  This function only
    changes rendering, so visualization density has no effect on grasp scores.
    """
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")

    # Object: point-cloud rendering avoids the jagged appearance and interaction
    # cost of a heavily decimated triangle mesh.
    object_limit = 1_200
    object_stride = max(1, (len(cloud.points) + object_limit - 1) // object_limit)
    object_points = cloud.points[::object_stride]
    axis.scatter(
        *object_points.T,
        s=6,
        alpha=0.48,
        color="#5b87ad",
        linewidths=0.0,
        depthshade=False,
        label="object point cloud",
    )

    # Hand: render each semantic region separately so the palm and fingers remain
    # recognizable even when points overlap in the projected view.
    labels = candidate.surface.labels
    posed_points = candidate.points
    unique_labels = sorted(int(value) for value in np.unique(labels))
    is_pika = len(candidate.surface.fractions) == 1
    body_label = 2 if is_pika else 6
    palm_label = None if is_pika else 5
    hand_colors = {
        0: "#ef4444",
        1: "#8b5cf6",
        2: "#06b6d4",
        3: "#22c55e",
        4: "#eab308",
        5: "#d97706",
        6: "#6b7280",
    }
    hand_names = {
        0: "finger 0",
        1: "finger 1",
        2: "finger 2",
        3: "finger 3",
        4: "thumb",
        5: "palm",
        6: "hand body",
    }
    if is_pika:
        hand_colors[2] = "#6b7280"
        hand_names.update({0: "left finger", 1: "right finger", 2: "gripper body"})
    for label in unique_labels:
        region = posed_points[labels == label]
        if not len(region):
            continue
        # Contact regions retain the most detail. The non-contact hand body is
        # deliberately sparse and translucent so it cannot hide the fingers.
        region_limit = 350 if label == body_label else 500 if label == palm_label else 650
        stride = max(1, (len(region) + region_limit - 1) // region_limit)
        region = region[::stride]
        axis.scatter(
            *region.T,
            s=5 if label == body_label else 9 if label == palm_label else 11,
            alpha=0.22 if label == body_label else 0.62 if label == palm_label else 0.88,
            color=hand_colors.get(label, "#6b7280"),
            linewidths=0.0,
            depthshade=False,
            label=hand_names.get(label, f"hand region {label}"),
        )

    if len(candidate.contact_points):
        axis.scatter(
            *candidate.contact_points.T,
            s=95,
            color="#111827",
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
            label="contacts",
        )
        axis.quiver(
            *candidate.contact_points.T,
            *candidate.contact_normals.T,
            length=0.02,
            color="#111827",
            linewidth=1.4,
        )

    translations, _ = approach(candidate)
    axis.plot(*translations.T, color="#2ca02c", linewidth=2.5, label="approach")
    axis.scatter(
        *translations[0],
        s=45,
        marker="o",
        color="#2ca02c",
        depthshade=False,
        label="pregrasp",
    )
    if candidate.approach_plan is not None:
        axis.plot(
            *candidate.approach_plan.grasp_translations.T,
            color="#dc2626",
            linewidth=2.5,
            label="checked closing trajectory",
        )

    # Draw the inferred support plane as a wire grid.  It is cheap and makes
    # table-clearance failures much easier to understand.
    visible = np.concatenate([cloud.points, candidate.points, translations])
    low, high = visible.min(0), visible.max(0)
    center = 0.5 * (low + high)
    radius = max(0.01, 0.55 * float(np.ptp(visible, axis=0).max()))
    table_z = float(cloud.points[:, 2].min())
    grid_values = np.linspace(-radius, radius, 7)
    for offset in grid_values:
        axis.plot(
            [center[0] - radius, center[0] + radius],
            [center[1] + offset, center[1] + offset],
            [table_z, table_z],
            color="#6b7280",
            alpha=0.18,
            linewidth=0.7,
        )
        axis.plot(
            [center[0] + offset, center[0] + offset],
            [center[1] - radius, center[1] + radius],
            [table_z, table_z],
            color="#6b7280",
            alpha=0.18,
            linewidth=0.7,
        )

    axis.set(
        xlim=(center[0] - radius, center[0] + radius),
        ylim=(center[1] - radius, center[1] + radius),
        zlim=(min(table_z - 0.01, center[2] - radius), center[2] + radius),
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
        title=(
            f"point-cloud grasp view {candidate.surface.fractions.tolist()}\n"
            f"score={candidate.score:.3f}, Efc={candidate.force_closure:.3f}, "
            f"Eg={candidate.gravity_balance_residual:.3f}, "
            f"Eworst={candidate.disturbance_residual:.3f}, valid={candidate.valid}"
        ),
    )
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=24, azim=-58)
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
