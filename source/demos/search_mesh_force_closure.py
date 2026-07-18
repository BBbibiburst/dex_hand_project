"""Search and visualize collision-free grasps for a supported end effector."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from source.envs.manipulation.object_catalog import DEFAULT_LIFT_OBJECT
from source.grasping.grasp_config_search import search_grasp_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mesh", type=Path)
    source.add_argument("--object-id", default=DEFAULT_LIFT_OBJECT)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument(
        "--joint-candidates",
        type=int,
        default=128,
        help="Joint wrist/depth/actuator candidates.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-size", type=float, default=0.09)
    parser.add_argument(
        "--end-effector",
        choices=("dex_hand", "pika_gripper"),
        default="dex_hand",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON (default: configs/grasps/<object_id>.json).",
    )
    parser.add_argument("--preview", type=Path)
    parser.add_argument(
        "--preview-image",
        type=Path,
        help="Save a PNG showing the point cloud, contacts, and normals.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open an interactive Matplotlib 3D viewer after optimization.",
    )
    return parser.parse_args()


def _draw_contacts(cloud, closure, *, output: Path | None, show: bool) -> None:
    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        cloud.points[:, 0],
        cloud.points[:, 1],
        cloud.points[:, 2],
        s=2,
        c="#8ba6c1",
        alpha=0.22,
        label="object point cloud",
    )
    colors = plt.get_cmap("tab10")(np.arange(closure.contact_points.shape[0]))
    axis.scatter(
        closure.contact_points[:, 0],
        closure.contact_points[:, 1],
        closure.contact_points[:, 2],
        s=90,
        c=colors,
        edgecolors="black",
        linewidths=0.8,
        depthshade=False,
        label="force-closure contacts",
    )
    hand = getattr(closure, "hand", getattr(closure, "gripper", None))
    hand_points = closure.points
    hand_tips = getattr(closure, "fingertip_centers", None)
    if hand_tips is None:
        local_centers = getattr(hand, "contact_centers", np.empty((0, 3)))
        hand_tips = (
            local_centers @ closure.rotation_matrix.T + closure.translation
        )
    hand_colors = np.asarray(
        ["#f4a261", "#e76f51", "#2a9d8f", "#457b9d", "#9b5de5", "#777777"]
    )
    for label in np.unique(hand.labels):
        selected = hand.labels == label
        axis.scatter(
            hand_points[selected, 0],
            hand_points[selected, 1],
            hand_points[selected, 2],
            s=2.5,
            c=hand_colors[int(label) % len(hand_colors)],
            alpha=0.32 if label == 5 else 0.55,
        )
    axis.scatter(
        hand_tips[:, 0],
        hand_tips[:, 1],
        hand_tips[:, 2],
        marker="x",
        s=75,
        c=hand_colors[: len(hand_tips)],
        linewidths=2.0,
        label="end-effector contact centers",
    )
    path = closure.approach_translations
    if path.size:
        axis.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color="#2ca02c",
            linewidth=2.5,
            label="collision-free approach path",
        )
        axis.scatter(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            s=28,
            c=np.linspace(0.0, 1.0, len(path)),
            cmap="viridis",
        )
    normal_length = 0.025
    axis.quiver(
        closure.contact_points[:, 0],
        closure.contact_points[:, 1],
        closure.contact_points[:, 2],
        closure.contact_normals[:, 0],
        closure.contact_normals[:, 1],
        closure.contact_normals[:, 2],
        length=normal_length,
        normalize=True,
        color=colors,
        linewidth=2.0,
    )
    for index, (finger, point) in enumerate(
        zip(closure.contacting_fingers, closure.contact_points, strict=True)
    ):
        axis.text(*point, f"  F{finger}", color=colors[index], fontsize=10)

    visible_points = np.concatenate(
        [cloud.points, hand_points, closure.contact_points, path]
    )
    low = visible_points.min(axis=0)
    high = visible_points.max(axis=0)
    center = 0.5 * (low + high)
    radius = 0.55 * float(np.max(high - low))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title(
        "End-effector mesh grasp search\n"
        f"penetration={closure.maximum_penetration * 1000:.1f}mm, "
        f"hand Efc={closure.force_closure_residual:.3f}, "
        f"fit={'PASS' if closure.success else 'FAIL'}"
    )
    axis.legend(loc="upper right")
    figure.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        print("Close the Matplotlib window to finish.")
        plt.show()
    plt.close(figure)


def run(args) -> None:
    result = search_grasp_config(
        object_id=None if args.mesh is not None else args.object_id,
        mesh=args.mesh,
        output=args.output,
        points=args.points,
        joint_candidates=args.joint_candidates,
        seed=args.seed,
        target_size=args.target_size,
        end_effector_name=args.end_effector,
    )
    mesh_path = result.mesh_path
    output = result.output_path
    cloud = result.cloud
    closure = result.grasp
    if args.preview:
        object_colors = np.tile(
            np.asarray([[70, 150, 255, 100]], dtype=np.uint8),
            (args.points, 1),
        )
        hand_colors = np.tile(
            np.asarray([[245, 135, 80, 190]], dtype=np.uint8),
            (closure.points.shape[0], 1),
        )
        scene = trimesh.Scene(
            [
                trimesh.points.PointCloud(cloud.points, colors=object_colors),
                trimesh.points.PointCloud(closure.points, colors=hand_colors),
                *[
                    trimesh.creation.uv_sphere(radius=0.004).apply_translation(point)
                    for point in closure.contact_points
                ],
            ]
        )
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        scene.export(args.preview)
    if args.preview_image is not None or args.viewer:
        _draw_contacts(
            cloud,
            closure,
            output=args.preview_image,
            show=args.viewer,
        )
    print(
        f"mesh={mesh_path} "
        f"end_effector={args.end_effector} "
        f"penetration={closure.maximum_penetration:.4f}m "
        f"rigid_penetration={closure.maximum_noncontact_penetration:.4f}m "
        f"contact_distance={closure.mean_contact_distance:.4f}m "
        f"contacts={closure.contacting_fingers} "
        f"hand_Efc={closure.force_closure_residual:.4f} "
        f"palmward_force="
        f"{getattr(closure, 'palmward_force_component', float('nan')):.4f} "
        f"palmward_depth="
        f"{getattr(closure, 'palmward_depth', float('nan')):.4f}m "
        f"table_clearance={closure.table_clearance:.4f}m "
        f"pca_axis={closure.pca_axis_index} "
        f"robustness={closure.robustness_margin:.4f} "
        f"preload={closure.preload_weights.tolist()} "
        f"hand_fit={closure.success} output={output}"
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
