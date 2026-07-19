"""Search and visualize collision-free grasps as lightweight point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from source.envs.manipulation.object_catalog import DEFAULT_LIFT_OBJECT
from source.grasping.grasp_config_search import draw, search_grasp_config


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
        help="Budget used to generate candidate end-effector shapes.",
    )
    parser.add_argument("--surface-anchors", type=int, default=24)
    parser.add_argument("--rolls-per-anchor", type=int, default=8)
    parser.add_argument("--coarse-keep", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--support-margin", type=float, default=0.008)
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
        help="Output JSON (default: configs/grasps/<end_effector>/<object_id>.json).",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Export a point-cloud 3D scene.",
    )
    parser.add_argument(
        "--preview-image",
        type=Path,
        help="Save a PNG showing object/hand point clouds, contacts, and approach path.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open an interactive Matplotlib 3D viewer after optimization.",
    )
    return parser.parse_args()


def run(args) -> None:
    for name in (
        "points",
        "joint_candidates",
        "surface_anchors",
        "rolls_per_anchor",
        "coarse_keep",
        "top_k",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.support_margin < 0.0:
        raise ValueError("--support-margin must be non-negative.")
    result = search_grasp_config(
        object_id=None if args.mesh is not None else args.object_id,
        mesh=args.mesh,
        output=args.output,
        points=args.points,
        joint_candidates=args.joint_candidates,
        surface_anchors=args.surface_anchors,
        rolls_per_anchor=args.rolls_per_anchor,
        coarse_keep=args.coarse_keep,
        top_k=args.top_k,
        support_margin=args.support_margin,
        seed=args.seed,
        target_size=args.target_size,
        end_effector_name=args.end_effector,
        require_valid=False,
        publish_invalid=args.output is not None,
    )
    mesh_path = result.mesh_path
    output = result.output_path
    cloud = result.cloud
    candidate = result.grasp
    if args.preview:
        object_colors = np.tile(
            np.asarray([[91, 135, 173, 150]], dtype=np.uint8),
            (len(cloud.points), 1),
        )
        hand_palette = np.asarray(
            [
                [239, 68, 68, 230],
                [139, 92, 246, 230],
                [6, 182, 212, 230],
                [34, 197, 94, 230],
                [234, 179, 8, 240],
                [217, 119, 6, 210],
                [107, 114, 128, 180],
            ],
            dtype=np.uint8,
        )
        display_indices = []
        body_label = 2 if len(candidate.surface.fractions) == 1 else 6
        for label in np.unique(candidate.surface.labels):
            indices = np.flatnonzero(candidate.surface.labels == label)
            point_limit = 350 if label == body_label else 650
            stride = max(1, (len(indices) + point_limit - 1) // point_limit)
            display_indices.extend(indices[::stride])
        display_indices = np.asarray(display_indices, dtype=np.int64)
        display_labels = candidate.surface.labels[display_indices]
        hand_colors = hand_palette[np.mod(display_labels, len(hand_palette))]
        scene = trimesh.Scene(
            [
                trimesh.points.PointCloud(cloud.points, colors=object_colors),
                trimesh.points.PointCloud(
                    candidate.points[display_indices],
                    colors=hand_colors,
                ),
                *[
                    trimesh.creation.uv_sphere(radius=0.004).apply_translation(point)
                    for point in candidate.contact_points
                ],
            ]
        )
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        scene.export(args.preview)
    if args.preview_image is not None or args.viewer:
        draw(
            cloud,
            candidate,
            output=args.preview_image,
            show=args.viewer,
        )
    print(
        f"mesh={mesh_path} "
        f"end_effector={args.end_effector} "
        f"penetration={candidate.penetration:.4f}m "
        f"rigid_penetration={candidate.rigid_penetration:.4f}m "
        f"contact_distance={candidate.mean_distance:.4f}m "
        f"contacts={candidate.contacts} "
        f"hand_Efc={candidate.force_closure:.4f} "
        f"gravity_E={candidate.gravity_balance_residual:.4f} "
        f"worst_disturbance_E={candidate.disturbance_residual:.4f} "
        f"normal_coverage={candidate.normal_coverage:.4f} "
        f"table_clearance={candidate.table_clearance:.4f}m "
        f"approach_clearance={candidate.approach_table_clearance:.4f}m "
        f"anchor={candidate.anchor_index} "
        f"hand_fit={candidate.valid} "
        f"output={output if result.published else '(not published)'}"
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
