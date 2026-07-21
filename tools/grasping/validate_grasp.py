"""Validate a searched grasp using only the selected end effector and object mesh."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from mujoco import viewer
import numpy as np

from source.grasping.constants import (
    DEFAULT_GRIP_PRELOAD,
    SUPPORTED_GRASP_CONFIG_SCHEMA_VERSIONS,
)
from source.grasping.standalone_validator import (
    build_standalone_model,
    execute_configured_grasp_trajectory,
    set_hand_targets,
    set_object_pose_for_hand_pose,
    validate_standalone,
)
from source.robots.registry import get_hand
from source.grasping.grasp_config_search import search_grasp_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grasp",
        nargs="?",
        type=Path,
        help="Existing grasp JSON. Omit to search from --object-id or --mesh.",
    )
    search_source = parser.add_mutually_exclusive_group()
    search_source.add_argument("--object-id")
    search_source.add_argument("--mesh", type=Path)
    parser.add_argument("--output", type=Path, help="Output path for a newly searched grasp.")
    parser.add_argument(
        "--end-effector",
        choices=("dex_hand", "pika_gripper"),
        default="dex_hand",
    )
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--joint-candidates", type=int, default=128)
    parser.add_argument("--surface-anchors", type=int, default=24)
    parser.add_argument("--rolls-per-anchor", type=int, default=8)
    parser.add_argument("--coarse-keep", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--support-margin", type=float, default=0.008)
    parser.add_argument("--target-size", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-speed", type=float, default=0.5)
    parser.add_argument(
        "--approach-steps-per-waypoint",
        type=int,
        default=12,
        help="Viewer animation steps for each searched approach waypoint.",
    )
    parser.add_argument(
        "--grip-preload",
        type=float,
        default=DEFAULT_GRIP_PRELOAD,
        help="Extra finger/thumb closure toward actuator maximum.",
    )
    return parser.parse_args()


def _resolve_grasp_path(args: argparse.Namespace) -> Path:
    requested_search = args.object_id is not None or args.mesh is not None
    if args.grasp is not None and requested_search:
        raise ValueError("Provide either an existing grasp JSON or --object-id/--mesh.")
    if args.grasp is not None:
        return args.grasp
    if not requested_search:
        raise ValueError("Provide an existing grasp JSON or one of --object-id/--mesh.")
    result = search_grasp_config(
        object_id=args.object_id,
        mesh=args.mesh,
        output=args.output,
        points=args.points,
        joint_candidates=args.joint_candidates,
        surface_anchors=args.surface_anchors,
        rolls_per_anchor=args.rolls_per_anchor,
        coarse_keep=args.coarse_keep,
        top_k=args.top_k,
        support_margin=args.support_margin,
        target_size=args.target_size,
        seed=args.seed,
        end_effector_name=args.end_effector,
    )
    print(f"Searched grasp config with the production strategy: {result.output_path}")
    return result.output_path


def run(args) -> None:
    if args.viewer_speed <= 0:
        raise ValueError("--viewer-speed must be positive.")
    if args.approach_steps_per_waypoint <= 0:
        raise ValueError("--approach-steps-per-waypoint must be positive.")
    grasp_path = _resolve_grasp_path(args)
    payload = json.loads(grasp_path.read_text(encoding="utf-8"))
    end_effector_name = payload.get("end_effector_name", "dex_hand")
    descriptor = get_hand(end_effector_name)
    actuator_names = tuple(descriptor.position_actuator_names)
    actuator_count = len(actuator_names)
    if payload.get("schema_version") not in SUPPORTED_GRASP_CONFIG_SCHEMA_VERSIONS:
        raise ValueError("Unsupported or missing grasp schema_version.")
    if not payload.get("hand_fit_success", False):
        raise ValueError("The grasp search result is marked as unsuccessful.")
    approach_translations = np.asarray(
        payload.get("approach_hand_translations", []),
        dtype=np.float64,
    )
    approach_rotations = np.asarray(
        payload.get("approach_hand_rotation_matrices", []),
        dtype=np.float64,
    )
    approach_fractions = np.asarray(
        payload.get("approach_hand_actuator_fractions", []),
        dtype=np.float64,
    )
    grasp_translations = np.asarray(
        payload.get("grasp_hand_translations", [payload["hand_translation"]]),
        dtype=np.float64,
    )
    grasp_rotations = np.asarray(
        payload.get(
            "grasp_hand_rotation_matrices",
            [payload["hand_rotation_matrix"]],
        ),
        dtype=np.float64,
    )
    grasp_fractions = np.asarray(
        payload.get(
            "grasp_hand_actuator_fractions",
            [payload["hand_actuator_fractions"]],
        ),
        dtype=np.float64,
    )
    waypoint_count = approach_translations.shape[0]
    if (
        waypoint_count < 2
        or approach_translations.shape != (waypoint_count, 3)
        or approach_rotations.shape != (waypoint_count, 3, 3)
        or approach_fractions.shape != (waypoint_count, actuator_count)
    ):
        raise ValueError("Grasp config has no valid approach path.")
    grasp_waypoint_count = grasp_translations.shape[0]
    if (
        grasp_waypoint_count < 1
        or grasp_translations.shape != (grasp_waypoint_count, 3)
        or grasp_rotations.shape != (grasp_waypoint_count, 3, 3)
        or grasp_fractions.shape != (grasp_waypoint_count, actuator_count)
    ):
        raise ValueError("Grasp config has no valid closing trajectory.")
    model, data = build_standalone_model(
        object_mesh=payload["mesh"],
        mesh_center=payload["mesh_center"],
        mesh_scale=payload["mesh_scale"],
        hand_translation=payload["hand_translation"],
        hand_rotation_matrix=payload["hand_rotation_matrix"],
        object_table_height=payload.get("object_table_height"),
        end_effector_name=end_effector_name,
    )

    def execute(handle):
        def show(model, data, step, total) -> None:
            if handle is None or not handle.is_running():
                return
            handle.sync()
            time.sleep(model.opt.timestep / args.viewer_speed)

        execute_configured_grasp_trajectory(
            model,
            data,
            payload,
            actuator_names=actuator_names,
            steps_per_waypoint=args.approach_steps_per_waypoint,
            step_callback=show,
        )
        set_object_pose_for_hand_pose(
            model,
            data,
            payload["hand_translation"],
            payload["hand_rotation_matrix"],
        )
        set_hand_targets(
            model,
            data,
            payload["hand_actuator_values"],
            grip_preload=args.grip_preload,
            preload_weights=payload.get("hand_preload_weights"),
            preload_directions=payload.get("hand_preload_directions"),
            actuator_names=actuator_names,
        )
        result = validate_standalone(
            model,
            data,
            seconds=args.seconds,
            step_callback=show,
        )
        return result

    def print_result(result) -> None:
        print(
            f"object_id={payload.get('object_id')} "
            f"end_effector={end_effector_name} "
            f"waypoints={waypoint_count} "
            f"table_clearance="
            f"{float(payload.get('hand_table_clearance', float('nan'))):.4f}m "
            f"roll_index="
            f"{payload.get('hand_orientation_roll_index', payload.get('hand_pca_axis_index'))} "
            f"contact_margin="
            f"{float(payload.get('hand_contact_distance_margin', payload.get('hand_robustness_margin', float('nan')))):.4f} "
            f"preload={payload.get('hand_preload_weights')} "
            f"stable={result.stable} seconds={result.simulated_seconds:.2f} "
            f"grip_preload={args.grip_preload:.2f} "
            f"initial_displacement={result.initial_displacement:.4f}m "
            f"post_seating_drift={result.position_drift:.4f}m "
            f"rotation_drift={result.rotation_drift:.3f}rad "
            f"vertical_drop={result.vertical_drop:.4f}m "
            f"initial_contacts={result.initial_contacts} "
            f"final_contacts={result.final_contacts}"
        )

    if args.viewer:
        handle = viewer.launch_passive(model, data)
        result = execute(handle)
        print_result(result)
        sys.stdout.flush()
        sys.stderr.flush()
        # MuJoCo 3.10.0 can segfault while tearing down the passive GLFW
        # thread, even through its context manager.  This is a standalone CLI,
        # so let the OS reclaim native Viewer resources after flushing results.
        os._exit(0)

    result = execute(None)
    print_result(result)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
