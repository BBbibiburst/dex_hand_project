"""Validate a searched grasp using only Dex Hand MJCF and the object mesh."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mujoco
from mujoco import viewer
import numpy as np

from source.grasping.standalone_validator import (
    build_standalone_model,
    set_hand_fraction_targets,
    set_hand_targets,
    set_object_pose_for_hand_pose,
    validate_standalone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grasp", type=Path, help="JSON from search_mesh_force_closure.")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-speed", type=float, default=1.0)
    parser.add_argument(
        "--approach-steps-per-waypoint",
        type=int,
        default=12,
        help="Viewer animation steps for each searched approach waypoint.",
    )
    parser.add_argument(
        "--grip-preload",
        type=float,
        default=0.25,
        help="Extra finger/thumb closure toward actuator maximum.",
    )
    return parser.parse_args()


def run(args) -> None:
    if args.viewer_speed <= 0:
        raise ValueError("--viewer-speed must be positive.")
    if args.approach_steps_per_waypoint <= 0:
        raise ValueError("--approach-steps-per-waypoint must be positive.")
    payload = json.loads(args.grasp.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
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
    waypoint_count = approach_translations.shape[0]
    if (
        waypoint_count < 2
        or approach_translations.shape != (waypoint_count, 3)
        or approach_rotations.shape != (waypoint_count, 3, 3)
        or approach_fractions.shape != (waypoint_count, 6)
    ):
        raise ValueError("Grasp config has no valid approach path.")
    model, data = build_standalone_model(
        object_mesh=payload["mesh"],
        mesh_center=payload["mesh_center"],
        mesh_scale=payload["mesh_scale"],
        hand_translation=payload["hand_translation"],
        hand_rotation_matrix=payload["hand_rotation_matrix"],
        object_table_height=payload.get("object_table_height"),
    )
    def execute(handle):
        def show(model, data, step, total) -> None:
            if handle is None or not handle.is_running():
                return
            handle.sync()
            time.sleep(model.opt.timestep / args.viewer_speed)

        if handle is not None:
            for translation, rotation, fractions in zip(
                approach_translations,
                approach_rotations,
                approach_fractions,
                strict=True,
            ):
                set_hand_fraction_targets(model, data, fractions)
                for _ in range(args.approach_steps_per_waypoint):
                    set_object_pose_for_hand_pose(
                        model,
                        data,
                        translation,
                        rotation,
                    )
                    mujoco.mj_step(model, data)
                    set_object_pose_for_hand_pose(
                        model,
                        data,
                        translation,
                        rotation,
                    )
                    show(model, data, 0, 0)
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
            f"waypoints={waypoint_count} "
            f"table_clearance="
            f"{float(payload.get('hand_table_clearance', float('nan'))):.4f}m "
            f"pca_axis={payload.get('hand_pca_axis_index')} "
            f"robustness="
            f"{float(payload.get('hand_robustness_margin', float('nan'))):.4f} "
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
