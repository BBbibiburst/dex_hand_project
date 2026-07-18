"""Validate a searched grasp using only Dex Hand MJCF and the object mesh."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mujoco import viewer

from source.grasping.standalone_validator import (
    build_standalone_model,
    set_hand_targets,
    validate_standalone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grasp", type=Path, help="JSON from search_mesh_force_closure.")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-speed", type=float, default=1.0)
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
    payload = json.loads(args.grasp.read_text(encoding="utf-8"))
    if not payload.get("hand_fit_success", False):
        raise ValueError("The grasp search result is marked as unsuccessful.")
    model, data = build_standalone_model(
        object_mesh=payload["mesh"],
        mesh_center=payload["mesh_center"],
        mesh_scale=payload["mesh_scale"],
        hand_translation=payload["hand_translation"],
        hand_rotation_matrix=payload["hand_rotation_matrix"],
    )
    set_hand_targets(
        model,
        data,
        payload["hand_actuator_values"],
        grip_preload=args.grip_preload,
    )
    handle = viewer.launch_passive(model, data) if args.viewer else None

    def show(model, data, step, total) -> None:
        if handle is None:
            return
        if not handle.is_running():
            return
        handle.sync()
        time.sleep(model.opt.timestep / args.viewer_speed)

    try:
        result = validate_standalone(
            model,
            data,
            seconds=args.seconds,
            step_callback=show,
        )
    finally:
        if handle is not None:
            handle.close()
    print(
        f"stable={result.stable} seconds={result.simulated_seconds:.2f} "
        f"grip_preload={args.grip_preload:.2f} "
        f"initial_displacement={result.initial_displacement:.4f}m "
        f"post_seating_drift={result.position_drift:.4f}m "
        f"rotation_drift={result.rotation_drift:.3f}rad "
        f"vertical_drop={result.vertical_drop:.4f}m "
        f"initial_contacts={result.initial_contacts} "
        f"final_contacts={result.final_contacts}"
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
