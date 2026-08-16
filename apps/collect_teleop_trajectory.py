"""Collect replayable raw Lift trajectories from Vive + stretch glove.

This collector deliberately reuses the same :class:`TeleopSession` as the
LeRobot collector. Pressing R restores an exact robot/object scene snapshot,
so a failed grasp cannot permanently displace the object while editing a seed
trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from source.cli.robot_config import add_robot_config_args, make_configured_manipulation_env
from source.teleop.config import load_teleop_config
from source.teleop.session import (
    TeleopSession,
    add_teleop_session_args,
    validate_teleop_session_args,
)
from source.teleop.trajectory.contracts import TeleopTrajectoryBuffer


def build_parser() -> argparse.ArgumentParser:
    config = load_teleop_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True, help="Lift catalogue object, e.g. ycb:011_banana")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/teleop_trajectories/raw"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    add_teleop_session_args(parser, config=config, default_device="hardware")
    add_robot_config_args(parser)
    return parser


def _slug(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def _make_env(args):
    return make_configured_manipulation_env(
        args,
        "lift",
        task_config={
            "object_id": args.object_id,
            "reward_shaping": True,
            "terminate_on_success": False,
        },
        control_mode="ik",
        render_mode=None,
        control_dt=1.0 / args.fps,
        episode_length=np.iinfo(np.int32).max,
    )


def _output_path(root: Path, object_id: str, index: int) -> Path:
    directory = root / _slug(object_id)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"raw_{index:04d}.npz"
    while candidate.exists():
        index += 1
        candidate = directory / f"raw_{index:04d}.npz"
    return candidate


def _metadata(args, env, session: TeleopSession) -> dict:
    return {
        "schema_version": 1,
        "trajectory_kind": "raw",
        "collector": "apps.collect_teleop_trajectory",
        "teleop_runtime": "source.teleop.session.TeleopSession",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "lift",
        "object_id": args.object_id,
        "task_config": {
            "object_id": args.object_id,
            "reward_shaping": True,
            "terminate_on_success": False,
        },
        "env_config": asdict(env.config),
        "seed": int(args.seed),
        "control_dt": float(env.config.control_dt),
        "fps": int(args.fps),
        "camera": args.camera,
        "hand_action_size": int(env.controller.hand_controller.action_size),
        "teleop": {
            "position_scale": args.position_scale,
            "workspace_yaw_degrees": args.workspace_yaw_degrees,
            "neutral_hand_pitch_degrees": args.neutral_hand_pitch_degrees,
            "thumb_rotation": args.thumb_rotation,
            "glove_smoothing": args.glove_smoothing,
            "glove_deadzone": args.glove_deadzone,
            "glove_closed_deadzone": args.glove_closed_deadzone,
            "finger_curve_gamma": args.finger_curve_gamma,
            "ik_posture_weight": args.ik_posture_weight,
            "ik_posture_qpos": session.posture.tolist(),
            "arm_home_qpos": session.home.tolist(),
        },
    }


def run(args: argparse.Namespace) -> int:
    if args.episodes <= 0 or args.max_frames <= 0:
        raise ValueError("episodes and max-frames must be positive.")
    validate_teleop_session_args(args)

    env = _make_env(args)
    session = TeleopSession(env, args, episodes=args.episodes, frame_limit=args.max_frames)
    buffer: TeleopTrajectoryBuffer | None = None
    last_recorded_action: np.ndarray | None = None
    saved = 0

    try:
        session.connect()
        session.reset_home(args.seed)
        session.calibrate(wait_for_dashboard_confirmation=True, episode_index=0, frames=0)
        scene_snapshot = session.snapshot()

        print(
            "Controls: SPACE start/pause recording | N save raw trajectory | "
            "R discard+restore exact scene | C recalibrate | Q quit"
        )
        print(
            "R restores the exact qpos/qvel/ctrl snapshot; PAUSED holds the last "
            "recorded command once a trajectory has started."
        )
        deadline = time.monotonic()
        while saved < args.episodes and not session.ui.quit_requested:
            if not session.is_open:
                print("teleoperation dashboard closed; stopping collection")
                break
            if session.ui.consume_calibration_request():
                session.ui.recording = False
                session.calibrate(
                    wait_for_dashboard_confirmation=False,
                    episode_index=saved,
                    frames=0 if buffer is None else buffer.frame_count,
                )
            if session.ui.consume_discard_request():
                session.restore(scene_snapshot)
                session.calibrate(
                    wait_for_dashboard_confirmation=True,
                    episode_index=saved,
                    frames=0,
                )
                buffer = None
                last_recorded_action = None
                session.ui.recording = False
                print("scene restored: robot + object returned to the exact saved snapshot")
                continue
            if session.ui.consume_save_request():
                if buffer is None or buffer.frame_count == 0:
                    print("trajectory is empty; nothing saved")
                else:
                    path = _output_path(args.output_root, args.object_id, saved)
                    trajectory = buffer.build()
                    trajectory.metadata["saved_task_success"] = bool(trajectory.task_success[-1])
                    trajectory.save(path)
                    print(
                        f"saved={path} frames={trajectory.horizon} "
                        f"success={bool(np.any(trajectory.task_success))}"
                    )
                    saved += 1
                    if saved < args.episodes:
                        session.restore(scene_snapshot)
                        session.calibrate(
                            wait_for_dashboard_confirmation=True,
                            episode_index=saved,
                            frames=0,
                        )
                        buffer = None
                        last_recorded_action = None
                        session.ui.recording = False
                continue

            control = session.read_control()
            frames = 0 if buffer is None else buffer.frame_count
            if control is None:
                session.render(
                    info={},
                    state="TRACKING LOST",
                    episode_index=saved,
                    frames=frames,
                    success=False,
                    message="WAITING FOR A VALID VIVE POSE",
                )
                deadline += session.period
                time.sleep(max(0.0, deadline - time.monotonic()))
                continue

            mapped_action = control.action
            # Pausing after recording begins must not advance simulation with
            # hidden commands that will be absent from the saved replay.
            action = (
                last_recorded_action
                if buffer is not None and not session.ui.recording and last_recorded_action is not None
                else mapped_action
            )
            if session.ui.recording and buffer is None:
                buffer = TeleopTrajectoryBuffer(
                    metadata=_metadata(args, env, session),
                    initial_qpos=env.data.qpos.copy(),
                    initial_qvel=env.data.qvel.copy(),
                    initial_ctrl=env.data.ctrl.copy(),
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                )

            observation, _, _, _, info = env.step(action)
            session.observation = observation
            success = bool(info.get("task_success", False))
            if session.ui.recording and buffer is not None:
                buffer.add_frame(
                    observation=observation,
                    action=action,
                    glove=control.glove,
                    vive=control.vive,
                    success=success,
                    timestamp=time.monotonic(),
                )
                last_recorded_action = np.asarray(action, dtype=np.float32).copy()
                if buffer.frame_count >= args.max_frames:
                    session.ui.recording = False
                    print("frame limit reached; press N to save or R to discard")

            frames = 0 if buffer is None else buffer.frame_count
            session.render(
                info=info,
                state="REC" if session.ui.recording else "PAUSED",
                episode_index=saved,
                frames=frames,
                success=success,
                message="R = EXACT SCENE RESTORE",
            )
            deadline += session.period
            time.sleep(max(0.0, deadline - time.monotonic()))
        return 0
    finally:
        session.close()
        env.close()


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
