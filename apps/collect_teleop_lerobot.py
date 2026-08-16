"""Collect Vive + stretch-glove demonstrations into a LeRobot dataset.

The default mock devices make the full control loop testable before hardware
drivers are implemented. Use ``--dry-run`` to skip the LeRobot dependency.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from source.cli.robot_config import add_robot_config_args, make_configured_manipulation_env
from source.data.lerobot_recorder import LeRobotEpisodeRecorder
from source.envs.manipulation import registered_tasks
from source.teleop.config import load_teleop_config
from source.teleop.session import (
    TeleopSession,
    add_teleop_session_args,
    validate_teleop_session_args,
)


def parse_args():
    teleop_config = load_teleop_config()
    parser = argparse.ArgumentParser(description="Collect teleoperated LeRobot demonstrations.")
    parser.add_argument("--task", choices=registered_tasks(), default="lift")
    parser.add_argument("--repo-id", default="local/dex-hand-demonstrations")
    parser.add_argument("--output", type=Path, default=Path("datasets/lerobot"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-frames", type=int, default=300)
    add_teleop_session_args(parser, config=teleop_config, default_device="sine")
    parser.add_argument("--dry-run", action="store_true", help="Control and render without writing data.")
    parser.add_argument(
        "--no-video", action="store_true", help="Store images instead of encoded MP4."
    )
    add_robot_config_args(parser)
    return parser.parse_args()


def _make_env(args):
    overrides = {
        "control_dt": 1.0 / args.fps,
        # Interactive episodes are ended explicitly with N / R / Q. Keep the
        # Gymnasium time limit from expiring while the operator is calibrating,
        # positioning, or recording a long demonstration.
        "episode_length": np.iinfo(np.int32).max,
    }
    return make_configured_manipulation_env(
        args,
        args.task,
        task_config={"reward_shaping": True},
        control_mode="ik",
        render_mode=None,
        **overrides,
    )


def run(args) -> None:
    if args.episodes <= 0 or args.episode_frames <= 0:
        raise ValueError("episodes and episode-frames must be positive.")
    validate_teleop_session_args(args)

    env = _make_env(args)
    session = TeleopSession(
        env,
        args,
        episodes=args.episodes,
        frame_limit=args.episode_frames,
    )
    recorder = None
    episode = 0
    episode_frames = 0
    success = False

    def reset_episode(seed: int) -> None:
        nonlocal episode_frames, success
        session.reset_home(seed)
        episode_frames = 0
        success = False
        session.ui.recording = False
        session.calibrate(
            wait_for_dashboard_confirmation=True,
            episode_index=episode,
            frames=0,
        )

    try:
        session.connect()
        reset_episode(0)
        first_image = session.camera_image()
        if not args.dry_run:
            recorder = LeRobotEpisodeRecorder(
                repo_id=args.repo_id,
                root=args.output,
                fps=args.fps,
                state_dim=env.model.nq + env.model.nv + env.model.nu,
                action_dim=env.action_space.shape[0],
                tactile_shape=np.asarray(session.observation["tactile"]).shape,
                image_shape=first_image.shape,
                use_videos=not args.no_video,
            )

        print(
            "Controls: SPACE record/pause | N save episode | R discard/reset | "
            "C recalibrate | Q quit"
        )
        deadline = time.monotonic()
        while episode < args.episodes and not session.ui.quit_requested:
            if not session.is_open:
                print("teleoperation dashboard closed; stopping collection")
                break
            if session.ui.consume_calibration_request():
                session.ui.recording = False
                session.calibrate(
                    wait_for_dashboard_confirmation=False,
                    episode_index=episode,
                    frames=episode_frames,
                )
            if session.ui.consume_discard_request():
                if recorder is not None:
                    recorder.clear_episode()
                reset_episode(episode)
                print(f"episode={episode} discarded and reset")
                continue
            if session.ui.consume_save_request():
                if episode_frames == 0:
                    print("episode is empty; nothing saved")
                else:
                    if recorder is not None:
                        recorder.save_episode()
                    print(f"episode={episode} saved success={success} frames={episode_frames}")
                    episode += 1
                    if episode < args.episodes:
                        reset_episode(episode)
                    continue

            control = session.read_control()
            if control is None:
                session.render(
                    info={},
                    state="TRACKING LOST",
                    episode_index=episode,
                    frames=episode_frames,
                    success=success,
                    message="WAITING FOR A VALID VIVE POSE",
                )
                deadline += session.period
                time.sleep(max(0.0, deadline - time.monotonic()))
                continue

            observation, _, terminated, truncated, info = env.step(control.action)
            session.observation = observation
            success = bool(info.get("task_success", False))
            image = session.render(
                info=info,
                state="REC" if session.ui.recording else "PAUSED",
                episode_index=episode,
                frames=episode_frames,
                success=success,
            )
            if session.ui.recording and episode_frames < args.episode_frames:
                if recorder is not None:
                    recorder.add_frame(
                        observation=observation,
                        image=image,
                        action=control.action,
                        glove=control.glove,
                        vive=control.vive,
                        task=args.task,
                    )
                episode_frames += 1
                if episode_frames == args.episode_frames:
                    session.ui.recording = False
                    print("frame limit reached; press N to save or R to discard")
            if terminated or truncated:
                session.ui.recording = False
                print("environment ended; press N to save or R to discard")
            deadline += session.period
            time.sleep(max(0.0, deadline - time.monotonic()))
    finally:
        if recorder is not None:
            if recorder.frame_count:
                print(f"discarding {recorder.frame_count} unsaved frames")
                recorder.clear_episode()
            recorder.finalize()
        session.close()
        env.close()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
