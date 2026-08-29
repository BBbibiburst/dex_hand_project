"""Record a compact MP4 from any of the five manipulation-task episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source.envs.core.registry import make_task
from source.envs.manipulation import make_manipulation_env
from source.envs.rl_env import RLEnvConfig, RobotGymEnv
from source.viz.task_demo_video import (
    SUPPORTED_TASKS,
    load_recorded_task_episode,
    record_task_episode_video,
)


def _environment_factory(episode, episode_length: int):
    if episode.env_config:
        config_payload = dict(episode.env_config)
        config_payload["episode_length"] = episode_length
        config_payload["enable_tactile_sensors"] = False
        task = make_task(episode.task, **episode.task_config)
        return RobotGymEnv(
            task=task,
            config=RLEnvConfig(**config_payload),
            render_mode=None,
        )
    return make_manipulation_env(
        episode.task,
        task_config=episode.task_config,
        control_mode="ik",
        enable_tactile_sensors=False,
        render_mode=None,
        episode_length=episode_length,
    )


def _json_object(value: str) -> dict[str, object]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("--task-config must be a JSON object.")
    return payload


def _camera(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", choices=SUPPORTED_TASKS)
    parser.add_argument("--task-config", type=_json_object, default={})
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--hold-last-seconds", type=float, default=0.75)
    parser.add_argument("--camera", type=_camera)
    parser.add_argument("--no-overlay", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episode = load_recorded_task_episode(
        args.manifest,
        task=args.task,
        task_config_override=args.task_config,
    )
    output = record_task_episode_video(
        episode,
        args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frame_stride=args.frame_stride,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        hold_last_seconds=args.hold_last_seconds,
        camera=args.camera,
        overlay=not args.no_overlay,
        environment_factory=_environment_factory,
    )
    print(f"[done] video={output} metadata={output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
