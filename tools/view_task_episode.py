"""Replay a recorded manipulation-task episode in the interactive MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from mujoco import viewer

from source.viz.overlays import clear_markers, draw_label
from source.viz.task_demo_video import SUPPORTED_TASKS, load_recorded_task_episode
from tools.record_task_demo import _environment_factory, _json_object


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", choices=SUPPORTED_TASKS)
    parser.add_argument("--task-config", type=_json_object, default={})
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-loop", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive.")
    episode = load_recorded_task_episode(
        args.manifest,
        task=args.task,
        task_config_override=args.task_config,
    )
    arrays = episode.arrays
    env = _environment_factory(episode, len(arrays["qpos"]) + 10)
    manifest = json.loads(episode.manifest_path.read_text(encoding="utf-8"))
    env.reset(seed=int(manifest.get("seed", 0)))
    period = float(episode.env_config.get("control_dt", env.config.control_dt)) / args.speed
    try:
        with viewer.launch_passive(env.model, env.data) as handle:
            while handle.is_running():
                for frame in range(len(arrays["qpos"])):
                    if not handle.is_running():
                        break
                    started = time.monotonic()
                    env.data.qpos[:] = arrays["qpos"][frame]
                    if "qvel" in arrays and arrays["qvel"].shape[1:] == env.data.qvel.shape:
                        env.data.qvel[:] = arrays["qvel"][frame]
                    if "ctrl" in arrays and arrays["ctrl"].shape[1:] == env.data.ctrl.shape:
                        env.data.ctrl[:] = arrays["ctrl"][frame]
                    mujoco.mj_forward(env.model, env.data)
                    stage = int(arrays["stage"][frame]) if "stage" in arrays else -1
                    stage_name = episode.stage_names.get(stage, str(stage))
                    clear_markers(handle)
                    draw_label(
                        handle,
                        np.asarray([0.38, -0.28, 1.08], dtype=np.float32),
                        f"{episode.task} | frame {frame:04d} | {stage_name}",
                    )
                    handle.sync()
                    time.sleep(max(0.0, period - (time.monotonic() - started)))
                if args.no_loop:
                    while handle.is_running():
                        handle.sync()
                        time.sleep(0.02)
                    break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
