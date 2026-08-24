"""Replay verified grasp/Lattice/PPO trajectories into a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from source.data.lerobot_recorder import LeRobotEpisodeRecorder
from source.envs.manipulation import make_lift_env
from source.verification.profiles import FINAL_PROFILE
from source.verification.strict_replay import load_replay_controls, strict_replay_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        action="append",
        type=Path,
        dest="trajectories",
        help="Verified grasp/Lattice/PPO trajectory directory or manifest; repeat as needed.",
    )
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        dest="input_roots",
        help="Recursively discover best_trajectory/best_attempt manifests under this directory.",
    )
    parser.add_argument("--repo-id", default="local/dex-hand-grasp-demonstrations")
    parser.add_argument("--output", type=Path, default=Path("datasets/grasp_lerobot"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Export trajectories that fail final C MuJoCo verification (diagnostics only).",
    )
    return parser


def _manifest(path: Path) -> Path:
    candidate = path / "manifest.json" if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(f"Trajectory manifest does not exist: {candidate}")
    return candidate.resolve()


def discover_trajectory_manifests(
    trajectories: list[Path] | None,
    input_roots: list[Path] | None,
) -> tuple[Path, ...]:
    candidates = [_manifest(path) for path in trajectories or []]
    for root in input_roots or []:
        if not root.is_dir():
            raise FileNotFoundError(f"Input root does not exist: {root}")
        for pattern in ("**/best_trajectory/manifest.json", "**/best_attempt/manifest.json"):
            candidates.extend(path.resolve() for path in root.glob(pattern))
    return tuple(dict.fromkeys(sorted(candidates)))


def _export_one(
    manifest: Path,
    *,
    args: argparse.Namespace,
    recorder: LeRobotEpisodeRecorder | None,
) -> LeRobotEpisodeRecorder | None:
    verification = strict_replay_manifest(manifest, profile=FINAL_PROFILE, use_cache=True)
    if not verification.success and not args.allow_unverified:
        print(f"SKIP unverified={manifest} status={verification.verification_status}", flush=True)
        return recorder

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    object_id = str(payload["object_id"])
    metadata = dict(payload.get("metadata", {}))
    control_dt = float(metadata.get("control_dt", 0.05))
    source_seed = int(metadata.get("source_seed", payload.get("seed", 0)))
    expected_fps = int(round(1.0 / control_dt))
    if expected_fps != args.fps:
        raise ValueError(
            f"Trajectory {manifest} uses control_dt={control_dt}, equivalent to {expected_fps} "
            f"FPS; requested dataset FPS is {args.fps}."
        )

    env = make_lift_env(
        task_config={
            "object_id": object_id,
            "reward_shaping": False,
            "terminate_on_success": False,
        },
        control_mode="position",
        enable_tactile_sensors=True,
        render_mode=None,
        control_dt=control_dt,
        episode_length=int(payload.get("frames", 300)) + 20,
    )
    renderer = None
    try:
        observation, _ = env.reset(seed=source_seed)
        controls, initial_qpos, initial_qvel, initial_ctrl = load_replay_controls(manifest, env)
        env.data.qpos[:] = initial_qpos
        env.data.qvel[:] = initial_qvel
        if initial_ctrl is not None and np.asarray(initial_ctrl).shape == env.data.ctrl.shape:
            env.data.ctrl[:] = initial_ctrl
        mujoco.mj_forward(env.model, env.data)

        if args.dry_run:
            print(
                f"VALID object={object_id} frames={len(controls)} manifest={manifest}",
                flush=True,
            )
            return recorder

        renderer = mujoco.Renderer(env.model, height=args.image_height, width=args.image_width)
        renderer.update_scene(env.data, camera=args.camera)
        first_image = renderer.render()
        if recorder is None:
            recorder = LeRobotEpisodeRecorder(
                repo_id=args.repo_id,
                root=args.output,
                fps=args.fps,
                state_dim=env.model.nq + env.model.nv + env.model.nu,
                action_dim=env.action_space.shape[0],
                tactile_shape=np.asarray(observation["tactile"]).shape,
                image_shape=first_image.shape,
                use_videos=not args.no_video,
                include_operator_metadata=False,
            )

        saved_frames = 0
        for control in np.asarray(controls, dtype=np.float32):
            observation, _, terminated, truncated, _ = env.step(control)
            renderer.update_scene(env.data, camera=args.camera)
            recorder.add_frame(
                observation=observation,
                image=renderer.render().copy(),
                action=control,
                task=f"lift:{object_id}",
            )
            saved_frames += 1
            if terminated or truncated:
                break
        recorder.save_episode()
        print(f"SAVED object={object_id} frames={saved_frames} manifest={manifest}", flush=True)
        return recorder
    except Exception:
        if recorder is not None:
            recorder.clear_episode()
        raise
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def run(args: argparse.Namespace) -> int:
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    manifests = discover_trajectory_manifests(args.trajectories, args.input_roots)
    if not manifests:
        raise ValueError("Supply at least one --trajectory or --input-root containing results.")
    recorder = None
    try:
        for manifest in manifests:
            recorder = _export_one(manifest, args=args, recorder=recorder)
    finally:
        if recorder is not None:
            if recorder.frame_count:
                recorder.clear_episode()
            recorder.finalize()
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
