"""Export locally verified Hold-recovery episodes into an appendable LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from apps.trace_diffusion_rollout import _make_env
from source.data.lerobot_recorder import LeRobotEpisodeRecorder
from source.imitation.observation import real_observable_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifests = []
    for path in sorted(args.recovery_root.glob("*/manifest.json")):
        payload = json.loads(path.read_text())
        if payload.get("verification_status") == "LOCAL_RECOVERY_VERIFIED_FINAL_CRITERIA":
            manifests.append((path, payload))
    if not manifests:
        raise ValueError("No locally verified recovery episodes found.")
    source = Path(manifests[0][1]["source_manifest"])
    source_payload = json.loads(source.read_text())
    env = _make_env(source_payload, 500)
    env.reset(seed=int(source_payload.get("metadata", {}).get("source_seed", 0)))
    renderer = mujoco.Renderer(env.model, args.image_height, args.image_width)
    recorder = None
    try:
        for manifest, payload in manifests:
            artifact = Path(payload["artifact"])
            with np.load(artifact, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            if recorder is None:
                recorder = LeRobotEpisodeRecorder(
                    repo_id=args.repo_id,
                    root=args.output,
                    fps=20,
                    state_dim=39,
                    action_dim=13,
                    tactile_shape=(712,),
                    image_shape=(args.image_height, args.image_width, 3),
                    use_videos=False,
                    include_operator_metadata=False,
                )
            for index, action in enumerate(arrays["action"]):
                env.data.qpos[:] = arrays["qpos"][index]
                env.data.qvel[:] = arrays["qvel"][index]
                env.data.ctrl[:] = arrays["ctrl"][index]
                mujoco.mj_forward(env.model, env.data)
                renderer.update_scene(env.data, camera="agentview")
                observation = {
                    "qpos": arrays["qpos"][index],
                    "qvel": arrays["qvel"][index],
                    "ctrl": arrays["ctrl"][index],
                    "tactile": arrays["tactile"][index],
                }
                recorder.add_frame(
                    observation=observation,
                    image=renderer.render().copy(),
                    action=action,
                    task="lift:ycb:004_sugar_box:hold_recovery",
                    state=real_observable_state(env),
                )
            recorder.save_episode()
            print(f"SAVED recovery={payload['id']} frames={len(arrays['action'])}", flush=True)
    finally:
        if recorder is not None:
            recorder.finalize()
        renderer.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
