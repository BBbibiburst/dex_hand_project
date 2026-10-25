"""Build root-separated Hold/Lift Cartesian supervision from verified episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from apps.trace_diffusion_rollout import _make_env, _reset_exact
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import STAGE_CODES
from source.imitation.cartesian_action import ContinuousCartesianAdapter
from source.imitation.observation import real_observable_state
from source.verification.strict_replay import load_replay_controls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--test-roots", type=int, default=2)
    return parser


def _pad_history(values: list[np.ndarray], end: int, length: int) -> np.ndarray:
    start = max(0, end - length + 1)
    selected = values[start : end + 1]
    return np.stack([selected[0]] * (length - len(selected)) + selected)


def build_episode(manifest: Path, horizon: int, history: int) -> dict[str, np.ndarray]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    episode = DemonstrationEpisode.load(manifest)
    env = _make_env(payload, len(episode.arrays["action"]) + 30)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", payload["seed"])))
    controls, qpos, qvel, ctrl = load_replay_controls(manifest, env)
    if ctrl is None:
        ctrl = env.data.ctrl.copy()
    observation = _reset_exact(env, qpos, qvel, ctrl)
    _ = observation
    adapter = ContinuousCartesianAdapter(env)
    states: list[np.ndarray] = []
    previous_xyz: list[np.ndarray] = []
    wrist: list[np.ndarray] = []
    target_position: list[np.ndarray] = []
    hand: list[np.ndarray] = []
    qposes: list[np.ndarray] = []
    qvels: list[np.ndarray] = []
    ctrls: list[np.ndarray] = []
    objects: list[np.ndarray] = []
    try:
        for control in np.asarray(controls, dtype=np.float32):
            states.append(real_observable_state(env))
            measured_position, _ = adapter.measured_pose()
            wrist.append(measured_position)
            target, _ = adapter.pose_for_joint_target(control[:7])
            target_position.append(target)
            previous_xyz.append(target - measured_position)
            hand.append(control[7:].copy())
            qposes.append(env.data.qpos.copy())
            qvels.append(env.data.qvel.copy())
            ctrls.append(env.data.ctrl.copy())
            binding = env.task._require_bindings().objects["object"]
            objects.append(env.data.xpos[binding.body_id].copy())
            env.step(control)
    finally:
        env.close()
    all_stages = np.asarray(episode.arrays["stage"], dtype=np.int16)
    # strict replay deliberately starts at Approach.  Keep its controls and
    # stage labels on the same suffix; the full episode also contains
    # settle/transit/pregrasp frames that are not returned by load controls.
    stage = all_stages[-len(controls) :]
    states_a = np.asarray(states, dtype=np.float32)
    previous_a = np.asarray(previous_xyz, dtype=np.float32)
    wrist_a = np.asarray(wrist, dtype=np.float32)
    target_a = np.asarray(target_position, dtype=np.float32)
    indices = np.flatnonzero(stage >= STAGE_CODES["hold"])
    state_history, action_history, labels, hands = [], [], [], []
    for index in indices:
        final = min(index + horizon - 1, len(stage) - 1)
        future = np.arange(index, final + 1)
        future = np.pad(future, (0, horizon - len(future)), constant_values=final)
        state_history.append(_pad_history(list(states_a), int(index), history))
        action_history.append(_pad_history(list(previous_a), int(index), history))
        labels.append(target_a[future] - wrist_a[index])
        hands.append(np.asarray(hand, dtype=np.float32)[future])
    hold_index = int(indices[0])
    return {
        "state_history": np.asarray(state_history, dtype=np.float32),
        "action_history": np.asarray(action_history, dtype=np.float32),
        "target_xyz": np.asarray(labels, dtype=np.float32),
        "expert_hand": np.asarray(hands, dtype=np.float32),
        "stage": stage[indices],
        "source_step": indices.astype(np.int32),
        "hold_qpos": np.asarray(qposes[hold_index]),
        "hold_qvel": np.asarray(qvels[hold_index]),
        "hold_ctrl": np.asarray(ctrls[hold_index]),
        "hold_object": np.asarray(objects[hold_index]),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    manifests = [Path(path) for path in audit["accepted_manifests"]]
    if len(manifests) <= args.test_roots:
        raise ValueError("Need more independent roots than held-out roots.")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for root_id, manifest in enumerate(manifests):
        arrays = build_episode(manifest, args.horizon, args.history)
        path = args.output / f"root_{root_id:02d}.npz"
        np.savez_compressed(path, **arrays)
        rows.append(
            {
                "root_id": root_id,
                "split": "test" if root_id >= len(manifests) - args.test_roots else "train",
                "manifest": str(manifest),
                "path": str(path),
                "samples": len(arrays["target_xyz"]),
                "hold_object_position": arrays["hold_object"].tolist(),
            }
        )
        print(f"root={root_id:02d} samples={rows[-1]['samples']} {rows[-1]['split']}", flush=True)
    summary = {
        "schema_version": 1,
        "state_schema": "8-step history of deployable 39D actuator position/velocity/command",
        "action_history_schema": "8-step executed XYZ target relative to measured wrist",
        "target_schema": "4 future FK target positions relative to current shared anchor",
        "privileged_inputs": [],
        "horizon": args.horizon,
        "history": args.history,
        "roots": len(rows),
        "train_roots": sum(row["split"] == "train" for row in rows),
        "test_roots": sum(row["split"] == "test" for row in rows),
        "train_samples": sum(row["samples"] for row in rows if row["split"] == "train"),
        "test_samples": sum(row["samples"] for row in rows if row["split"] == "test"),
        "rows": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
