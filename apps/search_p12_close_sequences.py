"""Bounded expert-sequence search after the P12 paired takeover diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from apps.diagnose_p12_close_takeover import (
    PHASE_STEPS,
    _hash,
    _probe,
    _restore,
    _stack,
)
from apps.trace_diffusion_rollout import _make_env, _snapshot
from source.imitation.cartesian_action import XYZ9, ContinuousCartesianAdapter
from source.verification.strict_replay import load_replay_controls


def alignment_candidates(env, expert: np.ndarray, snapshot: dict[str, np.ndarray], step: int):
    """Return four fixed-budget offline diagnostic alignments."""
    arm = env.controller.arm_controller
    hand_ids = env.controller.hand_controller.actuator_ids
    window = np.arange(35, min(106, len(expert)))
    current_arm = np.asarray(snapshot["ctrl"])[arm.actuator_ids]
    current_hand = np.asarray(snapshot["ctrl"])[hand_ids]
    arm_error = np.linalg.norm(expert[window, :7] - current_arm, axis=1)
    hand_error = np.linalg.norm(expert[window, 7:] - current_hand, axis=1)
    arm_scale = max(float(np.median(arm_error)), 1e-6)
    hand_scale = max(float(np.median(hand_error)), 1e-6)
    nearest_arm = int(window[np.argmin(arm_error)])
    nearest_joint_hand = int(window[np.argmin(arm_error / arm_scale + hand_error / hand_scale)])
    return (
        ("aligned_current_orientation", step + 1, False),
        ("aligned_expert_orientation", step + 1, True),
        ("nearest_arm_current_orientation", nearest_arm, False),
        ("nearest_joint_hand_expert_orientation", nearest_joint_hand, True),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", nargs="*", type=int, default=range(5))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads((args.dataset / "manifest.json").read_text())
    source_rows = {int(row["root_id"]): row for row in dataset["rows"]}
    records = []
    for root_id in args.roots:
        source = Path(source_rows[root_id]["manifest"])
        payload = json.loads(source.read_text())
        env = _make_env(payload, 100)
        env.reset(seed=int(payload.get("metadata", {}).get("source_seed", payload["seed"])))
        expert, _, _, _ = load_replay_controls(source, env)
        expert = np.asarray(expert, dtype=np.float32)
        try:
            for phase, step in PHASE_STEPS.items():
                snapshot_path = args.cross / "snapshots" / f"root_{root_id:02d}_{phase}.npz"
                with np.load(snapshot_path) as archive:
                    snapshot = {key: np.asarray(archive[key]) for key in archive.files}
                _restore(env, snapshot)
                candidates = alignment_candidates(env, expert, snapshot, step)
                sequence_steps = 30 if phase == "close_early" else 15
                for name, expert_start, use_expert_orientation in candidates:
                    started = time.perf_counter()
                    _restore(env, snapshot)
                    adapter = ContinuousCartesianAdapter(env)
                    adapter.reset(np.asarray(snapshot["ctrl"])[env.controller.arm_controller.actuator_ids])
                    if use_expert_orientation:
                        orientation = adapter.pose_for_joint_target(expert[expert_start, :7])[1]
                    else:
                        orientation = np.asarray(snapshot["fixed_orientation_wxyz"])
                    emitted, rows = [], []
                    ik_failures = 0
                    for index in range(expert_start, min(expert_start + sequence_steps, len(expert))):
                        action = expert[index]
                        anchor = adapter.measured_pose()
                        target_position, _ = adapter.pose_for_joint_target(action[:7])
                        label = np.r_[target_position - anchor[0], action[7:]]
                        control, ik = adapter.decode(
                            label, anchor, XYZ9, fixed_orientation_wxyz=orientation
                        )
                        ik_failures += int(not ik.success)
                        control = np.clip(control, env.action_space.low, env.action_space.high)
                        env.step(control)
                        emitted.append(control)
                        rows.append(_snapshot(env, control, False))
                    prepared = {
                        "qpos": env.data.qpos.copy(),
                        "qvel": env.data.qvel.copy(),
                        "ctrl": env.data.ctrl.copy(),
                        "qacc_warmstart": env.data.qacc_warmstart.copy(),
                        "fixed_orientation_wxyz": orientation.copy(),
                    }
                    hand = env.data.ctrl[env.controller.hand_controller.actuator_ids].copy()
                    probe = _probe(env, adapter, orientation, hand)
                    destination = args.output / f"root_{root_id:02d}" / phase / name
                    destination.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(destination / "prepared_snapshot.npz", **prepared)
                    np.savez_compressed(
                        destination / "trace.npz",
                        emitted_controls=np.asarray(emitted),
                        **{f"sequence_{key}": _stack(rows, key) for key in rows[0]},
                        **{f"probe_{key}": _stack(probe["rows"], key) for key in probe["rows"][0]},
                    )
                    record = {
                        "candidate_id": hashlib.sha256(f"{root_id}:{phase}:{name}".encode()).hexdigest()[:16],
                        "root_id": root_id,
                        "phase": phase,
                        "snapshot_step": step,
                        "candidate": name,
                        "expert_start_step": expert_start,
                        "sequence_steps": len(emitted),
                        "uses_oracle_orientation": use_expert_orientation,
                        "ik_failures": ik_failures,
                        "action_hash": _hash(np.asarray(emitted)),
                        **{key: value for key, value in probe.items() if key != "rows"},
                        "runtime_seconds": time.perf_counter() - started,
                        "prepared_snapshot": str((destination / "prepared_snapshot.npz").resolve()),
                    }
                    (destination / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
                    records.append(record)
                    print(f"root={root_id} phase={phase} candidate={name} ready={record['preparation_validated']} follow={record['object_following_ratio']:.3f}", flush=True)
        finally:
            env.close()
    validated_roots = sorted({row["root_id"] for row in records if row["preparation_validated"]})
    summary = {
        "schema_version": 1,
        "attempts": len(records),
        "budget_per_root": 8,
        "preparation_validated": sum(row["preparation_validated"] for row in records),
        "validated_roots": validated_roots,
        "results": records,
        "label_note": "Offline expert/action-alignment search; oracle orientation candidates are diagnostic only.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
