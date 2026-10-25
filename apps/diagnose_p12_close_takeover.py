"""Paired Close-stage oracle takeovers with the frozen P9 load-bearing probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from apps.collect_bc_hold_snapshots import snapshot_payload
from apps.search_grasp_preparation import preparation_validated
from apps.trace_diffusion_rollout import _make_env, _reset_exact, _snapshot
from source.geometry import quat_conjugate, quat_multiply, quat_to_rotvec
from source.imitation.cartesian_action import XYZ9, ContinuousCartesianAdapter
from source.verification.strict_replay import load_replay_controls


PHASE_STEPS = {"close_early": 55, "close_mid": 76}
TAKEOVER_END = 91
VARIANTS = {
    "bc_xyz_bc_hand": (False, False),
    "expert_xyz_bc_hand": (True, False),
    "bc_xyz_expert_hand": (False, True),
    "expert_xyz_expert_hand": (True, True),
}


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype=np.float32).tobytes()).hexdigest()


def _restore(env, snapshot: dict[str, np.ndarray]) -> None:
    _reset_exact(env, snapshot["qpos"], snapshot["qvel"], snapshot["ctrl"])
    env.data.qacc_warmstart[:] = snapshot["qacc_warmstart"]
    mujoco.mj_forward(env.model, env.data)


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(quat_to_rotvec(quat_multiply(second, quat_conjugate(first)))))


def _table_contact(env) -> bool:
    binding = env.task._require_bindings().objects["object"]
    table = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.task.arena.table_geom_name)
    for index in range(env.data.ncon):
        pair = {int(env.data.contact[index].geom1), int(env.data.contact[index].geom2)}
        if table in pair and pair.intersection(binding.geom_ids):
            return True
    return False


def _probe(env, adapter, orientation, hand, *, rise=0.005, steps=10, hold=10):
    binding = env.task._require_bindings().objects["object"]
    wrist_before, wrist_quat_before = adapter.measured_pose()
    object_before = env.data.xpos[binding.body_id].copy()
    object_quat_before = env.data.xquat[binding.body_id].copy()
    target = wrist_before + np.asarray([0.0, 0.0, rise])
    rows = []
    for _ in range(steps + hold):
        anchor = adapter.measured_pose()
        label = np.r_[target - anchor[0], hand]
        control, result = adapter.decode(
            label, anchor, XYZ9, fixed_orientation_wxyz=orientation
        )
        env.step(np.clip(control, env.action_space.low, env.action_space.high))
        row = _snapshot(env, control, False)
        row["object_table_contact"] = _table_contact(env)
        rows.append(row)
        if not result.success:
            break
    wrist_after = rows[-1]["wrist"]
    object_after = rows[-1]["object"]
    wrist_rise = float(wrist_after[2] - wrist_before[2])
    object_rise = float(object_after[2] - object_before[2])
    following = object_rise / wrist_rise if wrist_rise > 1e-6 else 0.0
    contact = float(np.mean([row["contact"] for row in rows]))
    opposition = float(np.mean([
        row["digit_contact_count"][4] > 0
        and np.sum(row["digit_contact_count"][:4] > 0) >= 2
        for row in rows
    ]))
    multi = float(np.mean([np.sum(row["digit_contact_count"] > 0) >= 3 for row in rows]))
    return {
        "wrist_rise_m": wrist_rise,
        "object_rise_m": object_rise,
        "object_following_ratio": following,
        "contact_fraction": contact,
        "opposition_fraction": opposition,
        "multi_digit_fraction": multi,
        "object_relative_translation_m": float(
            np.linalg.norm((object_after - wrist_after) - (object_before - wrist_before))
        ),
        "object_rotation_rad": _rotation_distance(object_quat_before, rows[-1]["object_quat_wxyz"]),
        "wrist_rotation_rad": _rotation_distance(wrist_quat_before, rows[-1]["wrist_quat_wxyz"]),
        "object_table_contact_fraction": float(np.mean([row["object_table_contact"] for row in rows])),
        "preparation_validated": preparation_validated(object_rise, following, contact, opposition),
        "rows": rows,
    }


def _stack(rows, key):
    return np.asarray([row[key] for row in rows])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", nargs="*", type=int, default=range(5))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads((args.dataset / "manifest.json").read_text())
    source_rows = {int(row["root_id"]): row for row in dataset["rows"]}
    records = []
    restore_records = []
    for root_id in args.roots:
        source = Path(source_rows[root_id]["manifest"])
        payload = json.loads(source.read_text())
        trace_path = args.bc_root / f"root_{root_id:02d}" / "rollout_trace.npz"
        with np.load(trace_path) as archive:
            trace = {key: np.asarray(archive[key]) for key in archive.files}
        env = _make_env(payload, 180)
        env.reset(seed=int(payload.get("metadata", {}).get("source_seed", payload["seed"])))
        expert, _, _, _ = load_replay_controls(source, env)
        expert = np.asarray(expert, dtype=np.float32)
        try:
            for phase, step in PHASE_STEPS.items():
                snapshot = snapshot_payload(trace, step)
                snapshot_path = args.output / "snapshots" / f"root_{root_id:02d}_{phase}.npz"
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(snapshot_path, **snapshot)
                _restore(env, snapshot)
                env.step(np.asarray(trace["joint13"][step + 1], dtype=np.float32))
                restore = {
                    "root_id": root_id,
                    "phase": phase,
                    "step": step,
                    "qpos_next_max_abs_error": float(np.max(np.abs(env.data.qpos - trace["bc_qpos"][step + 1]))),
                    "qvel_next_max_abs_error": float(np.max(np.abs(env.data.qvel - trace["bc_qvel"][step + 1]))),
                    "ctrl_next_max_abs_error": float(np.max(np.abs(env.data.ctrl - trace["bc_ctrl"][step + 1]))),
                    "snapshot": str(snapshot_path.resolve()),
                }
                restore_records.append(restore)
                for variant, (expert_xyz, expert_hand) in VARIANTS.items():
                    started = time.perf_counter()
                    _restore(env, snapshot)
                    adapter = ContinuousCartesianAdapter(env)
                    adapter.reset(np.asarray(snapshot["ctrl"])[env.controller.arm_controller.actuator_ids])
                    orientation = np.asarray(snapshot["fixed_orientation_wxyz"])
                    takeover_rows = []
                    emitted = []
                    ik_failures = 0
                    for local in range(step + 1, min(TAKEOVER_END + 1, len(trace["joint13"]))):
                        bc_action = np.asarray(trace["joint13"][local], dtype=np.float32)
                        expert_action = expert[local]
                        xyz_action = expert_action if expert_xyz else bc_action
                        hand_action = expert_action if expert_hand else bc_action
                        anchor = adapter.measured_pose()
                        target_position, _ = adapter.pose_for_joint_target(xyz_action[:7])
                        label = np.r_[target_position - anchor[0], hand_action[7:]]
                        control, ik = adapter.decode(
                            label, anchor, XYZ9, fixed_orientation_wxyz=orientation
                        )
                        ik_failures += int(not ik.success)
                        control = np.clip(control, env.action_space.low, env.action_space.high)
                        env.step(control)
                        row = _snapshot(env, control, False)
                        row["object_table_contact"] = _table_contact(env)
                        takeover_rows.append(row)
                        emitted.append(control)
                    prepared = {
                        "qpos": env.data.qpos.copy(),
                        "qvel": env.data.qvel.copy(),
                        "ctrl": env.data.ctrl.copy(),
                        "qacc_warmstart": env.data.qacc_warmstart.copy(),
                        "fixed_orientation_wxyz": orientation.copy(),
                    }
                    hand = env.data.ctrl[env.controller.hand_controller.actuator_ids].copy()
                    probe = _probe(env, adapter, orientation, hand)
                    destination = args.output / "takeovers" / f"root_{root_id:02d}" / phase / variant
                    destination.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(destination / "prepared_snapshot.npz", **prepared)
                    np.savez_compressed(
                        destination / "trace.npz",
                        emitted_controls=np.asarray(emitted),
                        **{f"takeover_{key}": _stack(takeover_rows, key) for key in takeover_rows[0]},
                        **{f"probe_{key}": _stack(probe["rows"], key) for key in probe["rows"][0]},
                    )
                    record = {
                        "root_id": root_id,
                        "source_manifest": str(source.resolve()),
                        "phase": phase,
                        "snapshot_step": step,
                        "takeover_end_step": TAKEOVER_END,
                        "variant": variant,
                        "expert_xyz": expert_xyz,
                        "expert_hand": expert_hand,
                        "takeover_steps": len(emitted),
                        "ik_failures": ik_failures,
                        "action_hash": _hash(np.asarray(emitted)),
                        **{key: value for key, value in probe.items() if key != "rows"},
                        "runtime_seconds": time.perf_counter() - started,
                        "prepared_snapshot": str((destination / "prepared_snapshot.npz").resolve()),
                    }
                    record["failure_reason"] = (
                        None if record["preparation_validated"] else
                        "IK_FAILURE" if ik_failures else
                        "NO_CONTACT" if record["contact_fraction"] < 0.8 else
                        "NO_OPPOSITION" if record["opposition_fraction"] < 0.8 else
                        "INSUFFICIENT_FOLLOWING" if record["object_following_ratio"] < 0.5 else
                        "INSUFFICIENT_OBJECT_RISE"
                    )
                    (destination / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
                    records.append(record)
                    print(f"root={root_id} phase={phase} variant={variant} ready={record['preparation_validated']} follow={record['object_following_ratio']:.3f}", flush=True)
        finally:
            env.close()
    summary = {
        "schema_version": 1,
        "phase_steps": PHASE_STEPS,
        "takeover_end_step": TAKEOVER_END,
        "probe": {"rise_m": 0.005, "steps": 10, "hold_steps": 10, "thresholds": {"object_rise_m": 0.003, "following": 0.5, "contact_fraction": 0.8, "opposition_fraction": 0.8}},
        "roots": len(args.roots),
        "attempts": len(records),
        "preparation_validated": sum(row["preparation_validated"] for row in records),
        "restore_checks": restore_records,
        "results": records,
        "label_note": "Expert actions are offline oracle diagnostics. PREPARATION_VALIDATED is not FINAL_VERIFIED.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
