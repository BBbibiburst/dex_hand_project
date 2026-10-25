"""Bounded local Approach-Close demonstrations from pre-divergence BC states."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from apps.collect_bc_hold_snapshots import snapshot_payload
from apps.diagnose_p12_close_takeover import _hash, _probe, _restore, _stack
from apps.trace_diffusion_rollout import _make_env, _snapshot
from source.imitation.cartesian_action import XYZ9, ContinuousCartesianAdapter
from source.verification.strict_replay import load_replay_controls


STARTS = {"approach_late": 20, "preclose": 35, "close_early": 45}
SEQUENCE_END = 85
MAX_CANDIDATES_PER_ROOT = 20


@dataclass(frozen=True)
class LocalCandidate:
    name: str
    expert_xyz_weight: float
    hand_source: str
    hand_delay_steps: int = 0

    @property
    def candidate_id(self):
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def candidate_family() -> tuple[LocalCandidate, ...]:
    return (
        LocalCandidate("bc_xyz_bc_hand", 0.0, "bc"),
        LocalCandidate("expert_xyz_expert_hand", 1.0, "expert"),
        LocalCandidate("expert_xyz_hand_delay8", 1.0, "expert", 8),
        LocalCandidate("expert_xyz_hand_delay12", 1.0, "expert", 12),
        LocalCandidate("blend_xyz_hand_delay8", 0.5, "expert", 8),
    )


def select_validation_mechanisms(summary: dict, maximum: int = 3):
    rows = list(summary["results"])
    rows.sort(
        key=lambda row: (
            bool(row["preparation_validated"]),
            float(row["object_rise_m"]),
            float(row["object_following_ratio"]),
            -float(row["object_relative_translation_m"]),
        ),
        reverse=True,
    )
    selected = []
    seen = set()
    for row in rows:
        signature = (row["start_name"], row["candidate_id"])
        if signature in seen:
            continue
        selected.append((row["start_name"], LocalCandidate(**row["candidate"])))
        seen.add(signature)
        if len(selected) == maximum:
            break
    return tuple(selected)


def _restore_error(env, trace, step):
    snapshot = snapshot_payload(trace, step)
    _restore(env, snapshot)
    env.step(np.asarray(trace["joint13"][step + 1], dtype=np.float32))
    return snapshot, {
        "qpos": float(np.max(np.abs(env.data.qpos - trace["bc_qpos"][step + 1]))),
        "qvel": float(np.max(np.abs(env.data.qvel - trace["bc_qvel"][step + 1]))),
        "ctrl": float(np.max(np.abs(env.data.ctrl - trace["bc_ctrl"][step + 1]))),
    }


def _execute(env, trace, expert, snapshot, start, candidate):
    _restore(env, snapshot)
    adapter = ContinuousCartesianAdapter(env)
    adapter.reset(np.asarray(snapshot["ctrl"])[env.controller.arm_controller.actuator_ids])
    orientation = np.asarray(snapshot["fixed_orientation_wxyz"])
    controls, rows = [], []
    ik_failures = limit_violations = 0
    for index in range(start + 1, min(SEQUENCE_END + 1, len(expert), len(trace["joint13"]))):
        anchor = adapter.measured_pose()
        expert_action = np.asarray(expert[index], dtype=np.float64)
        bc_action = np.asarray(trace["joint13"][index], dtype=np.float64)
        expert_position, _ = adapter.pose_for_joint_target(expert_action[:7])
        bc_position, _ = adapter.pose_for_joint_target(bc_action[:7])
        weight = candidate.expert_xyz_weight
        position = weight * expert_position + (1.0 - weight) * bc_position
        if candidate.hand_source == "bc":
            hand = bc_action[7:]
        else:
            hand_index = max(0, index - candidate.hand_delay_steps)
            hand = np.asarray(expert[hand_index, 7:], dtype=np.float64)
        label = np.r_[position - anchor[0], hand]
        control, ik = adapter.decode(label, anchor, XYZ9, fixed_orientation_wxyz=orientation)
        ik_failures += int(not ik.success)
        limit_violations += int(ik.limit_violation)
        control = np.clip(control, env.action_space.low, env.action_space.high)
        env.step(control)
        controls.append(control)
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
    return controls, rows, prepared, probe, ik_failures, limit_violations


def _failure_reason(record):
    if record["preparation_validated"]:
        return None
    if record["ik_failures"]:
        return "IK_FAILURE"
    if record["joint_limit_violations"]:
        return "JOINT_LIMIT"
    if record["contact_fraction"] < 0.8:
        return "CONTACT_LOSS"
    if record["opposition_fraction"] < 0.8:
        return "OPPOSITION_LOSS"
    if record["object_rise_m"] < 0.003:
        return "INSUFFICIENT_OBJECT_RISE"
    return "INSUFFICIENT_FOLLOWING"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("search", "validate"), required=True)
    parser.add_argument("--bc-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-summary", type=Path)
    parser.add_argument("--roots", nargs="*", type=int)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "search":
        roots = [3] if args.roots is None else args.roots
        mechanisms = tuple((name, candidate) for name in STARTS for candidate in candidate_family())
    else:
        if args.search_summary is None:
            raise ValueError("--search-summary is required in validate mode")
        roots = [0, 1, 2, 4] if args.roots is None else args.roots
        mechanisms = select_validation_mechanisms(json.loads(args.search_summary.read_text()))
    if len(mechanisms) > MAX_CANDIDATES_PER_ROOT:
        raise ValueError("P14 per-root candidate budget exceeded")
    dataset = json.loads((args.dataset / "manifest.json").read_text())
    source_rows = {int(row["root_id"]): row for row in dataset["rows"]}
    records, restore_checks = [], []
    for root_id in roots:
        source = Path(source_rows[root_id]["manifest"])
        payload = json.loads(source.read_text())
        with np.load(args.bc_root / f"root_{root_id:02d}" / "rollout_trace.npz") as archive:
            trace = {key: np.asarray(archive[key]) for key in archive.files}
        env = _make_env(payload, 120)
        env.reset(seed=int(payload.get("metadata", {}).get("source_seed", payload["seed"])))
        expert, _, _, _ = load_replay_controls(source, env)
        try:
            snapshots = {}
            for start_name in {name for name, _ in mechanisms}:
                snapshot, errors = _restore_error(env, trace, STARTS[start_name])
                snapshots[start_name] = snapshot
                restore_checks.append({"root_id": root_id, "start_name": start_name, **errors})
            for start_name, candidate in mechanisms:
                started = time.perf_counter()
                controls, rows, prepared, probe, ik_failures, limit_violations = _execute(
                    env, trace, expert, snapshots[start_name], STARTS[start_name], candidate
                )
                unique = f"{root_id}:{start_name}:{candidate.candidate_id}"
                run_id = hashlib.sha256(unique.encode()).hexdigest()[:16]
                destination = args.output / f"root_{root_id:02d}" / run_id
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination / "prepared_snapshot.npz", **prepared)
                np.savez_compressed(
                    destination / "trace.npz",
                    emitted_controls=np.asarray(controls),
                    **{f"sequence_{key}": _stack(rows, key) for key in rows[0]},
                    **{f"probe_{key}": _stack(probe["rows"], key) for key in probe["rows"][0]},
                )
                record = {
                    "run_id": run_id,
                    "candidate_id": candidate.candidate_id,
                    "candidate": asdict(candidate),
                    "root_id": root_id,
                    "seed": int(payload.get("metadata", {}).get("source_seed", payload["seed"])),
                    "source_manifest": str(source.resolve()),
                    "start_name": start_name,
                    "start_step": STARTS[start_name],
                    "sequence_end_step": SEQUENCE_END,
                    "execution_steps": len(controls),
                    "action_hash": _hash(np.asarray(controls)),
                    "ik_failures": ik_failures,
                    "joint_limit_violations": limit_violations,
                    **{key: value for key, value in probe.items() if key != "rows"},
                    "runtime_seconds": time.perf_counter() - started,
                    "prepared_snapshot": str((destination / "prepared_snapshot.npz").resolve()),
                    "trace": str((destination / "trace.npz").resolve()),
                }
                record["failure_reason"] = _failure_reason(record)
                (destination / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
                records.append(record)
                print(f"root={root_id} start={start_name} candidate={candidate.name} ready={record['preparation_validated']} rise={1000*record['object_rise_m']:.2f} follow={record['object_following_ratio']:.3f}", flush=True)
        finally:
            env.close()
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "roots": roots,
        "attempts": len(records),
        "candidates_per_root": len(mechanisms),
        "budget_limit_per_root": MAX_CANDIDATES_PER_ROOT,
        "preparation_validated": sum(row["preparation_validated"] for row in records),
        "validated_roots": sorted({row["root_id"] for row in records if row["preparation_validated"]}),
        "restore_checks": restore_checks,
        "results": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
