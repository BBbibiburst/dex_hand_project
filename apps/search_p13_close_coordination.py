"""Feedback-guided local XYZ9 + hand6 Close-early sequence search."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from apps.diagnose_p12_close_takeover import _hash, _probe, _restore, _stack
from apps.trace_diffusion_rollout import _make_env, _snapshot
from source.imitation.cartesian_action import XYZ9, ContinuousCartesianAdapter
from source.verification.strict_replay import load_replay_controls


FINGER_CLOSURE_IDS = (0, 1, 2, 3, 5)
MAX_CANDIDATES_PER_ROOT = 20


@dataclass(frozen=True)
class CoordinationCandidate:
    name: str
    xyz_mm: tuple[float, float, float]
    closure_mm: float
    thumb_rotation_mm: float = 0.0
    ramp_steps: int = 30
    repeat: int = 0

    @property
    def candidate_id(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def tuning_candidates() -> tuple[CoordinationCandidate, ...]:
    values = [
        CoordinationCandidate(f"baseline_repeat_{repeat}", (0, 0, 0), 0, repeat=repeat)
        for repeat in range(3)
    ]
    for z_mm in (0.0, -1.0, -2.0):
        for closure_mm in (0.25, 0.5, 0.75):
            values.append(
                CoordinationCandidate(
                    f"z{z_mm:+.1f}_close{closure_mm:.2f}",
                    (0.0, 0.0, z_mm),
                    closure_mm,
                )
            )
    for axis, delta in (("x", -1.0), ("x", 1.0), ("y", -1.0), ("y", 1.0)):
        xyz = {"x": (delta, 0.0, -1.0), "y": (0.0, delta, -1.0)}[axis]
        values.append(CoordinationCandidate(f"{axis}{delta:+.1f}_close0.50", xyz, 0.5))
    values.extend(
        [
            CoordinationCandidate("fast_ramp_close0.50", (0, 0, -1), 0.5, ramp_steps=10),
            CoordinationCandidate("mid_ramp_close0.50", (0, 0, -1), 0.5, ramp_steps=20),
        ]
    )
    if len(values) > MAX_CANDIDATES_PER_ROOT:
        raise RuntimeError("P13 tuning budget exceeds the per-root limit")
    return tuple(values)


def refinement_candidates() -> tuple[CoordinationCandidate, ...]:
    """Use the final two root-3 budget slots around the best first-round point."""
    return (
        CoordinationCandidate("z-2.0_close0.10", (0, 0, -2), 0.10),
        CoordinationCandidate("z-2.0_close0.15", (0, 0, -2), 0.15),
    )


def select_validation_candidates(summary: dict, maximum: int = 3) -> tuple[CoordinationCandidate, ...]:
    rows = [row for row in summary["results"] if not row["candidate"]["name"].startswith("baseline")]
    rows.sort(
        key=lambda row: (
            bool(row["preparation_validated"]),
            float(row["object_rise_m"]),
            float(row["object_following_ratio"]),
            -float(row["object_relative_translation_m"]),
        ),
        reverse=True,
    )
    return tuple(CoordinationCandidate(**row["candidate"]) for row in rows[:maximum])


def training_eligible(records: list[dict], minimum_roots: int = 2) -> bool:
    """Require validated examples from distinct roots before dataset creation."""
    roots = {int(row["root_id"]) for row in records if row["preparation_validated"]}
    return len(roots) >= minimum_roots


def _execute_candidate(env, expert, snapshot, candidate: CoordinationCandidate):
    _restore(env, snapshot)
    adapter = ContinuousCartesianAdapter(env)
    adapter.reset(np.asarray(snapshot["ctrl"])[env.controller.arm_controller.actuator_ids])
    orientation = np.asarray(snapshot["fixed_orientation_wxyz"])
    rows, controls = [], []
    ik_failures = limit_violations = 0
    start, steps = 56, 30
    for local, expert_index in enumerate(range(start, start + steps), start=1):
        fraction = min(1.0, local / max(1, candidate.ramp_steps))
        action = np.asarray(expert[expert_index], dtype=np.float64).copy()
        anchor = adapter.measured_pose()
        position, _ = adapter.pose_for_joint_target(action[:7])
        position += fraction * np.asarray(candidate.xyz_mm, dtype=np.float64) / 1000.0
        hand = action[7:].copy()
        hand[list(FINGER_CLOSURE_IDS)] += fraction * candidate.closure_mm / 1000.0
        hand[4] += fraction * candidate.thumb_rotation_mm / 1000.0
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


def _failure_reason(record: dict) -> str | None:
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
    parser.add_argument("--mode", choices=("tune", "refine", "validate"), required=True)
    parser.add_argument("--p12", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning-summary", type=Path)
    parser.add_argument("--roots", nargs="*", type=int)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "tune":
        roots = [3] if args.roots is None else args.roots
        candidates = tuning_candidates()
    elif args.mode == "refine":
        roots = [3] if args.roots is None else args.roots
        candidates = refinement_candidates()
    else:
        if args.tuning_summary is None:
            raise ValueError("--tuning-summary is required in validate mode")
        roots = [0, 1, 2, 4] if args.roots is None else args.roots
        candidates = select_validation_candidates(json.loads(args.tuning_summary.read_text()))
    if len(candidates) > MAX_CANDIDATES_PER_ROOT:
        raise ValueError("Candidate count exceeds fixed per-root budget")
    dataset = json.loads((args.dataset / "manifest.json").read_text())
    source_rows = {int(row["root_id"]): row for row in dataset["rows"]}
    records = []
    for root_id in roots:
        source = Path(source_rows[root_id]["manifest"])
        payload = json.loads(source.read_text())
        snapshot_path = args.p12 / "cross_takeover" / "snapshots" / f"root_{root_id:02d}_close_early.npz"
        with np.load(snapshot_path) as archive:
            snapshot = {key: np.asarray(archive[key]) for key in archive.files}
        env = _make_env(payload, 100)
        env.reset(seed=int(payload.get("metadata", {}).get("source_seed", payload["seed"])))
        expert, _, _, _ = load_replay_controls(source, env)
        try:
            for candidate in candidates:
                started = time.perf_counter()
                controls, rows, prepared, probe, ik_failures, limit_violations = (
                    _execute_candidate(env, expert, snapshot, candidate)
                )
                destination = args.output / f"root_{root_id:02d}" / candidate.candidate_id
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination / "prepared_snapshot.npz", **prepared)
                np.savez_compressed(
                    destination / "trace.npz",
                    emitted_controls=np.asarray(controls),
                    **{f"sequence_{key}": _stack(rows, key) for key in rows[0]},
                    **{f"probe_{key}": _stack(probe["rows"], key) for key in probe["rows"][0]},
                )
                record = {
                    "candidate_id": candidate.candidate_id,
                    "candidate": asdict(candidate),
                    "root_id": root_id,
                    "snapshot_id": f"root_{root_id:02d}_close_early_step55",
                    "seed": int(payload.get("metadata", {}).get("source_seed", payload["seed"])),
                    "source_manifest": str(source.resolve()),
                    "action_hash": _hash(np.asarray(controls)),
                    "execution_steps": len(controls),
                    "restore_contract": "qpos+qvel+ctrl+qacc_warmstart+forward",
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
                print(f"root={root_id} candidate={candidate.name} ready={record['preparation_validated']} rise={1000*record['object_rise_m']:.2f}mm follow={record['object_following_ratio']:.3f}", flush=True)
        finally:
            env.close()
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "roots": roots,
        "candidates_per_root": len(candidates),
        "budget_limit_per_root": MAX_CANDIDATES_PER_ROOT,
        "attempts": len(records),
        "preparation_validated": sum(row["preparation_validated"] for row in records),
        "validated_roots": sorted({row["root_id"] for row in records if row["preparation_validated"]}),
        "training_eligible": training_eligible(records),
        "selected_candidates": [asdict(candidate) for candidate in candidates],
        "results": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
