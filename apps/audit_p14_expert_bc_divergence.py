"""Audit the first persistent Expert/BC Approach-Close divergence on five roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from apps.diagnose_p12_close_takeover import _restore, _rotation_distance, _table_contact
from apps.trace_diffusion_rollout import _make_env, _snapshot
from source.verification.strict_replay import load_replay_controls


THRESHOLDS = {
    "wrist_position_error_m": 0.005,
    "wrist_orientation_error_rad": float(np.deg2rad(5.0)),
    "arm_qpos_error_l2": 0.05,
    "hand_target_error_l2_m": 0.001,
    "relative_object_position_error_m": 0.005,
}
PERSISTENCE_STEPS = 5


def first_persistent(values: np.ndarray, threshold: float, steps: int = PERSISTENCE_STEPS):
    mask = np.asarray(values) > threshold
    if len(mask) < steps:
        return None
    sustained = np.convolve(mask.astype(np.int16), np.ones(steps, np.int16), mode="valid")
    indices = np.flatnonzero(sustained == steps)
    return None if not len(indices) else int(indices[0])


def first_contact_per_digit(counts: np.ndarray) -> list[int | None]:
    result = []
    for digit in range(counts.shape[1]):
        indices = np.flatnonzero(counts[:, digit] > 0)
        result.append(None if not len(indices) else int(indices[0]))
    return result


def _state_row(env, target):
    row = _snapshot(env, target, False)
    hand_ids = env.controller.hand_controller.actuator_ids
    row["hand_actual"] = env.data.actuator_length[hand_ids].copy()
    row["object_table_contact"] = _table_contact(env)
    return row


def _stack(rows, key):
    return np.asarray([row[key] for row in rows])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", nargs="*", type=int, default=range(5))
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads((args.dataset / "manifest.json").read_text())
    source_rows = {int(row["root_id"]): row for row in dataset["rows"]}
    summaries = []
    for root_id in args.roots:
        existing = args.output / f"root_{root_id:02d}" / "summary.json"
        if args.reuse_existing and existing.is_file():
            summaries.append(json.loads(existing.read_text()))
            continue
        source = Path(source_rows[root_id]["manifest"])
        payload = json.loads(source.read_text())
        with np.load(args.bc_root / f"root_{root_id:02d}" / "rollout_trace.npz") as archive:
            bc_trace = {key: np.asarray(archive[key]) for key in archive.files}
        horizon = len(bc_trace["bc_qpos"])
        env = _make_env(payload, horizon + 20)
        env.reset(seed=int(payload.get("metadata", {}).get("source_seed", payload["seed"])))
        expert, initial_qpos, initial_qvel, initial_ctrl = load_replay_controls(source, env)
        if initial_ctrl is None:
            initial_ctrl = env.data.ctrl.copy()
        expert_rows = []
        try:
            env.data.qpos[:] = initial_qpos
            env.data.qvel[:] = initial_qvel
            env.data.ctrl[:] = initial_ctrl
            mujoco.mj_forward(env.model, env.data)
            for action in np.asarray(expert)[:horizon]:
                env.step(action)
                expert_rows.append(_state_row(env, action))
            bc_rows = []
            for index in range(horizon):
                snapshot = {
                    "qpos": bc_trace["bc_qpos"][index],
                    "qvel": bc_trace["bc_qvel"][index],
                    "ctrl": bc_trace["bc_ctrl"][index],
                    "qacc_warmstart": bc_trace["bc_qacc_warmstart"][index],
                }
                _restore(env, snapshot)
                bc_rows.append(_state_row(env, bc_trace["joint13"][index]))
        finally:
            env.close()
        expert_wrist = _stack(expert_rows, "wrist")
        bc_wrist = _stack(bc_rows, "wrist")
        expert_object = _stack(expert_rows, "object")
        bc_object = _stack(bc_rows, "object")
        metrics = {
            "wrist_position_error_m": np.linalg.norm(bc_wrist - expert_wrist, axis=1),
            "wrist_orientation_error_rad": np.asarray([
                _rotation_distance(first, second)
                for first, second in zip(
                    _stack(expert_rows, "wrist_quat_wxyz"),
                    _stack(bc_rows, "wrist_quat_wxyz"),
                    strict=True,
                )
            ]),
            "arm_qpos_error_l2": np.linalg.norm(
                _stack(bc_rows, "arm_qpos") - _stack(expert_rows, "arm_qpos"), axis=1
            ),
            "arm_target_error_l2": np.linalg.norm(
                _stack(bc_rows, "arm_target") - _stack(expert_rows, "arm_target"), axis=1
            ),
            "hand_target_error_l2_m": np.linalg.norm(
                _stack(bc_rows, "hand_target") - _stack(expert_rows, "hand_target"), axis=1
            ),
            "hand_actual_error_l2_m": np.linalg.norm(
                _stack(bc_rows, "hand_actual") - _stack(expert_rows, "hand_actual"), axis=1
            ),
            "relative_object_position_error_m": np.linalg.norm(
                (bc_object - bc_wrist) - (expert_object - expert_wrist), axis=1
            ),
            "object_position_error_m": np.linalg.norm(bc_object - expert_object, axis=1),
        }
        persistent = {
            name: first_persistent(metrics[name], threshold)
            for name, threshold in THRESHOLDS.items()
        }
        expert_digits = _stack(expert_rows, "digit_contact_count")
        bc_digits = _stack(bc_rows, "digit_contact_count")
        summary = {
            "root_id": root_id,
            "seed": int(payload.get("metadata", {}).get("source_seed", payload["seed"])),
            "source_manifest": str(source.resolve()),
            "bc_trace": str((args.bc_root / f"root_{root_id:02d}" / "rollout_trace.npz").resolve()),
            "horizon": horizon,
            "thresholds": THRESHOLDS,
            "persistence_steps": PERSISTENCE_STEPS,
            "first_persistent_divergence": persistent,
            "first_expert_contact_step": int(np.flatnonzero(_stack(expert_rows, "contact"))[0]),
            "first_bc_contact_step": int(np.flatnonzero(_stack(bc_rows, "contact"))[0]),
            "expert_first_contact_per_digit": first_contact_per_digit(expert_digits),
            "bc_first_contact_per_digit": first_contact_per_digit(bc_digits),
            "expert_first_opposition_step": first_persistent(
                (expert_digits[:, 4] > 0) & (np.sum(expert_digits[:, :4] > 0, axis=1) >= 2),
                0.5,
                1,
            ),
            "bc_first_opposition_step": first_persistent(
                (bc_digits[:, 4] > 0) & (np.sum(bc_digits[:, :4] > 0, axis=1) >= 2),
                0.5,
                1,
            ),
        }
        destination = args.output / f"root_{root_id:02d}"
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "alignment.npz",
            **metrics,
            expert_wrist=expert_wrist,
            bc_wrist=bc_wrist,
            expert_object=expert_object,
            bc_object=bc_object,
            expert_arm_qpos=_stack(expert_rows, "arm_qpos"),
            bc_arm_qpos=_stack(bc_rows, "arm_qpos"),
            expert_arm_target=_stack(expert_rows, "arm_target"),
            bc_arm_target=_stack(bc_rows, "arm_target"),
            expert_hand_target=_stack(expert_rows, "hand_target"),
            bc_hand_target=_stack(bc_rows, "hand_target"),
            expert_hand_actual=_stack(expert_rows, "hand_actual"),
            bc_hand_actual=_stack(bc_rows, "hand_actual"),
            expert_digit_contact=expert_digits,
            bc_digit_contact=bc_digits,
            expert_object_table_contact=_stack(expert_rows, "object_table_contact"),
            bc_object_table_contact=_stack(bc_rows, "object_table_contact"),
        )
        (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(f"root={root_id} persistent={persistent} contact=expert:{summary['first_expert_contact_step']}/bc:{summary['first_bc_contact_step']}", flush=True)
    earliest = {}
    for name in THRESHOLDS:
        values = [
            row["first_persistent_divergence"][name]
            for row in summaries
            if row["first_persistent_divergence"][name] is not None
        ]
        earliest[name] = min(values) if values else None
    aggregate = {
        "schema_version": 1,
        "roots": len(summaries),
        "thresholds": THRESHOLDS,
        "persistence_steps": PERSISTENCE_STEPS,
        "earliest_across_roots": earliest,
        "results": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
