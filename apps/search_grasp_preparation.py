"""Bounded local preparation search with an offline 5 mm load-bearing probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from apps.trace_diffusion_rollout import _make_env, _reset_exact, _snapshot
from source.imitation.cartesian_action import XYZ9, ContinuousCartesianAdapter


@dataclass(frozen=True)
class Candidate:
    name: str
    xyz_mm: tuple[float, float, float]
    hand_delta: tuple[float, float, float, float, float, float]


CANDIDATES = (
    Candidate("none", (0, 0, 0), (0, 0, 0, 0, 0, 0)),
    Candidate("x_plus_2", (2, 0, 0), (0, 0, 0, 0, 0, 0)),
    Candidate("x_minus_2", (-2, 0, 0), (0, 0, 0, 0, 0, 0)),
    Candidate("y_plus_2", (0, 2, 0), (0, 0, 0, 0, 0, 0)),
    Candidate("y_minus_2", (0, -2, 0), (0, 0, 0, 0, 0, 0)),
    Candidate("z_minus_2", (0, 0, -2), (0, 0, 0, 0, 0, 0)),
    Candidate("hand_close_05", (0, 0, 0), (0.0005,) * 6),
    Candidate("xyz_down_hand_close", (0, 0, -2), (0.0005,) * 6),
)


def preparation_validated(
    object_rise: float, following: float, contact_fraction: float, opposition_fraction: float
) -> bool:
    return bool(
        object_rise >= 0.003
        and following >= 0.5
        and contact_fraction >= 0.8
        and opposition_fraction >= 0.8
    )


def _execute_target(env, adapter, position, orientation, hand, steps):
    controls = []
    rows = []
    for _ in range(steps):
        anchor = adapter.measured_pose()
        label = np.r_[position - anchor[0], hand]
        control, result = adapter.decode(
            label, anchor, XYZ9, fixed_orientation_wxyz=orientation
        )
        control = np.clip(control, env.action_space.low, env.action_space.high)
        env.step(control)
        controls.append(control)
        rows.append(_snapshot(env, control, False))
        if not result.success:
            break
    return controls, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation-steps", type=int, default=12)
    parser.add_argument("--probe-rise-mm", type=float, default=5.0)
    parser.add_argument("--probe-steps", type=int, default=10)
    parser.add_argument("--probe-hold-steps", type=int, default=10)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.snapshots / "manifest.json").read_text())
    results = []
    for root in manifest["rows"]:
        payload = json.loads(Path(root["source_manifest"]).read_text())
        with np.load(root["snapshot"]) as archive:
            snapshot = {key: np.asarray(archive[key]) for key in archive.files}
        env = _make_env(
            payload, args.preparation_steps + args.probe_steps + args.probe_hold_steps + 20
        )
        try:
            for candidate in CANDIDATES:
                started = time.perf_counter()
                env.reset(seed=0)
                _reset_exact(env, snapshot["qpos"], snapshot["qvel"], snapshot["ctrl"])
                env.data.qacc_warmstart[:] = snapshot["qacc_warmstart"]
                mujoco.mj_forward(env.model, env.data)
                adapter = ContinuousCartesianAdapter(env)
                adapter.reset()
                initial_pose = adapter.measured_pose()
                initial_hand = env.data.ctrl[env.controller.hand_controller.actuator_ids].copy()
                prep_position = initial_pose[0] + np.asarray(candidate.xyz_mm) / 1000.0
                prep_hand = initial_hand + np.asarray(candidate.hand_delta)
                prep_controls, _ = _execute_target(
                    env,
                    adapter,
                    prep_position,
                    initial_pose[1],
                    prep_hand,
                    args.preparation_steps,
                )
                prepared = {
                    "qpos": env.data.qpos.copy(),
                    "qvel": env.data.qvel.copy(),
                    "ctrl": env.data.ctrl.copy(),
                    "qacc_warmstart": env.data.qacc_warmstart.copy(),
                    "fixed_orientation_wxyz": initial_pose[1].copy(),
                }
                binding = env.task._require_bindings().objects["object"]
                wrist_before = adapter.measured_pose()[0]
                object_before = env.data.xpos[binding.body_id].copy()
                probe_position = wrist_before + np.asarray([0, 0, args.probe_rise_mm / 1000])
                _, probe_rows = _execute_target(
                    env,
                    adapter,
                    probe_position,
                    initial_pose[1],
                    prep_hand,
                    args.probe_steps,
                )
                _, hold_rows = _execute_target(
                    env,
                    adapter,
                    probe_position,
                    initial_pose[1],
                    prep_hand,
                    args.probe_hold_steps,
                )
                rows = probe_rows + hold_rows
                wrist_after = rows[-1]["wrist"]
                object_after = rows[-1]["object"]
                wrist_rise = float(wrist_after[2] - wrist_before[2])
                object_rise = float(object_after[2] - object_before[2])
                following = object_rise / wrist_rise if wrist_rise > 1e-6 else 0.0
                contact_fraction = float(np.mean([row["contact"] for row in rows]))
                opposition_fraction = float(
                    np.mean(
                        [
                            row["digit_contact_count"][4] > 0
                            and np.sum(row["digit_contact_count"][:4] > 0) >= 2
                            for row in rows
                        ]
                    )
                )
                validated = preparation_validated(
                    object_rise, following, contact_fraction, opposition_fraction
                )
                destination = args.output / f"root_{root['root_id']:02d}" / candidate.name
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination / "prepared_snapshot.npz",
                    **prepared,
                    preparation_controls=np.asarray(prep_controls),
                )
                record = {
                    "root_id": root["root_id"],
                    "candidate": asdict(candidate),
                    "preparation_validated": validated,
                    "wrist_rise_m": wrist_rise,
                    "object_rise_m": object_rise,
                    "object_following_ratio": following,
                    "contact_fraction": contact_fraction,
                    "opposition_fraction": opposition_fraction,
                    "runtime_seconds": time.perf_counter() - started,
                    "prepared_snapshot": str(destination / "prepared_snapshot.npz"),
                }
                (destination / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
                results.append(record)
                print(
                    f"root={root['root_id']:02d} candidate={candidate.name} "
                    f"ready={validated} follow={following:.3f}",
                    flush=True,
                )
        finally:
            env.close()
    summary = {
        "config": {
            "preparation_steps": args.preparation_steps,
            "probe_rise_mm": args.probe_rise_mm,
            "probe_steps": args.probe_steps,
            "probe_hold_steps": args.probe_hold_steps,
            "candidates_per_root": len(CANDIDATES),
        },
        "roots": len(manifest["rows"]),
        "attempts": len(results),
        "preparation_validated": sum(row["preparation_validated"] for row in results),
        "results": results,
        "label_note": "Offline privileged dynamics probe; PREPARATION_VALIDATED is not Strict Lift success.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
