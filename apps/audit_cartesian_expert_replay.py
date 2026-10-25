"""Convert one Joint13 expert to Cartesian contracts and strictly replay it."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from apps.trace_diffusion_rollout import _make_env, _reset_exact
from source.grasp_pipeline.trajectory import GraspTrajectory
from source.imitation.cartesian_action import (
    JOINT13,
    POSE12,
    XYZ9,
    CartesianActionContract,
    ContinuousCartesianAdapter,
)
from source.verification.profiles import FINAL_PROFILE
from source.verification.strict_replay import load_replay_controls, strict_replay_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    return parser


def _orientation_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0)))


def _converted_controls(env, controls, initial, mode, fixed_orientation):
    initial_qpos, initial_qvel, initial_ctrl = initial
    _reset_exact(env, initial_qpos, initial_qvel, initial_ctrl)
    adapter = ContinuousCartesianAdapter(env)
    adapter.reset()
    fixed_quat = adapter.measured_pose()[1] if fixed_orientation else None
    rebuilt, labels, diagnostics = [], [], []
    wrist_positions, wrist_quats = [], []
    for step, expert in enumerate(controls):
        anchor = adapter.measured_pose()
        label = adapter.encode(expert, anchor, mode)
        control, result = adapter.decode(label, anchor, mode, fixed_orientation_wxyz=fixed_quat)
        observation, _, terminated, truncated, _ = env.step(control)
        _ = observation
        current_position, current_quat = adapter.measured_pose()
        target_position, target_quat = adapter.pose_for_joint_target(expert[:7])
        diagnostics.append(
            {
                "step": step,
                **asdict(result),
                "joint_rmse": float(np.sqrt(np.mean((control[:7] - expert[:7]) ** 2))),
                "fk_position_error": float(np.linalg.norm(current_position - target_position)),
                "fk_orientation_error": _orientation_error(current_quat, target_quat),
            }
        )
        rebuilt.append(control)
        labels.append(label)
        wrist_positions.append(current_position)
        wrist_quats.append(current_quat)
        if terminated or truncated:
            break
    return (
        np.asarray(rebuilt, np.float32),
        np.asarray(labels, np.float32),
        diagnostics,
        np.asarray(wrist_positions),
        np.asarray(wrist_quats),
    )


def _save_trajectory(output, payload, source, controls, initial):
    initial_qpos, initial_qvel, _ = initial
    trajectory = GraspTrajectory(
        object_id=str(payload["object_id"]),
        source_manifest=str(source.resolve()),
        start_stage="approach",
        action_mode="absolute_position",
        residual_actions=np.zeros((len(controls), 1), dtype=np.float32),
        controls=controls,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        success=False,
        episode_return=0.0,
        metadata={
            "control_dt": float(payload.get("metadata", {}).get("control_dt", 0.05)),
            "source_seed": int(
                payload.get("metadata", {}).get("source_seed", payload.get("seed", 0))
            ),
            "p6_cartesian_conversion": True,
        },
    )
    return trajectory.save(output)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    env = _make_env(payload, int(payload.get("frames", 500)) + 30)
    env.reset(seed=int(payload.get("metadata", {}).get("source_seed", 0)))
    controls, qpos, qvel, ctrl = load_replay_controls(args.manifest, env)
    controls = np.asarray(controls, np.float32)
    if args.max_steps:
        controls = controls[: args.max_steps]
    if ctrl is None:
        ctrl = env.data.ctrl.copy()
    initial = (qpos, qvel, ctrl)
    summary = {
        "source_manifest": str(args.manifest),
        "source_frames": len(controls),
        "control_dt": float(payload.get("metadata", {}).get("control_dt", 0.05)),
        "contracts": {},
        "online_seed": "previous commanded joint target; current measured qpos at reset",
        "oracle_future_seed": False,
    }
    trajectories = {}
    try:
        source_verification = strict_replay_manifest(
            args.manifest, profile=FINAL_PROFILE, use_cache=False
        )
        summary["source_strict"] = asdict(source_verification)
        for mode, fixed in ((XYZ9, False), (XYZ9, True), (POSE12, False)):
            key = (
                f"{mode}_{'fixed_initial_orientation' if fixed else 'measured_anchor_orientation'}"
            )
            started = time.perf_counter()
            rebuilt, labels, rows, positions, quats = _converted_controls(
                env, controls, initial, mode, fixed
            )
            directory = args.output / key
            directory.mkdir(parents=True, exist_ok=True)
            manifest = _save_trajectory(
                directory / "trajectory", payload, args.manifest, rebuilt, initial
            )
            verification = strict_replay_manifest(manifest, profile=FINAL_PROFILE, use_cache=False)
            numeric = {
                name: np.asarray([row[name] for row in rows], dtype=np.float64)
                for name in (
                    "position_error",
                    "orientation_error",
                    "joint_delta",
                    "minimum_singular_value",
                    "joint_rmse",
                    "fk_position_error",
                    "fk_orientation_error",
                )
            }
            np.savez_compressed(
                directory / "roundtrip_trace.npz",
                original_joint13=controls[: len(rebuilt)],
                cartesian_labels=labels,
                reconstructed_joint13=rebuilt,
                wrist_position=positions,
                wrist_quat_wxyz=quats,
                **numeric,
            )
            result = {
                "contract": asdict(CartesianActionContract(mode)),
                "orientation_reference": "fixed initial measured"
                if fixed
                else "per-step measured anchor",
                "ik_success_count": int(sum(bool(row["success"]) for row in rows)),
                "ik_failure_count": int(sum(not bool(row["success"]) for row in rows)),
                "joint_limit_violation_count": int(
                    sum(bool(row["limit_violation"]) for row in rows)
                ),
                "position_error_max_m": float(numeric["position_error"].max()),
                "orientation_error_max_rad": float(numeric["orientation_error"].max()),
                "fk_execution_position_rmse_m": float(
                    np.sqrt(np.mean(numeric["fk_position_error"] ** 2))
                ),
                "fk_execution_orientation_rmse_rad": float(
                    np.sqrt(np.mean(numeric["fk_orientation_error"] ** 2))
                ),
                "joint_target_rmse_rad": float(
                    np.sqrt(np.mean((rebuilt[:, :7] - controls[: len(rebuilt), :7]) ** 2))
                ),
                "maximum_joint_step_rad": float(np.max(np.abs(np.diff(rebuilt[:, :7], axis=0)))),
                "runtime_seconds": time.perf_counter() - started,
                "strict_replay": asdict(verification),
                "trajectory_manifest": str(manifest),
            }
            (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            summary["contracts"][key] = result
            trajectories[key] = (positions, quats)
    finally:
        env.close()

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for key, (positions, quats) in trajectories.items():
        axes[0].plot(positions[:, 2], label=key)
        axes[1].plot([_orientation_error(quats[0], quat) for quat in quats], label=key)
    axes[0].set_ylabel("Wrist world z (m)")
    axes[1].set_ylabel("Orientation change (rad)")
    axes[1].set_xlabel("Control step")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.output / "expert_cartesian_replay_comparison.png", dpi=180)
    plt.close(figure)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "action_contracts.json").write_text(
        json.dumps(
            {name: asdict(CartesianActionContract(name)) for name in (JOINT13, XYZ9, POSE12)},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
