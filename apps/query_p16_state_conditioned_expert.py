"""Replan and strictly verify grasp suffixes from real frozen-BC states."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from apps.collect_bc_hold_snapshots import snapshot_payload
from apps.trace_diffusion_rollout import _make_env, _reset_exact
from source.grasping.contracts import DemonstrationEpisode
from source.grasping.executor import (
    STAGE_CODES,
    _Recorder,
    actuator_targets_from_fractions,
    candidate_world_pose,
    grasp_hand_targets,
    hand_attach_rotation,
)
from source.imitation.cartesian_action import XYZ9, ContinuousCartesianAdapter
from source.verification.profiles import FINAL_PROFILE
from source.verification.strict_replay import load_replay_controls, strict_replay_manifest


QUERY_STEP = 10
MAX_ATTEMPTS_PER_ROOT = 3


def _record_position_step(env, recorder, control, stage):
    observation, reward, terminated, truncated, info = env.step(control)
    recorder.append(
        env,
        observation,
        control,
        stage,
        reward,
        bool(info.get("task_success", False)),
    )
    return bool(terminated or truncated)


def _append_initial(recorder, env, stage="pregrasp"):
    observation = env._get_observation()
    recorder.append(env, observation, env.data.ctrl.copy(), stage, 0.0, False)


def _cartesian_segment(
    env,
    recorder,
    adapter,
    *,
    target_position,
    target_orientation,
    start_hand,
    target_hand,
    steps,
    stage,
):
    start_position, _ = adapter.measured_pose()
    ik_failures = limit_violations = 0
    for index in range(steps):
        fraction = (index + 1) / steps
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        desired = (1.0 - fraction) * start_position + fraction * target_position
        hand = (1.0 - fraction) * start_hand + fraction * target_hand
        anchor = adapter.measured_pose()
        label = np.r_[desired - anchor[0], hand]
        control, result = adapter.decode(
            label,
            anchor,
            XYZ9,
            fixed_orientation_wxyz=target_orientation,
        )
        ik_failures += int(not result.success)
        limit_violations += int(result.limit_violation)
        control = np.clip(control, env.action_space.low, env.action_space.high)
        if _record_position_step(env, recorder, control, stage):
            return ik_failures, limit_violations, True
    return ik_failures, limit_violations, False


def _run_query(env, trace, source, candidate_episode, *, query_step, variant):
    controls, initial_qpos, initial_qvel, initial_ctrl = load_replay_controls(source, env)
    if initial_ctrl is None:
        initial_ctrl = env.data.ctrl.copy()
    _reset_exact(env, initial_qpos, initial_qvel, initial_ctrl)
    recorder = _Recorder()
    _append_initial(recorder, env)

    # Preserve the actual frozen-BC prefix. This is not an Expert prefix.
    for index in range(query_step + 1):
        if _record_position_step(env, recorder, trace["joint13"][index], "approach"):
            break
    snapshot = snapshot_payload(trace, query_step)
    restore_error = {
        "qpos": float(np.max(np.abs(env.data.qpos - snapshot["qpos"]))),
        "qvel": float(np.max(np.abs(env.data.qvel - snapshot["qvel"]))),
        "ctrl": float(np.max(np.abs(env.data.ctrl - snapshot["ctrl"]))),
        "qacc_warmstart": float(
            np.max(np.abs(env.data.qacc_warmstart - snapshot["qacc_warmstart"]))
        ),
    }
    if max(restore_error.values()) > 1e-6:
        raise RuntimeError(f"BC prefix does not reproduce saved snapshot: {restore_error}")

    candidate = candidate_episode.candidate
    binding = env.task._require_bindings().objects["object"]
    object_position = env.data.xpos[binding.body_id].copy()
    object_quaternion = env.data.xquat[binding.body_id].copy()
    grasp_position, hand_rotation, grasp_orientation = candidate_world_pose(
        candidate,
        object_position,
        object_quaternion,
        hand_attach_rotation(env),
    )
    pregrasp_distance = 0.09
    pregrasp_position = grasp_position - hand_rotation @ np.asarray(
        [0.0, pregrasp_distance, 0.0]
    )
    approach_steps = 35 if variant != "slow" else 55
    close_steps = 45 if variant != "slow" else 60

    config = candidate_episode.metadata.get("execution_config", {})
    class HandConfig:
        finger_preload = float(config.get("finger_preload", 0.35))
        thumb_grasp_preload = float(config.get("thumb_grasp_preload", 0.40))

    open_fraction, grip_fraction = grasp_hand_targets(candidate.actuator_fractions, HandConfig())
    open_hand = actuator_targets_from_fractions(env, open_fraction)
    grip_hand = actuator_targets_from_fractions(env, grip_fraction)
    adapter = ContinuousCartesianAdapter(env)
    adapter.reset(env.data.ctrl[env.controller.arm_controller.actuator_ids])
    ik_failures = limit_violations = 0
    ended = False
    current_hand = env.data.ctrl[env.controller.hand_controller.actuator_ids].copy()
    phases = (
        ("pregrasp", pregrasp_position, current_hand, open_hand, 20),
        ("approach", grasp_position, open_hand, open_hand, approach_steps),
        ("close", grasp_position, open_hand, grip_hand, close_steps),
        ("hold", grasp_position, grip_hand, grip_hand, 40),
        ("lift", grasp_position + np.asarray([0.0, 0.0, 0.075]), grip_hand, grip_hand, 90),
        ("verify", grasp_position + np.asarray([0.0, 0.0, 0.075]), grip_hand, grip_hand, 60),
    )
    for stage, position, start_hand, target_hand, steps in phases:
        failed, limited, ended = _cartesian_segment(
            env,
            recorder,
            adapter,
            target_position=position,
            target_orientation=grasp_orientation,
            start_hand=start_hand,
            target_hand=target_hand,
            steps=steps,
            stage=stage,
        )
        ik_failures += failed
        limit_violations += limited
        if ended:
            break
    arrays = recorder.arrays()
    initial_object = arrays["object_position"][0]
    final_lift = float(arrays["object_position"][-1, 2] - initial_object[2])
    metadata = {
        "control_dt": 0.05,
        "position_actuator_names": list(env.controller.actuator_names),
        "action_contract": "arm7_absolute_joint_target_plus_hand6",
        "state_conditioned_query": True,
        "query_step": query_step,
        "query_variant": variant,
        "bc_source_manifest": str(source.resolve()),
        "candidate_source_manifest": str(candidate_episode.metadata.get("source_manifest", "")),
        "object_initial_position": initial_object.tolist(),
        "object_initial_quaternion_wxyz": arrays["object_quaternion_wxyz"][0].tolist(),
        "query_object_position": object_position.tolist(),
        "query_object_quaternion_wxyz": object_quaternion.tolist(),
        "replanned_grasp_position": grasp_position.tolist(),
        "replanned_grasp_orientation_wxyz": grasp_orientation.tolist(),
        "restore_error": restore_error,
        "ik_failures": ik_failures,
        "joint_limit_violations": limit_violations,
        "object_lift": final_lift,
        "stage_codes": STAGE_CODES,
    }
    episode = DemonstrationEpisode(
        object_id=candidate.object_id,
        seed=int(json.loads(source.read_text())["seed"]),
        candidate=candidate,
        arrays=arrays,
        success=bool(final_lift >= 0.04),
        terminal_stage="verify" if not ended else "environment_end",
        failure_reason=None if final_lift >= 0.04 else "object did not reach diagnostic lift",
        metadata=metadata,
    )
    return episode, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-root", type=Path, required=True)
    parser.add_argument("--p14-summary", type=Path, required=True)
    parser.add_argument("--candidate-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", nargs="*", type=int, default=range(5))
    parser.add_argument("--query-step", type=int, default=QUERY_STEP)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if len(args.candidate_manifest) != 2:
        raise ValueError("Provide exactly two verified candidate manifests.")
    if not 5 <= args.query_step <= 20:
        raise ValueError("Query step must remain in the preregistered Approach [5, 20] window.")
    p14 = json.loads(args.p14_summary.read_text())
    sources = {int(row["root_id"]): Path(row["source_manifest"]) for row in p14["results"]}
    candidate_episodes = [DemonstrationEpisode.load(path) for path in args.candidate_manifest]
    attempts = []
    for root_id in args.roots:
        source = sources[root_id]
        payload = json.loads(source.read_text())
        trace_path = args.bc_root / f"root_{root_id:02d}" / "rollout_trace.npz"
        with np.load(trace_path) as archive:
            trace = {key: np.asarray(archive[key]) for key in archive.files}
        plans = (
            ("xminus", candidate_episodes[0], "normal"),
            ("pitch", candidate_episodes[1], "normal"),
            ("xminus_slow", candidate_episodes[0], "slow"),
        )
        if len(plans) > MAX_ATTEMPTS_PER_ROOT:
            raise RuntimeError("P16 per-root query budget exceeded.")
        env = _make_env(payload, 420)
        try:
            for name, candidate_episode, variant in plans:
                started = time.perf_counter()
                destination = args.output / f"root_{root_id:02d}" / name
                episode, metadata = _run_query(
                    env,
                    trace,
                    source,
                    candidate_episode,
                    query_step=args.query_step,
                    variant=variant,
                )
                manifest = episode.save(destination)
                strict = strict_replay_manifest(manifest, profile=FINAL_PROFILE, use_cache=False)
                row = {
                    "root_id": root_id,
                    "query_step": args.query_step,
                    "plan": name,
                    "source_manifest": str(source.resolve()),
                    "manifest": str(manifest.resolve()),
                    "state_conditioned_replan": True,
                    "strict_success": bool(strict.success),
                    "strict": asdict(strict),
                    "ik_failures": metadata["ik_failures"],
                    "joint_limit_violations": metadata["joint_limit_violations"],
                    "restore_error": metadata["restore_error"],
                    "runtime_seconds": time.perf_counter() - started,
                }
                attempts.append(row)
                (destination / "query_result.json").write_text(
                    json.dumps(row, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"root={root_id} plan={name} strict={strict.success} "
                    f"lift={1000.0 * strict.final_lift:.2f}mm",
                    flush=True,
                )
        finally:
            env.close()
    summary = {
        "schema_version": 1,
        "query_step": args.query_step,
        "attempts": len(attempts),
        "strict_successes": sum(row["strict_success"] for row in attempts),
        "successful_roots": sorted({row["root_id"] for row in attempts if row["strict_success"]}),
        "failure_statuses": {
            status: sum(
                row["strict"]["verification_status"] == status
                for row in attempts
                if not row["strict_success"]
            )
            for status in sorted(
                {row["strict"]["verification_status"] for row in attempts if not row["strict_success"]}
            )
        },
        "results": attempts,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if summary["strict_successes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
