"""Automatically collect phase-scripted demonstrations into LeRobot format."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import mujoco
import numpy as np

from source.evaluation.grasp_schema import (
    BENCHMARK_SCHEMA_VERSION,
    TRAJECTORY_STABLE,
    VALIDATION_SEMANTICS,
)
from source.cli.robot_config import add_robot_config_args
from source.cli.robot_config import make_configured_manipulation_env
from source.cli.grasp_search import (
    add_scripted_grasp_search_args,
    scripted_grasp_search_options,
    validate_scripted_grasp_search_args,
)
from source.scripted import create_strategy, registered_strategies
from source.envs.manipulation.placement import FixedTablePlacementSampler
from source.envs.manipulation.object_catalog import lift_object_ids
from source.teleop.devices import GloveSample, ViveSample
from source.data.lerobot_recorder import LeRobotEpisodeRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=registered_strategies(), default="lift")
    parser.add_argument("--repo-id", default="local/dex-hand-scripted-demonstrations")
    parser.add_argument("--output", type=Path, default=Path("datasets/scripted_lerobot"))
    parser.add_argument("--episodes", type=int, default=20, help="Number of successful episodes.")
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--grasp-benchmark-report",
        type=Path,
        help=(
            "Evaluate trajectory-stable grasps, or collect successful randomized Lift episodes "
            "when --coverage-search is enabled."
        ),
    )
    parser.add_argument("--trials-per-object", type=int, default=10)
    parser.add_argument(
        "--coverage-search",
        action="store_true",
        help="Search seed/candidate combinations until each object has a success.",
    )
    parser.add_argument("--target-successes-per-object", type=int, default=1)
    parser.add_argument("--max-coverage-seeds", type=int, default=30)
    parser.add_argument("--max-coverage-candidates", type=int, default=16)
    parser.add_argument("--evaluation-output", type=Path)
    parser.add_argument("--resume-evaluation", action="store_true")
    parser.add_argument("--dataset", choices=("all", "ycb", "egad"), default="all")
    parser.add_argument("--object-id", action="append", dest="object_ids")
    parser.add_argument("--limit", type=int)
    add_scripted_grasp_search_args(parser)
    add_robot_config_args(parser)
    return parser.parse_args()


def _make_env(args, *, object_id: str | None = None, task_scene: dict | None = None):
    overrides = {
        "control_dt": 1.0 / args.fps,
        "episode_length": args.max_steps,
    }
    task_config = {"reward_shaping": True, "terminate_on_success": False}
    if object_id is not None:
        task_config["object_id"] = object_id
    if task_scene is not None:
        task_config["placement_sampler"] = FixedTablePlacementSampler(
            xy=tuple(task_scene["object_xy"]),
            yaw=float(task_scene["object_yaw"]),
        )
    return make_configured_manipulation_env(
        args,
        args.task,
        # The strategy owns the final hold-and-verify phase. Do not terminate
        # the environment on the first transient success sample.
        task_config=task_config,
        control_mode="ik",
        render_mode=None,
        **overrides,
    )


def _operator_samples(action: np.ndarray, env, timestamp: float):
    hand_size = env.controller.hand_controller.action_size
    hand = np.asarray(action[-hand_size:], dtype=np.float32)
    low = np.asarray(env.action_space.low[-hand_size:], dtype=np.float32)
    high = np.asarray(env.action_space.high[-hand_size:], dtype=np.float32)
    denominator = np.maximum(high - low, 1e-8)
    normalized_opening = np.clip((hand - low) / denominator, 0.0, 1.0)
    if hand_size == 1:
        normalized_opening = np.repeat(normalized_opening, 6)
    glove = GloveSample(1.0 - normalized_opening, timestamp)
    vive = ViveSample(action[:3].copy(), action[3:7].copy(), timestamp)
    return glove, vive


def _yaw_from_quaternion(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _write_evaluation_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _rotated_candidate_indices(seed_index: int, candidate_count: int) -> tuple[int, ...]:
    """Rotate candidate priority so early candidates do not dominate the dataset."""
    start = seed_index % candidate_count
    return tuple((start + offset) % candidate_count for offset in range(candidate_count))


def _successful_diversity(trials: list[dict], *, candidate_count: int) -> dict:
    """Summarize diversity among saved successful episodes."""
    successful = [trial for trial in trials if trial["success"]]
    if not successful:
        return {
            "unique_successful_seeds": 0,
            "unique_successful_candidates": 0,
            "candidate_coverage_rate": 0.0,
            "initial_position_span_xyz": [0.0, 0.0, 0.0],
            "initial_yaw_bins": 0,
        }
    positions = np.asarray([trial["initial_position"] for trial in successful], dtype=np.float64)
    yaw_bins = {
        int(np.floor((float(trial["initial_yaw"]) + np.pi) / (2.0 * np.pi) * 12.0)) % 12
        for trial in successful
    }
    candidates = {int(trial["candidate_index"]) for trial in successful}
    return {
        "unique_successful_seeds": len({int(trial["seed"]) for trial in successful}),
        "unique_successful_candidates": len(candidates),
        "candidate_coverage_rate": len(candidates) / candidate_count,
        "initial_position_span_xyz": np.ptp(positions, axis=0).tolist(),
        "initial_yaw_bins": len(yaw_bins),
    }


def _evaluation_target_met(item: dict, *, coverage_search: bool, target: int) -> bool:
    """Return whether an existing object row is complete for this run mode."""
    return not coverage_search or int(item.get("successes", 0)) >= target


def _evaluate_episode(
    env,
    strategy,
    *,
    seed: int,
    max_steps: int,
    frame_callback=None,
) -> dict:
    observation, info = env.reset(seed=seed)
    initial_position = np.asarray(observation["object_pos"], dtype=np.float64).copy()
    initial_quaternion = np.asarray(observation["object_quat"], dtype=np.float64).copy()
    strategy.reset()
    episode_return = 0.0
    success = False
    steps = 0
    error = None
    try:
        for step in range(max_steps):
            action, _ = strategy.tick(observation, info, step, env)
            observation, reward, terminated, truncated, info = env.step(action)
            if frame_callback is not None:
                frame_callback(observation, action)
            steps = step + 1
            episode_return += reward
            success = bool(strategy.state.strategy_verified_success)
            if success or strategy.finished or terminated or truncated or strategy.aborted:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "seed": seed,
        "success": success,
        "steps": steps,
        "return": episode_return,
        "final_phase": strategy.phase_name,
        "aborted": strategy.aborted,
        "initial_position": initial_position.tolist(),
        "initial_quaternion_wxyz": initial_quaternion.tolist(),
        "initial_yaw": _yaw_from_quaternion(initial_quaternion),
        "error": error,
    }


def _run_catalog_evaluation(args) -> None:
    if args.task != "lift":
        raise ValueError("--grasp-benchmark-report currently supports only --task lift.")
    if args.trials_per_object <= 0:
        raise ValueError("--trials-per-object must be positive.")
    for name in (
        "target_successes_per_object",
        "max_coverage_seeds",
        "max_coverage_candidates",
    ):
        if getattr(args, name, 1) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.coverage_search and args.target_successes_per_object > args.max_coverage_seeds:
        raise ValueError(
            "--target-successes-per-object cannot exceed --max-coverage-seeds: "
            "diversity collection saves at most one success per randomized seed."
        )
    if getattr(args, "no_tactile", False):
        raise ValueError("Randomized Lift evaluation requires tactile sensors.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    source_report = json.loads(args.grasp_benchmark_report.read_text(encoding="utf-8"))
    source_parameters = source_report.get("parameters", {})
    if (
        source_report.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or source_parameters.get("validation_semantics") != VALIDATION_SEMANTICS
    ):
        raise ValueError(
            "The grasp benchmark uses legacy or ambiguous stability semantics. "
            "Regenerate it with the current benchmark pipeline."
        )
    available = set(lift_object_ids())
    requested = None if not args.object_ids else set(args.object_ids)
    rows = []
    for row in source_report.get("objects", []):
        object_id = row.get("object_id")
        if row.get("status") != TRAJECTORY_STABLE or object_id not in available:
            continue
        if not (row.get("robot_lift") or {}).get("robot_lift_verified"):
            continue
        if args.dataset != "all" and not object_id.startswith(f"{args.dataset}:"):
            continue
        if requested is not None and object_id not in requested:
            continue
        rows.append(row)
    if requested is not None:
        missing = sorted(requested - {row["object_id"] for row in rows})
        if missing:
            raise ValueError(
                f"Requested objects have no Lift-verified benchmark task: {missing}"
            )
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No Lift-verified benchmark tasks match the selection.")

    output = args.evaluation_output or args.grasp_benchmark_report.with_name(
        f"{args.grasp_benchmark_report.stem}_lift_evaluation.json"
    )
    result = {
        "schema_version": 5,
        "source_grasp_report": str(args.grasp_benchmark_report),
        "source_validation_semantics": source_parameters["validation_semantics"],
        "parameters": {
            "task": "lift",
            "trials_per_object": args.trials_per_object,
            "coverage_search": args.coverage_search,
            "target_successes_per_object": args.target_successes_per_object,
            "max_coverage_seeds": args.max_coverage_seeds,
            "max_coverage_candidates": args.max_coverage_candidates,
            "one_success_per_seed": True,
            "candidate_order": "rotated_by_seed",
            "seed": args.seed,
            "max_steps": args.max_steps,
            "fps": args.fps,
            "dataset": args.dataset,
            "object_ids": args.object_ids,
            "limit": args.limit,
        },
        "summary": {},
        "objects": [],
    }
    if args.resume_evaluation and output.is_file():
        stored = json.loads(output.read_text(encoding="utf-8"))
        if (
            stored.get("schema_version") != result["schema_version"]
            or stored.get("source_grasp_report") != result["source_grasp_report"]
            or stored.get("parameters") != result["parameters"]
        ):
            raise ValueError(f"Cannot resume {output} with different parameters.")
        result = stored
    stored_by_id = {item["object_id"]: item for item in result["objects"]}
    retained = [
        item
        for item in result["objects"]
        if _evaluation_target_met(
            item,
            coverage_search=args.coverage_search,
            target=args.target_successes_per_object,
        )
    ]
    completed_ids = {item["object_id"] for item in retained}
    result["objects"] = retained
    total_successes = sum(item["successes"] for item in retained)
    total_trials = sum(item["trials"] for item in retained)
    recorder = None
    for object_index, row in enumerate(rows):
        object_id = row["object_id"]
        if object_id in completed_ids:
            print(f"SKIP {object_id}", flush=True)
            continue
        config_path = Path(row["config"])
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        task_scene = (row.get("robot_lift") or {}).get("task_scene")
        env = _make_env(args, object_id=object_id, task_scene=task_scene)
        renderer = None
        if args.coverage_search and not getattr(args, "dry_run", False):
            initial_observation, _ = env.reset(seed=args.seed)
            renderer = mujoco.Renderer(env.model, height=args.image_height, width=args.image_width)
            renderer.update_scene(env.data, camera=args.camera)
            first_image = renderer.render()
            if recorder is None:
                recorder = LeRobotEpisodeRecorder(
                    repo_id=args.repo_id,
                    root=args.output,
                    fps=args.fps,
                    state_dim=env.model.nq + env.model.nv + env.model.nu,
                    action_dim=env.action_space.shape[0],
                    tactile_shape=np.asarray(initial_observation["tactile"]).shape,
                    image_shape=first_image.shape,
                    use_videos=not args.no_video,
                )

        def record_frame(observation, action) -> None:
            if recorder is None or renderer is None:
                return
            renderer.update_scene(env.data, camera=args.camera)
            image = renderer.render().copy()
            glove, vive = _operator_samples(action, env, float(env.data.time))
            recorder.add_frame(
                observation=observation,
                image=image,
                action=action,
                glove=glove,
                vive=vive,
                task=f"lift:{object_id}",
            )

        candidate_count = 1
        if args.coverage_search:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            archive = config_payload.get("lift_verified_candidates")
            if not isinstance(archive, list) or not archive:
                archive = config_payload.get("trajectory_stable_candidates")
            if isinstance(archive, list) and archive:
                candidate_count = min(len(archive), args.max_coverage_candidates)
        previous = stored_by_id.get(object_id, {})
        trials = list(previous.get("episodes", []))
        attempted_seeds = {int(trial["seed"]) for trial in trials}
        if trials:
            print(
                f"RESUME {object_id} successes={sum(item['success'] for item in trials)}/"
                f"{args.target_successes_per_object} previous_trials={len(trials)}",
                flush=True,
            )
        try:
            seed_count = args.max_coverage_seeds if args.coverage_search else args.trials_per_object
            for seed_index in range(seed_count):
                seed = args.seed + object_index * seed_count + seed_index
                if seed in attempted_seeds:
                    continue
                candidate_indices = (
                    _rotated_candidate_indices(seed_index, candidate_count)
                    if args.coverage_search
                    else range(candidate_count)
                )
                for candidate_index in candidate_indices:
                    strategy = create_strategy(
                        "lift",
                        reuse_grasp_config=True,
                        grasp_config_path=config_path,
                        grasp_candidate_index=(candidate_index if args.coverage_search else None),
                        grasp_search_options=scripted_grasp_search_options(args),
                    )
                    episode_kwargs = {
                        "seed": seed,
                        "max_steps": args.max_steps,
                    }
                    if args.coverage_search:
                        episode_kwargs["frame_callback"] = record_frame
                    trial = _evaluate_episode(env, strategy, **episode_kwargs)
                    trial["candidate_index"] = candidate_index
                    trials.append(trial)
                    if recorder is not None:
                        if trial["success"]:
                            recorder.save_episode()
                        else:
                            recorder.clear_episode()
                    outcome = "SUCCESS" if trial["success"] else "FAILED"
                    print(
                        f"[{object_index + 1}/{len(rows)}] {object_id} "
                        f"attempt={len(trials)} seed={seed} candidate={candidate_index} "
                        f"{outcome} phase={trial['final_phase']}",
                        flush=True,
                    )
                    # A randomized seed defines one object pose. Saving more
                    # than one successful candidate for that seed would add
                    # near-duplicate demonstrations, so move to a new pose.
                    if args.coverage_search and trial["success"]:
                        break
                if args.coverage_search and sum(item["success"] for item in trials) >= (
                    args.target_successes_per_object
                ):
                    break
        finally:
            if renderer is not None:
                renderer.close()
            env.close()
        successes = sum(trial["success"] for trial in trials)
        total_successes += successes
        total_trials += len(trials)
        object_result = {
            "object_id": object_id,
            "grasp_config": str(config_path),
            "successes": successes,
            "trials": len(trials),
            "success_rate": successes / len(trials),
            "coverage_success": successes >= args.target_successes_per_object,
            "successful_combinations": [
                {"seed": trial["seed"], "candidate_index": trial["candidate_index"]}
                for trial in trials
                if trial["success"]
            ],
            "episodes": trials,
        }
        object_result.update(_successful_diversity(trials, candidate_count=candidate_count))
        result["objects"].append(object_result)
        result["objects"].sort(
            key=lambda item: next(
                index
                for index, source_row in enumerate(rows)
                if source_row["object_id"] == item["object_id"]
            )
        )
        object_rates = [item["success_rate"] for item in result["objects"]]
        result["summary"] = {
            "selected_objects": len(rows),
            "completed_objects": len(result["objects"]),
            "successful_episodes": total_successes,
            "total_episodes": total_trials,
            "micro_success_rate": total_successes / total_trials,
            "macro_object_success_rate": float(np.mean(object_rates)),
            "objects_with_success": sum(
                item["successes"] >= args.target_successes_per_object for item in result["objects"]
            ),
            "object_coverage_rate": sum(
                item["successes"] >= args.target_successes_per_object for item in result["objects"]
            )
            / len(result["objects"]),
        }
        _write_evaluation_report(output, result)
    if recorder is not None:
        if recorder.frame_count:
            recorder.clear_episode()
        recorder.finalize()
    print(
        f"lift_evaluation={total_successes}/{total_trials} "
        f"success_rate={total_successes / total_trials:.1%} report={output}"
    )


def run(args) -> None:
    for name in ("episodes", "max_attempts", "max_steps", "fps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    validate_scripted_grasp_search_args(args)
    if args.grasp_benchmark_report is not None:
        _run_catalog_evaluation(args)
        return
    env = _make_env(args)
    strategy = create_strategy(
        args.task,
        reuse_grasp_config=args.reuse_grasp_config,
        grasp_search_options=scripted_grasp_search_options(args),
    )
    renderer = None
    recorder = None
    successful_episodes = 0
    attempts = 0
    try:
        observation, info = env.reset(seed=args.seed)
        if not args.dry_run:
            renderer = mujoco.Renderer(env.model, height=args.image_height, width=args.image_width)
            renderer.update_scene(env.data, camera=args.camera)
            first_image = renderer.render()
            recorder = LeRobotEpisodeRecorder(
                repo_id=args.repo_id,
                root=args.output,
                fps=args.fps,
                state_dim=env.model.nq + env.model.nv + env.model.nu,
                action_dim=env.action_space.shape[0],
                tactile_shape=np.asarray(observation["tactile"]).shape,
                image_shape=first_image.shape,
                use_videos=not args.no_video,
            )
        while successful_episodes < args.episodes and attempts < args.max_attempts:
            seed = args.seed + attempts
            observation, info = env.reset(seed=seed)
            strategy.reset()
            success = False
            steps = 0
            episode_return = 0.0
            previous_phase = strategy.phase_name
            for step in range(args.max_steps):
                action, _ = strategy.tick(observation, info, step, env)
                if strategy.phase_name != previous_phase:
                    print(
                        f"phase complete: {previous_phase} -> {strategy.phase_name} "
                        f"(attempt={attempts + 1}, step={step})"
                    )
                    previous_phase = strategy.phase_name
                observation, reward, terminated, truncated, info = env.step(action)
                steps = step + 1
                episode_return += reward
                success = bool(strategy.state.strategy_verified_success)
                if recorder is not None and renderer is not None:
                    renderer.update_scene(env.data, camera=args.camera)
                    image = renderer.render().copy()
                    glove, vive = _operator_samples(action, env, float(env.data.time))
                    recorder.add_frame(
                        observation=observation,
                        image=image,
                        action=action,
                        glove=glove,
                        vive=vive,
                        task=args.task,
                    )
                if success or strategy.finished or terminated or truncated or strategy.aborted:
                    break

            attempts += 1
            should_save = success or args.save_failures
            if recorder is not None:
                if should_save:
                    recorder.save_episode()
                else:
                    recorder.clear_episode()
            if success:
                successful_episodes += 1
            outcome = "SUCCESS" if success else "DISCARDED"
            print(
                f"attempt={attempts} seed={seed} outcome={outcome} steps={steps} "
                f"return={episode_return:.3f} phase={strategy.phase_name}"
            )
    except KeyboardInterrupt:
        print("Scripted collection interrupted.")
    finally:
        if recorder is not None:
            if recorder.frame_count:
                recorder.clear_episode()
            recorder.finalize()
        if renderer is not None:
            renderer.close()
        env.close()
    print(f"collected={successful_episodes}/{args.episodes} attempts={attempts}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
