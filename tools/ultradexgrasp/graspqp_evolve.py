"""Run the complete GraspQP -> DexEvolve -> MuJoCo grasp pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from source.envs.manipulation import make_lift_env
from source.ultradexgrasp.catalog import load_object_geometry
from source.ultradexgrasp.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from source.ultradexgrasp.dexevolve import (
    DexEvolveConfig,
    disturbance_lifetime,
    episode_fitness,
    evolve_candidates,
)
from source.ultradexgrasp.executor import STAGE_CODES, execute_grasp, rank_candidates_for_scene
from source.ultradexgrasp.graspqp_adapter import GraspQPConfig, refine_candidates_with_graspqp
from source.ultradexgrasp.hand_surrogate import load_or_calibrate_surrogate
from source.ultradexgrasp.dexevolve_mjwarp import MjWarpLifetimeConfig, MjWarpLifetimeEvaluator
from source.ultradexgrasp.synthesizer import synthesize_enclosure_priors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", default="ycb:002_master_chef_can")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graspqp-seeds", type=int, default=16)
    parser.add_argument("--graspqp-steps", type=int, default=100)
    parser.add_argument("--graspqp-executions", type=int, default=6)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--offspring", type=int, default=6)
    parser.add_argument("--generations", type=int, default=4)
    return parser


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.graspqp_seeds, args.graspqp_steps, args.graspqp_executions) <= 0:
        raise ValueError("GraspQP budgets must be positive.")
    pipeline = load_pipeline_config(args.config)
    surrogate = load_or_calibrate_surrogate(pipeline.surrogate_cache, **pipeline.surrogate_options)
    geometry = load_object_geometry(
        args.object_id,
        target_size=pipeline.target_size,
        maximum_horizontal_diameter=pipeline.maximum_horizontal_diameter,
        surface_points=pipeline.surface_points,
        seed=args.seed,
    )
    bank_size = max(args.graspqp_seeds * 8, args.graspqp_seeds)
    bank = synthesize_enclosure_priors(
        geometry,
        surrogate,
        replace(
            pipeline.synthesis,
            enclosure_prior_count=bank_size,
            contact_partition_prior_count=0,
            seed=args.seed,
        ),
    )
    selected_indices = np.linspace(0, len(bank) - 1, args.graspqp_seeds, dtype=int)
    seeds = tuple(bank[int(index)] for index in selected_indices)
    print(
        f"[graspqp] seeds={len(seeds)} steps={args.graspqp_steps} device={args.device}",
        flush=True,
    )
    candidates = refine_candidates_with_graspqp(
        geometry,
        surrogate,
        seeds,
        GraspQPConfig(steps=args.graspqp_steps, device=args.device),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _write(
        args.output / "graspqp_candidates.json",
        {
            "schema_version": 1,
            "object_id": args.object_id,
            "candidates": [item.to_dict() for item in candidates],
        },
    )
    env = make_lift_env(
        task_config={"object_id": args.object_id, "terminate_on_success": False},
        control_mode="ik",
        enable_tactile_sensors=False,
        render_mode=None,
        episode_length=pipeline.execution.maximum_steps + 20,
    )
    try:
        observation, _ = env.reset(seed=args.seed)
        ranked = rank_candidates_for_scene(
            env,
            candidates,
            observation["object_pos"],
            observation["object_quat"],
            pregrasp_distance=pipeline.execution.pregrasp_distance,
        )
        reachable = [
            item
            for item in ranked
            if item.maximum_position_error <= pipeline.execution.position_tolerance
            and item.maximum_orientation_error <= pipeline.execution.orientation_tolerance
        ]
        if not reachable:
            raise RuntimeError("GraspQP produced no RM75B-reachable candidates.")
        evaluated = []
        for rank, item in enumerate(reachable[: args.graspqp_executions]):
            episode = execute_grasp(
                item.candidate,
                seed=args.seed,
                config=pipeline.execution,
                environment=env,
            )
            fitness, metrics = episode_fitness(episode)
            evaluated.append((fitness, episode, metrics))
            episode.save(args.output / "graspqp_attempts" / f"rank_{rank:02d}")
            print(
                f"[graspqp:execute] rank={rank} lifetime={metrics['lifetime']:.3f} "
                f"success={episode.success}",
                flush=True,
            )
        _, seed_episode, seed_metrics = max(evaluated, key=lambda item: item[0])
        seed_manifest = seed_episode.save(args.output / "graspqp_seed")
        print(f"[dexevolve] seed_lifetime={seed_metrics['lifetime']:.3f}", flush=True)
        lifetime_evaluator = MjWarpLifetimeEvaluator(
            args.object_id,
            max(args.population, args.offspring),
            MjWarpLifetimeConfig(device=args.device),
        )
        evolution_config = DexEvolveConfig(
            population_size=args.population,
            offspring=args.offspring,
            generations=args.generations,
            seed=args.seed,
        )
        archive, history = evolve_candidates(
            tuple(item.candidate for item in reachable[: args.graspqp_executions]),
            environment=env,
            geometry=geometry,
            surrogate=surrogate,
            execution=replace(
                pipeline.execution,
                position_tolerance=max(pipeline.execution.position_tolerance, 0.04),
                orientation_tolerance=max(pipeline.execution.orientation_tolerance, 0.35),
            ),
            config=evolution_config,
            progress_callback=lambda metrics: print(
                f"[dexevolve] generation={int(metrics['generation'])} "
                f"fitness={metrics['best_fitness']:.3f} "
                f"lifetime={metrics['lifetime']:.3f} "
                f"success={bool(metrics['best_success'])}",
                flush=True,
            ),
            lifetime_evaluator=lifetime_evaluator,
        )
        mjwarp_metrics = lifetime_evaluator.metrics()
        for item in archive:
            gpu_lifetime = float(item.metrics.get("lifetime", 0.0))
            mujoco_lifetime = (
                0.0
                if item.episode is None
                else disturbance_lifetime(item.episode, env, evolution_config)
            )
            item.metrics["mjwarp_lifetime"] = gpu_lifetime
            item.metrics["lifetime"] = mujoco_lifetime
            item.metrics["mujoco_lifetime"] = mujoco_lifetime
            item.fitness = (
                evolution_config.lifetime_weight * mujoco_lifetime
                + evolution_config.transport_contact_weight
                * item.metrics.get("transport_contact_fraction", 0.0)
                + evolution_config.verify_contact_weight
                * item.metrics.get("verify_contact_fraction", 0.0)
                - evolution_config.distance_weight * item.metrics.get("distance_energy", 0.1)
                - evolution_config.penetration_weight * item.metrics.get("penetration_energy", 0.1)
            )
        archive = tuple(sorted(archive, key=lambda item: item.fitness, reverse=True))
        observation, _ = env.reset(seed=args.seed)
        strict_ranked = rank_candidates_for_scene(
            env,
            tuple(item.candidate for item in archive),
            observation["object_pos"],
            observation["object_quat"],
            pregrasp_distance=pipeline.execution.pregrasp_distance,
        )
        strict_reachable = {
            item.candidate.seed_index
            for item in strict_ranked
            if item.maximum_position_error <= pipeline.execution.position_tolerance
            and item.maximum_orientation_error <= pipeline.execution.orientation_tolerance
        }
        archive = tuple(item for item in archive if item.candidate.seed_index in strict_reachable)
        if not archive:
            raise RuntimeError("DexEvolve archive has no strictly RM75B-reachable survivors.")
        mujoco_lifetime = float(archive[0].metrics.get("mujoco_lifetime", 0.0))
        strict_reachable_count = len(archive)
    finally:
        if "lifetime_evaluator" in locals():
            lifetime_evaluator.close()
        env.close()
    best = archive[0]
    _write(
        args.output / "dexevolve_archive.json",
        {
            "schema_version": 1,
            "object_id": args.object_id,
            "lifetime_evaluator": mjwarp_metrics,
            "best_mujoco_lifetime": mujoco_lifetime,
            "strict_reachable_count": strict_reachable_count,
            "individuals": [
                {
                    "fitness": item.fitness,
                    "success": item.success,
                    "metrics": item.metrics,
                    "candidate": item.candidate.to_dict(),
                }
                for item in archive
            ],
        },
    )
    final_episode = None
    strict_execution_attempts = 0
    for survivor in archive:
        strict_execution_attempts += 1
        attempt = execute_grasp(
            survivor.candidate,
            seed=args.seed,
            config=pipeline.execution,
        )
        _, attempt_metrics = episode_fitness(attempt)
        opposed_contact_ok = (
            attempt_metrics["transport_contact_fraction"]
            >= evolution_config.minimum_opposed_contact_fraction
            and attempt_metrics["verify_contact_fraction"]
            >= evolution_config.minimum_opposed_contact_fraction
        )
        if np.any(attempt.arrays["stage"] == STAGE_CODES["hold"]) and opposed_contact_ok:
            best = survivor
            final_episode = attempt
            break
    if final_episode is None:
        raise RuntimeError(
            "DexEvolve archive has no strict execution with sustained thumb-opposed contact."
        )
    mujoco_lifetime = float(best.metrics.get("mujoco_lifetime", 0.0))
    evolution_manifest = None
    if best.episode is not None:
        evolution_manifest = best.episode.save(args.output / "dexevolve_best")
    final_episode.metadata["graspqp_dexevolve"] = {
        "graspqp_seed_manifest": str(seed_manifest),
        "fitness": best.fitness,
        "history": list(history),
        "lifetime_evaluator": mjwarp_metrics,
        "best_mujoco_lifetime": mujoco_lifetime,
        "strict_reachable_count": strict_reachable_count,
        "strict_execution_attempts": strict_execution_attempts,
    }
    manifest = final_episode.save(args.output)
    _write(
        args.output / "run.json",
        {
            "schema_version": 1,
            "object_id": args.object_id,
            "success": final_episode.success,
            "manifest": manifest.name,
            "evolution_manifest": (
                None
                if evolution_manifest is None
                else str(evolution_manifest.relative_to(args.output))
            ),
            "best_fitness": best.fitness,
            "history": list(history),
            "lifetime_evaluator": mjwarp_metrics,
            "best_mujoco_lifetime": mujoco_lifetime,
            "strict_reachable_count": strict_reachable_count,
            "strict_execution_attempts": strict_execution_attempts,
        },
    )
    print(
        f"[done] success={final_episode.success} fitness={best.fitness:.3f} manifest={manifest}",
        flush=True,
    )
    return 0 if final_episode.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
