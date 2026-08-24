"""DexEvolve-style refinement adapted to the six-drive underactuated hand."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from source.ultradexgrasp.catalog import ObjectGeometry
from source.ultradexgrasp.contracts import DemonstrationEpisode, GraspCandidate
from source.ultradexgrasp.dexevolve_contacts import depenetrate_pose, resample_contact_command
from source.ultradexgrasp.executor import (
    STAGE_CODES,
    ExecutionConfig,
    _robot_object_contact_summary,
    execute_grasp,
)
from source.ultradexgrasp.hand_surrogate import DexHandSurrogate


@dataclass(frozen=True)
class DexEvolveConfig:
    population_size: int = 32
    offspring: int = 16
    generations: int = 32
    tournament_size: int = 4
    mutation_probability: float = 0.75
    translation_sigma: float = 0.025
    orientation_sigma: float = 0.05
    actuator_sigma: float = 0.04
    crossover_probability: float = 0.2
    novelty_threshold: float = 0.1
    density_radius: float = 0.65
    density_power: float = 2.0
    maximum_archive: int = 1024
    contact_threshold: float = 0.012
    active_contacts: int = 12
    desired_contact_travel: float = 0.010
    depenetration_steps: int = 2
    # Convex-hull penetration is deliberately conservative for concave hand
    # skins; 25 mm corresponds to roughly 6 mm under the mesh-SDF metric used
    # by the Isaac implementation, based on the can calibration scene.
    maximum_penetration: float = 0.025
    lifetime_weight: float = 10.0
    transport_contact_weight: float = 10.0
    verify_contact_weight: float = 30.0
    distance_weight: float = 40.0
    penetration_weight: float = 100.0
    disturbance_steps: int = 80
    lateral_disturbance_force: float = 3.0
    upward_force_ratio: float = 2.0
    maximum_relative_drift: float = 0.035
    minimum_opposed_contact_fraction: float = 0.5
    seed: int = 0

    def validate(self):
        if (
            min(
                self.population_size,
                self.offspring,
                self.generations,
                self.tournament_size,
                self.maximum_archive,
                self.active_contacts,
            )
            <= 0
        ):
            raise ValueError("DexEvolve population settings must be positive.")
        if not 0 <= self.mutation_probability <= 1 or not 0 <= self.crossover_probability <= 1:
            raise ValueError("Genetic probabilities must lie in [0, 1].")


@dataclass
class EvolvedIndividual:
    candidate: GraspCandidate
    fitness: float = -np.inf
    episode: DemonstrationEpisode | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def success(self):
        return bool(self.episode is not None and self.episode.success)


def candidate_embedding(c):
    return np.concatenate(
        [
            10 * c.hand_translation,
            Rotation.from_matrix(c.hand_rotation_matrix).as_euler("xyz"),
            c.actuator_fractions,
        ]
    )


def candidate_distance(a, b):
    d = candidate_embedding(a) - candidate_embedding(b)
    return 0.5 * (np.sqrt(np.mean(d[:6] ** 2)) + np.sqrt(np.mean(d[6:] ** 2)))


def mutate_candidate(candidate, rng, config, *, seed_index):
    if rng.random() >= config.mutation_probability:
        return GraspCandidate(**{**candidate.__dict__, "seed_index": seed_index})
    e = Rotation.from_matrix(candidate.hand_rotation_matrix).as_euler("xyz")
    return GraspCandidate(
        **{
            **candidate.__dict__,
            "seed_index": seed_index,
            "hand_translation": candidate.hand_translation
            + rng.normal(0, config.translation_sigma, 3),
            "hand_rotation_matrix": Rotation.from_euler(
                "xyz", e + rng.normal(0, config.orientation_sigma, 3)
            ).as_matrix(),
            "actuator_fractions": np.clip(
                candidate.actuator_fractions + rng.normal(0, config.actuator_sigma, 6), 0, 1
            ),
            "metrics": {**candidate.metrics, "dexevolve_mutation": 1.0},
            "backend": f"{candidate.backend}+dexevolve",
        }
    )


def crossover_candidates(first, second, rng, *, seed_index):
    pose, hand = (first, second) if rng.integers(2) else (second, first)
    return GraspCandidate(
        **{
            **pose.__dict__,
            "seed_index": seed_index,
            "actuator_fractions": hand.actuator_fractions,
            "metrics": {**pose.metrics, "dexevolve_crossover": 1.0},
            "backend": f"{pose.backend}+dexevolve",
        }
    )


def _materialize(c, geometry, surrogate, config):
    translation = depenetrate_pose(
        geometry,
        surrogate,
        translation=c.hand_translation,
        rotation=c.hand_rotation_matrix,
        fractions=c.actuator_fractions,
        steps=config.depenetration_steps,
    )
    command = resample_contact_command(
        geometry,
        surrogate,
        translation=translation,
        rotation=c.hand_rotation_matrix,
        contact_fractions=c.actuator_fractions,
        contact_threshold=config.contact_threshold,
        active_contacts=config.active_contacts,
        desired_travel=config.desired_contact_travel,
    )
    metrics = {
        **c.metrics,
        "dexevolve_distance_energy": command.distance_energy,
        "dexevolve_penetration_energy": command.penetration_energy,
    }
    for i, v in enumerate(command.contact_fractions):
        metrics[f"dexevolve_q_{i}"] = float(v)
    for i, v in enumerate(command.command_delta):
        metrics[f"dexevolve_dq_{i}"] = float(v)
    return GraspCandidate(
        **{
            **c.__dict__,
            "hand_translation": translation,
            "actuator_fractions": command.grip_fractions,
            "contact_points": command.contact_points,
            "contact_normals": command.contact_normals,
            "contact_distances": command.contact_distances,
            "metrics": metrics,
            "backend": f"{c.backend}+adaptive-command",
        }
    )


def episode_fitness(episode):
    stage = np.asarray(episode.arrays["stage"])
    contacts = np.asarray(episode.arrays["robot_object_digit_contact_count"])
    mask = stage >= STAGE_CODES["hold"]
    alive = mask & (contacts[:, 4] > 0) & (contacts[:, :4].max(axis=1) > 0)
    run = best = 0
    for value in alive:
        run = run + 1 if value else 0
        best = max(best, run)
    lifetime = best / max(int(mask.sum()), 1)
    transport = stage >= STAGE_CODES["lift"]
    transport_opposed = (contacts[:, 4] > 0) & (contacts[:, :4].max(axis=1) > 0) & transport
    transport_contact = float(transport_opposed.sum() / max(int(transport.sum()), 1))
    verify = stage == STAGE_CODES["verify"]
    verify_opposed = (contacts[:, 4] > 0) & (contacts[:, :4].max(axis=1) > 0) & verify
    verify_contact = float(verify_opposed.sum() / max(int(verify.sum()), 1))
    return lifetime, {
        "lifetime": lifetime,
        "transport_contact_fraction": transport_contact,
        "verify_contact_fraction": verify_contact,
        "distance_energy": float(episode.candidate.metrics.get("dexevolve_distance_energy", 0.1)),
        "penetration_energy": float(
            episode.candidate.metrics.get("dexevolve_penetration_energy", 0.1)
        ),
    }


def disturbance_lifetime(episode, environment, config):
    """Replay the closed state and test survival under five force windows."""
    stages = np.asarray(episode.arrays["stage"])
    hold = np.flatnonzero(stages == STAGE_CODES["hold"])
    if not len(hold):
        return 0.0
    frame = int(hold[-1])
    env = environment
    env.data.qpos[:] = episode.arrays["qpos"][frame]
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = episode.arrays["ctrl"][frame]
    env.data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    binding = env.task.bindings.objects["object"]
    body = binding.body_id
    site = env.controller.arm_controller.site_id
    initial_relative = env.data.xpos[body] - env.data.site_xpos[site]
    mass = float(env.model.body_mass[body])
    lateral = config.lateral_disturbance_force
    forces = (
        np.asarray([0.0, 0.0, config.upward_force_ratio * mass * 9.81]),
        np.asarray([lateral, 0.0, config.upward_force_ratio * mass * 9.81]),
        np.asarray([-lateral, 0.0, config.upward_force_ratio * mass * 9.81]),
        np.asarray([0.0, lateral, config.upward_force_ratio * mass * 9.81]),
        np.asarray([0.0, -lateral, config.upward_force_ratio * mass * 9.81]),
    )
    survived = 0
    for force in forces:
        for _ in range(config.disturbance_steps):
            env.data.xfrc_applied[body, :3] = force
            mujoco.mj_step(env.model, env.data)
        _, _, digit_counts, _ = _robot_object_contact_summary(env)
        relative = env.data.xpos[body] - env.data.site_xpos[site]
        opposed = digit_counts[4] > 0 and digit_counts[:4].max() > 0
        if (
            not opposed
            or np.linalg.norm(relative - initial_relative) > config.maximum_relative_drift
        ):
            break
        survived += 1
    env.data.xfrc_applied[body] = 0.0
    return survived / len(forces)


def _evaluate(c, environment, execution, geometry, surrogate, config):
    try:
        c = _materialize(c, geometry, surrogate, config)
        pen = c.metrics["dexevolve_penetration_energy"]
        if pen > config.maximum_penetration:
            return EvolvedIndividual(
                c,
                -config.penetration_weight * pen,
                None,
                {"lifetime": 0.0, "distance_energy": 0.0, "penetration_energy": pen},
            )
        episode = execute_grasp(c, seed=config.seed, config=execution, environment=environment)
        _, m = episode_fitness(episode)
        m["lifetime"] = disturbance_lifetime(episode, environment, config)
        score = (
            config.lifetime_weight * m["lifetime"]
            + config.transport_contact_weight * m["transport_contact_fraction"]
            + config.verify_contact_weight * m["verify_contact_fraction"]
            - config.distance_weight * m["distance_energy"]
            - config.penetration_weight * m["penetration_energy"]
        )
        return EvolvedIndividual(c, score, episode, m)
    except Exception as exc:
        return EvolvedIndividual(
            c, -1e6, None, {"evaluation_error": 1.0, "error_hash": float(hash(str(exc)) & 0xFFFF)}
        )


def _evaluate_batch(candidates, environment, execution, geometry, surrogate, config, evaluator):
    """Prepare candidates serially, then evaluate all valid snapshots in one GPU batch."""
    results = []
    pending_indices = []
    pending_episodes = []
    for candidate in candidates:
        try:
            executable = _materialize(candidate, geometry, surrogate, config)
            pen = executable.metrics["dexevolve_penetration_energy"]
            if pen > config.maximum_penetration:
                results.append(
                    EvolvedIndividual(
                        executable,
                        -config.penetration_weight * pen,
                        None,
                        {"lifetime": 0.0, "distance_energy": 0.0, "penetration_energy": pen},
                    )
                )
                continue
            episode = execute_grasp(
                executable, seed=config.seed, config=execution, environment=environment
            )
            # Only a completed close/hold state can enter the disturbance batch.
            stages = np.asarray(episode.arrays["stage"])
            if not np.any(stages == STAGE_CODES["hold"]):
                results.append(
                    EvolvedIndividual(
                        executable,
                        -1e3,
                        episode,
                        {
                            "lifetime": 0.0,
                            "transport_contact_fraction": 0.0,
                            "verify_contact_fraction": 0.0,
                            "distance_energy": executable.metrics["dexevolve_distance_energy"],
                            "penetration_energy": pen,
                        },
                    )
                )
                continue
            pending_indices.append(len(results))
            pending_episodes.append(episode)
            results.append(None)
        except Exception as exc:
            results.append(
                EvolvedIndividual(
                    candidate,
                    -1e6,
                    None,
                    {"evaluation_error": 1.0, "error_hash": float(hash(str(exc)) & 0xFFFF)},
                )
            )
    if pending_episodes:
        lifetimes = evaluator.evaluate(pending_episodes)
        for index, episode, lifetime in zip(pending_indices, pending_episodes, lifetimes):
            distance = float(episode.candidate.metrics["dexevolve_distance_energy"])
            penetration = float(episode.candidate.metrics["dexevolve_penetration_energy"])
            _, episode_metrics = episode_fitness(episode)
            transport = episode_metrics["transport_contact_fraction"]
            verify_contact = episode_metrics["verify_contact_fraction"]
            metrics = {
                "lifetime": float(lifetime),
                "transport_contact_fraction": transport,
                "verify_contact_fraction": verify_contact,
                "distance_energy": distance,
                "penetration_energy": penetration,
            }
            score = (
                config.lifetime_weight * float(lifetime)
                + config.transport_contact_weight * transport
                + config.verify_contact_weight * verify_contact
                - config.distance_weight * distance
                - config.penetration_weight * penetration
            )
            results[index] = EvolvedIndividual(episode.candidate, score, episode, metrics)
    return results


def _density(archive, config):
    scores = []
    for a in archive:
        d = np.array([candidate_distance(a.candidate, b.candidate) for b in archive])
        near = d < config.density_radius
        rho = np.sum(1 - (d[near] / config.density_radius) ** config.density_power)
        scores.append(a.fitness / max(float(rho), 1.0))
    return np.asarray(scores)


def _insert(archive, item, config):
    if archive:
        d = np.array([candidate_distance(item.candidate, x.candidate) for x in archive])
        i = int(np.argmin(d))
        if d[i] < config.novelty_threshold:
            if item.fitness > archive[i].fitness:
                archive[i] = item
            return
    archive.append(item)
    if len(archive) > config.maximum_archive:
        archive.sort(key=lambda x: x.fitness, reverse=True)
        del archive[int(0.75 * config.maximum_archive) :]


def evolve_candidates(
    seed_candidates: Sequence[GraspCandidate],
    *,
    environment,
    geometry: ObjectGeometry,
    surrogate: DexHandSurrogate,
    execution=None,
    config=None,
    progress_callback: Callable | None = None,
    lifetime_evaluator=None,
):
    config = config or DexEvolveConfig()
    execution = execution or ExecutionConfig()
    config.validate()
    if not seed_candidates:
        raise ValueError("DexEvolve requires analytical seeds.")
    rng = np.random.default_rng(config.seed)
    next_id = 3_000_000
    population = list(seed_candidates[: config.population_size])
    while len(population) < config.population_size:
        population.append(
            mutate_candidate(
                seed_candidates[len(population) % len(seed_candidates)],
                rng,
                config,
                seed_index=next_id,
            )
        )
        next_id += 1
    archive = (
        _evaluate_batch(
            population, environment, execution, geometry, surrogate, config, lifetime_evaluator
        )
        if lifetime_evaluator is not None
        else [_evaluate(c, environment, execution, geometry, surrogate, config) for c in population]
    )
    history = []
    for generation in range(config.generations):
        density = _density(archive, config)
        children = []
        for _ in range(config.offspring):
            ids = rng.choice(len(archive), min(config.tournament_size, len(archive)), replace=False)
            child = archive[int(ids[np.argmax(density[ids])])].candidate
            if len(archive) > 1 and rng.random() < config.crossover_probability:
                child = crossover_candidates(
                    child,
                    archive[int(rng.integers(len(archive)))].candidate,
                    rng,
                    seed_index=next_id,
                )
                next_id += 1
            child = mutate_candidate(child, rng, config, seed_index=next_id)
            next_id += 1
            children.append(child)
        evaluated = (
            _evaluate_batch(
                children, environment, execution, geometry, surrogate, config, lifetime_evaluator
            )
            if lifetime_evaluator is not None
            else [
                _evaluate(child, environment, execution, geometry, surrogate, config)
                for child in children
            ]
        )
        for item in evaluated:
            _insert(archive, item, config)
        best = max(archive, key=lambda x: x.fitness)
        m = {
            "generation": float(generation + 1),
            "archive_size": float(len(archive)),
            "best_fitness": float(best.fitness),
            "best_success": float(best.success),
            **best.metrics,
        }
        history.append(m)
        if progress_callback:
            progress_callback(m)
    archive.sort(key=lambda x: x.fitness, reverse=True)
    return tuple(archive), tuple(history)


def evolve_candidate(seed_candidate, **kwargs):
    return evolve_candidates((seed_candidate,), **kwargs)
