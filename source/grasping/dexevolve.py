"""MuJoCo simulator-in-the-loop evolutionary refinement inspired by DexEvolve."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from source.grasping.standalone_validator import validate_grasp_payload_direct
from source.robots.registry import get_hand


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 32
    offspring: int = 16
    generations: int = 20
    tournament_size: int = 4
    mutation_probability: float = 0.75
    crossover_probability: float = 0.2
    translation_sigma: float = 0.005
    orientation_sigma: float = 0.05
    actuator_sigma: float = 0.04
    novelty_threshold: float = 0.1
    density_radius: float = 0.65
    density_power: float = 2.0
    max_archive: int = 1024
    seconds: float = 1.5
    settle_seconds: float = 0.4
    jobs: int = 4
    seed: int = 0


@dataclass
class Individual:
    payload: dict
    fitness: float = -np.inf
    stable: bool = False
    metrics: dict | None = None


def embedding(payload: dict) -> np.ndarray:
    position = 10.0 * np.asarray(payload["hand_translation"], dtype=np.float64)
    angles = Rotation.from_matrix(payload["hand_rotation_matrix"]).as_euler("xyz")
    joints = np.asarray(payload["hand_actuator_fractions"], dtype=np.float64)
    return np.concatenate([position, angles, joints])


@lru_cache(maxsize=None)
def _actuator_ranges(end_effector_name: str) -> tuple[tuple[float, float], ...]:
    """Load immutable actuator limits once per evolution process."""
    descriptor = get_hand(end_effector_name)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(descriptor.xml_path))
    ranges = []
    for name in descriptor.position_actuator_names:
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        low, high = model.actuator_ctrlrange[actuator]
        ranges.append((float(low), float(high)))
    return tuple(ranges)


def _actuator_values(payload: dict, fractions: np.ndarray) -> list[float]:
    ranges = _actuator_ranges(payload.get("end_effector_name", "dex_hand"))
    return [
        low + float(fraction) * (high - low)
        for (low, high), fraction in zip(ranges, fractions, strict=True)
    ]


def synchronize_trajectory(
    payload: dict,
    *,
    previous_rotation: np.ndarray,
    previous_fractions: np.ndarray,
) -> None:
    """Propagate a mutated final pose through its executable trajectory."""
    rotation = np.asarray(payload["hand_rotation_matrix"], dtype=np.float64)
    rotation_delta = rotation @ np.asarray(previous_rotation, dtype=np.float64).T
    for key in ("approach_hand_rotation_matrices", "grasp_hand_rotation_matrices"):
        if key in payload:
            matrices = np.asarray(payload[key], dtype=np.float64)
            payload[key] = np.einsum("ij,njk->nik", rotation_delta, matrices).tolist()

    if "grasp_hand_actuator_fractions" in payload:
        trajectory = np.asarray(payload["grasp_hand_actuator_fractions"], dtype=np.float64)
        delta = (
            np.asarray(payload["hand_actuator_fractions"], dtype=np.float64)
            - np.asarray(previous_fractions, dtype=np.float64)
        )
        progress = (
            np.ones((1, 1), dtype=np.float64)
            if len(trajectory) == 1
            else np.linspace(0.0, 1.0, len(trajectory))[:, None]
        )
        payload["grasp_hand_actuator_fractions"] = np.clip(
            trajectory + progress * delta,
            0.0,
            1.0,
        ).tolist()


def mutate(payload: dict, rng: np.random.Generator, config: EvolutionConfig) -> dict:
    child = deepcopy(payload)
    old_translation = np.asarray(child["hand_translation"], dtype=np.float64)
    translation = old_translation + rng.normal(0.0, config.translation_sigma, 3)
    old_rotation = np.asarray(child["hand_rotation_matrix"], dtype=np.float64)
    old_fractions = np.asarray(child["hand_actuator_fractions"], dtype=np.float64)
    rotation = old_rotation @ Rotation.from_rotvec(
        rng.normal(0.0, config.orientation_sigma, 3)
    ).as_matrix()
    fractions = np.clip(
        np.asarray(child["hand_actuator_fractions"], dtype=np.float64)
        + rng.normal(0.0, config.actuator_sigma, len(child["hand_actuator_fractions"])),
        0.0,
        1.0,
    )
    delta = translation - old_translation
    child["hand_translation"] = translation.tolist()
    child["hand_rotation_matrix"] = rotation.tolist()
    child["hand_actuator_fractions"] = fractions.tolist()
    child["hand_actuator_values"] = _actuator_values(child, fractions)
    for key in ("approach_hand_translations", "grasp_hand_translations"):
        if key in child:
            child[key] = (np.asarray(child[key], dtype=np.float64) + delta).tolist()
    synchronize_trajectory(
        child,
        previous_rotation=old_rotation,
        previous_fractions=old_fractions,
    )
    return child


def crossover(first: dict, second: dict, rng: np.random.Generator) -> dict:
    child = deepcopy(first)
    blend = float(rng.uniform(0.25, 0.75))
    first_pos = np.asarray(first["hand_translation"], dtype=np.float64)
    second_pos = np.asarray(second["hand_translation"], dtype=np.float64)
    translation = (1.0 - blend) * first_pos + blend * second_pos
    delta = translation - first_pos
    child["hand_translation"] = translation.tolist()
    for key in ("approach_hand_translations", "grasp_hand_translations"):
        if key in child:
            child[key] = (np.asarray(child[key], dtype=np.float64) + delta).tolist()
    first_q = np.asarray(first["hand_actuator_fractions"], dtype=np.float64)
    second_q = np.asarray(second["hand_actuator_fractions"], dtype=np.float64)
    fractions = np.where(rng.random(len(first_q)) < 0.5, first_q, second_q)
    child["hand_actuator_fractions"] = fractions.tolist()
    child["hand_actuator_values"] = _actuator_values(child, fractions)
    first_rot = Rotation.from_matrix(first["hand_rotation_matrix"])
    second_rot = Rotation.from_matrix(second["hand_rotation_matrix"])
    relative = first_rot.inv() * second_rot
    child["hand_rotation_matrix"] = (
        first_rot * Rotation.from_rotvec(blend * relative.as_rotvec())
    ).as_matrix().tolist()
    synchronize_trajectory(
        child,
        previous_rotation=first_rot.as_matrix(),
        previous_fractions=first_q,
    )
    return child


def _evaluate_task(task: tuple[dict, float, float]) -> Individual:
    payload, seconds, settle_seconds = task
    try:
        result = validate_grasp_payload_direct(
            payload, seconds=seconds, settle_seconds=settle_seconds
        )
        metrics = asdict(result)
        fitness = (
            100.0 * float(result.stable)
            + 0.5 * result.final_contacts
            - 500.0 * max(result.vertical_drop, 0.0)
            - 200.0 * result.position_drift
            - 2.0 * result.rotation_drift
        )
        return Individual(payload, fitness, result.stable, metrics)
    except Exception as exc:
        return Individual(payload, -1e6, False, {"error": str(exc)})


def evaluate_population(
    payloads: list[dict],
    config: EvolutionConfig,
    *,
    executor: ProcessPoolExecutor | None = None,
) -> list[Individual]:
    tasks = [(payload, config.seconds, config.settle_seconds) for payload in payloads]
    if config.jobs == 1:
        return [_evaluate_task(task) for task in tasks]
    if executor is not None:
        return list(executor.map(_evaluate_task, tasks))
    with ProcessPoolExecutor(max_workers=config.jobs) as temporary_executor:
        return list(temporary_executor.map(_evaluate_task, tasks))


def _density_adjusted(archive: list[Individual], config: EvolutionConfig) -> np.ndarray:
    features = np.stack([embedding(item.payload) for item in archive])
    distances = np.linalg.norm(features[:, None] - features[None, :], axis=-1)
    density = np.maximum((distances < config.density_radius).sum(1) - 1, 0)
    scores = np.asarray([item.fitness for item in archive])
    return scores / np.power(1.0 + density, config.density_power)


def _insert(archive: list[Individual], child: Individual, config: EvolutionConfig) -> None:
    feature = embedding(child.payload)
    if archive:
        distances = np.asarray([np.linalg.norm(feature - embedding(item.payload)) for item in archive])
        nearest = int(np.argmin(distances))
        if distances[nearest] < config.novelty_threshold:
            if child.fitness > archive[nearest].fitness:
                archive[nearest] = child
            return
    archive.append(child)
    if len(archive) > config.max_archive:
        archive.sort(key=lambda item: item.fitness, reverse=True)
        del archive[config.max_archive :]


def evolve(seed_payload: dict, config: EvolutionConfig) -> tuple[list[Individual], list[dict]]:
    if config.jobs == 1:
        return _evolve(seed_payload, config, executor=None)
    with ProcessPoolExecutor(max_workers=config.jobs) as executor:
        return _evolve(seed_payload, config, executor=executor)


def _evolve(
    seed_payload: dict,
    config: EvolutionConfig,
    *,
    executor: ProcessPoolExecutor | None,
) -> tuple[list[Individual], list[dict]]:
    rng = np.random.default_rng(config.seed)
    seeds = [deepcopy(seed_payload)]
    seeds.extend(mutate(seed_payload, rng, config) for _ in range(config.population_size - 1))
    archive = evaluate_population(seeds, config, executor=executor)
    history = []
    for generation in range(config.generations):
        adjusted = _density_adjusted(archive, config)
        children = []
        for _ in range(config.offspring):
            choices = rng.choice(len(archive), min(config.tournament_size, len(archive)), replace=False)
            parent = archive[int(choices[np.argmax(adjusted[choices])])].payload
            child = deepcopy(parent)
            if len(archive) > 1 and rng.random() < config.crossover_probability:
                other = archive[int(rng.integers(len(archive)))].payload
                child = crossover(child, other, rng)
            if rng.random() < config.mutation_probability:
                child = mutate(child, rng, config)
            children.append(child)
        evaluated = evaluate_population(children, config, executor=executor)
        for child in evaluated:
            _insert(archive, child, config)
        history.append(
            {
                "generation": generation + 1,
                "archive": len(archive),
                "stable": sum(item.stable for item in archive),
                "best_fitness": max(item.fitness for item in archive),
            }
        )
    archive.sort(key=lambda item: (not item.stable, -item.fitness))
    return archive, history
