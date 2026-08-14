"""MuJoCo simulator-in-the-loop evolutionary refinement inspired by DexEvolve."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import importlib.util
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

from source.grasping.standalone_validator import validate_grasp_payload_direct
from source.grasping.dex_hand_surface import load_posed_dex_hand_surface
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
    minimum_table_clearance: float = 0.005
    preferred_table_clearance: float = 0.025
    robustness_samples: int = 2
    robustness_translation_sigma: float = 0.0025
    robustness_orientation_sigma: float = 0.025
    backend: str = "cpu"
    mjwarp_device: str = "cuda:0"
    mjwarp_batch_size: int = 32
    mjwarp_nconmax: int = 128
    mjwarp_njmax: int = 512
    mjwarp_fallback: bool = True


@dataclass
class Individual:
    payload: dict
    fitness: float = -np.inf
    direct_hold_stable: bool = False
    metrics: dict | None = None


class _MjWarpWithCpuFallback:
    """Switch a failed device batch to the existing CPU validator for this object."""

    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator
        self.backend = "mjwarp"
        self.fallback_error: str | None = None

    def evaluate(self, payloads: list[dict], *, seconds: float, settle_seconds: float):
        if self.backend == "mjwarp":
            try:
                return self.evaluator.evaluate(
                    payloads,
                    seconds=seconds,
                    settle_seconds=settle_seconds,
                )
            except Exception as exc:
                self.backend = "cpu_fallback"
                self.fallback_error = str(exc)
        return [
            validate_grasp_payload_direct(
                payload,
                seconds=seconds,
                settle_seconds=settle_seconds,
            )
            for payload in payloads
        ]


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


@lru_cache(maxsize=16)
def _dex_hand_vertices(fractions: tuple[float, ...]) -> np.ndarray:
    surface = load_posed_dex_hand_surface(
        actuator_fractions=np.asarray(fractions, dtype=np.float64),
        max_points_per_geom=32,
        seed=0,
    )
    return np.concatenate([mesh.vertices for mesh in surface.meshes])


def table_clearance_metrics(payload: dict) -> dict[str, float] | None:
    """Recompute full-hand table clearance for every executable waypoint."""
    table_z = payload.get("object_table_height")
    if table_z is None or payload.get("end_effector_name", "dex_hand") != "dex_hand":
        return None
    final_fractions = np.asarray(payload["hand_actuator_fractions"], dtype=np.float64)
    final_vertices = _dex_hand_vertices(
        tuple(float(value) for value in np.round(final_fractions, decimals=6))
    )
    trajectory_fraction_groups = [
        np.asarray(payload.get("approach_hand_actuator_fractions", [final_fractions]))[0],
    ]
    grasp_fractions = np.asarray(
        payload.get("grasp_hand_actuator_fractions", [final_fractions]), dtype=np.float64
    )
    trajectory_fraction_groups.extend(
        (grasp_fractions[0], grasp_fractions[len(grasp_fractions) // 2], grasp_fractions[-1])
    )
    unique_fraction_groups = {
        tuple(float(value) for value in np.round(group, decimals=6))
        for group in trajectory_fraction_groups
    }
    # A conservative swept-shape approximation: apply the union of open,
    # mid-closing and final full collision meshes at every path waypoint.
    trajectory_vertices = np.concatenate(
        [_dex_hand_vertices(group) for group in sorted(unique_fraction_groups)]
    )

    def minimum(translations, rotations, vertices) -> float:
        result = np.inf
        for translation, rotation in zip(translations, rotations, strict=True):
            posed = vertices @ np.asarray(rotation, dtype=np.float64).T + np.asarray(
                translation, dtype=np.float64
            )
            result = min(result, float(posed[:, 2].min() - float(table_z)))
        return float(result)

    final_rotation = np.asarray(payload["hand_rotation_matrix"], dtype=np.float64)
    grasp_rotations = np.asarray(
        payload.get("grasp_hand_rotation_matrices", [final_rotation]), dtype=np.float64
    )
    rotation_delta = final_rotation @ grasp_rotations[-1].T
    approach_rotations = np.einsum(
        "ij,njk->nik",
        rotation_delta,
        np.asarray(
            payload.get("approach_hand_rotation_matrices", [final_rotation]),
            dtype=np.float64,
        ),
    )
    grasp_rotations = np.einsum("ij,njk->nik", rotation_delta, grasp_rotations)
    final_clearance = minimum([payload["hand_translation"]], [final_rotation], final_vertices)
    approach_clearance = minimum(
        payload.get("approach_hand_translations", [payload["hand_translation"]]),
        approach_rotations,
        trajectory_vertices,
    )
    grasp_clearance = minimum(
        payload.get("grasp_hand_translations", [payload["hand_translation"]]),
        grasp_rotations,
        trajectory_vertices,
    )
    return {
        "hand_table_clearance": final_clearance,
        "approach_minimum_table_clearance": approach_clearance,
        "grasp_minimum_table_clearance": grasp_clearance,
        "trajectory_minimum_table_clearance": min(
            final_clearance,
            approach_clearance,
            grasp_clearance,
        ),
    }


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
        delta = np.asarray(payload["hand_actuator_fractions"], dtype=np.float64) - np.asarray(
            previous_fractions, dtype=np.float64
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
    rotation = (
        old_rotation
        @ Rotation.from_rotvec(rng.normal(0.0, config.orientation_sigma, 3)).as_matrix()
    )
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
        (first_rot * Rotation.from_rotvec(blend * relative.as_rotvec())).as_matrix().tolist()
    )
    synchronize_trajectory(
        child,
        previous_rotation=first_rot.as_matrix(),
        previous_fractions=first_q,
    )
    return child


def _evaluate_task(task: tuple[dict, float, float, float, float]) -> Individual:
    (
        payload,
        seconds,
        settle_seconds,
        minimum_clearance,
        preferred_clearance,
        robustness_samples,
        robustness_translation_sigma,
        robustness_orientation_sigma,
    ) = task
    try:
        clearance = table_clearance_metrics(payload)
        if clearance is not None:
            payload.update(clearance)
            actual_minimum = clearance["trajectory_minimum_table_clearance"]
            if actual_minimum < minimum_clearance:
                return Individual(
                    payload,
                    -1e6 - 1e3 * (minimum_clearance - actual_minimum),
                    False,
                    {**clearance, "rejection_reason": "table_clearance"},
                )
        result = validate_grasp_payload_direct(
            payload, seconds=seconds, settle_seconds=settle_seconds
        )
        robustness_results = _evaluate_robustness(
            payload,
            samples=robustness_samples,
            translation_sigma=robustness_translation_sigma,
            orientation_sigma=robustness_orientation_sigma,
            seconds=min(seconds, 0.8),
            settle_seconds=min(settle_seconds, 0.25),
        )
        return _individual_from_result(
            payload,
            result,
            clearance=clearance,
            preferred_clearance=preferred_clearance,
            robustness_results=robustness_results,
        )
    except Exception as exc:
        return Individual(payload, -1e6, False, {"error": str(exc)})


def _individual_from_result(
    payload: dict,
    result,
    *,
    clearance: dict[str, float] | None,
    preferred_clearance: float,
    robustness_results: list | None = None,
) -> Individual:
    metrics = asdict(result)
    fitness = (
        100.0 * float(result.direct_hold_stable)
        + 0.5 * result.final_contacts
        - 500.0 * max(result.vertical_drop, 0.0)
        - 200.0 * result.position_drift
        - 2.0 * result.rotation_drift
    )
    if clearance is not None:
        fitness -= 20.0 * max(
            preferred_clearance - clearance["trajectory_minimum_table_clearance"],
            0.0,
        )
        metrics.update(clearance)
    if robustness_results:
        stable_count = sum(item.direct_hold_stable for item in robustness_results)
        robustness_rate = stable_count / len(robustness_results)
        fitness += 30.0 * robustness_rate
        fitness -= 100.0 * (1.0 - robustness_rate)
        metrics.update(
            robustness_samples=len(robustness_results),
            robustness_stable=stable_count,
            robustness_rate=robustness_rate,
        )
    return Individual(payload, fitness, result.direct_hold_stable, metrics)


def _evaluate_robustness(
    payload: dict,
    *,
    samples: int,
    translation_sigma: float,
    orientation_sigma: float,
    seconds: float,
    settle_seconds: float,
) -> list:
    """Evaluate deterministic small wrist perturbations around a candidate."""
    if samples <= 0:
        return []
    signature = np.concatenate(
        [
            np.asarray(payload["hand_translation"], dtype=np.float64),
            np.asarray(payload["hand_actuator_fractions"], dtype=np.float64),
        ]
    )
    seed = int(abs(float(np.dot(signature, np.arange(1, len(signature) + 1)))) * 1e6) % 2**32
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(samples):
        perturbed = deepcopy(payload)
        translation_delta = rng.normal(0.0, translation_sigma, 3)
        perturbed["hand_translation"] = (
            np.asarray(payload["hand_translation"], dtype=np.float64) + translation_delta
        ).tolist()
        rotation_delta = Rotation.from_rotvec(rng.normal(0.0, orientation_sigma, 3)).as_matrix()
        perturbed["hand_rotation_matrix"] = (
            np.asarray(payload["hand_rotation_matrix"], dtype=np.float64) @ rotation_delta
        ).tolist()
        results.append(
            validate_grasp_payload_direct(
                perturbed,
                seconds=seconds,
                settle_seconds=settle_seconds,
            )
        )
    return results


def mjwarp_available() -> bool:
    """Return whether optional MJWarp modules are installed without initialising CUDA."""

    return (
        importlib.util.find_spec("mujoco_warp") is not None
        and importlib.util.find_spec("warp") is not None
    )


def _evaluate_population_mjwarp(
    payloads: list[dict],
    config: EvolutionConfig,
    evaluator: Any,
) -> list[Individual]:
    individuals: list[Individual | None] = [None] * len(payloads)
    accepted: list[dict] = []
    accepted_indices: list[int] = []
    clearances: list[dict[str, float] | None] = []
    for index, payload in enumerate(payloads):
        clearance = table_clearance_metrics(payload)
        if clearance is not None:
            payload.update(clearance)
            actual_minimum = clearance["trajectory_minimum_table_clearance"]
            if actual_minimum < config.minimum_table_clearance:
                individuals[index] = Individual(
                    payload,
                    -1e6 - 1e3 * (config.minimum_table_clearance - actual_minimum),
                    False,
                    {**clearance, "rejection_reason": "table_clearance"},
                )
                continue
        accepted.append(payload)
        accepted_indices.append(index)
        clearances.append(clearance)

    batch_size = max(1, int(config.mjwarp_batch_size))
    for start in range(0, len(accepted), batch_size):
        batch = accepted[start : start + batch_size]
        results = evaluator.evaluate(
            batch,
            seconds=config.seconds,
            settle_seconds=config.settle_seconds,
        )
        for offset, result in enumerate(results):
            accepted_index = start + offset
            original_index = accepted_indices[accepted_index]
            individuals[original_index] = _individual_from_result(
                batch[offset],
                result,
                clearance=clearances[accepted_index],
                preferred_clearance=config.preferred_table_clearance,
                robustness_results=None,
            )
    if any(item is None for item in individuals):
        raise RuntimeError("MJWarp population evaluator returned an incomplete batch.")
    return [item for item in individuals if item is not None]


def evaluate_population(
    payloads: list[dict],
    config: EvolutionConfig,
    *,
    executor: ProcessPoolExecutor | None = None,
    batch_evaluator: Any = None,
) -> list[Individual]:
    if batch_evaluator is not None:
        return _evaluate_population_mjwarp(payloads, config, batch_evaluator)
    tasks = [
        (
            payload,
            config.seconds,
            config.settle_seconds,
            config.minimum_table_clearance,
            config.preferred_table_clearance,
            config.robustness_samples,
            config.robustness_translation_sigma,
            config.robustness_orientation_sigma,
        )
        for payload in payloads
    ]
    if batch_evaluator is not None or config.jobs == 1:
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
        distances = np.asarray(
            [np.linalg.norm(feature - embedding(item.payload)) for item in archive]
        )
        nearest = int(np.argmin(distances))
        if distances[nearest] < config.novelty_threshold:
            if child.fitness > archive[nearest].fitness:
                archive[nearest] = child
            return
    archive.append(child)
    if len(archive) > config.max_archive:
        archive.sort(key=lambda item: item.fitness, reverse=True)
        del archive[config.max_archive :]


def evolve(
    seed_payload: dict,
    config: EvolutionConfig,
    *,
    progress_callback: Callable[[int, int, dict], None] | None = None,
) -> tuple[list[Individual], list[dict]]:
    backend = config.backend
    if backend not in {"cpu", "mjwarp", "auto"}:
        raise ValueError(f"Unknown evolution backend {backend!r}.")
    if backend == "auto":
        backend = "mjwarp" if mjwarp_available() else "cpu"
    batch_evaluator = None
    fallback_error = None
    if backend == "mjwarp":
        try:
            from source.grasping.mjwarp_evaluator import MjWarpPopulationEvaluator

            batch_evaluator = _MjWarpWithCpuFallback(
                MjWarpPopulationEvaluator(
                    seed_payload,
                    device=config.mjwarp_device,
                    nconmax=config.mjwarp_nconmax,
                    njmax=config.mjwarp_njmax,
                )
            )
        except Exception as exc:
            if not config.mjwarp_fallback:
                raise
            fallback_error = str(exc)
            backend = "cpu"
    if config.jobs == 1:
        return _evolve(
            seed_payload,
            config,
            executor=None,
            batch_evaluator=batch_evaluator,
            backend=backend,
            fallback_error=fallback_error,
            progress_callback=progress_callback,
        )
    with ProcessPoolExecutor(max_workers=config.jobs) as executor:
        return _evolve(
            seed_payload,
            config,
            executor=executor,
            batch_evaluator=batch_evaluator,
            backend=backend,
            fallback_error=fallback_error,
            progress_callback=progress_callback,
        )


def _evolve(
    seed_payload: dict,
    config: EvolutionConfig,
    *,
    executor: ProcessPoolExecutor | None,
    batch_evaluator: Any = None,
    backend: str = "cpu",
    fallback_error: str | None = None,
    progress_callback: Callable[[int, int, dict], None] | None = None,
) -> tuple[list[Individual], list[dict]]:
    rng = np.random.default_rng(config.seed)
    seeds = [deepcopy(seed_payload)]
    seeds.extend(mutate(seed_payload, rng, config) for _ in range(config.population_size - 1))
    archive = evaluate_population(
        seeds,
        config,
        executor=executor,
        batch_evaluator=batch_evaluator,
    )
    history = []
    for generation in range(config.generations):
        adjusted = _density_adjusted(archive, config)
        children = []
        for _ in range(config.offspring):
            choices = rng.choice(
                len(archive), min(config.tournament_size, len(archive)), replace=False
            )
            parent = archive[int(choices[np.argmax(adjusted[choices])])].payload
            child = deepcopy(parent)
            if len(archive) > 1 and rng.random() < config.crossover_probability:
                other = archive[int(rng.integers(len(archive)))].payload
                child = crossover(child, other, rng)
            if rng.random() < config.mutation_probability:
                child = mutate(child, rng, config)
            children.append(child)
        evaluated = evaluate_population(
            children,
            config,
            executor=executor,
            batch_evaluator=batch_evaluator,
        )
        for child in evaluated:
            _insert(archive, child, config)
        generation_summary = {
            "generation": generation + 1,
            "archive": len(archive),
            "direct_hold_stable": sum(item.direct_hold_stable for item in archive),
            "best_fitness": max(item.fitness for item in archive),
            "backend": getattr(batch_evaluator, "backend", backend),
            "backend_fallback_error": getattr(
                batch_evaluator,
                "fallback_error",
                fallback_error,
            ),
        }
        history.append(generation_summary)
        if progress_callback is not None:
            progress_callback(generation + 1, config.generations, generation_summary)
    archive.sort(key=lambda item: (not item.direct_hold_stable, -item.fitness))
    return archive, history
