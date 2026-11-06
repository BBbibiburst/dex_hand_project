"""Small task-specific scene/goal perturbations, applied before physics reset."""

from dataclasses import replace
import math

import numpy as np
from scipy.spatial.transform import Rotation

from source.envs.manipulation.arenas import BinsArena


FIELDS = {
    "lift": ("dx_m", "dy_m", "yaw_deg"),
    "pick_place": ("dx_m", "dy_m", "yaw_deg", "target_dx_m", "target_dy_m"),
    "stack": ("target_dx_m", "target_dy_m"),
    "push": ("dx_m", "dy_m", "yaw_deg", "push_direction_delta_deg"),
    "nut_assembly": ("dx_m", "dy_m", "yaw_deg"),
}


def validate_conditions(task, values):
    if task not in FIELDS or set(values) - set(FIELDS[task]):
        raise ValueError("Unsupported task perturbation fields")
    for key, value in values.items():
        limit = 2. if key.endswith("deg") else .01
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or abs(value) > limit:
            raise ValueError(f"Perturbation {key} exceeds the bounded pilot range")
    return {key: float(values.get(key, 0.)) for key in FIELDS[task]}


def sampling_contract(contract=None):
    if contract is None:
        contract = {'schema': 'scene-perturbation-ranges-v1', 'tasks': {
            task: {key: [-.5, .5] if key.endswith('deg') else [-.002, .002] for key in fields}
            for task, fields in FIELDS.items()}}
    if (not isinstance(contract, dict) or set(contract) != {'schema', 'tasks'}
            or contract['schema'] != 'scene-perturbation-ranges-v1'
            or not isinstance(contract['tasks'], dict) or set(contract['tasks']) != set(FIELDS)):
        raise ValueError('Invalid versioned five-task perturbation contract')
    tasks = {}
    for task, fields in FIELDS.items():
        values = contract['tasks'][task]
        if not isinstance(values, dict) or set(values) != set(fields):
            raise ValueError(f'Perturbation ranges must specify exactly the supported fields: {task}')
        tasks[task] = {}
        for key in fields:
            bounds = values[key]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f'Expected lower/upper range for {task}/{key}')
            validate_conditions(task, {key: bounds[0]})
            validate_conditions(task, {key: bounds[1]})
            if bounds[0] > bounds[1]:
                raise ValueError(f'Perturbation range is reversed: {task}/{key}')
            tasks[task][key] = [float(bounds[0]), float(bounds[1])]
    return {'schema': contract['schema'], 'tasks': tasks}


def sample_conditions(task, seed, index, contract=None):
    """Deterministic candidate conditions; successes must still be measured."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Candidate index must be nonnegative")
    ranges = sampling_contract(contract)['tasks'][task]
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    values = {key: float(rng.uniform(*ranges[key]))
              for key in FIELDS[task]}
    return validate_conditions(task, values)


def shifted_pose(position, quaternion, values):
    position = np.asarray(position, dtype=float).copy()
    position[:2] += [values.get("dx_m", 0.), values.get("dy_m", 0.)]
    quaternion = np.asarray(quaternion, dtype=float)
    rotation = Rotation.from_euler("z", values.get("yaw_deg", 0.), degrees=True) * Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    return position, rotation.as_quat()[[3, 0, 1, 2]]


def environment_conditions(task, config, values):
    """Change scene initial/goal coordinates, never masses or contact parameters."""
    values = validate_conditions(task, values)
    config = dict(config)
    if task == "stack":
        center = config.get("support_center", (.46, .18))
        config["support_center"] = (center[0] + values["target_dx_m"], center[1] + values["target_dy_m"])
    elif task == "pick_place":
        arena = config.get("arena", BinsArena())
        config["arena"] = replace(arena, target_center=(
            arena.target_center[0] + values["target_dx_m"], arena.target_center[1] + values["target_dy_m"]))
    elif task in ("push", "nut_assembly"):
        prefix = "fixed_object" if task == "push" else "fixed_bolt"
        if prefix + "_position" in config:
            position, quaternion = shifted_pose(config[prefix + "_position"],
                                                config[prefix + "_quaternion_wxyz"], values)
            config[prefix + "_position"] = tuple(position)
            config[prefix + "_quaternion_wxyz"] = tuple(quaternion)
    return config
