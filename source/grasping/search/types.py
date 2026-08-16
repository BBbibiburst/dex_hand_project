"""Data contracts shared by grasp-search stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class Device:
    name: str
    xml: Path
    root_body: str
    actuators: tuple[str, ...]
    contact_labels: tuple[int, ...]


@dataclass
class Cloud:
    points: np.ndarray
    normals: np.ndarray
    center: np.ndarray
    scale: float
    mesh: trimesh.Trimesh
    tree: cKDTree


@dataclass
class Surface:
    points: np.ndarray
    labels: np.ndarray
    meshes: list[tuple[np.ndarray, np.ndarray]]
    actuator_values: np.ndarray
    fractions: np.ndarray
    midpoint: np.ndarray


@dataclass(frozen=True)
class ApproachPlan:
    approach_translations: np.ndarray
    approach_fractions: np.ndarray
    grasp_translations: np.ndarray
    grasp_fractions: np.ndarray
    direction: np.ndarray
    maximum_penetration: float
    minimum_object_clearance: float
    maximum_grasp_penetration: float
    maximum_grasp_rigid_penetration: float
    minimum_table_clearance: float
    collision_free: bool


@dataclass
class Candidate:
    surface: Surface
    rotation: np.ndarray
    translation: np.ndarray
    points: np.ndarray
    contacts: tuple[int, ...]
    contact_points: np.ndarray
    contact_normals: np.ndarray
    penetration: float
    rigid_penetration: float
    mean_distance: float
    force_closure: float
    gravity_balance_residual: float
    disturbance_residual: float
    normal_coverage: float
    table_clearance: float
    approach_table_clearance: float
    roll_index: int
    score: float
    valid: bool
    rejection_reasons: tuple[str, ...]
    anchor_index: int
    approach_plan: ApproachPlan | None = None
    approach_alternatives: tuple[ApproachPlan, ...] = ()


@dataclass(frozen=True)
class GraspConfigSearchResult:
    """Artifacts returned by the reusable grasp-search API."""

    output_path: Path
    mesh_path: Path
    cloud: Cloud
    candidates: tuple[Candidate, ...]
    config: dict
    published: bool

    @property
    def grasp(self) -> Candidate:
        """Return the selected best candidate."""
        return self.candidates[0]


@dataclass(frozen=True)
class ValidatedGraspConfigResult:
    """A grasp candidate that passed standalone dynamics validation."""

    output_path: Path
    selected_seed: int
    attempts_used: int
    validation: TrajectoryValidationResult
