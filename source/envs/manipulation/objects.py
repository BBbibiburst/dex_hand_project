# -*- coding: utf-8 -*-
"""Composable task-object specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import mujoco
import numpy as np

from source.assets import asset_path
from source.envs.manipulation.object_catalog import resolve_record, resolve_record_path


def _configure_free_joint(joint: mujoco.MjsJoint, name: str) -> None:
    """Configure a task-object free joint without robot-joint defaults."""
    joint.name = name
    joint.type = mujoco.mjtJoint.mjJNT_FREE
    # Objects are attached to the arm's merged MjSpec and would otherwise
    # inherit its joint damping, dry friction, and armature. Those values are
    # appropriate for powered robot joints but can freeze small free objects
    # in physically unstable poses.
    joint.damping = np.zeros(3, dtype=np.float64)
    joint.frictionloss = 0.0
    joint.armature = 0.0


class ManipulationObjectSpec(ABC):
    name: str

    @property
    @abstractmethod
    def body_name(self) -> str: ...

    @property
    @abstractmethod
    def joint_name(self) -> str: ...

    @property
    @abstractmethod
    def geom_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def horizontal_radius(self) -> float: ...

    @property
    @abstractmethod
    def bottom_offset(self) -> float: ...

    @abstractmethod
    def add_to_spec(self, spec: mujoco.MjSpec, initial_pos: np.ndarray) -> None: ...


def _read_obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
    if not vertices:
        raise ValueError(f"OBJ mesh contains no vertices: {path}")
    points = np.asarray(vertices, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)


@dataclass(frozen=True)
class MeshObjectSpec(ManipulationObjectSpec):
    """Free YCB/EGAD mesh with separate visual and collision geoms."""

    name: str
    object_id: str
    target_size: float = 0.09
    density: float = 500.0
    friction: Tuple[float, float, float] = (1.0, 0.005, 0.0001)

    def __post_init__(self) -> None:
        record = resolve_record(self.object_id)
        root = resolve_record_path(record, "source_path")
        model_files = tuple(record.get("model_files", ()))
        visual = next(
            (root / item for item in model_files if Path(item).name == "textured.obj"),
            None,
        )
        if visual is None:
            visual = next(
                (root / item for item in model_files if Path(item).suffix.lower() == ".obj"),
                None,
            )
        if visual is None or not visual.is_file():
            raise FileNotFoundError(f"No OBJ visual mesh for {self.object_id}")
        low, high = _read_obj_bounds(visual)
        extent = high - low
        scale = self.target_size / max(float(extent.max()), 1e-8)
        object.__setattr__(self, "_visual_path", visual)
        # MuJoCo builds a convex collision hull from OBJ meshes. The YCB bundle
        # also contains collision.ply, but MuJoCo's built-in mesh decoder does
        # not support that file on all platforms.
        object.__setattr__(self, "_collision_path", visual)
        object.__setattr__(self, "_center", 0.5 * (low + high))
        object.__setattr__(self, "_extent", extent * scale)
        object.__setattr__(self, "_scale", scale)
        texture = root / "texture_map.png"
        object.__setattr__(self, "_texture_path", texture if texture.is_file() else None)

    @property
    def body_name(self) -> str:
        return f"{self.name}_body"

    @property
    def joint_name(self) -> str:
        return f"{self.name}_freejoint"

    @property
    def geom_names(self) -> tuple[str, ...]:
        return (f"{self.name}_collision",)

    @property
    def horizontal_radius(self) -> float:
        return 0.5 * float(np.linalg.norm(self._extent[:2]))

    @property
    def bottom_offset(self) -> float:
        return 0.5 * float(self._extent[2])

    def add_to_spec(self, spec: mujoco.MjSpec, initial_pos: np.ndarray) -> None:
        prefix = self.name.replace(":", "_")
        visual_mesh = spec.add_mesh()
        visual_mesh.name = f"{prefix}_visual_mesh"
        visual_mesh.file = str(self._visual_path.resolve())
        visual_mesh.scale = [self._scale] * 3
        visual_mesh.refpos = self._center.tolist()

        collision_mesh = spec.add_mesh()
        collision_mesh.name = f"{prefix}_collision_mesh"
        collision_mesh.file = str(self._collision_path.resolve())
        collision_mesh.scale = [self._scale] * 3
        collision_mesh.refpos = self._center.tolist()

        material_name = ""
        if self._texture_path is not None:
            texture = spec.add_texture()
            texture.name = f"{prefix}_texture"
            texture.type = mujoco.mjtTexture.mjTEXTURE_2D
            texture.file = str(self._texture_path.resolve())
            material = spec.add_material()
            material.name = f"{prefix}_material"
            material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
            material_name = material.name

        body = spec.worldbody.add_body()
        body.name = self.body_name
        body.pos = np.asarray(initial_pos, dtype=float).tolist()
        joint = body.add_joint()
        _configure_free_joint(joint, self.joint_name)

        collision = body.add_geom()
        collision.name = self.geom_names[0]
        collision.type = mujoco.mjtGeom.mjGEOM_MESH
        collision.meshname = collision_mesh.name
        collision.density = self.density
        collision.friction = list(self.friction)
        collision.contype = 1
        collision.conaffinity = 1
        collision.rgba = [0.0, 0.0, 0.0, 0.0]

        visual = body.add_geom()
        visual.name = f"{self.name}_visual"
        visual.type = mujoco.mjtGeom.mjGEOM_MESH
        visual.meshname = visual_mesh.name
        visual.contype = 0
        visual.conaffinity = 0
        visual.density = 0.0
        visual.rgba = [0.65, 0.75, 0.90, 1.0]
        if material_name:
            visual.material = material_name


@dataclass(frozen=True)
class FreeBoxSpec(ManipulationObjectSpec):
    """Free box; retained as the compatible name used by existing tasks."""

    name: str
    half_size: Tuple[float, float, float]
    rgba: Tuple[float, float, float, float]
    density: float = 500.0
    friction: Tuple[float, float, float] = (1.0, 0.005, 0.0001)
    duplicate_collision_geoms: bool = True

    @property
    def body_name(self) -> str:
        return f"{self.name}_body"

    @property
    def joint_name(self) -> str:
        return f"{self.name}_freejoint"

    @property
    def geom_name(self) -> str:
        return f"{self.name}_geom"

    @property
    def geom_names(self) -> tuple[str, ...]:
        return (self.geom_name,)

    @property
    def horizontal_radius(self) -> float:
        return float(np.linalg.norm(self.half_size[:2]))

    @property
    def bottom_offset(self) -> float:
        return float(self.half_size[2])

    def add_to_spec(self, spec: mujoco.MjSpec, initial_pos: np.ndarray) -> None:
        body = spec.worldbody.add_body()
        body.name = self.body_name
        body.pos = np.asarray(initial_pos, dtype=float).tolist()
        joint = body.add_joint()
        _configure_free_joint(joint, self.joint_name)
        geom = body.add_geom()
        geom.name = self.geom_name
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = list(self.half_size)
        geom.density = self.density
        geom.friction = list(self.friction)
        geom.rgba = list(self.rgba)
        geom.condim = 3
        geom.contype = 1
        geom.conaffinity = 1
        if self.duplicate_collision_geoms:
            visual = body.add_geom()
            visual.name = f"{self.name}_visual"
            visual.type = mujoco.mjtGeom.mjGEOM_BOX
            visual.size = list(self.half_size)
            visual.contype = 0
            visual.conaffinity = 0
            visual.density = 0.0
            visual.rgba = list(self.rgba)


@dataclass(frozen=True)
class FreeCylinderSpec(ManipulationObjectSpec):
    """Free upright cylinder used by can-like task objects."""

    name: str
    radius: float
    half_height: float
    rgba: Tuple[float, float, float, float]
    density: float = 500.0

    @property
    def body_name(self) -> str:
        return f"{self.name}_body"

    @property
    def joint_name(self) -> str:
        return f"{self.name}_freejoint"

    @property
    def geom_names(self) -> tuple[str, ...]:
        return (f"{self.name}_geom",)

    @property
    def horizontal_radius(self) -> float:
        return self.radius

    @property
    def bottom_offset(self) -> float:
        return self.half_height

    def add_to_spec(self, spec: mujoco.MjSpec, initial_pos: np.ndarray) -> None:
        body = spec.worldbody.add_body()
        body.name = self.body_name
        body.pos = np.asarray(initial_pos, dtype=float).tolist()
        joint = body.add_joint()
        _configure_free_joint(joint, self.joint_name)
        geom = body.add_geom()
        geom.name = self.geom_names[0]
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [self.radius, self.half_height, 0.0]
        geom.density = self.density
        geom.rgba = list(self.rgba)
        geom.friction = [1.0, 0.005, 0.0001]


@dataclass(frozen=True)
class FreeNutSpec(ManipulationObjectSpec):
    """Free four-bar nut with a physical central opening."""

    name: str
    outer_radius: float
    inner_radius: float
    half_height: float
    rgba: Tuple[float, float, float, float]

    @property
    def body_name(self) -> str:
        return f"{self.name}_body"

    @property
    def joint_name(self) -> str:
        return f"{self.name}_freejoint"

    @property
    def geom_names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}_bar_{i}" for i in range(4))

    @property
    def horizontal_radius(self) -> float:
        return self.outer_radius

    @property
    def bottom_offset(self) -> float:
        return self.half_height

    def add_to_spec(self, spec: mujoco.MjSpec, initial_pos: np.ndarray) -> None:
        body = spec.worldbody.add_body()
        body.name = self.body_name
        body.pos = np.asarray(initial_pos, dtype=float).tolist()
        joint = body.add_joint()
        _configure_free_joint(joint, self.joint_name)
        thickness = 0.5 * (self.outer_radius - self.inner_radius)
        center = 0.5 * (self.outer_radius + self.inner_radius)
        half_length = self.outer_radius
        for index, (pos, size) in enumerate(
            (
                ((0.0, center, 0.0), (half_length, thickness, self.half_height)),
                ((0.0, -center, 0.0), (half_length, thickness, self.half_height)),
                ((center, 0.0, 0.0), (thickness, self.inner_radius, self.half_height)),
                ((-center, 0.0, 0.0), (thickness, self.inner_radius, self.half_height)),
            )
        ):
            geom = body.add_geom()
            geom.name = self.geom_names[index]
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.pos = list(pos)
            geom.size = list(size)
            geom.rgba = list(self.rgba)
            geom.density = 500.0
            geom.friction = [1.0, 0.005, 0.0001]


@dataclass(frozen=True)
class XmlNutSpec(ManipulationObjectSpec):
    """Procedurally constructed nut retained under its compatible public name."""

    name: str
    xml_filename: str
    radius: float = 0.11
    bottom: float = 0.05

    @property
    def body_name(self) -> str:
        return f"{self.name}_body"

    @property
    def joint_name(self) -> str:
        return f"{self.name}_freejoint"

    @property
    def geom_names(self) -> tuple[str, ...]:
        ring_count = 32 if self.xml_filename == "round-nut.xml" else 4
        return (
            *(f"{self.name}_ring_collision_{index}" for index in range(ring_count)),
            f"{self.name}_handle_collision",
            f"{self.name}_grip_pad_collision",
        )

    @property
    def horizontal_radius(self) -> float:
        return self.radius

    @property
    def bottom_offset(self) -> float:
        return self.bottom

    @property
    def handle_site_name(self) -> str:
        return f"{self.name}_handle_site"

    def add_to_spec(self, spec: mujoco.MjSpec, initial_pos: np.ndarray) -> None:
        if self.xml_filename not in {"square-nut.xml", "round-nut.xml"}:
            raise ValueError(f"Unknown nut design: {self.xml_filename!r}")
        material = self._add_material(spec)
        object_body = spec.worldbody.add_body()
        object_body.name = self.body_name
        object_body.pos = np.asarray(initial_pos, dtype=float).tolist()
        joint = object_body.add_joint()
        _configure_free_joint(joint, self.joint_name)
        self._add_refined_collisions(object_body)
        self._add_refined_visuals(object_body, material)
        handle_site = object_body.add_site()
        handle_site.name = self.handle_site_name
        handle_site.pos = [0.084, 0.0, 0.0]
        handle_site.size = [0.005, 0.0, 0.0]
        handle_site.rgba = [1.0, 0.0, 0.0, 0.0]
        center_site = object_body.add_site()
        center_site.name = f"{self.name}_center_site"
        center_site.size = [0.003, 0.0, 0.0]
        center_site.rgba = [1.0, 0.0, 0.0, 0.0]

    def _add_material(self, spec: mujoco.MjSpec) -> str:
        round_nut = self.xml_filename == "round-nut.xml"
        texture = spec.add_texture()
        texture.name = f"{self.name}_metal_texture"
        texture.type = mujoco.mjtTexture.mjTEXTURE_CUBE
        texture.file = str(
            asset_path(
                "textures",
                "steel-scratched.png" if round_nut else "brass-ambra.png",
            ).resolve()
        )
        material = spec.add_material()
        material.name = f"{self.name}_metal"
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
        material.reflectance = 1.0
        material.shininess = 1.0
        material.specular = 1.0
        material.texrepeat = [1.0, 1.0]
        material.texuniform = True
        return material.name

    def _add_refined_collisions(self, body: mujoco.MjsBody) -> None:
        if self.xml_filename == "round-nut.xml":
            segments = 32
            radius = 0.04245
            tube_radius = 0.01125
            points = tuple(
                (
                    radius * np.cos(2.0 * np.pi * index / segments),
                    radius * np.sin(2.0 * np.pi * index / segments),
                    0.0,
                )
                for index in range(segments)
            )
        else:
            tube_radius = 0.0105
            half = 0.03325
            points = (
                (-half, -half, 0.0),
                (half, -half, 0.0),
                (half, half, 0.0),
                (-half, half, 0.0),
            )

        for index, start in enumerate(points):
            self._add_capsule_collision(
                body,
                f"{self.name}_ring_collision_{index}",
                start,
                points[(index + 1) % len(points)],
                radius=tube_radius,
            )

        handle_start = (0.040, 0.0, 0.0) if len(points) == 32 else (0.030, 0.0, 0.0)
        self._add_capsule_collision(
            body,
            f"{self.name}_handle_collision",
            handle_start,
            (0.084, 0.0, 0.0),
            radius=0.0095,
        )
        pad = body.add_geom()
        pad.name = f"{self.name}_grip_pad_collision"
        pad.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        pad.pos = [0.084, 0.0, 0.0]
        pad.size = [0.014, 0.019, 0.010]
        pad.density = 100.0
        pad.friction = [0.95, 0.3, 0.1]
        pad.condim = 4
        pad.contype = 1
        pad.conaffinity = 1
        pad.rgba = [0.0, 0.0, 0.0, 0.0]

    def _add_refined_visuals(self, body: mujoco.MjsBody, material: str) -> None:
        if self.xml_filename == "round-nut.xml":
            self._add_round_ring_visual(body, material)
            handle_start = (0.040, 0.0, 0.0)
            handle_end = (0.085, 0.0, 0.0)
        else:
            self._add_square_ring_visual(body, material)
            handle_start = (0.030, 0.0, 0.0)
            handle_end = (0.082, 0.0, 0.0)
        self._add_capsule_visual(
            body,
            f"{self.name}_handle_visual",
            handle_start,
            handle_end,
            radius=0.0095,
            material=material,
        )
        # A shallow grip pad makes the handle look manufactured rather than
        # like another collision bar.
        pad = body.add_geom()
        pad.name = f"{self.name}_grip_pad_visual"
        pad.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        pad.pos = [0.084, 0.0, 0.0]
        pad.size = [0.014, 0.019, 0.010]
        pad.material = material
        pad.contype = 0
        pad.conaffinity = 0
        pad.density = 0.0

    def _add_round_ring_visual(self, body: mujoco.MjsBody, material: str) -> None:
        segments = 32
        radius = 0.043
        for index in range(segments):
            first = 2.0 * np.pi * index / segments
            second = 2.0 * np.pi * (index + 1) / segments
            self._add_capsule_visual(
                body,
                f"{self.name}_ring_visual_{index}",
                (radius * np.cos(first), radius * np.sin(first), 0.0),
                (radius * np.cos(second), radius * np.sin(second), 0.0),
                radius=0.0095,
                material=material,
            )

    def _add_square_ring_visual(self, body: mujoco.MjsBody, material: str) -> None:
        half = 0.033
        corners = (
            (-half, -half, 0.0),
            (half, -half, 0.0),
            (half, half, 0.0),
            (-half, half, 0.0),
        )
        for index, first in enumerate(corners):
            self._add_capsule_visual(
                body,
                f"{self.name}_ring_visual_{index}",
                first,
                corners[(index + 1) % len(corners)],
                radius=0.0095,
                material=material,
            )

    @staticmethod
    def _add_capsule_collision(
        body: mujoco.MjsBody,
        name: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        *,
        radius: float,
    ) -> None:
        geom = body.add_geom()
        geom.name = name
        geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        geom.fromto = [*start, *end]
        geom.size = [radius, 0.0, 0.0]
        geom.density = 100.0
        geom.friction = [0.95, 0.3, 0.1]
        geom.condim = 4
        geom.contype = 1
        geom.conaffinity = 1
        geom.rgba = [0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def _add_capsule_visual(
        body: mujoco.MjsBody,
        name: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        *,
        radius: float,
        material: str,
    ) -> None:
        geom = body.add_geom()
        geom.name = name
        geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        geom.fromto = [*start, *end]
        geom.size = [radius, 0.0, 0.0]
        geom.material = material
        geom.contype = 0
        geom.conaffinity = 0
        geom.density = 0.0
