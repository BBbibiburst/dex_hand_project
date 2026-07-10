# -*- coding: utf-8 -*-
"""Composable task-object specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import mujoco
import numpy as np

from source.assets import asset_path


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
        joint.name = self.joint_name
        joint.type = mujoco.mjtJoint.mjJNT_FREE
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
        joint.name = self.joint_name
        joint.type = mujoco.mjtJoint.mjJNT_FREE
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
        joint.name = self.joint_name
        joint.type = mujoco.mjtJoint.mjJNT_FREE
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
    """Free nut loaded from a project-local copy of robosuite's MJCF asset."""

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
        count = 9 if self.xml_filename == "round-nut.xml" else 5
        return tuple(f"{self.name}_geom_{index}" for index in range(count))

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
        xml_path = asset_path("objects", self.xml_filename)
        if not xml_path.exists():
            raise FileNotFoundError(f"Nut object XML not found: {xml_path}")

        object_spec = mujoco.MjSpec.from_file(str(xml_path))
        wrapper = object_spec.worldbody.first_body()
        if wrapper is None or wrapper.first_body() is None:
            raise ValueError(f"Nut XML has no nested object body: {xml_path}")
        object_body = wrapper.first_body()
        object_body.name = self.body_name
        object_body.pos = np.asarray(initial_pos, dtype=float).tolist()

        joint = object_body.add_joint()
        joint.name = self.joint_name
        joint.type = mujoco.mjtJoint.mjJNT_FREE
        for geom, name in zip(object_body.geoms, self.geom_names):
            geom.name = name
        for site in object_body.sites:
            if site.name == "handle_site":
                site.name = self.handle_site_name
            elif site.name:
                site.name = f"{self.name}_{site.name}"
        frame = spec.worldbody.add_frame()
        # Attach the actual object body at world level. MuJoCo only permits a
        # free joint on a top-level body; the XML's unnamed wrapper only owns
        # offset metadata sites used by robosuite's Python object wrapper.
        frame.attach_body(object_body, prefix="", suffix="")
