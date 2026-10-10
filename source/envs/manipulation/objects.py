"""Composable task-object specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from source.assets import asset_path
from source.envs.manipulation.object_catalog import (
    DEFAULT_MAX_HORIZONTAL_DIAMETER,
    limited_mesh_scale,
    resolve_record,
    resolve_record_path,
)
from source.collision_decomposition import convex_decomposition_paths

_OBJECT_CONTACT_SOLREF = (0.001, 2.0)
_OBJECT_CONTACT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)


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


def _configure_object_collision(
    geom: mujoco.MjsGeom,
    *,
    friction: tuple[float, float, float],
    condim: int,
) -> None:
    """Set object contact parameters without inheriting robot MJCF defaults.

    Task objects are appended to the already-loaded arm ``MjSpec``. Without
    explicit values they inherit the arm's default geom class, which silently
    couples YCB contact behaviour to whichever robot happens to host the task.
    Objects remain rigid; the higher-priority hand pad material supplies the
    compliant side of hand-object contacts.
    """
    geom.condim = condim
    geom.friction = list(friction)
    geom.solref = list(_OBJECT_CONTACT_SOLREF)
    geom.solimp = list(_OBJECT_CONTACT_SOLIMP)
    geom.priority = 0
    geom.solmix = 1.0
    geom.contype = 1
    geom.conaffinity = 1


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
    """Free catalogue mesh with separate visual and collision geoms."""

    name: str
    object_id: str
    # None preserves physical scale. A positive value remains available for
    # explicitly size-normalised ablations and stacking tasks.
    target_size: float | None = None
    maximum_horizontal_diameter: float | None = DEFAULT_MAX_HORIZONTAL_DIAMETER
    density: float = 500.0
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)

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
        scale = limited_mesh_scale(
            record,
            extent,
            target_size=self.target_size,
            maximum_horizontal_diameter=self.maximum_horizontal_diameter,
        )
        object.__setattr__(self, "_visual_path", visual)
        collision_source = next(
            (root / item for item in model_files if Path(item).name == "collision.ply"),
            visual,
        )
        object.__setattr__(self, "_collision_paths", convex_decomposition_paths(collision_source))
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
        return tuple(
            f"{self.name}_collision_{index:03d}" for index in range(len(self._collision_paths))
        )

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

        collision_mesh_names = []
        for index, path in enumerate(self._collision_paths):
            collision_mesh = spec.add_mesh()
            collision_mesh.name = f"{prefix}_collision_mesh_{index:03d}"
            collision_mesh.file = str(path.resolve())
            collision_mesh.scale = [self._scale] * 3
            collision_mesh.refpos = self._center.tolist()
            collision_mesh_names.append(collision_mesh.name)

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

        for name, mesh_name in zip(self.geom_names, collision_mesh_names, strict=True):
            collision = body.add_geom()
            collision.name = name
            collision.type = mujoco.mjtGeom.mjGEOM_MESH
            collision.meshname = mesh_name
            collision.density = self.density
            _configure_object_collision(collision, friction=self.friction, condim=4)
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
    half_size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    density: float = 500.0
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    duplicate_collision_geoms: bool = True
    fixed_to_world: bool = False

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
        _configure_object_collision(geom, friction=self.friction, condim=3)
        geom.rgba = list(self.rgba)
        if self.duplicate_collision_geoms:
            visual = body.add_geom()
            visual.name = f"{self.name}_visual"
            visual.type = mujoco.mjtGeom.mjGEOM_BOX
            visual.size = list(self.half_size)
            visual.contype = 0
            visual.conaffinity = 0
            visual.density = 0.0
            visual.rgba = list(self.rgba)
        if self.fixed_to_world:
            # Retain a free joint so the object continues to satisfy the
            # common task-object binding contract, while a physical weld
            # makes the support an immovable part of the scene.
            weld = spec.add_equality()
            weld.name = f"{self.name}_world_weld"
            weld.type = mujoco.mjtEq.mjEQ_WELD
            weld.objtype = mujoco.mjtObj.mjOBJ_BODY
            weld.name1 = self.body_name
            weld.name2 = ""


@dataclass(frozen=True)
class FreeCylinderSpec(ManipulationObjectSpec):
    """Free upright cylinder used by can-like task objects."""

    name: str
    radius: float
    half_height: float
    rgba: tuple[float, float, float, float]
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
        _configure_object_collision(
            geom,
            friction=(1.0, 0.005, 0.0001),
            condim=4,
        )


@dataclass(frozen=True)
class FreeBoltSpec(ManipulationObjectSpec):
    """Free upright bolt with a separate handle, stop shoulder, and mating shaft."""

    name: str
    polygon_sides: int
    shaft_radius: float
    # Human-hand-scale tool geometry. The original benchmark used a 68 mm
    # diameter, 110 mm tall handle on a 74 mm shaft, which made the complete
    # part almost 190 mm tall. These dimensions keep it readily graspable by
    # the dexterous hand without making it look like an industrial fixture.
    shaft_half_height: float = 0.0245
    handle_radius: float = 0.024
    handle_half_height: float = 0.036
    handle_center_z: float = 0.040
    shoulder_center_z: float = 0.0
    shoulder_half_height: float = 0.004
    rgba: tuple[float, float, float, float] = (0.72, 0.75, 0.80, 1.0)

    def __post_init__(self) -> None:
        if self.polygon_sides != 0 and self.polygon_sides < 3:
            raise ValueError("polygon_sides must be zero or at least three.")
        if (
            min(
                self.shaft_radius,
                self.shaft_half_height,
                self.handle_radius,
                self.handle_half_height,
            )
            <= 0.0
        ):
            raise ValueError("Bolt dimensions must be positive.")

    @property
    def body_name(self) -> str:
        return f"{self.name}_body"

    @property
    def joint_name(self) -> str:
        return f"{self.name}_freejoint"

    @property
    def geom_names(self) -> tuple[str, ...]:
        return (
            f"{self.name}_shaft",
            f"{self.name}_handle",
            f"{self.name}_shoulder",
        )

    @property
    def horizontal_radius(self) -> float:
        return max(self.handle_radius, self.shoulder_radius, self.shaft_radius)

    @property
    def shoulder_radius(self) -> float:
        return max(0.030, self.shaft_radius + 0.006)

    @property
    def bottom_offset(self) -> float:
        return 2.0 * self.shaft_half_height

    def add_to_spec(self, spec: mujoco.MjSpec, initial_pos: np.ndarray) -> None:
        body = spec.worldbody.add_body()
        body.name = self.body_name
        body.pos = np.asarray(initial_pos, dtype=float).tolist()
        joint = body.add_joint()
        _configure_free_joint(joint, self.joint_name)
        shaft = body.add_geom()
        shaft.name = self.geom_names[0]
        shaft.pos = [0.0, 0.0, -self.shaft_half_height]
        shaft.type = mujoco.mjtGeom.mjGEOM_MESH
        shaft.meshname = self._add_shaft_mesh(spec)
        shaft.density = 150.0
        shaft.rgba = list(self.rgba)
        _configure_object_collision(
            shaft,
            friction=(1.2, 0.01, 0.001),
            condim=4,
        )
        handle = body.add_geom()
        handle.name = self.geom_names[1]
        handle.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        handle.pos = [0.0, 0.0, self.handle_center_z]
        handle.size = [self.handle_radius, self.handle_half_height, 0.0]
        handle.density = 150.0
        handle.rgba = [0.30, 0.34, 0.42, 1.0]
        _configure_object_collision(
            handle,
            friction=(3.0, 1.0, 0.2),
            condim=4,
        )
        shoulder = body.add_geom()
        shoulder.name = self.geom_names[2]
        shoulder.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        shoulder.pos = [0.0, 0.0, self.shoulder_center_z]
        shoulder.size = [self.shoulder_radius, self.shoulder_half_height, 0.0]
        shoulder.density = 150.0
        shoulder.rgba = [0.48, 0.51, 0.58, 1.0]
        _configure_object_collision(
            shoulder,
            friction=(2.0, 0.1, 0.01),
            condim=4,
        )

    def _add_shaft_mesh(self, spec: mujoco.MjSpec) -> str:
        sides = 24 if self.polygon_sides == 0 else self.polygon_sides
        phase = np.pi / 4.0 if sides == 4 else 0.0
        angles = [phase + 2.0 * np.pi * i / sides for i in range(sides)]
        # The shaft is full-size over nearly its entire insertion length. Only
        # the terminal 5 mm is chamfered to help initial alignment; the old
        # full-length taper created a visually tiny, mechanically irrelevant
        # lower pin while the oversized handle accidentally became the fit.
        chamfer = min(0.005, 0.5 * self.shaft_half_height)
        rings = (
            (-self.shaft_half_height, 0.75),
            (-self.shaft_half_height + chamfer, 1.0),
            (self.shaft_half_height, 1.0),
        )
        vertices = [
            (
                scale * self.shaft_radius * np.cos(angle),
                scale * self.shaft_radius * np.sin(angle),
                z,
            )
            for z, scale in rings
            for angle in angles
        ]
        faces: list[tuple[int, int, int]] = []
        for index in range(1, sides - 1):
            faces.extend(
                (
                    (0, index + 1, index),
                    (2 * sides, 2 * sides + index, 2 * sides + index + 1),
                )
            )
        for ring in range(len(rings) - 1):
            lower = ring * sides
            upper = (ring + 1) * sides
            for index in range(sides):
                following = (index + 1) % sides
                faces.extend(
                    (
                        (lower + index, lower + following, upper + following),
                        (lower + index, upper + following, upper + index),
                    )
                )
        mesh = spec.add_mesh()
        mesh.name = f"{self.name}_shaft_mesh"
        mesh.uservert = np.asarray(vertices, dtype=np.float32).reshape(-1).tolist()
        mesh.userface = np.asarray(faces, dtype=np.int32).reshape(-1).tolist()
        return mesh.name


@dataclass(frozen=True)
class FreeNutSpec(ManipulationObjectSpec):
    """Free four-bar nut with a physical central opening."""

    name: str
    outer_radius: float
    inner_radius: float
    half_height: float
    rgba: tuple[float, float, float, float]

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
            _configure_object_collision(
                geom,
                friction=(1.0, 0.005, 0.0001),
                condim=4,
            )


@dataclass(frozen=True)
class XmlNutSpec(ManipulationObjectSpec):
    """Procedurally constructed nut retained under its compatible public name."""

    name: str
    xml_filename: str
    radius: float = 0.11
    bottom: float = 0.05
    polygon_sides: int | None = None
    ring_radius: float = 0.0425
    tube_radius: float = 0.0105

    @property
    def ring_segments(self) -> int:
        if self.polygon_sides is not None:
            return 32 if self.polygon_sides == 0 else self.polygon_sides
        return 32 if self.xml_filename == "round-nut.xml" else 4

    @property
    def body_name(self) -> str:
        return f"{self.name}_body"

    @property
    def joint_name(self) -> str:
        return f"{self.name}_freejoint"

    @property
    def geom_names(self) -> tuple[str, ...]:
        ring_count = self.ring_segments
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
        if self.xml_filename not in {"square-nut.xml", "round-nut.xml", "polygon-nut.xml"}:
            raise ValueError(f"Unknown nut design: {self.xml_filename!r}")
        if self.ring_segments < 3:
            raise ValueError("A polygon nut must have at least three sides.")
        if self.ring_radius <= self.tube_radius:
            raise ValueError("ring_radius must exceed tube_radius.")
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
        round_nut = self.ring_segments == 32
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
        segments = self.ring_segments
        # Square keeps a face horizontal like the original asset. Other
        # regular polygons place a vertex on +x; 32 sides is the round case.
        phase = np.pi / 4.0 if segments == 4 else 0.0
        points = tuple(
            (
                self.ring_radius * np.cos(phase + 2.0 * np.pi * index / segments),
                self.ring_radius * np.sin(phase + 2.0 * np.pi * index / segments),
                0.0,
            )
            for index in range(segments)
        )

        for index, start in enumerate(points):
            self._add_capsule_collision(
                body,
                f"{self.name}_ring_collision_{index}",
                start,
                points[(index + 1) % len(points)],
                radius=self.tube_radius,
            )

        handle_start = (self.ring_radius * 0.82, 0.0, 0.0)
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
        _configure_object_collision(
            pad,
            friction=(3.0, 1.0, 0.2),
            condim=4,
        )
        pad.rgba = [0.0, 0.0, 0.0, 0.0]

    def _add_refined_visuals(self, body: mujoco.MjsBody, material: str) -> None:
        self._add_polygon_ring_visual(body, material)
        handle_start = (self.ring_radius * 0.82, 0.0, 0.0)
        handle_end = (0.085, 0.0, 0.0)
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

    def _add_polygon_ring_visual(self, body: mujoco.MjsBody, material: str) -> None:
        segments = self.ring_segments
        phase = np.pi / 4.0 if segments == 4 else 0.0
        points = tuple(
            (
                self.ring_radius * np.cos(phase + 2.0 * np.pi * index / segments),
                self.ring_radius * np.sin(phase + 2.0 * np.pi * index / segments),
                0.0,
            )
            for index in range(segments)
        )
        for index, first in enumerate(points):
            self._add_capsule_visual(
                body,
                f"{self.name}_ring_visual_{index}",
                first,
                points[(index + 1) % segments],
                radius=max(self.tube_radius - 0.001, 0.004),
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
        _configure_object_collision(
            geom,
            friction=(3.0, 1.0, 0.2),
            condim=4,
        )
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
