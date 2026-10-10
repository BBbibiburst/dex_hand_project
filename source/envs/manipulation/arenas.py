# -*- coding: utf-8 -*-
"""XML-backed arena helpers for manipulation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import mujoco
import numpy as np

from source.assets import asset_path


@dataclass(frozen=True)
class TableArena:
    """robosuite XML-backed table arena for single-arm manipulation tasks.

    The arena is loaded from the copied robosuite XML template under
    ``assets/arenas``. Like robosuite's Python arena class, this wrapper then
    applies the requested table size and table-top offset to the loaded table
    body. Static world geometry, cameras, lights, walls, skybox, and materials
    come from the XML template instead of being recreated by hand.
    """

    table_full_size: Tuple[float, float, float] = (0.8, 0.8, 0.05)
    # The table top is level with the default base's arm_mount site (z=0.5).
    table_offset: Tuple[float, float, float] = (0.55, 0.0, 0.5)
    table_friction: Tuple[float, float, float] = (1.0, 0.005, 0.0001)
    table_has_legs: bool = True
    arena_xml_path: Optional[Path] = None
    table_name: str = "table"
    table_geom_name: str = "table_collision"
    table_visual_name: str = "table_visual"
    table_top_site_name: str = "table_top"

    @property
    def table_half_size(self) -> np.ndarray:
        return 0.5 * np.asarray(self.table_full_size, dtype=np.float64)

    @property
    def table_top_pos(self) -> np.ndarray:
        return np.asarray(self.table_offset, dtype=np.float64)

    @property
    def table_top_z(self) -> float:
        return float(self.table_top_pos[2])

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        """Load the original arena XML and merge it into ``spec``."""
        arena_path = self.arena_xml_path or asset_path("arenas", "table_arena.xml")
        arena_spec = mujoco.MjSpec.from_file(str(arena_path))
        self._merge_arena_spec(spec, arena_spec, Path(arena_path).parent)
        self._configure_table_from_robosuite_template(spec)
        self._configure_agentview_camera(spec)

    def _configure_agentview_camera(self, spec: mujoco.MjSpec) -> None:
        """Aim agentview about 30 degrees down and 30 degrees off a side view."""
        camera = spec.camera("agentview")
        if camera is None:
            return
        target = self.table_top_pos + np.asarray([-0.10, 0.0, 0.20])
        # Stay primarily side-on so the palm / gripper opening remains visible,
        # while a 30-degree forward offset provides useful depth cues.
        horizontal = 1.15
        angle = np.deg2rad(30.0)
        position = target + np.asarray(
            [
                horizontal * np.sin(angle),
                -horizontal * np.cos(angle),
                horizontal * np.tan(angle),
            ]
        )
        view = target - position
        view /= np.linalg.norm(view)
        local_z = -view
        local_x = np.cross(np.asarray([0.0, 0.0, 1.0]), local_z)
        local_x /= np.linalg.norm(local_x)
        local_y = np.cross(local_z, local_x)
        rotation = np.column_stack((local_x, local_y, local_z))
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
        camera.pos = position.tolist()
        camera.quat = quaternion.tolist()
        camera.fovy = 55.0

    def _merge_arena_spec(
        self,
        target: mujoco.MjSpec,
        source: mujoco.MjSpec,
        source_dir: Path,
    ) -> None:
        for texture in source.textures:
            cloned = target.add_texture()
            self._copy_mjs_attributes(texture, cloned)
            if getattr(texture, "file", ""):
                cloned.file = str((source_dir / texture.file).resolve())

        for material in source.materials:
            cloned = target.add_material()
            self._copy_mjs_attributes(material, cloned)
            for role in (
                mujoco.mjtTextureRole.mjTEXROLE_RGB,
                mujoco.mjtTextureRole.mjTEXROLE_RGBA,
            ):
                try:
                    cloned.textures[role] = material.textures[role]
                except Exception:
                    pass

        for geom in source.worldbody.geoms:
            self._clone_geom(target.worldbody, geom)
        for body in source.worldbody.bodies:
            self._clone_body(target.worldbody, body)
        for light in source.worldbody.lights:
            self._clone_light(target.worldbody, light)
        for camera in source.worldbody.cameras:
            self._clone_camera(target.worldbody, camera)
        for site in source.worldbody.sites:
            self._clone_site(target.worldbody, site)

    def _clone_body(self, parent: mujoco.MjsBody, source_body: mujoco.MjsBody) -> mujoco.MjsBody:
        body = parent.add_body()
        self._copy_mjs_attributes(source_body, body)
        for geom in source_body.geoms:
            self._clone_geom(body, geom)
        for site in source_body.sites:
            self._clone_site(body, site)
        for light in source_body.lights:
            self._clone_light(body, light)
        for camera in source_body.cameras:
            self._clone_camera(body, camera)
        for child in source_body.bodies:
            self._clone_body(body, child)
        return body

    def _clone_geom(self, parent: mujoco.MjsBody, source_geom: mujoco.MjsGeom) -> None:
        geom = parent.add_geom()
        self._copy_mjs_attributes(source_geom, geom)

    def _clone_site(self, parent: mujoco.MjsBody, source_site: mujoco.MjsSite) -> None:
        site = parent.add_site()
        self._copy_mjs_attributes(source_site, site)

    def _clone_light(self, parent: mujoco.MjsBody, source_light: mujoco.MjsLight) -> None:
        light = parent.add_light()
        self._copy_mjs_attributes(source_light, light)

    def _clone_camera(self, parent: mujoco.MjsBody, source_camera: mujoco.MjsCamera) -> None:
        camera = parent.add_camera()
        self._copy_mjs_attributes(source_camera, camera)

    def _copy_mjs_attributes(self, source: Any, target: Any) -> None:
        skipped = {
            "bodies",
            "cameras",
            "frames",
            "geoms",
            "joints",
            "lights",
            "sites",
            "textures",
            "materials",
            "meshes",
        }
        for name in dir(source):
            if name.startswith("_") or name in skipped:
                continue
            try:
                value = getattr(source, name)
            except Exception:
                continue
            if callable(value):
                continue
            try:
                setattr(target, name, value)
            except Exception:
                pass

    def _configure_table_from_robosuite_template(self, spec: mujoco.MjSpec) -> None:
        table_half = self.table_half_size
        table_body = spec.body(self.table_name)
        table_body.pos = [
            float(self.table_top_pos[0]),
            float(self.table_top_pos[1]),
            float(self.table_top_pos[2] - table_half[2]),
        ]
        collision = spec.geom(self.table_geom_name)
        collision.size = table_half.tolist()
        collision.friction = list(self.table_friction)
        visual = spec.geom(self.table_visual_name)
        visual.size = table_half.tolist()
        site = spec.site(self.table_top_site_name)
        site.pos = [0.0, 0.0, float(table_half[2])]

        if self.table_has_legs:
            self._configure_table_legs(table_body, table_half)
        else:
            for leg in table_body.geoms:
                if leg.name.startswith("table_leg"):
                    leg.contype = 0
                    leg.conaffinity = 0
                    leg.rgba = [0.0, 0.0, 0.0, 0.0]

    def _configure_table_legs(
        self,
        table_body: mujoco.MjsBody,
        table_half: np.ndarray,
    ) -> None:
        leg_radius = 0.025
        leg_length = max(float(self.table_top_pos[2] - table_half[2]), 0.05)
        leg_z = -0.5 * leg_length - float(table_half[2])
        offsets = ((1, 1), (-1, 1), (-1, -1), (1, -1))
        for index, (x_sign, y_sign) in enumerate(offsets, 1):
            leg_name = f"table_leg{index}_visual"
            leg = next((geom for geom in table_body.geoms if geom.name == leg_name), None)
            if leg is None:
                leg = table_body.add_geom()
                leg.name = leg_name
            leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            leg.size = [leg_radius, 0.5 * leg_length, 0.0]
            leg.pos = [
                x_sign * max(float(table_half[0]) - 0.10, 0.0),
                y_sign * max(float(table_half[1]) - 0.10, 0.0),
                leg_z,
            ]
            leg.contype = 0
            leg.conaffinity = 0
            leg.rgba = [0.35, 0.35, 0.35, 1.0]
            leg.material = "table_legs_metal"


@dataclass(frozen=True)
class BinsArena(TableArena):
    """Table arena with painted source and target placement regions."""

    bin_half_size: Tuple[float, float, float] = (0.18, 0.10, 0.06)
    source_center: Tuple[float, float] = (0.46, -0.02)
    target_center: Tuple[float, float] = (0.46, 0.20)

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        super().augment_spec(spec)
        hx, hy, _ = self.bin_half_size
        for name, center, color in (
            ("source_region", self.source_center, (0.88, 0.55, 0.16, 0.32)),
            ("target_region", self.target_center, (0.12, 0.48, 0.88, 0.38)),
        ):
            # A site is a painted task annotation, not a rigid body: it adds no
            # mass, contact, height change, or imaginary obstacle to the scene.
            site = spec.worldbody.add_site()
            site.name = name
            site.type = mujoco.mjtGeom.mjGEOM_BOX
            site.pos = [center[0], center[1], self.table_top_z + 0.001]
            site.size = [hx, hy, 0.001]
            site.rgba = list(color)
            site.group = 2


@dataclass(frozen=True)
class PushArena(TableArena):
    """Flat table with a non-colliding circular planar-pushing target."""

    target_site_name: str = "push_target"
    target_radius: float = 0.075
    initial_target_xy: Tuple[float, float] = (0.58, 0.18)

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        super().augment_spec(spec)
        site = spec.worldbody.add_site()
        site.name = self.target_site_name
        site.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        site.pos = [
            float(self.initial_target_xy[0]),
            float(self.initial_target_xy[1]),
            self.table_top_z + 0.001,
        ]
        site.size = [self.target_radius, 0.0015, 0.0]
        site.rgba = [0.12, 0.72, 0.34, 0.55]
        site.group = 2


@dataclass(frozen=True)
class PegsArena(TableArena):
    """Table arena with one square and one round peg."""

    peg_centers: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.50, -0.12), (0.50, 0.12))
    square_peg_half_width: float = 0.018
    round_peg_radius: float = 0.025
    peg_half_height: float = 0.08

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        super().augment_spec(spec)
        for index, center in enumerate(self.peg_centers):
            cross_section = self.square_peg_half_width if index == 0 else self.round_peg_radius
            body = spec.worldbody.add_body()
            body.name = f"peg{index + 1}_body"
            body.pos = [center[0], center[1], self.table_top_z + self.peg_half_height]
            geom = body.add_geom()
            geom.name = f"peg{index + 1}_geom"
            if index == 0:
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.size = [
                    cross_section,
                    cross_section,
                    self.peg_half_height,
                ]
            else:
                geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                geom.size = [cross_section, self.peg_half_height, 0.0]
            geom.rgba = [0.0, 0.0, 0.0, 0.0]

            color = [0.20, 0.42, 0.82, 1.0] if index == 0 else [0.88, 0.68, 0.12, 1.0]
            visual = body.add_geom()
            visual.name = f"peg{index + 1}_visual"
            visual.type = geom.type
            visual.size = list(geom.size)
            visual.rgba = color
            visual.contype = 0
            visual.conaffinity = 0
            visual.density = 0.0

            collar = body.add_geom()
            collar.name = f"peg{index + 1}_collar_visual"
            collar.type = (
                mujoco.mjtGeom.mjGEOM_BOX if index == 0 else mujoco.mjtGeom.mjGEOM_CYLINDER
            )
            collar.pos = [0.0, 0.0, -self.peg_half_height + 0.006]
            collar.size = (
                [cross_section * 1.35, cross_section * 1.35, 0.006]
                if index == 0
                else [cross_section * 1.35, 0.006, 0.0]
            )
            collar.rgba = [color[0] * 0.75, color[1] * 0.75, color[2] * 0.75, 1.0]
            collar.contype = 0
            collar.conaffinity = 0
            collar.density = 0.0

            cap = body.add_geom()
            cap.name = f"peg{index + 1}_cap_visual"
            cap.type = mujoco.mjtGeom.mjGEOM_BOX if index == 0 else mujoco.mjtGeom.mjGEOM_CYLINDER
            cap.pos = [0.0, 0.0, self.peg_half_height - 0.003]
            cap.size = (
                [cross_section * 1.05, cross_section * 1.05, 0.003]
                if index == 0
                else [cross_section * 1.05, 0.003, 0.0]
            )
            cap.rgba = [min(value * 1.18, 1.0) for value in color[:3]] + [1.0]
            cap.contype = 0
            cap.conaffinity = 0
            cap.density = 0.0


@dataclass(frozen=True)
class ProceduralPegArena(TableArena):
    """One round or regular-polygon peg used by the 100-case nut benchmark."""

    peg_center: Tuple[float, float] = (0.50, 0.10)
    polygon_sides: int = 4
    peg_radius: float = 0.022
    peg_half_height: float = 0.065

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        super().augment_spec(spec)
        if self.polygon_sides != 0 and self.polygon_sides < 3:
            raise ValueError("polygon_sides must be zero (round) or at least three.")
        body = spec.worldbody.add_body()
        body.name = "procedural_peg_body"
        body.pos = [
            self.peg_center[0],
            self.peg_center[1],
            self.table_top_z + self.peg_half_height,
        ]
        geom = body.add_geom()
        geom.name = "procedural_peg_geom"
        if self.polygon_sides == 0:
            geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            geom.size = [self.peg_radius, self.peg_half_height, 0.0]
        else:
            mesh_name = self._add_prism_mesh(spec)
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = mesh_name
        geom.rgba = [0.0, 0.0, 0.0, 0.0]
        geom.friction = [0.9, 0.01, 0.001]
        visual = body.add_geom()
        visual.name = "procedural_peg_visual"
        visual.type = geom.type
        if self.polygon_sides == 0:
            visual.size = list(geom.size)
        else:
            visual.meshname = geom.meshname
        visual.rgba = [0.20, 0.48, 0.82, 1.0]
        visual.contype = 0
        visual.conaffinity = 0
        visual.density = 0.0

    def _add_prism_mesh(self, spec: mujoco.MjSpec) -> str:
        sides = self.polygon_sides
        phase = np.pi / 4.0 if sides == 4 else 0.0
        ring = [
            (
                self.peg_radius * np.cos(phase + 2.0 * np.pi * i / sides),
                self.peg_radius * np.sin(phase + 2.0 * np.pi * i / sides),
            )
            for i in range(sides)
        ]
        vertices = [
            (x, y, z) for z in (-self.peg_half_height, self.peg_half_height) for x, y in ring
        ]
        faces: list[tuple[int, int, int]] = []
        for i in range(1, sides - 1):
            faces.append((0, i + 1, i))
            faces.append((sides, sides + i, sides + i + 1))
        for i in range(sides):
            j = (i + 1) % sides
            faces.extend(((i, j, sides + j), (i, sides + j, sides + i)))
        mesh = spec.add_mesh()
        mesh.name = "procedural_peg_mesh"
        mesh.uservert = np.asarray(vertices, dtype=np.float32).reshape(-1).tolist()
        mesh.userface = np.asarray(faces, dtype=np.int32).reshape(-1).tolist()
        return mesh.name


@dataclass(frozen=True)
class FixedNutArena(TableArena):
    """Table-mounted polygon socket used by the bolt-insertion benchmark."""

    nut_center: Tuple[float, float] = (0.50, 0.08)
    polygon_sides: int = 4
    inner_apothem: float = 0.020
    wall_thickness: float = 0.0045
    wall_half_height: float = 0.025
    guide_height: float = 0.022
    guide_expansion: float = 0.007

    @property
    def environment_geom_names(self) -> tuple[str, ...]:
        segments = 24 if self.polygon_sides == 0 else self.polygon_sides
        return tuple(
            name
            for index in range(segments)
            for name in (
                f"fixed_nut_wall_{index:02d}",
                f"fixed_nut_guide_{index:02d}",
            )
        )

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        super().augment_spec(spec)
        if self.polygon_sides != 0 and self.polygon_sides < 3:
            raise ValueError("polygon_sides must be zero or at least three.")
        segments = 24 if self.polygon_sides == 0 else self.polygon_sides
        phase = np.pi / 4.0 if segments == 4 else 0.0
        centerline_apothem = self.inner_apothem + self.wall_thickness
        radius = centerline_apothem / np.cos(np.pi / segments)
        points = tuple(
            np.asarray(
                [
                    radius * np.cos(phase + 2.0 * np.pi * index / segments),
                    radius * np.sin(phase + 2.0 * np.pi * index / segments),
                ],
                dtype=np.float64,
            )
            for index in range(segments)
        )
        body = spec.worldbody.add_body()
        body.name = "fixed_nut_body"
        body.pos = [
            self.nut_center[0],
            self.nut_center[1],
            self.table_top_z + self.wall_half_height,
        ]
        for index, start in enumerate(points):
            end = points[(index + 1) % segments]
            edge = end - start
            midpoint = 0.5 * (start + end)
            yaw = float(np.arctan2(edge[1], edge[0]))
            lower_height = 2.0 * self.wall_half_height - self.guide_height
            geom = body.add_geom()
            geom.name = f"fixed_nut_wall_{index:02d}"
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.pos = [
                float(midpoint[0]),
                float(midpoint[1]),
                -self.wall_half_height + 0.5 * lower_height,
            ]
            geom.quat = [float(np.cos(0.5 * yaw)), 0.0, 0.0, float(np.sin(0.5 * yaw))]
            geom.size = [
                0.5 * float(np.linalg.norm(edge)),
                self.wall_thickness,
                0.5 * lower_height,
            ]
            geom.rgba = [0.66, 0.48, 0.20, 1.0]
            geom.friction = [1.2, 0.01, 0.001]

            # A physical entry chamfer guides a slightly tilted shaft into the
            # true-size lower socket. The underactuated hand supplies passive
            # compliance; no object pose is teleported or numerically snapped.
            radial = midpoint / np.linalg.norm(midpoint)
            guide_midpoint = midpoint + 0.5 * self.guide_expansion * radial
            half_vertical = 0.5 * self.guide_height
            tilt = float(np.arctan2(0.5 * self.guide_expansion, half_vertical))
            cz, sz = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
            ct, st = np.cos(0.5 * tilt), np.sin(0.5 * tilt)
            guide = body.add_geom()
            guide.name = f"fixed_nut_guide_{index:02d}"
            guide.type = mujoco.mjtGeom.mjGEOM_BOX
            guide.pos = [
                float(guide_midpoint[0]),
                float(guide_midpoint[1]),
                self.wall_half_height - half_vertical,
            ]
            guide.quat = [
                float(cz * ct),
                float(cz * st),
                float(sz * st),
                float(sz * ct),
            ]
            guide.size = [
                0.5
                * float(np.linalg.norm(edge))
                * (centerline_apothem + 0.5 * self.guide_expansion)
                / centerline_apothem,
                self.wall_thickness,
                float(np.hypot(half_vertical, 0.5 * self.guide_expansion)),
            ]
            guide.rgba = [0.76, 0.57, 0.27, 1.0]
            guide.friction = [0.8, 0.005, 0.0005]
