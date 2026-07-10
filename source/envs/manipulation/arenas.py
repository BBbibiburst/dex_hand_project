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
    table_offset: Tuple[float, float, float] = (0.55, 0.0, 0.8)
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
    """Table arena with a source bin and four target compartments."""

    bin_half_size: Tuple[float, float, float] = (0.18, 0.18, 0.06)
    source_center: Tuple[float, float] = (0.46, -0.20)
    target_center: Tuple[float, float] = (0.46, 0.20)

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        super().augment_spec(spec)
        for prefix, center in (
            ("source_bin", self.source_center),
            ("target_bin", self.target_center),
        ):
            body = spec.worldbody.add_body()
            body.name = prefix
            body.pos = [center[0], center[1], self.table_top_z]
            hx, hy, hz = self.bin_half_size
            for name, pos, size in (
                ("bottom", (0, 0, 0.005), (hx, hy, 0.005)),
                ("left", (0, -hy, hz), (hx, 0.005, hz)),
                ("right", (0, hy, hz), (hx, 0.005, hz)),
                ("front", (-hx, 0, hz), (0.005, hy, hz)),
                ("back", (hx, 0, hz), (0.005, hy, hz)),
            ):
                geom = body.add_geom()
                geom.name = f"{prefix}_{name}"
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.pos, geom.size = list(pos), list(size)
                geom.rgba = [0.25, 0.35, 0.45, 1.0]
            if prefix == "target_bin":
                for axis, pos, size in (
                    ("x", (0, 0, hz), (0.005, hy, hz)),
                    ("y", (0, 0, hz), (hx, 0.005, hz)),
                ):
                    geom = body.add_geom()
                    geom.name = f"target_divider_{axis}"
                    geom.type = mujoco.mjtGeom.mjGEOM_BOX
                    geom.pos, geom.size = list(pos), list(size)
                    geom.rgba = [0.25, 0.35, 0.45, 1.0]


@dataclass(frozen=True)
class PegsArena(TableArena):
    """Table arena with one square and one round peg."""

    peg_centers: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.50, -0.12), (0.50, 0.12))
    peg_radius: float = 0.012
    peg_half_height: float = 0.08

    def augment_spec(self, spec: mujoco.MjSpec) -> None:
        super().augment_spec(spec)
        for index, center in enumerate(self.peg_centers):
            body = spec.worldbody.add_body()
            body.name = f"peg{index + 1}_body"
            body.pos = [center[0], center[1], self.table_top_z + self.peg_half_height]
            geom = body.add_geom()
            geom.name = f"peg{index + 1}_geom"
            if index == 0:
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.size = [
                    self.peg_radius,
                    self.peg_radius,
                    self.peg_half_height,
                ]
            else:
                geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                geom.size = [self.peg_radius, self.peg_half_height, 0.0]
            geom.rgba = [0.25, 0.45, 0.85, 1.0] if index == 0 else [0.85, 0.75, 0.15, 1.0]
