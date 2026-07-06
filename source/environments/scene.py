# -*- coding: utf-8 -*-
"""Shared MuJoCo scene augmentation helpers."""

from __future__ import annotations

import mujoco


def add_preview_scene(spec: mujoco.MjSpec) -> None:
    """Add a textured ground plane and light for standalone model previews."""
    skybox_tex = spec.add_texture()
    skybox_tex.name = "skybox_tex"
    skybox_tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    skybox_tex.rgb1 = [0.3, 0.5, 0.7]
    skybox_tex.rgb2 = [0.0, 0.0, 0.0]
    skybox_tex.width = 512
    skybox_tex.height = 3072

    ground_tex = spec.add_texture()
    ground_tex.name = "groundplane_tex"
    ground_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    ground_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    ground_tex.rgb1 = [0.2, 0.3, 0.4]
    ground_tex.rgb2 = [0.1, 0.2, 0.3]
    ground_tex.width = 512
    ground_tex.height = 512

    ground_mat = spec.add_material()
    ground_mat.name = "groundplane"
    ground_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = ground_tex.name
    ground_mat.texrepeat = [5, 5]
    ground_mat.reflectance = 0.2
    ground_mat.shininess = 0.1
    ground_mat.specular = 0.1

    spec.worldbody.add_light(
        name="top_light",
        pos=[0.0, 0.0, 4.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[2.0, 2.0, 2.0],
        ambient=[0.8, 0.8, 0.8],
        specular=[0.3, 0.3, 0.3],
    )

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.material = ground_mat.name


def add_basic_scene(spec: mujoco.MjSpec) -> None:
    """Add a lightweight ground plane and light for RL environments."""
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.rgba = [0.25, 0.25, 0.25, 1.0]
    spec.worldbody.add_light(
        name="rl_top_light",
        pos=[0.0, 0.0, 4.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[1.5, 1.5, 1.5],
        ambient=[0.5, 0.5, 0.5],
    )
