# -*- coding: utf-8 -*-
"""Shared MuJoCo scene augmentation helpers."""

from __future__ import annotations

import mujoco

from source.assets import asset_path


def _add_skybox(spec: mujoco.MjSpec, *, name: str) -> None:
    skybox_tex = spec.add_texture()
    skybox_tex.name = name
    skybox_tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    skybox_tex.rgb1 = [0.55, 0.65, 0.75]
    skybox_tex.rgb2 = [0.08, 0.10, 0.14]
    skybox_tex.width = 512
    skybox_tex.height = 3072


def add_preview_scene(spec: mujoco.MjSpec) -> None:
    """Add a textured ground plane and light for standalone model previews."""
    _add_skybox(spec, name="skybox_tex")

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
    """Add a visually distinct fallback scene for task-free RL environments."""
    _add_skybox(spec, name="rl_skybox_tex")

    floor_texture = spec.add_texture()
    floor_texture.name = "rl_floor_tex"
    floor_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_texture.file = str(asset_path("textures", "light-gray-floor-tile.png").resolve())

    floor_material = spec.add_material()
    floor_material.name = "rl_floor_mat"
    floor_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = floor_texture.name
    floor_material.texrepeat = [4.0, 4.0]
    floor_material.texuniform = True
    floor_material.reflectance = 0.02
    floor_material.shininess = 0.05
    floor_material.specular = 0.05

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.material = floor_material.name
    spec.worldbody.add_light(
        name="rl_top_light",
        pos=[0.0, 0.0, 4.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[1.5, 1.5, 1.5],
        ambient=[0.5, 0.5, 0.5],
    )
    spec.worldbody.add_camera(
        name="agentview",
        mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
        pos=[1.25, 0.0, 1.35],
        quat=[0.653, 0.271, 0.271, 0.653],
    )
