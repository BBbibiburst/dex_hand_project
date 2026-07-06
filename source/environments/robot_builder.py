# -*- coding: utf-8 -*-
"""
Robot arm and dexterous hand model assembly tool.

This module loads the RM75B arm, base, and dexterous hand XML files, and mounts
the hand model under the specified body of the arm (default ``right_hand``).
The public API deliberately separates ``MjSpec`` construction from model
compilation, so the caller can continue adding cameras, objects, task logic,
or sensors before calling ``spec.compile()``.
"""

import os
from pathlib import Path
import traceback
from typing import Optional, Tuple, Union

import mujoco
from mujoco import viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from source.environments.tactile_layout import write_augmented_dex_hand_xml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_ARM_PATH = PROJECT_ROOT / "assets" / "robots" / "rm75b" / "rm75b.xml"
DEFAULT_HAND_PATH = PROJECT_ROOT / "assets" / "grippers" / "dex_hand" / "dex_hand.xml"
DEFAULT_BASE_PATH = PROJECT_ROOT / "assets" / "bases" / "rethink_minimal_mount.xml"

# Installation pose of the hand relative to the arm mount point, using xyz
# Euler angles in degrees.
DEFAULT_HAND_ROT_XYZ_DEG = (-90.0, -90.0, 0.0)

BASE_PREFIX = "mount_"
DEFAULT_HAND_PREFIX = "dexhand_"
DEFAULT_ATTACH_POINT_NAME = "right_hand"
DEFAULT_BASE_ARM_MOUNT_SITE_NAME = "arm_mount"

PathLike = Union[str, Path]
RotXyzDeg = Tuple[float, float, float]


def _resolve_path(path: Optional[PathLike], default_path: Path) -> Path:
    """Resolve an optional path; fall back to the default when None is passed."""
    return Path(path) if path is not None else default_path


def _load_spec_or_raise(path: Path, description: str) -> mujoco.MjSpec:
    """Load an XML file as ``MjSpec``, raising a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{description} XML file not found: {path}")
    return mujoco.MjSpec.from_file(str(path))


def _load_hand_spec_or_raise(path: Path, *, add_tactile: bool) -> mujoco.MjSpec:
    """Load the dex hand XML, optionally injecting generated touch sensors."""
    if not path.exists():
        raise FileNotFoundError(f"hand model XML file not found: {path}")
    if not add_tactile:
        return mujoco.MjSpec.from_file(str(path))

    augmented_path = write_augmented_dex_hand_xml(path)
    try:
        return mujoco.MjSpec.from_file(str(augmented_path))
    finally:
        try:
            os.unlink(augmented_path)
        except OSError:
            pass


def _first_body_or_raise(spec: mujoco.MjSpec, description: str) -> mujoco.MjsBody:
    """Return the first body under worldbody; raise with context if absent."""
    body = spec.worldbody.first_body()
    if body is None:
        raise ValueError(f"{description} XML has no body under <worldbody>.")
    return body


def _site_or_raise(
    spec: mujoco.MjSpec,
    site_name: str,
    description: str,
) -> mujoco.MjsSite:
    """Look up a site by name; list available sites on failure."""
    try:
        return spec.site(site_name)
    except KeyError as exc:
        available = [site.name for site in spec.sites()]
        raise ValueError(
            f"{description} XML has no site '{site_name}'. "
            f"Available sites: {available}"
        ) from exc


def _euler_deg_to_wxyz(rot_xyz_deg: RotXyzDeg) -> list:
    """Convert xyz Euler angles (degrees) to the wxyz quaternion used by MuJoCo."""
    x, y, z, w = R.from_euler("xyz", rot_xyz_deg, degrees=True).as_quat()
    return [w, x, y, z]


def _reset_body_pos(body: mujoco.MjsBody) -> None:
    """Zero out the root body offset so the parent attach frame handles placement."""
    if np.linalg.norm(np.asarray(body.pos, dtype=float)) > 1e-6:
        body.pos = [0.0, 0.0, 0.0]


def _mount_arm_on_base(
    arm_spec: mujoco.MjSpec,
    base_path: Path,
    mount_site_name: str,
) -> None:
    """Attach the base to worldbody and place the arm root at the base mount site."""
    base_spec = _load_spec_or_raise(base_path, "base model")
    base_root = _first_body_or_raise(base_spec, "base model")
    mount_site = _site_or_raise(base_spec, mount_site_name, "base model")

    mount_frame = arm_spec.worldbody.add_frame()
    mount_frame.attach_body(base_root, prefix=BASE_PREFIX, suffix="")

    arm_root = arm_spec.worldbody.first_body()
    if arm_root is None:
        return

    arm_root.pos = list(mount_site.pos)
    arm_root.quat = list(mount_site.quat)


def _attach_hand_to_arm(
    arm_spec: mujoco.MjSpec,
    hand_root: mujoco.MjsBody,
    attach_point_name: str,
    rot_xyz_deg: RotXyzDeg,
    hand_prefix: str,
) -> None:
    """Attach the hand model root under the specified arm body via a rotated frame."""
    try:
        attach_point = arm_spec.body(attach_point_name)
    except KeyError as exc:
        available = [body.name for body in arm_spec.worldbody.bodies()]
        raise ValueError(
            f"Arm model has no mount body '{attach_point_name}'. "
            f"Available bodies: {available}"
        ) from exc

    attach_frame = attach_point.add_frame()
    attach_frame.pos = [0.0, 0.0, 0.0]
    attach_frame.quat = _euler_deg_to_wxyz(rot_xyz_deg)
    attach_frame.attach_body(hand_root, prefix=hand_prefix, suffix="")


def _configure_solver(spec: mujoco.MjSpec) -> None:
    """Set more stable solver parameters for the merged multi-joint model."""
    spec.option.timestep = 0.001
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 100


def _add_default_scene(spec: mujoco.MjSpec) -> None:
    """Add a simple skybox, ground plane, and lights for standalone preview."""
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


def build_combined_spec(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    base_path: Optional[PathLike] = None,
    rot_xyz_deg: RotXyzDeg = DEFAULT_HAND_ROT_XYZ_DEG,
    attach_point_name: str = DEFAULT_ATTACH_POINT_NAME,
    base_mount_site_name: str = DEFAULT_BASE_ARM_MOUNT_SITE_NAME,
    hand_prefix: str = DEFAULT_HAND_PREFIX,
    add_tactile_sensors: bool = True,
) -> mujoco.MjSpec:
    """
    Build an uncompiled arm + dexterous hand ``MjSpec``.

    Args:
        arm_path: Path to the arm XML; defaults to ``DEFAULT_ARM_PATH``.
        hand_path: Path to the hand XML; defaults to ``DEFAULT_HAND_PATH``.
        base_path: Path to the base XML; defaults to ``DEFAULT_BASE_PATH``.
        rot_xyz_deg: xyz Euler angles (degrees) of the hand relative to
            ``attach_point_name``.
        attach_point_name: Name of the arm body used to mount the hand model.
        base_mount_site_name: Site in the base XML that declares the arm root
            position and orientation.

    Returns:
        A merged but uncompiled ``MjSpec``, ready for further customization or
        direct compilation.
    """
    arm_path = _resolve_path(arm_path, DEFAULT_ARM_PATH)
    hand_path = _resolve_path(hand_path, DEFAULT_HAND_PATH)
    base_path = _resolve_path(base_path, DEFAULT_BASE_PATH)

    arm_spec = _load_spec_or_raise(arm_path, "arm model")
    hand_spec = _load_hand_spec_or_raise(hand_path, add_tactile=add_tactile_sensors)
    _configure_solver(arm_spec)

    _mount_arm_on_base(arm_spec, base_path, base_mount_site_name)

    hand_root = _first_body_or_raise(hand_spec, "hand model")
    _reset_body_pos(hand_root)
    _attach_hand_to_arm(
        arm_spec=arm_spec,
        hand_root=hand_root,
        attach_point_name=attach_point_name,
        rot_xyz_deg=rot_xyz_deg,
        hand_prefix=hand_prefix,
    )

    return arm_spec


def build_combined_model(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    base_path: Optional[PathLike] = None,
    rot_xyz_deg: Optional[RotXyzDeg] = None,
    attach_point_name: str = DEFAULT_ATTACH_POINT_NAME,
    base_mount_site_name: str = DEFAULT_BASE_ARM_MOUNT_SITE_NAME,
    hand_prefix: str = DEFAULT_HAND_PREFIX,
    add_scene: bool = True,
    add_tactile_sensors: bool = True,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Build, optionally add a preview scene, and compile the merged robot model.

    ``rot_xyz_deg=None`` means use ``DEFAULT_HAND_ROT_XYZ_DEG``, so the preview
    entry and the spec builder share the same default installation pose.
    """
    spec = build_combined_spec(
        arm_path=arm_path,
        hand_path=hand_path,
        base_path=base_path,
        rot_xyz_deg=DEFAULT_HAND_ROT_XYZ_DEG if rot_xyz_deg is None else rot_xyz_deg,
        attach_point_name=attach_point_name,
        base_mount_site_name=base_mount_site_name,
        hand_prefix=hand_prefix,
        add_tactile_sensors=add_tactile_sensors,
    )

    if add_scene:
        _add_default_scene(spec)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


if __name__ == "__main__":
    print("--- Standalone preview: RM75B + Dexterous Hand ---")
    try:
        model, data = build_combined_model()

        with viewer.launch_passive(model, data) as v:
            while v.is_running():
                mujoco.mj_step(model, data)
                v.sync()

    except FileNotFoundError as e:
        print(f"\n[Error] Missing file: {e}")
    except Exception as e:
        print(f"\n[Error] Unexpected exception: {e}")
        traceback.print_exc()
