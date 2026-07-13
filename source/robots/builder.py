# -*- coding: utf-8 -*-
"""Descriptor-driven robot assembly utilities.

The public API separates ``MjSpec`` construction from model compilation, so
the caller can add cameras, objects, task logic, or sensors before
``spec.compile()``.

Tactile (or any other) sensing is wired in via ``TactileSensorBase.augment_spec``
directly on the loaded ``MjSpec`` object — there is no XML text rewriting or
temporary file involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from source.assets import PathLike, resolve_path
from source.robots.config import (
    apply_config_overrides,
    descriptors_from_robot_config,
    load_robot_config,
    optional_tuple,
)
from source.robots.defaults import DEFAULT_ARM, DEFAULT_BASE, DEFAULT_HAND
from source.robots.descriptors import ArmDescriptor, BaseDescriptor, EndEffectorDescriptor
from source.robots.scene import add_preview_scene
from source.sensors.base import TactileSensorBase

RotXyzDeg = Tuple[float, float, float]


def _items(value):
    """Return MuJoCo MjSpec child collections across API variants."""
    return value() if callable(value) else value


def _load_spec_or_raise(path: Path, description: str) -> mujoco.MjSpec:
    """Load an XML file as ``MjSpec``, raising a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{description} XML file not found: {path}")
    return mujoco.MjSpec.from_file(str(path))


def _first_body_or_raise(spec: mujoco.MjSpec, description: str) -> mujoco.MjsBody:
    """Return the first body under worldbody; raise with context if absent."""
    body = spec.worldbody.first_body()
    if body is None:
        raise ValueError(f"{description} XML has no body under <worldbody>.")
    return body


def _site_or_raise(spec: mujoco.MjSpec, site_name: str, description: str) -> mujoco.MjsSite:
    """Look up a site by name; list available sites on failure."""
    try:
        return spec.site(site_name)
    except KeyError as exc:
        available = [site.name for site in _items(spec.sites)]
        raise ValueError(
            f"{description} XML has no site '{site_name}'. Available sites: {available}"
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
    mount_prefix: str,
) -> None:
    """Attach the base to worldbody and place the arm root at the base mount site."""
    base_spec = _load_spec_or_raise(base_path, "base model")
    base_root = _first_body_or_raise(base_spec, "base model")
    mount_site = _site_or_raise(base_spec, mount_site_name, "base model")

    mount_frame = arm_spec.worldbody.add_frame()
    mount_frame.attach_body(base_root, prefix=mount_prefix, suffix="")

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
        available = [body.name for body in _items(arm_spec.worldbody.bodies)]
        raise ValueError(
            f"Arm model has no mount body '{attach_point_name}'. Available bodies: {available}"
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


def build_robot_spec(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    base_path: Optional[PathLike] = None,
    *,
    arm_descriptor: ArmDescriptor = DEFAULT_ARM,
    hand_descriptor: EndEffectorDescriptor = DEFAULT_HAND,
    base_descriptor: BaseDescriptor = DEFAULT_BASE,
    rot_xyz_deg: Optional[RotXyzDeg] = None,
    attach_point_name: Optional[str] = None,
    base_mount_site_name: Optional[str] = None,
    hand_prefix: Optional[str] = None,
    tactile_sensor: Optional[TactileSensorBase] = None,
    add_tactile_sensors: bool = True,
) -> mujoco.MjSpec:
    """Build an uncompiled arm + end-effector ``MjSpec`` from descriptors.

    Args:
        arm_path/hand_path/base_path: Override the descriptor's XML path.
        rot_xyz_deg: xyz Euler angles (degrees) of the hand relative to
            ``attach_point_name``; defaults to the arm descriptor's value.
        attach_point_name: Arm body used to mount the hand; defaults to the
            arm descriptor's value.
        base_mount_site_name: Site in the base XML declaring the arm root
            pose; defaults to the base descriptor's value.
        tactile_sensor: A ``TactileSensorBase`` instance whose
            ``augment_spec`` will be invoked on the hand's ``MjSpec`` before
            it is attached. If ``None`` and ``add_tactile_sensors`` is True,
            one is instantiated from ``hand_descriptor.tactile_sensor_factory``
            when available (pass your own instance if you need to keep a
            reference to it, e.g. to call ``bind``/``read`` later).
        add_tactile_sensors: Convenience flag; ignored if ``tactile_sensor``
            is explicitly provided.

    Returns:
        A merged but uncompiled ``MjSpec``, ready for further customization
        or direct compilation.
    """
    arm_path = resolve_path(arm_path, arm_descriptor.xml_path)
    hand_path = resolve_path(hand_path, hand_descriptor.xml_path)
    base_path = resolve_path(base_path, base_descriptor.xml_path)
    rot_xyz_deg = arm_descriptor.hand_attach_rot_xyz_deg if rot_xyz_deg is None else rot_xyz_deg
    attach_point_name = (
        arm_descriptor.hand_attach_body_name if attach_point_name is None else attach_point_name
    )
    base_mount_site_name = (
        base_descriptor.arm_mount_site_name
        if base_mount_site_name is None
        else base_mount_site_name
    )
    hand_prefix = hand_descriptor.default_prefix if hand_prefix is None else hand_prefix

    if tactile_sensor is None and add_tactile_sensors and hand_descriptor.tactile_sensor_factory:
        tactile_sensor = hand_descriptor.tactile_sensor_factory()

    arm_spec = _load_spec_or_raise(arm_path, "arm model")
    hand_spec = _load_spec_or_raise(hand_path, "hand model")
    _configure_solver(arm_spec)

    _mount_arm_on_base(arm_spec, base_path, base_mount_site_name, base_descriptor.mount_prefix)

    if tactile_sensor is not None:
        # Operate directly on the loaded MjSpec; no XML text round-trip.
        tactile_sensor.augment_spec(hand_spec)
        tactile_sensor.set_name_prefix(hand_prefix)

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


def build_robot_model(
    arm_path: Optional[PathLike] = None,
    hand_path: Optional[PathLike] = None,
    base_path: Optional[PathLike] = None,
    *,
    arm_descriptor: ArmDescriptor = DEFAULT_ARM,
    hand_descriptor: EndEffectorDescriptor = DEFAULT_HAND,
    base_descriptor: BaseDescriptor = DEFAULT_BASE,
    rot_xyz_deg: Optional[RotXyzDeg] = None,
    attach_point_name: Optional[str] = None,
    base_mount_site_name: Optional[str] = None,
    hand_prefix: Optional[str] = None,
    tactile_sensor: Optional[TactileSensorBase] = None,
    add_scene: bool = True,
    add_tactile_sensors: bool = True,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """Build, optionally add a preview scene, and compile the merged robot model."""
    spec = build_robot_spec(
        arm_path=arm_path,
        hand_path=hand_path,
        base_path=base_path,
        arm_descriptor=arm_descriptor,
        hand_descriptor=hand_descriptor,
        base_descriptor=base_descriptor,
        rot_xyz_deg=rot_xyz_deg,
        attach_point_name=attach_point_name,
        base_mount_site_name=base_mount_site_name,
        hand_prefix=hand_prefix,
        tactile_sensor=tactile_sensor,
        add_tactile_sensors=add_tactile_sensors,
    )

    if add_scene:
        add_preview_scene(spec)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def build_robot_model_from_config(
    robot_config_path: Optional[PathLike] = None,
    tactile_sensor: Optional[TactileSensorBase] = None,
    **overrides,
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """Build a robot model from the global robot config JSON."""
    config = apply_config_overrides(load_robot_config(robot_config_path), overrides)
    arm_descriptor, hand_descriptor, base_descriptor = descriptors_from_robot_config(config)

    enable_tactile = bool(config.get("enable_tactile_sensors", True))
    if tactile_sensor is None and enable_tactile and hand_descriptor.tactile_sensor_factory:
        tactile_sensor = hand_descriptor.tactile_sensor_factory(
            str(config.get("tactile_backend", "simple_box")),
            **dict(config.get("tactile_options") or {}),
        )

    return build_robot_model(
        arm_descriptor=arm_descriptor,
        hand_descriptor=hand_descriptor,
        base_descriptor=base_descriptor,
        rot_xyz_deg=optional_tuple(config, "hand_attach_rot_xyz_deg"),
        attach_point_name=config.get("attach_point_name"),
        base_mount_site_name=config.get("base_mount_site_name"),
        hand_prefix=config.get("hand_prefix"),
        tactile_sensor=tactile_sensor,
        add_scene=bool(config.get("add_preview_scene", True)),
        add_tactile_sensors=enable_tactile,
    )
