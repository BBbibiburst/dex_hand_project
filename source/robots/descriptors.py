# -*- coding: utf-8 -*-
"""Device descriptors for model-agnostic robot assembly and control.

Design principle
----------------
These dataclasses describe *what a device is* (paths, actuator names, mount
points) — never *how to simulate its sensors*. Tactile / force / any other
sensing behaviour is intentionally NOT modeled here as structured data
(patches, rows, cols, mesh names, ...). Instead each end effector supplies a
``tactile_sensor_factory``: a zero-argument callable that returns a
``TactileSensorBase`` instance (see ``source.environments.tactile_sensors``).

This keeps the framework layer free of any assumption about sensor
implementation (STL-fit taxel grids, contact-force summation, an external
model, or nothing at all). Swapping an end effector for one with a
completely different sensing strategy never requires touching this module,
the registry, the environment, or the controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:
    from source.environments.tactile_sensors import TactileSensorBase


TactileSensorFactory = Callable[[], "TactileSensorBase"]
ControllerFactory = Callable[..., object]


@dataclass(frozen=True)
class ArmDescriptor:
    """Describes an arm model and where/how a hand attaches to it."""

    name: str
    xml_path: Path
    position_actuator_names: Sequence[str]
    ee_site_name: str
    hand_attach_body_name: str
    hand_attach_rot_xyz_deg: tuple[float, float, float]
    controller_factory: Optional[ControllerFactory] = None


@dataclass(frozen=True)
class EndEffectorDescriptor:
    """Describes an end effector (hand / gripper) model.

    ``tactile_sensor_factory`` is optional. Leave it ``None`` for end
    effectors with no tactile sensing (e.g. a simple parallel gripper); the
    environment will fall back to ``NullTactileSensor`` automatically.
    """

    name: str
    xml_path: Path
    position_actuator_names: Sequence[str] = ()
    default_prefix: str = ""
    tactile_sensor_factory: Optional[TactileSensorFactory] = None
    controller_factory: Optional[ControllerFactory] = None


@dataclass(frozen=True)
class BaseDescriptor:
    """Describes a mounting base and the site where the arm root attaches."""

    name: str
    xml_path: Path
    arm_mount_site_name: str
    mount_prefix: str = "mount_"
