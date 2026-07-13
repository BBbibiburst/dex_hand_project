# -*- coding: utf-8 -*-
"""Pika parallel gripper descriptor."""

from __future__ import annotations

from source.assets import asset_path
from source.control.end_effectors import PikaGripperController
from source.robots.descriptors import EndEffectorDescriptor
from source.robots.registry import register_hand
from source.sensors.tactile.pika_gripper import create_pika_gripper_tactile_sensor

PIKA_GRIPPER_POSITION_ACTUATORS = ("gripper_position",)


PIKA_GRIPPER = register_hand(
    EndEffectorDescriptor(
        name="pika_gripper",
        xml_path=asset_path("grippers", "pika_gripper", "pika_gripper.xml"),
        position_actuator_names=PIKA_GRIPPER_POSITION_ACTUATORS,
        default_prefix="pika_",
        tactile_sensor_factory=create_pika_gripper_tactile_sensor,
        controller_factory=PikaGripperController,
    )
)
