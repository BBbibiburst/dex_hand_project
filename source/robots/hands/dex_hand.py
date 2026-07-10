# -*- coding: utf-8 -*-
"""Dex hand descriptor.

This file only describes the end effector: XML path, actuator names, default
name prefix, controller factory, and tactile sensor factory. STL fitting,
patch layouts, and taxel placement live in ``source.sensors.tactile.dex_hand``.
"""

from __future__ import annotations

from source.control.end_effectors import EndEffectorPositionController
from source.assets import asset_path
from source.robots.descriptors import EndEffectorDescriptor
from source.robots.registry import register_hand
from source.sensors.tactile.dex_hand import create_dex_hand_tactile_sensor


DEX_HAND_POSITION_ACTUATORS = (
    "act_push_0_j",
    "act_push_1_j",
    "act_push_2_j",
    "act_push_3_j",
    "thumb_rotate_act_push_j",
    "thumb_grasp_act_push_j",
)


DEX_HAND = register_hand(
    EndEffectorDescriptor(
        name="dex_hand",
        xml_path=asset_path("grippers", "dex_hand", "dex_hand.xml"),
        position_actuator_names=DEX_HAND_POSITION_ACTUATORS,
        default_prefix="dexhand_",
        tactile_sensor_factory=create_dex_hand_tactile_sensor,
        controller_factory=EndEffectorPositionController,
    )
)
