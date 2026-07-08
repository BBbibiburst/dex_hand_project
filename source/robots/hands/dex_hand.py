# -*- coding: utf-8 -*-
"""Dex hand descriptor.

Note how little this file knows: an XML path, actuator names, a default
name prefix, and a factory that produces a tactile sensor. It has no
knowledge of STL fitting, patch layouts, or taxel counts — all of that is
private to ``dex_hand_tactile.py``.
"""

from __future__ import annotations

from source.control.controllers import EndEffectorPositionController
from source.environments.assets import DEX_HAND_XML_PATH
from source.robots.descriptors import EndEffectorDescriptor
from source.robots.hands.dex_hand_tactile import DexHandTouchSensor
from source.robots.registry import register_hand


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
        xml_path=DEX_HAND_XML_PATH,
        position_actuator_names=DEX_HAND_POSITION_ACTUATORS,
        default_prefix="dexhand_",
        tactile_sensor_factory=DexHandTouchSensor,
        controller_factory=EndEffectorPositionController,
    )
)
