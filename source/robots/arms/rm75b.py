# -*- coding: utf-8 -*-
"""RM75B arm descriptor."""

from __future__ import annotations

from source.control.controllers import ArmPositionIkController
from source.environments.assets import RM75B_XML_PATH
from source.robots.descriptors import ArmDescriptor
from source.robots.registry import register_arm


RM75B_ARM = register_arm(
    ArmDescriptor(
        name="rm75b",
        xml_path=RM75B_XML_PATH,
        position_actuator_names=(
            "pos_joint1",
            "pos_joint2",
            "pos_joint3",
            "pos_joint4",
            "pos_joint5",
            "pos_joint6",
            "pos_joint7",
        ),
        ee_site_name="right_hand_site",
        hand_attach_body_name="right_hand",
        hand_attach_rot_xyz_deg=(-90.0, -90.0, 0.0),
        controller_factory=ArmPositionIkController,
    )
)
