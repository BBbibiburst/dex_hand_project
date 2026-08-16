"""Supported end-effector search descriptors."""

from __future__ import annotations

from source.assets import asset_path
from source.grasping.search.types import Device

DEX_XML = asset_path("grippers", "dex_hand", "dex_hand.xml")
PIKA_XML = asset_path("grippers", "pika_gripper", "pika_gripper.xml")

DEVICES = {
    "dex_hand": Device(
        "dex_hand", DEX_XML, "hand_root",
        ("act_push_0_j", "act_push_1_j", "act_push_2_j", "act_push_3_j",
         "thumb_rotate_act_push_j", "thumb_grasp_act_push_j"),
        (0, 1, 2, 3, 4),
    ),
    "pika_gripper": Device(
        "pika_gripper", PIKA_XML, "gripper_base_link", ("gripper_position",), (0, 1)
    ),
}
