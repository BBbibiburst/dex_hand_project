# -*- coding: utf-8 -*-
"""Rethink minimal mount base descriptor."""

from __future__ import annotations

from source.assets import asset_path
from source.robots.descriptors import BaseDescriptor
from source.robots.registry import register_base

RETHINK_MINIMAL_MOUNT = register_base(
    BaseDescriptor(
        name="rethink_minimal_mount",
        xml_path=asset_path("bases", "rethink_minimal_mount.xml"),
        arm_mount_site_name="arm_mount",
    )
)
