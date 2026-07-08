# -*- coding: utf-8 -*-
"""Rethink minimal mount base descriptor."""

from __future__ import annotations

from source.environments.assets import DEFAULT_BASE_XML_PATH
from source.robots.descriptors import BaseDescriptor
from source.robots.registry import register_base


RETHINK_MINIMAL_MOUNT = register_base(
    BaseDescriptor(
        name="rethink_minimal_mount",
        xml_path=DEFAULT_BASE_XML_PATH,
        arm_mount_site_name="arm_mount",
    )
)
