# -*- coding: utf-8 -*-
"""Default registered robot component descriptors."""

from __future__ import annotations

from source.robots.arms.rm75b import RM75B_ARM
from source.robots.bases.rethink_minimal_mount import RETHINK_MINIMAL_MOUNT
from source.robots.hands.dex_hand import DEX_HAND

DEFAULT_ARM = RM75B_ARM
DEFAULT_HAND = DEX_HAND
DEFAULT_BASE = RETHINK_MINIMAL_MOUNT

__all__ = ["DEFAULT_ARM", "DEFAULT_HAND", "DEFAULT_BASE"]
