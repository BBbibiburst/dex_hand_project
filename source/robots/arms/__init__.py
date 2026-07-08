# -*- coding: utf-8 -*-
"""Arm descriptors.

To add a new arm: create ``source/robots/arms/<name>.py`` following the
pattern in ``rm75b.py`` and import it below so its module-level
``register_arm(...)`` call executes.
"""

from source.robots.arms.rm75b import RM75B_ARM

__all__ = ["RM75B_ARM"]
