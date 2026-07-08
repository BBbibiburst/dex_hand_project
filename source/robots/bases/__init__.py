# -*- coding: utf-8 -*-
"""Base descriptors.

To add a new base: create ``source/robots/bases/<name>.py`` following the
pattern in ``rethink_minimal_mount.py`` and import it below.
"""

from source.robots.bases.rethink_minimal_mount import RETHINK_MINIMAL_MOUNT

__all__ = ["RETHINK_MINIMAL_MOUNT"]
