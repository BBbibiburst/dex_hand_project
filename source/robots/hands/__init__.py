# -*- coding: utf-8 -*-
"""End-effector descriptors.

To add a new hand or gripper: create ``source/robots/hands/<name>.py``
following the pattern in ``dex_hand.py``. If it has no tactile sensing,
simply omit ``tactile_sensor_factory`` (it defaults to ``None``) — the
environment falls back to ``NullTactileSensor`` automatically. If it does,
implement ``TactileSensorBase`` however makes sense for that device; nothing
outside your new module needs to know how.
"""

from source.robots.hands.dex_hand import DEX_HAND

__all__ = ["DEX_HAND"]
