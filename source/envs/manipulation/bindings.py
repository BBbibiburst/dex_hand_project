# -*- coding: utf-8 -*-
"""Compiled MuJoCo identifiers owned by a manipulation task."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectBinding:
    body_id: int
    joint_id: int
    qpos_adr: int
    qvel_adr: int
    geom_ids: frozenset[int]


@dataclass(frozen=True)
class TaskBindings:
    objects: dict[str, ObjectBinding]
    ee_site_id: int | None
    robot_geom_ids: frozenset[int]
    environment_geom_ids: frozenset[int]
