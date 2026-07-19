"""Shared grasp-search and validation contracts."""

from __future__ import annotations


GRASP_CONFIG_SCHEMA_VERSION = 2
SUPPORTED_GRASP_CONFIG_SCHEMA_VERSIONS = (1, GRASP_CONFIG_SCHEMA_VERSION)
GRASP_SEARCH_STRATEGY = "two_stage_support_aware_force_closure_v3"
DEFAULT_GRIP_PRELOAD = 0.40
