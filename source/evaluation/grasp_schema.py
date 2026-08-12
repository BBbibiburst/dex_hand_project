"""Canonical public schema constants for grasp evaluation."""

BENCHMARK_SCHEMA_VERSION = 4
VALIDATION_SEMANTICS = "trajectory-hold-v2"

TRAJECTORY_STABLE = "trajectory_stable"
DIRECT_HOLD_ONLY = "direct_hold_only"
UNSTABLE = "unstable"
VALIDATION_ERROR = "validation_error"
SEARCH_ERROR = "search_error"

CURRENT_BENCHMARK_STATUSES = frozenset(
    {
        TRAJECTORY_STABLE,
        DIRECT_HOLD_ONLY,
        UNSTABLE,
        VALIDATION_ERROR,
        SEARCH_ERROR,
    }
)

LEGACY_STABLE = "legacy_stable"
