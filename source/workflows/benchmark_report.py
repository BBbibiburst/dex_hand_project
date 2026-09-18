"""Comparable metrics for the five manipulation benchmark domains."""

from __future__ import annotations

from typing import Any, Mapping


def _is_success_status(value: Any) -> bool:
    status = str(value)
    return status == "SUCCESS" or status.endswith("_SUCCESS")


def _success_ids(summary: Mapping[str, Any], *, key: str = "object_id") -> set[str]:
    return {
        str(row[key])
        for row in summary.get("results", ())
        if _is_success_status(row.get("status")) and key in row
    }


def catalogue_task_metrics(
    domain: Mapping[str, Any],
    task_summary: Mapping[str, Any],
    *,
    full_catalogue_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return task-valid success and full-catalogue coverage separately."""
    valid_ids = {str(row["object_id"]) for row in domain.get("objects", ())}
    if not valid_ids:
        raise ValueError("Task-valid domain contains no objects.")
    task_success = _success_ids(task_summary)
    valid_success = len(valid_ids.intersection(task_success))
    source_count = int(domain["source_count"])
    full_source = task_summary if full_catalogue_summary is None else full_catalogue_summary
    full_success = len(_success_ids(full_source))
    return {
        "domain_version": domain.get("domain_version"),
        "task_valid_count": len(valid_ids),
        "task_valid_successes": valid_success,
        "task_valid_success_rate": valid_success / len(valid_ids),
        "full_catalogue_count": source_count,
        "full_catalogue_successes": full_success,
        "full_catalogue_coverage": full_success / source_count,
        "excluded_count": int(domain.get("excluded_count", source_count - len(valid_ids))),
        "excluded": list(domain.get("excluded", ())),
    }


def fixed_domain_metrics(summary: Mapping[str, Any], *, domain: str) -> dict[str, Any]:
    """Summarize Lift/PickPlace catalogue or NutAssembly procedural results."""
    count = int(summary.get("count", len(summary.get("results", ()))))
    successes = sum(_is_success_status(row.get("status")) for row in summary.get("results", ()))
    return {
        "domain": domain,
        "count": count,
        "successes": successes,
        "success_rate": successes / max(count, 1),
        "excluded_count": 0,
    }
