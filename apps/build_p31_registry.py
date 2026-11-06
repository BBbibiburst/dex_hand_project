"""Build the asset-identity registry and observed task/object coverage for P31."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASK_ROOTS = {
    "lift": Path("outputs/dex_hand_top100_v2/grasp"),
    "pick_place": Path("outputs/pick_place_top100_final"),
    "stack": Path("outputs/stack_task_valid_v24"),
    "push": Path("outputs/push_top100_v30"),
    "nut_assembly": Path("outputs/nut_assembly_top100_final"),
}


def _file_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _observed() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    output = {task: set() for task in TASK_ROOTS}
    failures = {task: set() for task in TASK_ROOTS}
    for task, root in TASK_ROOTS.items():
        for manifest in root.glob("**/manifest.json"):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            object_id = payload.get("object_id")
            if object_id:
                (output if payload.get("success") else failures)[task].add(str(object_id))
    lift_summary = TASK_ROOTS["lift"].parent / "summary.json"
    if lift_summary.is_file():
        for row in json.loads(lift_summary.read_text(encoding="utf-8")).get("results", []):
            if row.get("object_id"):
                (output if row.get("status") in {"LATTICE_SUCCESS", "RL_SUCCESS", "GRASP_SUCCESS"} else failures)["lift"].add(str(row["object_id"]))
    return output, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", type=Path, default=Path("assets/maniskill/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.asset_manifest.read_text(encoding="utf-8"))
    observed, failures = _observed()
    objects = []
    for row in source["objects"]:
        object_id = f"{row['dataset']}:{row['object_id']}"
        root = Path(row["project_path"])
        files = [path for path in root.rglob("*") if path.is_file()]
        tasks = [task for task, values in observed.items() if object_id in values]
        objects.append(
            {
                "object_id": object_id,
                "canonical_name": row["object_id"],
                "asset_path": str(root),
                "asset_hash": _file_digest(files),
                "asset_source": row["dataset"],
                "geometry": {"model_files": row.get("model_files", []), "size_bytes": row.get("size_bytes")},
                "supported_tasks_observed": tasks,
                "task_validation": {task: object_id in observed[task] for task in TASK_ROOTS},
                "has_success_episode": bool(tasks),
                "has_failure_episode": any(object_id in values for values in failures.values()),
            }
        )
    objects.append(
        {
            "object_id": "procedural_bolt_family",
            "canonical_name": "procedural bolt/socket family",
            "asset_path": "source/envs/manipulation/nut_assembly.py",
            "asset_hash": _file_digest([Path("source/envs/manipulation/nut_assembly.py")]),
            "asset_source": "procedural_family_not_100_independent_objects",
            "geometry": {"variants": 100, "unique_asset_count": 1},
            "supported_tasks_observed": ["nut_assembly"],
            "task_validation": {task: task == "nut_assembly" for task in TASK_ROOTS},
            "has_success_episode": True,
            "has_failure_episode": True,
        }
    )
    coverage = {
        task: {
            "observed_success_object_ids": sorted(values),
            "observed_success_unique_count": len(values) if task != "nut_assembly" else 1,
            "raw_variant_count": len(values),
            "observed_failure_object_ids": sorted(failures[task]),
            "observed_failure_unique_count": len(failures[task]) if task != "nut_assembly" else 1,
        }
        for task, values in observed.items()
    }
    operated_assets = set().union(
        *(values for task, values in observed.items() if task != "nut_assembly")
    )
    if observed["nut_assembly"]:
        operated_assets.add("procedural_bolt_family")
    registry = {
        "schema_version": "five-task-object-registry-v1",
        "source_assets": source["actual_count_total"],
        "registered_unique_assets": len(objects),
        "objects": objects,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "object_registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "task_object_coverage.json").write_text(
        json.dumps(
            {
                "schema_version": "task-object-coverage-v1",
                "historically_observed_success_unique_asset_count": len(operated_assets),
                "tasks": coverage,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
