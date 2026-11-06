"""Versioned contracts and transactional inventory for the five-task dataset."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np


SCHEMA_VERSION = "five-task-episode-v1"
TASKS = ("lift", "pick_place", "stack", "push", "nut_assembly")
TACTILE_STATUSES = (
    "PRESENT",
    "NOT_AVAILABLE_IN_SOURCE",
    "SENSOR_NOT_ENABLED",
    "NOT_APPLICABLE",
)


def classify_failure(reason: str | None, terminal_stage: str | None) -> dict:
    text = (reason or "").lower()
    return {
        "stage": terminal_stage,
        "collision": "collision" in text,
        "ik_failure": "ik" in text,
        "grasp_failure": any(token in text for token in ("grasp", "close", "hold")),
        "object_drop": any(token in text for token in ("drop", "fell", "fall")),
        "timeout": any(token in text for token in ("timeout", "time limit")),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tactile_status(archive: dict[str, np.ndarray], manifest: dict) -> str:
    if "tactile" in archive and archive["tactile"].ndim == 2:
        if archive["tactile"].shape[1] == 712:
            return "PRESENT"
        if archive["tactile"].shape[1] == 0:
            return "NOT_AVAILABLE_IN_SOURCE"
    enabled = manifest.get("metadata", {}).get("execution_config", {}).get(
        "enable_tactile_sensors"
    )
    return "SENSOR_NOT_ENABLED" if enabled is False else "NOT_AVAILABLE_IN_SOURCE"


def validate_source_episode(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = str(payload.get("metadata", {}).get("task") or "lift")
    if task not in TASKS:
        raise ValueError(f"Unsupported task {task!r}.")
    arrays_path = manifest_path.parent / str(payload.get("arrays", "episode.npz"))
    with np.load(arrays_path, allow_pickle=False) as source:
        archive = {name: np.asarray(source[name]) for name in source.files}
    required = {
        "qpos",
        "qvel",
        "ctrl",
        "action",
        "object_position",
        "object_quaternion_wxyz",
        "stage",
        "task_success",
    }
    missing = sorted(required - archive.keys())
    if missing:
        raise ValueError(f"Missing source arrays: {missing}")
    lengths = {name: len(archive[name]) for name in required}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) < 2:
        raise ValueError(f"Invalid source lengths: {lengths}")
    if archive["action"].shape[1:] != (13,):
        raise ValueError(f"Action must be [T,13], got {archive['action'].shape}.")
    if not all(np.isfinite(value).all() for value in archive.values()):
        raise ValueError("Source contains non-finite values.")
    return {
        "manifest": payload,
        "archive": archive,
        "arrays_path": arrays_path,
        "task": task,
        "tactile_status": tactile_status(archive, payload),
    }


def write_causal_source(source_manifest: Path, destination: Path) -> dict:
    """Shift post-step records into pre-action-observation/action pairs.

    Existing DemonstrationEpisode recorders append after ``env.step(action)``.
    For action ``t``, the only recorded pre-action state is row ``t-1``.  The
    first action is therefore deliberately excluded rather than mislabeled.
    """

    validated = validate_source_episode(source_manifest)
    archive = validated["archive"]
    destination.mkdir(parents=True, exist_ok=False)
    shifted: dict[str, np.ndarray] = {}
    for name, value in archive.items():
        if len(value) != len(archive["action"]):
            shifted[name] = value
        elif name == "action":
            shifted[name] = value[1:]
        else:
            shifted[name] = value[:-1]
    np.savez_compressed(destination / "episode.npz", **shifted)
    payload = dict(validated["manifest"])
    payload["arrays"] = "episode.npz"
    payload["frames"] = len(shifted["action"])
    payload["metadata"] = {
        **payload.get("metadata", {}),
        "p31_causal_pairing": "observation_source_row_t_minus_1_with_action_source_row_t",
        "p31_dropped_source_action_indices": [0],
        "p31_source_observation_index_offset": 0,
        "p31_source_action_index_offset": 1,
    }
    (destination / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return {
        **validated,
        "causal_manifest": destination / "manifest.json",
        "causal_frames": len(shifted["action"]),
    }


def validate_exported_arrays(path: Path, tactile: str) -> dict:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    if arrays["state"].ndim != 2 or arrays["state"].shape[1] != 39:
        raise ValueError(f"State contract failed: {arrays['state'].shape}")
    if arrays["action"].ndim != 2 or arrays["action"].shape[1] != 13:
        raise ValueError(f"Action contract failed: {arrays['action'].shape}")
    length = len(arrays["state"])
    if any(len(arrays[name]) != length for name in ("action", "timestamp", "sim_time")):
        raise ValueError("Exported arrays have inconsistent time lengths.")
    if length < 1 or not np.all(np.diff(arrays["timestamp"]) > 0):
        raise ValueError("Timestamp contract failed.")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Exported arrays contain non-finite values.")
    if tactile == "PRESENT":
        if "tactile" not in arrays or arrays["tactile"].shape != (length, 712):
            raise ValueError("PRESENT tactile must have shape [T,712].")
    elif "tactile" in arrays:
        raise ValueError("Missing tactile must not be materialized as an array.")
    return {"frames": length, "state_shape": list(arrays["state"].shape), "action_shape": list(arrays["action"].shape)}


class EpisodeIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=60000")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                lineage_hash TEXT NOT NULL UNIQUE,
                content_hash TEXT NOT NULL UNIQUE,
                task_name TEXT NOT NULL,
                object_id TEXT NOT NULL,
                success INTEGER NOT NULL,
                simulation_duration REAL NOT NULL,
                tactile_status TEXT NOT NULL,
                relative_path TEXT NOT NULL UNIQUE,
                bytes INTEGER NOT NULL,
                validation_status TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def find_lineage(self, lineage_hash: str) -> dict | None:
        self.connection.row_factory = sqlite3.Row
        row = self.connection.execute(
            "SELECT * FROM episodes WHERE lineage_hash = ?", (lineage_hash,)
        ).fetchone()
        return dict(row) if row else None

    def find_content(self, content_hash: str) -> dict | None:
        self.connection.row_factory = sqlite3.Row
        row = self.connection.execute(
            "SELECT * FROM episodes WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return dict(row) if row else None

    def insert(self, episode: dict) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode["episode_id"],
                    episode["lineage_hash"],
                    episode["content_hash"],
                    episode["task_name"],
                    episode["object_id"],
                    int(episode["success"]),
                    episode["simulation_duration"],
                    episode["tactile"]["status"],
                    episode["relative_path"],
                    episode["storage_bytes"],
                    episode["validation_status"],
                ),
            )

    def inventory(self) -> dict:
        self.connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in self.connection.execute("SELECT * FROM episodes")]
        tasks = {}
        for task in TASKS:
            selected = [row for row in rows if row["task_name"] == task]
            tasks[task] = {
                "episodes": len(selected),
                "successes": sum(row["success"] for row in selected),
                "failures": sum(not row["success"] for row in selected),
                "simulation_hours": sum(row["simulation_duration"] for row in selected) / 3600,
                "success_simulation_hours": sum(
                    row["simulation_duration"] for row in selected if row["success"]
                )
                / 3600,
                "failure_simulation_hours": sum(
                    row["simulation_duration"] for row in selected if not row["success"]
                )
                / 3600,
                "objects": len({row["object_id"] for row in selected}),
                "storage_bytes": sum(row["bytes"] for row in selected),
                "tactile_present": sum(row["tactile_status"] == "PRESENT" for row in selected),
                "success_rate": sum(row["success"] for row in selected) / max(len(selected), 1),
                "tactile_coverage_rate": sum(
                    row["tactile_status"] == "PRESENT" for row in selected
                )
                / max(len(selected), 1),
            }
        return {
            "schema_version": "five-task-inventory-v1",
            "episodes": len(rows),
            "unique_lineages": len({row["lineage_hash"] for row in rows}),
            "unique_contents": len({row["content_hash"] for row in rows}),
            "simulation_hours": sum(row["simulation_duration"] for row in rows) / 3600,
            "success_simulation_hours": sum(
                row["simulation_duration"] for row in rows if row["success"]
            )
            / 3600,
            "failure_simulation_hours": sum(
                row["simulation_duration"] for row in rows if not row["success"]
            )
            / 3600,
            "unique_objects": len({row["object_id"] for row in rows}),
            "validation_pass_rate": sum(row["validation_status"] == "PASS" for row in rows) / max(len(rows), 1),
            "tactile_coverage_rate": sum(
                row["tactile_status"] == "PRESENT" for row in rows
            )
            / max(len(rows), 1),
            "tasks": tasks,
        }

    def write_inventory(self, path: Path) -> dict:
        """Serialize the derived JSON cache while holding the SQLite writer lock."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            payload = self.inventory()
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
            self.connection.commit()
            return payload
        except Exception:
            self.connection.rollback()
            raise
