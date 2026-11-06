"""Transactionally export one five-task source trajectory into the P31 schema."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import uuid

from apps.check_p30_f_drive import check as check_drive
from apps.run_p30_full_episode import build_parser as build_replay_parser, run as run_replay
from source.data.five_task_dataset import (
    EpisodeIndex,
    SCHEMA_VERSION,
    canonical_hash,
    classify_failure,
    sha256_file,
    validate_exported_arrays,
    validate_source_episode,
    write_causal_source,
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _fsync_tree(path: Path) -> None:
    for item in path.rglob("*"):
        if item.is_file():
            with item.open("rb") as stream:
                os.fsync(stream.fileno())
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _asset_record(registry: Path | None, object_id: str, task_config: dict) -> dict:
    if registry and registry.is_file():
        payload = json.loads(registry.read_text(encoding="utf-8"))
        for row in payload.get("objects", []):
            if row["object_id"] == object_id:
                return row
            if object_id.startswith("procedural_bolt:") and row["object_id"] == "procedural_bolt_family":
                return row
    return {
        "object_id": object_id,
        "asset_hash": canonical_hash({"object_id": object_id, "task_config": task_config}),
        "asset_source": "procedural_or_unregistered",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-registry", type=Path)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--online-timing", type=Path)
    parser.add_argument("--encoder", default="libx264", choices=("libx264", "libx265", "libaom-av1"))
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--test-only-skip-volume-preflight", action="store_true")
    return parser


def export(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.test_only_skip_volume_preflight:
        if os.environ.get("P31_TESTING") != "1":
            raise RuntimeError("Volume preflight may only be skipped with P31_TESTING=1.")
    else:
        drive = check_drive(args.output_root)
        if not drive["ready"]:
            raise RuntimeError(f"Output root is not the expected F-volume capacity class: {drive}")
    validated = validate_source_episode(args.source_manifest.resolve())
    source_episode_hash = sha256_file(validated["arrays_path"])
    source_manifest_hash = sha256_file(args.source_manifest.resolve())
    task_config = dict(validated["manifest"].get("metadata", {}).get("task_config", {}))
    lineage_hash = canonical_hash(
        {
            "source_episode_hash": source_episode_hash,
            "causal_contract": "post_step_to_pre_action_v1",
        }
    )
    index = EpisodeIndex(args.output_root / "inventory.sqlite")
    try:
        duplicate = index.find_lineage(lineage_hash)
        if duplicate:
            return {
                "status": "DUPLICATE_LINEAGE",
                "existing": duplicate,
                "wall_seconds": time.perf_counter() - started,
            }

        episode_id = str(uuid.uuid4())
        work = args.output_root / f".work-{episode_id}"
        partial = args.output_root / f".partial-{episode_id}"
        final = args.output_root / "episodes" / episode_id
        if any(path.exists() for path in (work, partial, final)):
            raise FileExistsError(f"Transaction path collision for {episode_id}.")
        work.mkdir()
        causal = write_causal_source(args.source_manifest.resolve(), work / "causal_source")
        replay_args = build_replay_parser().parse_args(
            [
                "--manifest", str(causal["causal_manifest"]),
                "--output", str(partial),
                "--width", "1920", "--height", "1080", "--fps", "20",
                "--encoder", args.encoder,
                "--mode", "async", "--queue-size", str(args.queue_size),
            ]
        )
        replay = run_replay(replay_args)
        arrays_contract = validate_exported_arrays(
            partial / "modalities.npz", validated["tactile_status"]
        )
        online_timing = (
            json.loads(args.online_timing.read_text(encoding="utf-8"))
            if args.online_timing and args.online_timing.is_file()
            else None
        )
        asset = _asset_record(args.object_registry, validated["manifest"]["object_id"], task_config)
        content_hash = canonical_hash(
            {
                "video": replay["integrity"]["video_sha256"],
                "modalities": replay["integrity"]["modalities_sha256"],
            }
        )
        duplicate_content = index.find_content(content_hash)
        if duplicate_content:
            duplicate_dir = args.output_root / "duplicates" / episode_id
            duplicate_dir.parent.mkdir(exist_ok=True)
            partial.replace(duplicate_dir)
            (duplicate_dir / "duplicate.json").write_text(
                json.dumps(
                    {
                        "status": "DUPLICATE_CONTENT",
                        "attempt_id": args.attempt_id,
                        "content_hash": content_hash,
                        "existing": duplicate_content,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            shutil.rmtree(work)
            return {
                "status": "DUPLICATE_CONTENT",
                "existing": duplicate_content,
                "quarantine_path": str(duplicate_dir),
                "content_hash": content_hash,
                "wall_seconds": time.perf_counter() - started,
            }
        control_dt = float(validated["manifest"].get("metadata", {}).get("control_dt", 0.05))
        metadata = {
            "episode_id": episode_id,
            "schema_version": SCHEMA_VERSION,
            "task_name": validated["task"],
            "object_id": validated["manifest"]["object_id"],
            "object_asset_hash": asset["asset_hash"],
            "object_asset_source": asset.get("asset_source"),
            "random_seed": int(validated["manifest"].get("seed", 0)),
            "attempt_id": args.attempt_id,
            "source_trajectory_hash": source_episode_hash,
            "source_manifest_hash": source_manifest_hash,
            "lineage_hash": lineage_hash,
            "content_hash": content_hash,
            "code_commit": _git_head(),
            "environment_config_hash": canonical_hash(task_config),
            "start_time": started_at,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "simulation_duration": arrays_contract["frames"] * control_dt,
            "control_frequency": 1.0 / control_dt,
            "physics_frequency": 1000.0,
            "success": bool(validated["manifest"].get("success")),
            "failure_reason": validated["manifest"].get("failure_reason"),
            "failure": classify_failure(
                validated["manifest"].get("failure_reason"),
                validated["manifest"].get("terminal_stage"),
            ),
            "termination_condition": validated["manifest"].get("terminal_stage"),
            "validation_status": "PASS",
            "tactile": {"status": validated["tactile_status"], "shape": [arrays_contract["frames"], 712] if validated["tactile_status"] == "PRESENT" else None},
            "rgb": {"width": 1920, "height": 1080, "fps": 20, "codec": args.encoder},
            "state": {"shape": arrays_contract["state_shape"], "semantic": "pre_action_observation"},
            "action": {"shape": arrays_contract["action_shape"], "source_index_offset": 1},
            "object_state": {"primary_object": "position_plus_quaternion", "secondary_objects": "NOT_AVAILABLE_IN_SOURCE"},
            "online_timing": online_timing,
            "export_timing": {"wall_seconds": replay["wall_seconds"], "phases": replay["phases"]},
            "export_resources": replay["resources"],
            "video_integrity": replay["integrity"]["video"],
        }
        metadata["relative_path"] = str(Path("episodes") / episode_id)
        metadata_path = partial / "episode.json"
        metadata["storage_bytes"] = _directory_bytes(partial)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # Include episode.json itself in the persisted byte count.  A second
        # write is intentional: the first materializes the file so its size is
        # part of the final accounting.
        metadata["storage_bytes"] = _directory_bytes(partial)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _fsync_tree(partial)
        final.parent.mkdir(exist_ok=True)
        partial.replace(final)
        metadata["storage_bytes"] = _directory_bytes(final)
        try:
            index.insert(metadata)
        except sqlite3.IntegrityError:
            # A concurrent worker can commit the same lineage/content after
            # this worker's pre-check. Preserve the completed duplicate for
            # audit, but never register or count it as additional duration.
            duplicate = index.find_lineage(lineage_hash) or index.find_content(content_hash)
            if not duplicate:
                raise
            duplicate_dir = args.output_root / "duplicates" / episode_id
            duplicate_dir.parent.mkdir(exist_ok=True)
            final.replace(duplicate_dir)
            (duplicate_dir / "duplicate.json").write_text(
                json.dumps(
                    {
                        "status": "DUPLICATE_CONCURRENT",
                        "attempt_id": args.attempt_id,
                        "lineage_hash": lineage_hash,
                        "content_hash": content_hash,
                        "existing": duplicate,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            shutil.rmtree(work)
            return {
                "status": "DUPLICATE_CONTENT",
                "existing": duplicate,
                "quarantine_path": str(duplicate_dir),
                "content_hash": content_hash,
                "wall_seconds": time.perf_counter() - started,
            }
        inventory_path = args.output_root / "episode_inventory.json"
        index.write_inventory(inventory_path)
        shutil.rmtree(work)
        return {
            "status": "EXPORTED",
            "episode_id": episode_id,
            "path": str(final),
            "lineage_hash": lineage_hash,
            "content_hash": content_hash,
            "wall_seconds": time.perf_counter() - started,
            "metadata": metadata,
        }
    except Exception:
        raise
    finally:
        index.close()


def main() -> None:
    result = export(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
