"""Versioned Episode packaging, independent file checks and crash recovery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from source.data.five_task_dataset import canonical_hash, sha256_file
from source.data.production_recorder import validate_online_arrays, validate_stage_encoding
from source.data.production_video import verify_video
from source.data.production_provenance import verify_provenance
from source.data.production_failures import reported_task_outcome
from source.data.production_capture import capture_contract, recorded_rgb_contract


SCHEMA = "five-task-episode-v4"
FIELDS = {"state": "state", "action": "position_action", "tactile": "sensor_tactile",
          "object_state": "object_state", "timestamp": "timestamp",
          "next_timestamp": "next_timestamp", "stage": "stage", "success": "task_success"}


def copy_source_contents(source, destination):
    """Copy regular content without Unix attributes unsupported by DrvFS.

    Episode identity covers file bytes, not filesystem permission or mtime.
    Packaging still hashes, verifies and fsyncs the completed transaction.
    """
    if source.is_symlink() or not source.is_dir():
        raise ValueError('Episode source must be a regular directory')
    destination.mkdir()
    for path in sorted(source.rglob('*')):
        target = destination / path.relative_to(source)
        if path.is_symlink():
            raise ValueError('Symlinks are not allowed in Episode source')
        if path.is_dir():
            target.mkdir()
        elif path.is_file():
            shutil.copyfile(path, target)
        else:
            raise ValueError('Non-regular Episode source entry')


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def canonical_arrays(source):
    validate_online_arrays(source)
    extra = {key: source[value] for key, value in (
        ('integration_state', 'pre_integration_state'), ('control_exact', 'applied_ctrl64')) if value in source}
    return {**{key: source[value] for key, value in FIELDS.items()}, **extra,
            "frame_index": np.arange(len(source["timestamp"]), dtype=np.int64)}


def content_hash(arrays, decoded_video_hash):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        if key in ('integration_state', 'control_exact'):
            # Replay checkpoints must not turn identical observations into new data.
            continue
        array = np.asarray(arrays[key])
        if key in ("timestamp", "next_timestamp"):
            # Different absolute clocks do not make identical trajectories unique.
            array = np.round(array - arrays["timestamp"][0], 9)
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(decoded_video_hash.encode())
    return digest.hexdigest()


class _DecodedVideoProof:
    """One in-process full decode, bound to exact bytes/path/frame count."""

    def __init__(self, path, frames, capture=None):
        self.path = path.resolve()
        self.frames = frames
        self.capture = capture_contract(capture)
        self.file_sha256 = sha256_file(path)
        self.result = verify_video(path, frames) if capture is None else verify_video(path, frames, capture=capture)
        if sha256_file(path) != self.file_sha256:
            raise ValueError('Video changed during complete decode')

    def checked_result(self, path, frames, capture=None):
        if (path.resolve() != self.path or frames != self.frames
                or capture_contract(capture) != self.capture
                or sha256_file(path) != self.file_sha256):
            raise ValueError('Video differs from the in-process decode proof')
        return dict(self.result)


def verify_episode(directory, *, _video_proof=None):
    meta = json.loads((directory / "episode.json").read_text())
    if meta["schema"] not in (SCHEMA, "five-task-episode-v3", "five-task-episode-v2"):
        raise ValueError("Unsupported Episode schema")
    required_files = {"modalities.npz", "agentview.mp4", "source_episode/episode.npz", "source_episode/manifest.json"}
    if not required_files.issubset(meta["files"]):
        raise ValueError("Episode checksum manifest is incomplete")
    for name, expected in meta["files"].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or (directory / path).is_symlink():
            raise ValueError("Unsafe Episode file reference")
        if sha256_file(directory / path) != expected:
            raise ValueError(f"File content changed: {name}")
    with np.load(directory / "source_episode/episode.npz", allow_pickle=False) as source:
        source_arrays = {key: source[key] for key in source.files}
    with np.load(directory / "modalities.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    expected = canonical_arrays(source_arrays)
    if arrays.keys() != expected.keys() or any(
            arrays[key].dtype != expected[key].dtype or not np.array_equal(arrays[key], value)
            for key, value in expected.items()):
        raise ValueError("Canonical modalities differ from the original online recording")
    check = validate_online_arrays(source_arrays, require_causal=meta["schema"] != "five-task-episode-v2")
    if meta['schema'] == SCHEMA and check['schema'] != 'five-task-online-v4':
        raise ValueError('V4 Episode lacks full precision replay state')
    if len(meta["object_names"]) != arrays["object_state"].shape[1]:
        raise ValueError("Object identity count differs from object_state")
    source_meta = json.loads((directory / "source_episode/manifest.json").read_text())
    stage_encoding = validate_stage_encoding(source_arrays, source_meta['metadata'])
    if meta["success"] != source_meta["success"] or meta["object_id"] != source_meta["object_id"]:
        raise ValueError("Task outcome or object identity changed during packaging")
    if meta["object_names"] != source_meta["metadata"]["online_contract"]["object_names"]:
        raise ValueError("Object names changed during packaging")
    model_provenance = verify_provenance(directory / "source_episode", source_meta["metadata"]["online_contract"])
    if meta['schema'] == SCHEMA:
        contract = source_meta['metadata']['online_contract']
        if (not model_provenance.get('passed')
                or contract.get('integration_state_spec') != model_provenance['integration_state_spec']
                or source_arrays['pre_integration_state'].shape[1] != model_provenance['integration_state_size']
                or source_arrays['applied_ctrl64'].shape[1] != model_provenance['model_nu']
                or not np.allclose(source_arrays['pre_integration_state'][:, 0], arrays['timestamp'], rtol=0, atol=1e-10)):
            raise ValueError('V4 exact state dimensions or clock differ from compiled model')
    replay_evidence = {"present": False}
    if "independent_replay.json" in meta["files"]:
        replay_evidence = json.loads((directory / "independent_replay.json").read_text())
        if (replay_evidence.get("manifest_sha256") != meta["files"]["source_episode/manifest.json"]
                or not replay_evidence.get("passed")
                or replay_evidence.get("source_success") != meta["success"]
                or replay_evidence.get("task") != meta["task"]
                or replay_evidence.get("frames") != len(arrays["timestamp"])
                or not replay_evidence.get("unchanged_task_criterion")):
            raise ValueError("Independent replay evidence differs from packaged source")
        replay_evidence = {"present": True, **replay_evidence}
    expected_lineage = canonical_hash({"task": meta["task"], "asset_hash": meta["asset_hash"],
                                      "source_episode_sha256": meta["files"]["source_episode/episode.npz"]})
    if meta["lineage_hash"] != expected_lineage:
        raise ValueError("Lineage hash differs from the recorded source")
    if meta["success"] and not np.any(arrays["tactile"] > 0):
        raise ValueError("Successful interaction has no measured tactile response")
    rgb_frames, capture = recorded_rgb_contract(source_arrays, source_meta['metadata']['online_contract'])
    if _video_proof is None:
        video = (verify_video(directory / 'agentview.mp4', rgb_frames) if capture is None else
                 verify_video(directory / 'agentview.mp4', rgb_frames, capture=capture))
    elif isinstance(_video_proof, _DecodedVideoProof):
        video = _video_proof.checked_result(directory / 'agentview.mp4', rgb_frames, capture)
    else:
        raise ValueError('Video verification requires an in-process proof, never saved JSON')
    digest = content_hash(arrays, video["decoded_rgb_sha256"])
    if digest != meta["content_hash"]:
        raise ValueError("Semantic content hash differs")
    if meta.get("admission") == "production_verified":
        if (not model_provenance.get("passed") or not replay_evidence.get("passed")
                or not check["command_causality_checked"] or meta.get("objects_need_asset_role_resolution", True)):
            raise ValueError("Formal admission lacks model, replay, causal or object identity proof")
        if (replay_evidence.get('initial_state') != 'pre_action'
                or replay_evidence.get('skipped_initial_actions') != 0
                or replay_evidence.get('replay_final_success') != meta['success']):
            raise ValueError('Formal replay must cover the complete pre-action episode outcome')
        if meta['schema'] == SCHEMA and (not replay_evidence.get('exact_integration_state')
                                        or not replay_evidence.get('exact_recorded_ctrl')):
            raise ValueError('Formal V4 replay requires complete state and exact controls')
        if "job_payload.json" not in meta["files"]:
            raise ValueError("Formal admission lacks immutable job payload")
        payload = json.loads((directory / "job_payload.json").read_text())
        if capture_contract(capture) != capture_contract(payload.get('rgb_capture')):
            raise ValueError('RGB recording contract differs from the immutable job payload')
        if (canonical_hash(payload) != meta["job_payload_hash"] or payload.get("asset_hash") != meta["asset_hash"]
                or payload.get("task") != meta["task"]):
            raise ValueError("Formal job payload differs from Episode")
        contract = source_meta['metadata']['online_contract']
        if (contract.get('environment_conditions') != payload.get('conditions')
                or contract.get('reference_manifest_sha256') != payload.get('reference_manifest_sha256')):
            raise ValueError('Formal execution conditions differ from immutable job')
        if 'worker_result.json' not in meta['files']:
            raise ValueError('Formal admission lacks completed worker evidence')
        worker = json.loads((directory / 'worker_result.json').read_text())
        if (not worker.get('complete') or not worker.get('replay_passed') or worker.get('success') != meta['success']):
            raise ValueError('Formal worker outcome differs from Episode')
        primary = "cubeA" if meta["task"] == "stack" else "object"
        expected_identities = [{**entity,
            "identity_kind": "catalogue_asset" if entity["name"] == primary else "model_entity",
            "identity_sha256": meta["asset_hash"] if entity["name"] == primary else canonical_hash({
                "model": model_provenance["compiled_model_sha256"], "entity": entity})}
            for entity in model_provenance["objects"]]
        if (meta.get("object_identities") != expected_identities
                or sum(row["identity_kind"] == "catalogue_asset" for row in expected_identities) != 1):
            raise ValueError("Formal object identities differ from model and asset payload")
        if meta["task"] == "lift" and meta["success"]:
            if "strict_final.json" not in meta["files"]:
                raise ValueError("Formal Lift admission lacks FINAL Strict evidence")
            strict = json.loads((directory / "strict_final.json").read_text())
            if (strict.get("manifest_sha256") != meta["files"]["source_episode/manifest.json"]
                    or not strict.get("result", {}).get("success")
                    or strict['result'].get('profile') != 'final'
                    or strict['result'].get('verification_status') != 'FINAL_VERIFIED'):
                raise ValueError("Formal Lift FINAL Strict evidence differs")
    return {"passed": True, **check,
            "task_outcome": reported_task_outcome(meta['success'], source_meta.get('failure_reason')),
            "stage_encoding": stage_encoding, "video": video, "model_provenance": model_provenance,
            "independent_replay": replay_evidence, "content_hash": digest,
            "success": bool(meta["success"]), "lineage_hash": meta["lineage_hash"],
            "asset_hash": meta["asset_hash"], "task": meta["task"]}


def package_episode(case, root, job_id, asset_hash, *, fault=None, job_payload=None, formal=False):
    """Package one already executed case; a copy never represents new data time."""
    if not job_id or Path(job_id).name != job_id or job_id in ('.', '..'):
        raise ValueError("Invalid job ID")
    if len(asset_hash) != 64 or any(c not in '0123456789abcdef' for c in asset_hash):
        raise ValueError("Expected independent asset SHA-256")
    result = json.loads((case / "result.json").read_text())
    if not result.get("contract_passed") or result.get("rgb_status") != "DECODE_VERIFIED":
        raise ValueError("Case does not have complete verified online RGB and modalities")
    manifest = json.loads((case / "source_episode/manifest.json").read_text())
    contract = manifest["metadata"].get("online_contract", {})
    if contract.get("observation_timing") != "before_action":
        raise ValueError("No online pre-action provenance")
    with np.load(case / "source_episode/episode.npz", allow_pickle=False) as archive:
        source_arrays = {key: archive[key] for key in archive.files}
    arrays = canonical_arrays(source_arrays)
    lineage_hash = canonical_hash({
        "task": result["task"], "asset_hash": asset_hash,
        "source_episode_sha256": sha256_file(case / "source_episode/episode.npz")})
    transaction_root = root / ".transactions"
    episode_root = root / "episodes"
    transaction_root.mkdir(parents=True, exist_ok=True)
    episode_root.mkdir(parents=True, exist_ok=True)
    destination = episode_root / job_id
    if destination.exists():
        raise FileExistsError(destination)
    stage = transaction_root / f"{job_id}-{uuid.uuid4().hex}"
    stage.mkdir()
    copy_source_contents(case / "source_episode", stage / "source_episode")
    shutil.copyfile(case / "agentview.mp4", stage / "agentview.mp4")
    for report in ("independent_replay.json", "strict_final.json", "worker_result.json"):
        if (case / report).is_file():
            shutil.copyfile(case / report, stage / report)
    np.savez_compressed(stage / "modalities.npz", **arrays)
    rgb_frames, capture = recorded_rgb_contract(source_arrays, contract)
    video_proof = _DecodedVideoProof(stage / 'agentview.mp4', rgb_frames, capture)
    video = video_proof.checked_result(stage / 'agentview.mp4', rgb_frames, capture)
    files = {str(path.relative_to(stage)): sha256_file(path)
             for path in stage.rglob('*') if path.is_file()}
    artifact_schema = {'five-task-online-v4': SCHEMA, 'five-task-online-v3': 'five-task-episode-v3'}.get(
        contract.get('schema'), 'five-task-episode-v2')
    metadata = {"schema": artifact_schema, "job_id": job_id, "task": result["task"],
                "job_payload_hash": canonical_hash(job_payload or {}),
                "asset_hash": asset_hash, "object_id": manifest["object_id"],
                "object_names": contract["object_names"], "success": bool(manifest["success"]),
                "lineage_hash": lineage_hash,
                "content_hash": content_hash(arrays, video["decoded_rgb_sha256"]),
                "files": files, "origin_case": str(case),
                "admission": "development_verified_not_formal_production",
                "canonical_action": "arm7_hand6_position_command_v1",
                "sensor_provenance": contract.get("sensor"),
                "objects_need_asset_role_resolution": True}
    if formal:
        if not job_payload or job_payload.get("asset_hash") != asset_hash or job_payload.get("task") != result["task"]:
            raise ValueError("Formal packaging requires the immutable asset/task payload")
        write_json(stage / "job_payload.json", job_payload)
        metadata["files"]["job_payload.json"] = sha256_file(stage / "job_payload.json")
        model_proof = verify_provenance(stage / "source_episode", contract)
        if not model_proof.get("passed"):
            raise ValueError("Formal packaging requires a compiled model")
        primary = "cubeA" if result["task"] == "stack" else "object"
        identities = []
        for entity in model_proof["objects"]:
            identities.append({**entity, "identity_kind": "catalogue_asset" if entity["name"] == primary else "model_entity",
                               "identity_sha256": asset_hash if entity["name"] == primary else canonical_hash({
                                   "model": model_proof["compiled_model_sha256"], "entity": entity})})
        if sum(row["identity_kind"] == "catalogue_asset" for row in identities) != 1:
            raise ValueError("Formal primary object cannot be resolved in compiled model")
        metadata.update(admission="production_verified", objects_need_asset_role_resolution=False,
                        object_identities=identities)
    write_json(stage / "episode.json", metadata)
    verification = verify_episode(stage, _video_proof=video_proof)
    write_json(stage / "verification.json", verification)
    for path in stage.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    sync_directory(stage / "source_episode")
    sync_directory(stage)
    if fault == "before_rename":
        raise RuntimeError("Injected interruption before atomic rename")
    # A job may only be finalized once; transaction retries use recovery first.
    stage.rename(destination)
    sync_directory(episode_root)
    sync_directory(transaction_root)
    if fault == "after_rename":
        raise RuntimeError("Injected interruption after atomic rename")
    return destination, verification


def reconcile_episode(root, job_id):
    """Inspect completed transactions without deleting or rerunning physics."""
    paths = [root / kind / job_id for kind in ("episodes", "duplicates")]
    completed = [path for path in paths if path.is_dir()]
    if len(completed) > 1:
        raise ValueError("Multiple finalized directories for one job require review")
    if completed:
        path = completed[0]
        return {"state": "READY_TO_REGISTER", "path": str(path), "verification": verify_episode(path)}
    staged = []
    for stage in (root / ".transactions").glob(f"{job_id}-*"):
        try:
            verification = verify_episode(stage)
            staged.append({"path": str(stage), "verified": True, "verification": verification})
        except Exception as error:
            staged.append({"path": str(stage), "verified": False, "error": str(error)})
    return {"state": "STAGED_REQUIRES_REVIEW" if staged else "NO_ARTIFACT", "staged": staged}


def register_artifact(queue, job, root, *, allow_development=False):
    """Finish file-to-inventory transaction; safe to retry after a lost worker."""
    recovered = reconcile_episode(root, job["job_id"])
    if recovered["state"] == "STAGED_REQUIRES_REVIEW":
        candidates = [row for row in recovered["staged"] if row["verified"]]
        if len(candidates) != 1 or len(recovered["staged"]) != 1:
            raise ValueError("Ambiguous or invalid staged transactions require review")
        source = Path(candidates[0]["path"])
        destination = root / "episodes" / job["job_id"]
        source.rename(destination)
        sync_directory(destination.parent)
        recovered = reconcile_episode(root, job["job_id"])
    if recovered["state"] != "READY_TO_REGISTER":
        raise ValueError("No complete verified Episode to register")
    path = Path(recovered["path"])
    meta = json.loads((path / "episode.json").read_text())
    if meta["job_id"] != job["job_id"] or meta["asset_hash"] != job["asset_hash"] or meta["task"] != job["task"]:
        raise ValueError("Episode identity differs from claimed job")
    if meta["job_payload_hash"] != canonical_hash(job["payload"]):
        raise ValueError("Episode conditions differ from immutable job payload")
    if not allow_development and meta["admission"] != "production_verified":
        raise ValueError("Development package is not admitted to formal production")
    verification = recovered["verification"]
    queue.validating(job["job_id"], job["lease_token"])
    state = queue.register(job["job_id"], job["lease_token"],
                          lineage_hash=meta["lineage_hash"], content_hash=meta["content_hash"],
                          relative_path=str(path.relative_to(root)),
                          seconds=verification["duration_seconds"], success=meta["success"],
                          verification=verification)
    if state == "DUPLICATE":
        isolate_duplicate(root, job["job_id"])
    return state


def isolate_duplicate(root, job_id):
    source = root / "episodes" / job_id
    destination = root / "duplicates" / job_id
    if source.is_dir():
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        source.rename(destination)
        sync_directory(source.parent)
        sync_directory(destination.parent)
