import json
import time
import subprocess
import sys

import numpy as np
import pytest

from source.data.production_artifacts import package_episode, register_artifact, verify_episode, content_hash
from source.data.production_queue import ProductionQueue
from source.data.production_video import OnlineVideo
from source.data.production_inventory import audit_inventory
from source.data.five_task_dataset import sha256_file


def test_formal_nut_identity_binds_bolt_not_fixed_nut():
    from source.data.production_artifacts import model_object_identities

    bolt = {'name': 'assembly_bolt', 'model_kind': 'body', 'model_name': 'assembly_bolt_body'}
    fixture = {'name': 'fixed_nut_body', 'model_kind': 'body', 'model_name': 'fixed_nut_body'}
    proof = {'compiled_model_sha256': 'a' * 64, 'objects': [bolt, fixture]}
    identities = model_object_identities('nut_assembly', 'b' * 64, proof)
    assert identities[0]['identity_kind'] == 'catalogue_asset'
    assert identities[0]['identity_sha256'] == 'b' * 64
    assert identities[1]['identity_kind'] == 'model_entity'
    with pytest.raises(ValueError, match='primary object'):
        model_object_identities('nut_assembly', 'b' * 64, {**proof, 'objects': [fixture]})
    with pytest.raises(ValueError, match='primary object'):
        model_object_identities('nut_assembly', 'b' * 64, {**proof, 'objects': [bolt, bolt, fixture]})


def test_packaging_does_not_copy_linux_file_attributes(tmp_path, synthetic_case, monkeypatch):
    def unsupported(*args, **kwargs):
        raise PermissionError('DrvFS does not permit copying POSIX metadata')

    monkeypatch.setattr('shutil.copystat', unsupported)
    root = tmp_path / 'drvfs_format_fixture'
    directory, verification = package_episode(synthetic_case, root, 'job', 'a' * 64)
    assert verification['passed']
    for path in (synthetic_case / 'source_episode').rglob('*'):
        if path.is_file():
            assert sha256_file(path) == sha256_file(directory / 'source_episode' / path.relative_to(synthetic_case / 'source_episode'))


def test_content_tree_copy_rejects_links_and_existing_destination(tmp_path):
    from source.data.production_artifacts import copy_source_contents

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'outside_link').symlink_to(tmp_path / 'unrelated')
    with pytest.raises(ValueError, match='Symlinks'):
        copy_source_contents(source, tmp_path / 'copy')
    with pytest.raises(FileExistsError):
        copy_source_contents(source, tmp_path / 'copy')


def test_versioned_stage_mapping_is_checked_in_packaging(tmp_path, synthetic_case):
    from source.data.production_recorder import STAGE_ENCODING, STAGE_CODES_V1

    path = synthetic_case / 'source_episode/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['metadata']['stage_codes'] = dict(STAGE_CODES_V1)
    manifest['metadata']['online_contract']['stage_encoding'] = STAGE_ENCODING
    path.write_text(json.dumps(manifest))
    _, check = package_episode(synthetic_case, tmp_path / 'valid', 'job', 'a' * 64)
    assert check['stage_encoding']['checked']
    manifest['metadata']['stage_codes']['transit'], manifest['metadata']['stage_codes']['pregrasp'] = 2, 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='versioned contract'):
        package_episode(synthetic_case, tmp_path / 'bad', 'job', 'a' * 64)


def test_valid_failed_episode_exposes_reported_reason_to_file_audit(tmp_path, synthetic_case):
    source = synthetic_case / 'source_episode/manifest.json'
    manifest = json.loads(source.read_text())
    manifest.update(success=False, failure_reason='object was not released stably inside target region')
    source.write_text(json.dumps(manifest))
    directory, check = package_episode(synthetic_case, tmp_path / 'valid_failure', 'job', 'a' * 64)
    assert check['passed'] and not check['success']
    assert check['task_outcome']['failure_category'] == 'PLACEMENT_CRITERION'
    assert verify_episode(directory)['task_outcome']['failure_reason'] == manifest['failure_reason']


def test_replay_auxiliary_precision_does_not_manufacture_unique_observed_content():
    observed = {'timestamp': np.array([0., .05]), 'state': np.ones((2, 39), np.float32)}
    digest = content_hash(observed, 'a' * 64)
    with_checkpoint = {**observed, 'integration_state': np.ones((2, 20), np.float64),
                       'control_exact': np.ones((2, 13), np.float64)}
    assert content_hash(with_checkpoint, 'a' * 64) == digest


@pytest.fixture
def synthetic_case(tmp_path):
    """Artificial format fixture, never a physical simulation or production sample."""
    case = tmp_path / "artificial_format_fixture"
    source = case / "source_episode"
    source.mkdir(parents=True)
    n = 3
    arrays = {"timestamp": np.arange(n) * .05, "next_timestamp": (np.arange(n) + 1) * .05,
              "state": np.zeros((n, 39), np.float32),
              "position_action": np.ones((n, 13), np.float32),
              "ik_action": np.ones((n, 13), np.float32),
              "sensor_tactile": np.ones((n, 712), np.float32),
              "object_state": np.tile(np.array([0, 0, 0, 1, 0, 0, 0], np.float32), (n, 1, 1)),
              "stage": np.ones(n, np.int16), "task_success": np.ones(n, bool)}
    np.savez_compressed(source / "episode.npz", **arrays)
    (source / "manifest.json").write_text(json.dumps({
        "success": True, "object_id": "synthetic_format_fixture", "metadata": {"online_contract": {
            "observation_timing": "before_action", "object_names": ["object"]}}}))
    (case / "result.json").write_text(json.dumps({
        "task": "lift", "contract_passed": True, "rgb_status": "DECODE_VERIFIED"}))
    video = OnlineVideo(case / "agentview.mp4", timeout=20)
    for shade in (0, 80, 160):
        video.submit(np.full((1080, 1920, 3), shade, np.uint8))
    video.finish()
    return case


@pytest.mark.parametrize("fault", ["before_rename", "after_rename"])
def test_crash_recovery_registers_existing_file_without_regeneration(tmp_path, synthetic_case, fault):
    root = tmp_path / "dataset"
    queue = ProductionQueue(tmp_path / "queue.sqlite")
    queue.enqueue("job", "lift", "a" * 64, {})
    old = queue.claim()
    with pytest.raises(RuntimeError, match="Injected"):
        package_episode(synthetic_case, root, "job", "a" * 64, fault=fault)
    queue.recover(now=time.time() + 1000)
    job = queue.claim()
    assert old["lease_token"] != job["lease_token"]
    assert register_artifact(queue, job, root, allow_development=True) == "EXPORTED"
    assert queue.status()["episodes"] == 1
    assert queue.status()["eligible_hours"] == pytest.approx(.15 / 3600)
    assert len(list((root / "episodes").iterdir())) == 1


def test_file_tampering_and_formal_admission_are_rejected(tmp_path, synthetic_case):
    root = tmp_path / "dataset"
    queue = ProductionQueue(tmp_path / "queue.sqlite")
    queue.enqueue("job", "lift", "a" * 64, {})
    job = queue.claim()
    path, _ = package_episode(synthetic_case, root, "job", "a" * 64)
    with pytest.raises(ValueError, match="not admitted"):
        register_artifact(queue, job, root)
    with (path / "modalities.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="content changed"):
        verify_episode(path)
    assert queue.status()["episodes"] == 0


def test_self_declared_formal_admission_requires_verifiable_proofs(tmp_path, synthetic_case):
    path, _ = package_episode(synthetic_case, tmp_path / "dataset", "job", "a" * 64)
    meta_path = path / "episode.json"
    meta = json.loads(meta_path.read_text())
    meta.update(admission="production_verified", objects_need_asset_role_resolution=False)
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Formal admission lacks"):
        verify_episode(path)


def test_replay_proof_is_preserved_and_bound_to_source(tmp_path, synthetic_case):
    report = {"manifest_sha256": sha256_file(synthetic_case / "source_episode/manifest.json"),
              "task": "lift", "frames": 3, "source_success": True,
              "passed": True, "unchanged_task_criterion": True}
    (synthetic_case / "independent_replay.json").write_text(json.dumps(report))
    path, result = package_episode(synthetic_case, tmp_path / "dataset", "job", "a" * 64)
    assert result["independent_replay"]["present"]
    assert json.loads((path / "independent_replay.json").read_text()) == report
    report["manifest_sha256"] = "b" * 64
    (synthetic_case / "independent_replay.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="replay evidence differs"):
        package_episode(synthetic_case, tmp_path / "dataset", "wrong_source", "a" * 64)


def test_file_audit_checks_immutable_queue_payload_not_only_episode_totals(tmp_path, synthetic_case):
    root = tmp_path / 'dataset'
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    queue.enqueue('job', 'lift', 'a' * 64, {})
    job = queue.claim()
    package_episode(synthetic_case, root, 'job', 'a' * 64)
    assert register_artifact(queue, job, root, allow_development=True) == 'EXPORTED'
    with queue.transaction() as db:
        db.execute('UPDATE jobs SET payload=? WHERE job_id=?', ('{"changed":true}', 'job'))
    result = audit_inventory(root, queue.path)
    assert not result['passed']
    assert 'immutable queue plan' in result['errors'][0]['error']


def test_real_worker_exit_after_rename_is_recovered_without_waiting_lease(tmp_path, synthetic_case):
    root = tmp_path / "dataset"
    queue = ProductionQueue(tmp_path / "queue.sqlite")
    queue.enqueue("job", "lift", "a" * 64, {})
    program = """
import os,sys
from pathlib import Path
from source.data.production_queue import ProductionQueue
from source.data.production_artifacts import package_episode
q=ProductionQueue(Path(sys.argv[1])); q.claim(lease_seconds=300)
try:
    package_episode(Path(sys.argv[2]),Path(sys.argv[3]),'job','a'*64,fault='after_rename')
except RuntimeError:
    os._exit(42)
"""
    result = subprocess.run([sys.executable, "-c", program, str(queue.path), str(synthetic_case), str(root)],
                            timeout=20)
    assert result.returncode == 42
    assert queue.status()["episodes"] == 0
    assert queue.recover(abandoned=True) == 1
    job = queue.claim()
    assert register_artifact(queue, job, root, allow_development=True) == "EXPORTED"
    assert queue.status()["episodes"] == 1


def test_duplicate_copy_is_preserved_outside_valid_episode_directory(tmp_path, synthetic_case):
    root = tmp_path / "dataset"
    queue = ProductionQueue(tmp_path / "queue.sqlite")
    for key in ("one", "two"):
        queue.enqueue(key, "lift", "a" * 64, {})
        job = queue.claim()
        package_episode(synthetic_case, root, key, "a" * 64)
        state = register_artifact(queue, job, root, allow_development=True)
        assert state == ("EXPORTED" if key == "one" else "DUPLICATE")
    assert queue.status()["episodes"] == 1
    assert (root / "duplicates/two/agentview.mp4").is_file()
    assert not (root / "episodes/two").exists()
    audit = audit_inventory(root, queue.path)
    assert audit["passed"]
    assert audit["verified_episodes"] == 1
    assert not audit["all_formally_admitted"]


def test_packaging_decodes_once_but_registration_and_final_audit_decode_independently(tmp_path, synthetic_case, monkeypatch):
    import source.data.production_artifacts as module

    original = module.verify_video
    decoded = []

    def checked(path, frames):
        decoded.append(str(path))
        return original(path, frames)

    monkeypatch.setattr(module, 'verify_video', checked)
    root = tmp_path / 'dataset'
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    queue.enqueue('job', 'lift', 'a' * 64, {})
    job = queue.claim()
    package_episode(synthetic_case, root, 'job', 'a' * 64)
    assert len(decoded) == 1
    assert register_artifact(queue, job, root, allow_development=True) == 'EXPORTED'
    assert len(decoded) == 2
    assert audit_inventory(root, queue.path)['passed']
    assert len(decoded) == 3


def test_decode_proof_cannot_be_reused_for_changed_bytes_or_saved_json(tmp_path, synthetic_case):
    from source.data.production_artifacts import _DecodedVideoProof

    video = synthetic_case / 'agentview.mp4'
    proof = _DecodedVideoProof(video, 3)
    with pytest.raises(ValueError, match='differs'):
        proof.checked_result(video, 4)
    different = tmp_path / 'same_bytes_other_path.mp4'
    different.write_bytes(video.read_bytes())
    with pytest.raises(ValueError, match='differs'):
        proof.checked_result(different, 3)
    with video.open('ab') as stream:
        stream.write(b'changed after verification')
    with pytest.raises(ValueError, match='differs'):
        proof.checked_result(video, 3)
    # Public independent verification does not consume a persisted JSON verdict.
    path, _ = package_episode(synthetic_case, tmp_path / 'dataset', 'job', 'a' * 64)
    with pytest.raises(ValueError, match='never saved JSON'):
        verify_episode(path, _video_proof={'decoded_rgb_sha256': 'a' * 64})


def test_decode_proof_rejects_video_changed_while_decoding(synthetic_case, monkeypatch):
    import source.data.production_artifacts as module

    original = module.verify_video

    def decode_then_change(path, frames):
        result = original(path, frames)
        with path.open('ab') as stream:
            stream.write(b'changed concurrently')
        return result

    monkeypatch.setattr(module, 'verify_video', decode_then_change)
    with pytest.raises(ValueError, match='during complete decode'):
        module._DecodedVideoProof(synthetic_case / 'agentview.mp4', 3)
