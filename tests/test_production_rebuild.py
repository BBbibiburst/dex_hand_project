import json

import pytest

from test_production_artifacts import synthetic_case as format_fixture
from source.data.production_artifacts import package_episode
from source.data.production_rebuild import rebuild_inventory


@pytest.fixture
def synthetic_case(tmp_path):
    return format_fixture.__wrapped__(tmp_path)


def test_rebuild_checks_files_preserves_original_and_does_not_claim(tmp_path, synthetic_case):
    root = tmp_path / 'dataset'
    package_episode(synthetic_case, root, 'job', 'a' * 64)
    old = root / 'inventory.sqlite'
    old.write_bytes(b'corrupt database evidence')
    pending = root / 'work' / 'pending_job'
    pending.mkdir(parents=True)
    report = rebuild_inventory(root, tmp_path / 'recovered')
    assert old.read_bytes() == b'corrupt database evidence'
    assert report['audit']['passed']
    assert report['audit']['verified_episodes'] == 1
    assert report['unresolved_work'] == ['pending_job']
    assert report['new_simulation_seconds'] == 0
    assert not report['automatic_resume_allowed']
    import sqlite3

    with sqlite3.connect(report['candidate_database']) as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchall() == [('EXPORTED', 0)]
        assert db.execute("SELECT count(*) FROM events WHERE state='RUNNING'").fetchone()[0] == 0


def test_rebuild_refuses_tampered_files_and_existing_output(tmp_path, synthetic_case):
    root = tmp_path / 'dataset'
    path, _ = package_episode(synthetic_case, root, 'job', 'a' * 64)
    destination = tmp_path / 'candidate'
    destination.mkdir()
    with pytest.raises(ValueError, match='fresh'):
        rebuild_inventory(root, destination)
    metadata = json.loads((path / 'episode.json').read_text())
    metadata['job_payload_hash'] = 'b' * 64
    (path / 'episode.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='payload'):
        rebuild_inventory(root, tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists()
