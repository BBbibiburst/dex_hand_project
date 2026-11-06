import hashlib
import json

import numpy as np
import pytest

from apps.five_task_production import load_config
from source.data.production_duration import candidate_duration
from source.data.production_queue import ProductionQueue
from source.data.production_statistics import production_statistics
from pathlib import Path


def test_duration_requires_checked_arrays_and_actual_timestamps(tmp_path):
    path = tmp_path / 'modalities.npz'
    np.savez_compressed(path, timestamp=np.array([4., 4.05, 4.10]),
                        next_timestamp=np.array([4.05, 4.10, 4.15]))
    record = {'contract_passed': True, 'modalities_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / 'result.json').write_text(json.dumps(record))
    assert candidate_duration(tmp_path) == pytest.approx(.15)
    assert candidate_duration(None) is None
    path.write_bytes(b'changed')
    assert candidate_duration(tmp_path) is None


def test_pre_dedup_duration_never_adds_duplicate_to_registered_hours(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    config = {**load_config(Path('configs/five_task_production.json')), 'queue_database': str(queue.path)}
    for name in ('one', 'two'):
        queue.enqueue(name, 'lift', 'a' * 64, {})
        job = queue.claim()
        queue.validating(name, job['lease_token'])
        state = queue.register(name, job['lease_token'], lineage_hash='b' * 64, content_hash='c' * 64,
                               relative_path='episodes/' + name, seconds=10., success=True, verification={'passed': True})
        queue.record_attempt_metrics(name, 1, 20., state, candidate_seconds=10.)
    result = production_statistics(config)['assets'][0]
    assert result['pre_dedup_candidate_seconds'] == 20
    assert result['post_dedup_registered_seconds'] == 10
    assert result['known_duplicate_candidate_seconds'] == 10
    with pytest.raises(ValueError, match='immutable'):
        queue.record_attempt_metrics('two', 1, 20., 'DUPLICATE', candidate_seconds=11.)


def test_old_inventory_read_only_missing_duration_remains_unknown(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    queue.enqueue('one', 'lift', 'a' * 64, {})
    queue.claim()
    queue.record_attempt_metrics('one', 1, 5., 'QUARANTINED')
    with queue.connect() as db:
        db.execute('ALTER TABLE attempt_metrics DROP COLUMN candidate_seconds')
        before = list(db.iterdump())
    config = {**load_config(Path('configs/five_task_production.json')), 'queue_database': str(queue.path)}
    assert production_statistics(config)['assets'][0]['pre_dedup_candidate_seconds'] is None
    with queue.connect() as db:
        assert list(db.iterdump()) == before
    # Explicit opening for resumed execution upgrades only the monitoring schema.
    ProductionQueue(queue.path)
    assert production_statistics(config)['assets'][0]['pre_dedup_candidate_seconds'] is None
