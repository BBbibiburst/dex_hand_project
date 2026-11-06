import threading
import time
from pathlib import Path

import pytest

from source.data.production_liveness import LeaseKeeper
from source.data.production_queue import ProductionQueue


def job(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    queue.enqueue('fixture', 'lift', 'a' * 64, {})
    return queue, queue.claim(lease_seconds=.3)


def test_lease_survives_long_packaging_without_physics_worker(tmp_path):
    queue, claimed = job(tmp_path)
    stopped = threading.Event()
    keeper = LeaseKeeper(queue, claimed, stopped, interval=.03, lease_seconds=.3)
    keeper.start()
    try:
        time.sleep(.6)  # Longer than the initial lease, standing in for file verification.
        queue.validating(claimed['job_id'], claimed['lease_token'])
        keeper.check()
        assert not stopped.is_set()
    finally:
        keeper.close()


def test_guard_failure_stops_coordinator_and_is_not_hidden(tmp_path):
    queue, claimed = job(tmp_path)
    stopped = threading.Event()
    fail = threading.Event()

    def guard():
        if fail.is_set():
            raise RuntimeError('Simulated mount identity change')

    keeper = LeaseKeeper(queue, claimed, stopped, guard=guard, interval=.02)
    keeper.start()
    try:
        fail.set()
        assert stopped.wait(1)
        with pytest.raises(RuntimeError, match='mount identity change'):
            keeper.check()
    finally:
        keeper.close()


def test_registration_race_is_normal_completion_not_lease_failure(tmp_path):
    queue, claimed = job(tmp_path)
    stopped = threading.Event()
    keeper = LeaseKeeper(queue, claimed, stopped, interval=.02)
    keeper.start()
    try:
        queue.validating(claimed['job_id'], claimed['lease_token'])
        queue.register(claimed['job_id'], claimed['lease_token'], lineage_hash='b' * 64,
                       content_hash='c' * 64, relative_path='artificial/fixture', seconds=1,
                       success=True, verification={'passed': True, 'artificial_test': True})
        assert keeper.finished.wait(1)
        keeper.check()
        assert not stopped.is_set()
    finally:
        keeper.close()


def test_initial_guard_failure_stops_before_any_worker(tmp_path):
    queue, claimed = job(tmp_path)
    stopped = threading.Event()

    def unavailable():
        raise RuntimeError('Volume already unavailable')

    keeper = LeaseKeeper(queue, claimed, stopped, guard=unavailable)
    with pytest.raises(RuntimeError, match='already unavailable'):
        keeper.start()
    assert stopped.is_set()
    assert keeper.thread is None
    keeper.close()


def test_background_failure_callback_preserves_first_diagnostic(tmp_path):
    from source.data.production_monitor import record_runtime_failure
    import json

    queue, claimed = job(tmp_path)
    stopped = threading.Event()
    triggered = threading.Event()

    def guard():
        if triggered.is_set():
            raise RuntimeError('Simulated available memory below reserve: 123 bytes')

    records = []

    def record(error):
        records.append(record_runtime_failure(tmp_path, 0, {'error': str(error), 'job_id': claimed['job_id']}))

    keeper = LeaseKeeper(queue, claimed, stopped, guard=guard, interval=.02, on_failure=record)
    keeper.start()
    try:
        triggered.set()
        assert stopped.wait(1)
    finally:
        keeper.close()
    assert len(records) == 1
    assert '123 bytes' in json.loads(Path(records[0]).read_text())['error']


def test_failure_reporting_error_does_not_hide_original_guard_error(tmp_path):
    queue, claimed = job(tmp_path)

    def guard():
        raise RuntimeError('Primary guard failure')

    def report(_):
        raise OSError('Secondary logging failure')

    keeper = LeaseKeeper(queue, claimed, threading.Event(), guard=guard, on_failure=report)
    with pytest.raises(RuntimeError, match='Primary guard failure'):
        keeper.start()
    assert isinstance(keeper.failure_reporting_error, OSError)
    with pytest.raises(RuntimeError, match='Primary guard failure'):
        keeper.check()
    keeper.close()
