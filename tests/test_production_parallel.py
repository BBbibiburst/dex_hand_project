import copy
import json
from pathlib import Path
import threading
import time

import pytest

from apps.five_task_production import load_config
from source.data.production_parallel import POLICY, worker_plan, WorkerAdmission


GIB = 1024**3


def adaptive():
    config = json.loads(Path('configs/five_task_collection.json').read_text())
    return {**config, 'workers': 'auto', 'worker_policy': POLICY}


def test_server_gets_more_than_two_workers_without_host_specs():
    result = worker_plan(adaptive(), {'cpu_capacity': 32, 'total_memory_bytes': 96*GIB,
                                     'available_memory_bytes': 80*GIB})
    assert result['pool_workers'] == result['initial_admitted_workers'] == 16


def test_small_host_remains_memory_limited_and_auto_is_explicit(tmp_path):
    config = adaptive()
    result = worker_plan(config, {'cpu_capacity': 8, 'total_memory_bytes': 8*GIB,
                                  'available_memory_bytes': int(3.5*GIB)})
    assert result['pool_workers'] == 3
    assert result['initial_admitted_workers'] == 1
    path = tmp_path / 'auto.json'
    path.write_text(json.dumps(config))
    assert load_config(path)['workers'] == 'auto'
    config.pop('worker_policy')
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='explicit'):
        load_config(path)


def test_pending_processes_reserve_memory_before_their_rss_increases(monkeypatch):
    resources = {'cpu_capacity': 8, 'total_memory_bytes': 8*GIB, 'available_memory_bytes': int(3.5*GIB)}
    monkeypatch.setattr('source.data.production_parallel.hardware_resources', lambda: resources)
    gate = WorkerAdmission(adaptive())
    stopped = threading.Event()
    assert gate.acquire(0, stopped)
    # Available RAM has not fallen yet, but the starting process already owns a reservation.
    assert not gate.acquire(1, stopped, deadline=time.monotonic()+.02)
    gate.release(0)
    assert gate.acquire(1, threading.Event())
    gate.release(1)
    assert not gate.active


def test_worker_error_releases_admission_and_later_host_growth_is_used(monkeypatch):
    resources = {'cpu_capacity': 8, 'total_memory_bytes': 16*GIB, 'available_memory_bytes': 2*GIB}
    monkeypatch.setattr('source.data.production_parallel.hardware_resources', lambda: resources)
    gate = WorkerAdmission(adaptive())
    assert gate.plan['initial_admitted_workers'] == 0
    resources['available_memory_bytes'] = 10*GIB

    @gate.cleanup_on_exit
    def worker(slot):
        assert gate.acquire(slot, threading.Event())
        raise RuntimeError('test worker failed')

    with pytest.raises(RuntimeError):
        worker(0)
    assert not gate.active
    assert gate.acquire(1, threading.Event())
    gate.release(1)


def test_fixed_legacy_limits_and_explicit_adaptive_ceiling():
    config = adaptive()
    config.update(workers=12, maximum_workers=6)
    result = worker_plan(config, {'cpu_capacity': 64, 'total_memory_bytes': 128*GIB,
                                 'available_memory_bytes': 100*GIB})
    assert result['pool_workers'] == 6
    bad = copy.deepcopy(config)
    bad['estimated_worker_memory_bytes'] = True
    with pytest.raises(ValueError):
        worker_plan(bad)
    assert worker_plan({'workers': 2})['pool_workers'] == 2
