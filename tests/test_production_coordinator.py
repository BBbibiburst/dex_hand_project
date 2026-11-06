from pathlib import Path

import pytest

from apps.five_task_production import load_config
from source.data.production_coordinator import pilot
from source.data.production_queue import ProductionQueue
from source.data.five_task_dataset import canonical_hash


def test_pilot_cannot_start_under_formal_dataset_root():
    config = load_config(Path('configs/five_task_production.json'))
    with pytest.raises(ValueError, match='separate project outputs'):
        pilot(config, Path('/mnt/f/dex_hand_dataset_1000h'), 1, 32000)


def test_pilot_has_explicit_attempt_budget():
    config = load_config(Path('configs/five_task_production.json'))
    with pytest.raises(ValueError, match='1 to 5'):
        pilot(config, Path('outputs/should_not_be_created_by_test'), 1000, 32000)


def test_canary_does_not_offer_unbounded_or_formal_root_execution():
    from source.data.production_coordinator import canary

    config = load_config(Path('configs/five_task_production.json'))
    with pytest.raises(ValueError, match='outputs directory'):
        canary(config, Path('/mnt/f/formal_root'), 1, 1, wall_budget=10)
    with pytest.raises(ValueError, match='at most 40'):
        canary(config, Path('outputs/must_not_create_canary'), 41, 1, wall_budget=3600)
    with pytest.raises(ValueError, match='at most 40'):
        canary(config, Path('outputs/must_not_create_canary'), 1, 1, wall_budget=3601)


def test_formal_preflight_rejects_before_creating_root(tmp_path, monkeypatch):
    from source.data.production_coordinator import production
    import source.data.production_preflight as module

    config = load_config(Path('configs/five_task_production.json'))
    config['dataset_root'] = str(tmp_path / 'must_not_exist')
    monkeypatch.setattr(module, 'preflight', lambda _: {'infrastructure_ready': False, 'volume': {'ready': False}})
    with pytest.raises(ValueError, match='infrastructure preflight failed'):
        production(config, 1)
    assert not Path(config['dataset_root']).exists()


def test_common_loop_enqueues_more_candidates_and_obeys_attempt_budget(tmp_path, monkeypatch):
    """Scheduler integration using fake workers, not simulated or formal data."""
    import source.data.production_coordinator as module

    config = load_config(Path('configs/five_task_production.json'))
    config['workers'] = 1
    rows = [{'template_id': 'format_test', 'task': 'lift', 'asset_hash': 'a' * 64,
             'base_seed': 0, 'perturbation_fields': ['dx_m']}]
    monkeypatch.setattr(module, 'load_templates', lambda *_: rows)
    monkeypatch.setattr(module, 'reconcile_episode', lambda *_: {'state': 'NO_ARTIFACT'})
    monkeypatch.setattr(module, 'package_episode', lambda *a, **kw: None)

    class Process:
        pid = __import__('os').getpid()
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: Process())
    monkeypatch.setattr(module, 'stop_child', lambda _: None)  # No real child in this scheduler fixture.

    def register(queue, job, root, **_):
        queue.validating(job['job_id'], job['lease_token'])
        return queue.register(job['job_id'], job['lease_token'], lineage_hash=canonical_hash(job['job_id']),
                              content_hash=canonical_hash(job['job_id']), relative_path='artificial/' + job['job_id'],
                              seconds=1, success=True, verification={'passed': True, 'artificial_test': True})

    monkeypatch.setattr(module, 'register_artifact', register)
    root = tmp_path / 'artificial_coordinator'
    result = module._coordinate(config, root, 3, 42, append_candidates=True)
    assert result['attempted_jobs'] == 3
    assert result['inventory']['episodes'] == 3
    result = module._coordinate(config, root, 1, 42, resume=True, append_candidates=True)
    assert result['inventory']['episodes'] == 4
    stats, _ = ProductionQueue(root / 'inventory.sqlite').scheduling_stats()
    assert stats['format_test']['last_candidate_index'] == 3


def test_expired_session_budget_starts_no_worker(tmp_path, monkeypatch):
    import source.data.production_coordinator as module

    config = load_config(Path('configs/five_task_production.json'))
    monkeypatch.setattr(module, 'load_templates', lambda *_: [])
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No worker should start'))
    result = module._coordinate(config, tmp_path / 'budget_test', 1, 1, wall_budget=1e-20)
    assert result['attempted_jobs'] == 0
    assert result['stopped']


def test_formal_catalogue_cannot_silently_miss_asset_target(tmp_path, monkeypatch):
    import source.data.production_coordinator as module

    config = load_config(Path('configs/five_task_production.json'))
    monkeypatch.setattr(module, 'load_templates', lambda *_: [{'asset_hash': 'a' * 64}])
    root = tmp_path / 'not_created'
    with pytest.raises(ValueError, match='cannot cover'):
        module._coordinate(config, root, None, 1, formal=True)
    assert not root.exists()


def test_exited_worker_does_not_leave_encoder_like_child_alive():
    import os
    import signal
    import subprocess
    import sys
    import time

    from source.data.production_coordinator import stop_child
    from source.data.production_queue import process_identity

    process = subprocess.Popen([sys.executable, '-c',
        'import subprocess,sys,os; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"]); '
        'print(p.pid,flush=True); os._exit(0)'], start_new_session=True, stdout=subprocess.PIPE, text=True)
    try:
        child = int(process.stdout.readline())
        process.wait(timeout=5)
        assert process_identity(child) is not None
        stop_child(process)
        deadline = time.monotonic() + 2
        while process_identity(child) is not None and time.monotonic() < deadline:
            time.sleep(.01)
        assert process_identity(child) is None
    finally:
        process.stdout.close()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
