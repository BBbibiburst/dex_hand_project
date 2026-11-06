import errno
import json
from pathlib import Path
import subprocess
import sys

import pytest

from source.data.production_failures import failure_policy, WorkerFailure
from source.data.production_queue import ProductionQueue


def test_valid_task_failures_are_diagnostics_not_exception_retry_policies():
    from source.data.production_failures import reported_task_outcome

    placement = reported_task_outcome(False, 'object was not released stably inside target region')
    assert placement['failure_category'] == 'PLACEMENT_CRITERION'
    assert 'retryable' not in placement
    assert reported_task_outcome(True, None)['failure_category'] is None
    assert reported_task_outcome(False, None)['failure_category'] == 'UNKNOWN_TASK_FAILURE'
    assert reported_task_outcome(False, 'object fell? unknown cause')['failure_category'] == 'UNKNOWN_TASK_FAILURE'
    assert reported_task_outcome(False, 'pregrasp IK residual is too large: position=0.05')['failure_category'] == 'IK_RESIDUAL_GATE'
    assert reported_task_outcome(False, 'hold has no thumb-opposed skin contact')['failure_category'] == 'CONTACT_REQUIREMENT'


@pytest.mark.parametrize('error,phase,category,retryable,limit,stop', [
    (ValueError('changed hash'), 'INPUT_VALIDATION', 'INPUT_CONTRACT_ERROR', False, 1, False),
    (ValueError('bad replay'), 'REPLAY', 'OUTPUT_VALIDATION_ERROR', False, 1, False),
    (TimeoutError('deadline'), 'SIMULATION', 'EXECUTION_TIMEOUT', True, 2, False),
    (BrokenPipeError('encoder'), 'SIMULATION', 'ENCODER_PIPE_FAILURE', True, 2, False),
    (OSError(errno.ENOSPC, 'disk full'), 'SIMULATION', 'RESOURCE_EXHAUSTED', True, None, True),
    (OSError(errno.EBUSY, 'busy'), 'SIMULATION', 'TRANSIENT_IO', True, 3, False),
    (RuntimeError('unknown'), 'SIMULATION', 'UNKNOWN_EXECUTION_ERROR', False, 1, False),
    (InterruptedError('SIGTERM'), 'SIMULATION', 'SESSION_INTERRUPTED', True, None, True)])
def test_failure_categories_are_typed(error, phase, category, retryable, limit, stop):
    policy = failure_policy(error, phase)
    assert (policy['category'], policy['retryable'], policy['attempt_limit'], policy['stop_coordinator']) == (
        category, retryable, limit, stop)
    assert failure_policy(WorkerFailure(policy), 'RECOVER_OR_EXECUTE') == policy


def test_failure_limit_is_durable_and_never_exceeds_job_limit(tmp_path):
    path = tmp_path / 'queue.sqlite'
    queue = ProductionQueue(path)
    queue.enqueue('timeout', 'lift', 'a' * 64, {}, max_attempts=3)
    policy = failure_policy(TimeoutError('deadline'), 'SIMULATION')
    for attempt, state in ((1, 'PENDING'), (2, 'QUARANTINED')):
        queue = ProductionQueue(path)
        job = queue.claim()
        assert job['attempts'] == attempt
        assert queue.fail(job['job_id'], job['lease_token'], policy['error'],
                          retryable=policy['retryable'], attempt_limit=policy['attempt_limit'], details=policy) == state
    assert queue.claim() is None
    with queue.connect() as db:
        event = db.execute("SELECT details FROM events WHERE state='QUARANTINED'").fetchone()
    assert json.loads(event['details'])['failure_policy']['category'] == 'EXECUTION_TIMEOUT'
    queue.enqueue('one_try', 'lift', 'b' * 64, {}, max_attempts=1)
    job = queue.claim()
    assert queue.fail(job['job_id'], job['lease_token'], 'timeout', attempt_limit=2) == 'QUARANTINED'


def test_worker_policy_cannot_raise_its_own_retry_budget():
    policy = failure_policy(TimeoutError('deadline'), 'SIMULATION')
    policy['attempt_limit'] = 1000
    with pytest.raises(ValueError, match='Malformed'):
        WorkerFailure(policy)


def test_actual_worker_rejects_changed_asset_before_simulation(tmp_path):
    asset = tmp_path / 'asset.bin'
    asset.write_bytes(b'artificial input validation fixture')
    job = tmp_path / 'job.json'
    job.write_text(json.dumps({'reference_manifest': str(tmp_path / 'not_read.json'),
                               'asset_path': str(asset), 'asset_hash': 'a' * 64}))
    output = tmp_path / 'worker'
    process = subprocess.run([sys.executable, 'apps/production_worker.py', '--job', str(job),
                              '--output', str(output), '--budget-seconds', '5'],
                             capture_output=True, text=True, timeout=20)
    (tmp_path / 'worker_failure.log').write_text(process.stderr)
    result = json.loads((output / 'worker_result.json').read_text())
    assert process.returncode != 0
    assert result['failure']['category'] == 'INPUT_CONTRACT_ERROR'
    assert not result['failure']['retryable']
    assert not (output / 'source_episode').exists()


def test_coordinator_uses_worker_policy_instead_of_generic_exit(tmp_path, monkeypatch):
    from apps.five_task_production import load_config
    import source.data.production_coordinator as coordinator

    config = load_config(Path('configs/five_task_production.json'))
    config['workers'] = 1
    templates = [{'template_id': 'fixture', 'task': 'lift', 'asset_hash': 'a' * 64,
                  'base_seed': 0, 'perturbation_fields': ['dx_m']}]
    monkeypatch.setattr(coordinator, 'load_templates', lambda *_: templates)
    monkeypatch.setattr(coordinator, 'reconcile_episode', lambda *_: {'state': 'NO_ARTIFACT'})
    monkeypatch.setattr(coordinator, 'stop_child', lambda _: None)

    class FailedProcess:
        pid = __import__('os').getpid()
        returncode = 1

        def __init__(self, command, **_):
            output = Path(command[command.index('--output') + 1])
            output.mkdir()
            (output / 'worker_result.json').write_text(json.dumps({
                'complete': False, 'failure': failure_policy(TimeoutError('fixture deadline'), 'SIMULATION')}))

        def poll(self):
            return self.returncode

    monkeypatch.setattr(coordinator.subprocess, 'Popen', FailedProcess)
    result = coordinator._coordinate(config, tmp_path / 'dataset', 2, 42)
    assert [row['state'] for row in result['results']] == ['PENDING', 'QUARANTINED']
    assert all(row['failure_policy']['category'] == 'EXECUTION_TIMEOUT' for row in result['results'])
