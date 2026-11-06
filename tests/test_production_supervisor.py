import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import os
import signal
import threading

import pytest

from apps.five_task_production import load_config
from source.data.production_supervisor import supervise
from source.data.production_queue import ProductionQueue


@pytest.fixture
def supervision(tmp_path, monkeypatch):
    import source.data.production_supervisor as module

    config = {**load_config(Path('configs/five_task_production.json')),
              'dataset_root': str(tmp_path / 'dataset'), 'queue_database': str(tmp_path / 'control/queue.sqlite')}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(module, 'preflight', lambda _: {'infrastructure_ready': True})
    monkeypatch.setattr(module, 'execution_fingerprint', lambda _: {'fixture': 'frozen'})
    monkeypatch.setattr(module, 'live_dataset_processes', lambda _: [])
    return module, config, path, tmp_path / 'supervisor'


def fake_processes(module, config, monkeypatch, codes, termination='QUOTA_STALLED'):
    commands = []

    class Process:
        pid = 12345

        def __init__(self, command, **_):
            commands.append(command)
            self.returncode = codes[len(commands) - 1]
            Path(config['queue_database']).touch()
            if self.returncode == 0:
                root = Path(config['dataset_root'])
                root.mkdir(exist_ok=True)
                (root / 'summary.json').write_text(json.dumps({'termination': termination}))

        def poll(self):
            return self.returncode

    monkeypatch.setattr(module.subprocess, 'Popen', Process)
    return commands


def test_supervisor_restarts_corrupt_exit_then_uses_resume(supervision, monkeypatch):
    module, config, path, output = supervision
    commands = fake_processes(module, config, monkeypatch, [75, 0])
    result = supervise(config, path, 42, output, maximum_restarts=1, wall_budget=15)
    assert [c[2] for c in commands] == ['run', 'resume']
    assert result['termination'] == 'QUOTA_STALLED'
    assert not result['formal_dataset_accepted']
    assert len(result['sessions']) == 2


@pytest.mark.parametrize('code', [0, 1, 2, -15])
def test_supervisor_does_not_restart_normal_or_nonretryable_exit(supervision, monkeypatch, code):
    module, config, path, output = supervision
    commands = fake_processes(module, config, monkeypatch, [code])
    result = supervise(config, path, 42, output, maximum_restarts=3, wall_budget=15)
    assert len(commands) == 1
    assert len(result['sessions']) == 1


def test_supervisor_preflight_refusal_creates_no_control_files(supervision, monkeypatch):
    module, config, path, output = supervision
    monkeypatch.setattr(module, 'preflight', lambda _: {'infrastructure_ready': False})
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: pytest.fail('Must not launch'))
    result = supervise(config, path, 42, output)
    assert result['termination'] == 'SUPERVISOR_PREFLIGHT_REJECTED'
    assert not output.exists()
    assert not Path(config['queue_database']).parent.exists()


def test_attach_database_failure_cleans_up_actual_child(tmp_path, monkeypatch):
    import source.data.production_coordinator as module

    config = load_config(Path('configs/five_task_production.json'))
    config['workers'] = 1
    monkeypatch.setattr(module, 'load_templates', lambda *_: [
        {'template_id': 'fixture', 'task': 'lift', 'asset_hash': 'a' * 64,
         'base_seed': 0, 'perturbation_fields': ['dx_m']}])
    original = subprocess.Popen
    processes = []

    def launch(_command, **kwargs):
        child = original([sys.executable, '-c', 'import time;time.sleep(20)'], **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    monkeypatch.setattr(ProductionQueue, 'attach_worker', lambda *a: (_ for _ in ()).throw(
        sqlite3.DatabaseError('file is not a database')))
    with pytest.raises(sqlite3.DatabaseError):
        module._coordinate(config, tmp_path / 'dataset', 1, 42)
    assert processes and all(process.poll() is not None for process in processes)


def test_supervisor_sigterm_stops_actual_child_without_restart(supervision, monkeypatch):
    module, config, path, output = supervision
    original = subprocess.Popen
    processes = []

    def launch(_command, **kwargs):
        child = original([sys.executable, '-c', 'import time;time.sleep(20)'], **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    timer = threading.Timer(.3, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        result = supervise(config, path, 42, output, maximum_restarts=3, wall_budget=10)
    finally:
        timer.cancel()
        timer.join(timeout=1)
        for child in processes:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
    assert result['termination'] == 'SUPERVISOR_STOP_REQUESTED'
    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_restart_budget_is_finite(supervision, monkeypatch):
    module, config, path, output = supervision
    commands = fake_processes(module, config, monkeypatch, [75, 75])
    result = supervise(config, path, 42, output, maximum_restarts=1, wall_budget=15)
    assert result['termination'] == 'SUPERVISOR_RESTART_LIMIT'
    assert len(commands) == 2


def test_config_change_between_restarts_stops_supervisor(supervision, monkeypatch):
    module, config, path, output = supervision
    commands = fake_processes(module, config, monkeypatch, [75])
    original = module.subprocess.Popen

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        path.write_text('changed configuration')
        return process

    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    result = supervise(config, path, 42, output, maximum_restarts=1, wall_budget=15)
    assert result['termination'] == 'SUPERVISOR_CONTRACT_CHANGED'
    assert len(commands) == 1


def test_cli_marks_database_corruption_as_recoverable_exit(monkeypatch, capsys):
    import apps.five_task_production as cli
    import source.data.production_coordinator as coordinator

    monkeypatch.setattr(coordinator, 'production', lambda *a, **kw: (_ for _ in ()).throw(
        sqlite3.DatabaseError('file is not a database')))
    monkeypatch.setattr(sys, 'argv', ['five_task_production.py', 'resume'])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 75
    assert json.loads(capsys.readouterr().out)['termination'] == 'DATABASE_CORRUPTION'
