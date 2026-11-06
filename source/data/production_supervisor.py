"""Explicit production supervisor; never restart intentional or protective stops."""

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from source.data.five_task_dataset import sha256_file
from source.data.production_execution_contract import execution_fingerprint
from source.data.production_index import active_database, live_dataset_processes
from source.data.production_preflight import preflight
from source.data.production_bootstrap import intent_path


def supervise(config, config_path, sampling_seed, output, *, maximum_restarts=3, wall_budget=None):
    if isinstance(maximum_restarts, bool) or not isinstance(maximum_restarts, int) or not 0 <= maximum_restarts <= 10:
        raise ValueError('Supervisor allows zero to ten restarts')
    if wall_budget is not None and not 0 < wall_budget < float('inf'):
        raise ValueError('Supervisor wall budget must be positive and finite')
    output, config_path = Path(output), Path(config_path).resolve()
    root = Path(config['dataset_root'])
    if output.exists() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Supervisor needs a fresh control directory outside the dataset')
    checks = preflight(config)
    if not checks['infrastructure_ready']:
        return {'termination': 'SUPERVISOR_PREFLIGHT_REJECTED', 'checks': checks, 'sessions': [],
                'formal_dataset_accepted': False}
    project = Path(__file__).resolve().parents[2]
    frozen = execution_fingerprint(project)
    config_digest = sha256_file(config_path)
    output.mkdir(parents=True)
    control = Path(config['queue_database']).parent
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / 'supervisor.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('A supervisor already owns this control directory') from None
    stopped = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    sessions = []
    result = {'termination': 'SUPERVISOR_NOT_FINISHED', 'sessions': sessions,
              'formal_dataset_accepted': False, 'maximum_restarts': maximum_restarts,
              'config_file': str(config_path), 'config_sha256': config_digest,
              'execution_contract': frozen, 'initial_preflight': checks, 'sampling_seed': sampling_seed}

    def save():
        temporary = output / 'supervisor.json.tmp'
        with temporary.open('w') as stream:
            json.dump({**result, 'wall_seconds': time.monotonic() - started}, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output / 'supervisor.json')

    try:
        for index in range(maximum_restarts + 1):
            if stopped.is_set():
                result['termination'] = 'SUPERVISOR_STOP_REQUESTED'
                break
            remaining = None if wall_budget is None else wall_budget - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                result['termination'] = 'SUPERVISOR_WALL_BUDGET'
                break
            if sha256_file(config_path) != config_digest or execution_fingerprint(project) != frozen:
                result['termination'] = 'SUPERVISOR_CONTRACT_CHANGED'
                break
            orphan_deadline = time.monotonic() + config['episode_wall_timeout_seconds'] + 30
            live = live_dataset_processes(root)
            while live and not stopped.is_set() and time.monotonic() < orphan_deadline:
                if wall_budget is not None and time.monotonic() - started >= wall_budget:
                    break
                stopped.wait(.2)
                live = live_dataset_processes(root)
            if live or stopped.is_set():
                result.update(termination='SUPERVISOR_ORPHANS_OR_STOP', live_dataset_processes=live)
                break
            checks = preflight(config)
            if not checks['infrastructure_ready']:
                result.update(termination='SUPERVISOR_PREFLIGHT_REJECTED', checks=checks)
                break
            if stopped.is_set():
                result['termination'] = 'SUPERVISOR_STOP_REQUESTED'
                break
            remaining = None if wall_budget is None else wall_budget - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                result['termination'] = 'SUPERVISOR_WALL_BUDGET'
                break
            mode = 'resume' if (active_database(root, config['queue_database']).exists()
                                or root.exists() or intent_path(root).exists()) else 'run'
            command = [sys.executable, 'apps/five_task_production.py', mode, '--config', str(config_path),
                       '--sampling-seed', str(sampling_seed)]
            if remaining is not None:
                command += ['--wall-budget-seconds', str(remaining)]
            summary_path = root / 'summary.json'
            previous_summary = sha256_file(summary_path) if summary_path.is_file() else None
            row = {'index': index, 'command': command, 'started_unix_seconds': time.time()}
            sessions.append(row)
            save()
            with (output / f'session_{index:03d}.log').open('xb') as log:
                process = subprocess.Popen(command, cwd=project, start_new_session=True,
                    env={**os.environ, 'PYTHONPATH': str(project), 'MUJOCO_GL': 'egl'},
                    stdout=log, stderr=subprocess.STDOUT)
                row['pid'] = process.pid
                try:
                    while process.poll() is None:
                        if stopped.is_set() or (wall_budget is not None and time.monotonic() - started >= wall_budget):
                            process.send_signal(signal.SIGTERM)
                            try:
                                process.wait(timeout=90)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
                            break
                        stopped.wait(.2)
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGTERM)
                        try:
                            process.wait(timeout=90)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                row['returncode'] = process.returncode
            if stopped.is_set():
                result['termination'] = 'SUPERVISOR_STOP_REQUESTED'
                break
            if wall_budget is not None and time.monotonic() - started >= wall_budget:
                result['termination'] = 'SUPERVISOR_WALL_BUDGET'
                break
            if process.returncode == 0:
                if not summary_path.is_file() or sha256_file(summary_path) == previous_summary:
                    result['termination'] = 'SUPERVISOR_MISSING_NEW_SUMMARY'
                    break
                summary = json.loads(summary_path.read_text())
                result.update(termination=summary['termination'],
                    formal_dataset_accepted=summary.get('final_file_acceptance', {}).get('formal_dataset_accepted', False))
                break
            if process.returncode not in (75, -signal.SIGKILL, -signal.SIGSEGV, -signal.SIGABRT, -signal.SIGBUS):
                result['termination'] = 'SUPERVISOR_NONRETRYABLE_EXIT'
                break
            result['termination'] = 'SUPERVISOR_RESTART_LIMIT' if index == maximum_restarts else 'SUPERVISOR_RESTART_PENDING'
            save()
            if index < maximum_restarts:
                stopped.wait(2.)
        result['live_dataset_processes_after_exit'] = live_dataset_processes(root)
        if result['live_dataset_processes_after_exit']:
            result.update(termination='SUPERVISOR_ORPHANS_REMAIN', formal_dataset_accepted=False)
        save()
        return result
    except BaseException as error:
        result.update(termination='SUPERVISOR_EXCEPTION', error=f'{type(error).__name__}: {error}',
                      formal_dataset_accepted=False)
        try:
            save()
        except Exception as logging_error:
            print(f'Supervisor error record also failed: {logging_error}', file=sys.stderr, flush=True)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.close()
