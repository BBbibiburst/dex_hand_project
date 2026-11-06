"""Run a finite development canary with one planned coordinator interruption."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

from apps.five_task_production import load_config
from source.data.production_queue import process_identity
from source.data.production_statistics import ReadOnlyQueue
from source.data.production_index import active_database
from source.data.production_bootstrap import intent_path


def run(output, seconds, interrupt_after, maximum_attempts):
    project = Path(__file__).resolve().parents[1]
    if not output.resolve().is_relative_to(project / 'outputs') or output.exists():
        raise ValueError('Use a fresh development output directory under project outputs')
    if not 1 <= seconds <= 3600 or not 0 < interrupt_after < seconds or not 1 <= maximum_attempts <= 80:
        raise ValueError('Finite canary requires <=3600 seconds, one earlier interruption and <=80 total attempts')
    output.mkdir(parents=True)
    baseline = json.loads((project / 'outputs/five_task_p33_catalogue_frozen_2026-11-03/templates.json').read_text())
    ids = {'lift_sugar', 'pick_tomato', 'stack_ring', 'push_tomato', 'nut_000'}
    templates = [row for row in baseline['templates'] if row['template_id'] in ids]
    if len(templates) != 5:
        raise ValueError('Five frozen current baseline templates are required')
    library = output / 'templates.json'
    library.write_text(json.dumps({'schema': 'five-task-template-library-v1', 'templates': templates}, indent=2))
    config = {**load_config(project / 'configs/five_task_production.json'), 'template_library': str(library),
              'note': 'Finite development canary; no formal production admission'}
    config_path = output / 'config.json'
    config_path.write_text(json.dumps(config, indent=2))
    plan = {'maximum_wall_seconds': seconds, 'interrupt_after_seconds': interrupt_after,
            'maximum_total_attempts': maximum_attempts, 'maximum_workers': 2,
            'retained_stop_water_bytes': 4_000_000_000, 'planned_disk_budget_bytes': 12_000_000_000,
            'formal_production': False, 'sampling_seed': 41000}
    plan['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / 'plan.json').write_text(json.dumps(plan, indent=2))
    dataset = output / 'dataset'
    stopped = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    sessions = []
    fault_sent = False
    process = None
    try:
        while not stopped.is_set():
            remaining = seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            attempts = 0
            database = active_database(dataset, dataset / 'inventory.sqlite')
            if database.exists():
                with ReadOnlyQueue(database).connect() as db:
                    if db.execute("SELECT 1 FROM queue_meta WHERE key='file_rebuild_history'").fetchone():
                        raise RuntimeError('Canary attempt budget is unknown after inventory reconstruction')
                    attempts = db.execute('SELECT COALESCE(SUM(attempts),0) FROM jobs').fetchone()[0]
            if attempts >= maximum_attempts:
                break
            command = [sys.executable, 'apps/five_task_production.py', 'canary', '--config', str(config_path),
                       '--pilot-root', str(dataset), '--sampling-seed', '41000', '--max-episodes',
                       str(min(40, maximum_attempts - attempts)), '--wall-budget-seconds', str(remaining)]
            if dataset.exists() or intent_path(dataset).exists():
                command.append('--resume')
            index = len(sessions)
            with (output / f'session_{index:02d}.log').open('xb') as log:
                process = subprocess.Popen(command, cwd=project, env={**os.environ, 'PYTHONPATH': str(project), 'MUJOCO_GL': 'egl'},
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row = {'command': command, 'pid': process.pid, 'started_elapsed_seconds': time.monotonic() - started}
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if stopped.is_set() or elapsed >= seconds or (not fault_sent and elapsed >= interrupt_after):
                        process.send_signal(signal.SIGTERM)
                        if not fault_sent and elapsed >= interrupt_after and not stopped.is_set() and elapsed < seconds:
                            fault_sent = True
                            row['planned_sigterm_elapsed_seconds'] = elapsed
                        try:
                            process.wait(timeout=75)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
                            stopped.set()
                        break
                    stopped.wait(.2)
                row.update(returncode=process.returncode, ended_elapsed_seconds=time.monotonic() - started)
                sessions.append(row)
                (output / 'sessions.json').write_text(json.dumps(sessions, indent=2))
            # Never overlap a replacement coordinator with an orphaned worker.
            database = active_database(dataset, dataset / 'inventory.sqlite')
            if database.exists():
                with ReadOnlyQueue(database).connect() as db:
                    events = db.execute("SELECT details FROM events WHERE state='RUNNING'").fetchall()
                live = [worker for event in events if (worker := json.loads(event['details']).get('worker'))
                        and process_identity(worker['pid']) == worker]
                if live:
                    raise RuntimeError('Owned worker still alive after coordinator exit; resume refused')
            if process.returncode != 0:
                break
            summary = json.loads((dataset / 'summary.json').read_text())
            if summary['termination'] != 'ATTEMPT_BUDGET_OR_EMPTY_QUEUE' and not row.get('planned_sigterm_elapsed_seconds'):
                break
        result = {'scope': 'Bounded development endurance, not formal production', 'plan': plan,
                  'actual_wall_seconds': time.monotonic() - started, 'sessions': sessions,
                  'planned_interrupt_executed': fault_sent,
                  'hour_duration_observed': time.monotonic() - started >= 3600,
                  'new_formal_production_seconds': 0,
                  'file_acceptance_pending': True}
        (output / 'driver_result.json').write_text(json.dumps(result, indent=2))
        return result
    except BaseException as error:
        (output / 'driver_failure.json').write_text(json.dumps({
            'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(),
            'actual_wall_seconds': time.monotonic() - started, 'sessions': sessions,
            'new_formal_production_seconds': 0}, indent=2))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=75)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--interrupt-after', type=float, default=150)
    parser.add_argument('--maximum-attempts', type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seconds, args.interrupt_after, args.maximum_attempts), indent=2))
