"""Bounded pilot coordinator joining template planning, workers and inventory."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time

from source.data.production_artifacts import package_episode, reconcile_episode, register_artifact
from source.data.production_templates import load_templates
from source.data.production_scheduler import enqueue_next
from source.data.production_liveness import LeaseKeeper
from source.data.production_execution_contract import execution_fingerprint
from source.data.production_resources import memory_status, canary_storage_status
from source.data.production_monitor import write_worker_status, record_runtime_failure
from source.data.production_failures import failure_policy, WorkerFailure
from source.data.production_profile import PhaseProfile
from source.data.production_duration import candidate_duration
from source.data.production_eta import begin_session, finish_session
from source.data.production_parallel import WorkerAdmission


def stop_child(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    # The group can still contain FFmpeg after the worker itself has exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def pilot(config, root, max_episodes, sampling_seed, resume=False, append_candidates=False, *, wall_budget=None):
    project = Path(__file__).resolve().parents[2]
    if not root.resolve().is_relative_to(project / "outputs"):
        raise ValueError("Pilot must use a separate project outputs directory")
    if not 1 <= max_episodes <= 5:
        raise ValueError("Development pilot budget is 1 to 5 attempted jobs")
    return _coordinate(config, root, max_episodes, sampling_seed, resume, append_candidates, wall_budget=wall_budget)


def production(config, sampling_seed, *, resume=False, wall_budget=None):
    from source.data.production_preflight import preflight

    checks = preflight(config)
    if not checks['infrastructure_ready']:
        raise ValueError('Production infrastructure preflight failed: ' + json.dumps(checks))
    from source.data.production_index import active_database
    from source.data.production_bootstrap import bootstrap_pending

    database = active_database(config['dataset_root'], config['queue_database'])
    if (database.exists() and not resume) or (resume and not database.exists() and not bootstrap_pending(config['dataset_root'])):
        raise ValueError('Use resume for an existing queue; a new run requires a fresh queue')
    return _coordinate(config, Path(config['dataset_root']), None, sampling_seed, resume, True,
                      formal=True, wall_budget=wall_budget)


def canary(config, root, max_episodes, sampling_seed, *, resume=False, wall_budget=3600):
    project = Path(__file__).resolve().parents[2]
    if not root.resolve().is_relative_to(project / 'outputs'):
        raise ValueError('Canary must stay in a separate project outputs directory')
    if not 1 <= max_episodes <= 40 or not 0 < wall_budget <= 3600:
        raise ValueError('Canary budget is at most 40 attempts and 3600 wall seconds')
    return _coordinate(config, root, max_episodes, sampling_seed, resume, True,
                       wall_budget=wall_budget, canary_mode=True)


def _coordinate(config, root, max_episodes, sampling_seed, resume=False, append_candidates=False,
               *, formal=False, wall_budget=None, canary_mode=False):
    coordinator_started = time.monotonic()
    admission = WorkerAdmission(config)
    workers = admission.plan['pool_workers'] if max_episodes is None else min(admission.plan['pool_workers'], max_episodes)
    project = Path(__file__).resolve().parents[2]
    guarded = formal or canary_mode
    if wall_budget is not None and (not isinstance(wall_budget, (int, float)) or not 0 < wall_budget < float('inf')):
        raise ValueError('Wall budget must be positive and finite')
    templates = load_templates(Path(config.get("template_library", "configs/five_task_templates.json")),
                               Path(config.get("object_registry", "outputs/five_task_p31/object_registry.json")))
    if formal and len({row['asset_hash'] for row in templates}) < config['minimum_independent_assets']:
        raise ValueError('Template catalogue cannot cover the independent asset target')
    run_contract = None
    if guarded:
        run_contract = {'schema': 'production-run-contract-v1', 'sampling_seed': sampling_seed,
                        'mode': 'production' if formal else 'development_canary',
                        'template_sha256': hashlib.sha256(Path(config['template_library']).read_bytes()).hexdigest(),
                        'registry_sha256': hashlib.sha256(Path(config['object_registry']).read_bytes()).hexdigest(),
                        'execution': execution_fingerprint(project)}
    from source.data.production_bootstrap import reserve_bootstrap, install_metadata, finish_bootstrap

    database_path = Path(config['queue_database']) if formal else root / 'inventory.sqlite'
    startup_lock, bootstrap = reserve_bootstrap(root, database_path, config, run_contract, sampling_seed,
        workers if max_episodes is None else max_episodes, resume=resume)
    lock_stream = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        device = root.stat().st_dev
        lock_stream = (root / 'coordinator.lock').open('a')
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException as error:
        if lock_stream is not None:
            lock_stream.close()
        startup_lock.close()
        if isinstance(error, BlockingIOError):
            raise RuntimeError('A coordinator already owns this pilot') from error
        raise
    startup_lock.close()

    def storage_check():
        if formal:
            from source.data.production_preflight import check_dataset_volume

            result = check_dataset_volume(config)
            if root.stat().st_dev != device:
                result['ready'] = False
                result['errors'].append('Dataset filesystem identity changed during this session')
            return result
        return canary_storage_status(root, device, config['disk_reserve_fraction'])
    from source.data.production_index import open_inventory

    try:
        initialized = install_metadata(root, bootstrap)
        queue = open_inventory(root, database_path,
                               auto_rebuild=config.get('auto_rebuild_inventory', True),
                               maximum_retries=config['maximum_retries'], initializing=not initialized)
    except BaseException:
        lock_stream.close()
        raise
    stopped = threading.Event()
    stop_signals = []

    def stop_requested(signum, _):
        stop_signals.append({'signal': signum, 'unix_seconds': time.time()})
        stopped.set()

    previous_handler = signal.signal(signal.SIGINT, stop_requested)
    previous_term = signal.signal(signal.SIGTERM, stop_requested)
    counts_lock = threading.Lock()
    attempts = 0
    results = deque(maxlen=100) if formal else []
    guard_failures = []
    guard_failure_lock = threading.Lock()
    integrity_lock = threading.Lock()
    integrity_failures = []
    started = time.monotonic()
    try:
        runtime_session = begin_session(queue, workers, started=coordinator_started)
        finish_bootstrap(root, bootstrap, queue, lambda: enqueue_next(queue, templates, config, sampling_seed))
        queue.recover(abandoned=True)

        def periodic_integrity(*, startup=False):
            from source.data.production_periodic_audit import audit_after_attempt

            with integrity_lock:
                try:
                    result = audit_after_attempt(queue, root,
                        interval=config.get('integrity_audit_interval_attempts', 10),
                        force=startup, count_attempt=not startup)
                    if result['state'] != 'FAILED':
                        return
                except Exception as error:
                    result = {'state': 'FAILED', 'error': f'{type(error).__name__}: {error}'}
                    try:
                        result['record_path'] = record_runtime_failure(root, 'integrity', result)
                    except Exception as logging_error:
                        result['logging_error'] = str(logging_error)
                integrity_failures.append(result)
                stopped.set()

        if guarded:
            periodic_integrity(startup=True)

        @admission.cleanup_on_exit
        def work(slot):
            nonlocal attempts
            while not stopped.is_set():
                if not admission.acquire(slot, stopped, deadline=started + wall_budget if wall_budget is not None else None):
                    return
                with counts_lock:
                    if max_episodes is not None and attempts >= max_episodes:
                        return
                    if wall_budget is not None and time.monotonic() - started >= wall_budget:
                        stopped.set()
                        return
                    if guarded:
                        memory = memory_status(config)
                        if not memory['ready']:
                            stopped.set()
                            results.append({'state': 'MEMORY_STOP', 'memory': memory})
                            return
                        volume = storage_check()
                        if not volume['ready']:
                            stopped.set()
                            results.append({'state': 'VOLUME_STOP', 'volume': volume})
                            return
                    quota = queue.quota_status(config["task_hours"],
                                               min_assets=config["minimum_independent_assets"],
                                               count_failures=config["count_failed_episodes"],
                                               outcome_hours=config.get('task_outcome_hours'),
                                               coverage_success_only=config.get('coverage_requires_success', True))
                    if quota["complete"]:
                        return
                    if guarded:
                        progress = queue.observe_progress(sum(quota['remaining_task_hours'].values()), quota['remaining_assets'],
                            limit=config.get('maximum_no_progress_attempts', 1000))
                        if progress['stalled']:
                            stopped.set()
                            results.append({'state': 'QUOTA_STALLED', 'progress': progress})
                            return
                    job = queue.claim(quota["needed_tasks"], lease_seconds=300)
                    if job is None and append_candidates:
                        planned = enqueue_next(queue, templates, config, sampling_seed)
                        if planned['state'] == 'ENQUEUED':
                            job = queue.claim(quota['needed_tasks'], lease_seconds=300)
                        elif planned['state'] == 'WAITING_OR_BLOCKED':
                            results.append(planned)
                    if job is None:
                        return
                    attempts += 1
                    print(json.dumps({"job_id": job["job_id"], "state": "RUNNING"}), flush=True)
                phase = 'RECOVER_OR_EXECUTE'
                case = None
                attempt_started = time.monotonic()
                attempt_profile = PhaseProfile()
                state = 'NO_TERMINAL_RESULT'

                def runtime_guard():
                    if wall_budget is not None and time.monotonic() - started >= wall_budget:
                        raise TimeoutError('Coordinator session budget reached during processing')
                    if guarded:
                        memory = memory_status(config)
                        if not memory['ready']:
                            raise RuntimeError('System memory reserve violated: ' + json.dumps(memory))
                        volume = storage_check()
                        if not volume['ready']:
                            raise RuntimeError('Volume identity or water level changed: ' + json.dumps(volume))
                        if execution_fingerprint(project) != run_contract['execution']:
                            raise RuntimeError('Executable source, shared resources or package versions changed during production')
                        write_worker_status(root, slot, {'job_id': job['job_id'], 'task': job['task'],
                            'phase': phase, 'state': 'ACTIVE', 'memory': memory, 'volume': volume})

                def capture_failure(error):
                    event = {'job_id': job['job_id'], 'task': job['task'], 'phase': phase,
                             'detected_unix_seconds': time.time(), 'error': f'{type(error).__name__}: {error}',
                             'kind': 'SESSION_BUDGET' if isinstance(error, TimeoutError) and 'session budget' in str(error)
                             else 'RUNTIME_GUARD'}
                    event['record_path'] = record_runtime_failure(root, slot, event)
                    with guard_failure_lock:
                        guard_failures.append(event)

                keeper = LeaseKeeper(queue, job, stopped, guard=runtime_guard, on_failure=capture_failure)
                try:
                    keeper.start()
                    existing = attempt_profile.call('recover_files', reconcile_episode, root, job['job_id'])
                    if existing["state"] == "NO_ARTIFACT":
                        work_root = root / "work" / job["job_id"]
                        work_root.mkdir(parents=True, exist_ok=True)
                        # Reuse a completed worker output after coordinator loss.
                        completed = []
                        for path in work_root.glob("attempt_*/worker_result.json"):
                            if json.loads(path.read_text()).get("complete"):
                                completed.append(path.parent)
                        if completed:
                            case = sorted(completed)[0]
                        else:
                            case = work_root / f"attempt_{job['attempts']:03d}"
                            job_file = work_root / f"job_{job['attempts']:03d}.json"
                            with job_file.open("x") as stream:
                                json.dump(job["payload"], stream, indent=2)
                            command = [sys.executable, "apps/production_worker.py", "--job", str(job_file),
                                       "--output", str(case), "--budget-seconds", str(int(config["episode_wall_timeout_seconds"]))]
                            with (work_root / f"attempt_{job['attempts']:03d}.log").open("xb") as log:
                                env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(project)}
                                if admission.plan['policy'] == 'resource-adaptive-v1':
                                    threads = str(config.get('cpu_threads_per_worker', 2))
                                    env.update({key: threads for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                                        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
                                process = subprocess.Popen(command, cwd=project, env=env, stdout=log,
                                                           stderr=subprocess.STDOUT, start_new_session=True)
                                admission.attach_process(slot, process.pid)
                                worker_started = time.monotonic()
                                try:
                                    queue.attach_worker(job["job_id"], job["lease_token"], process.pid)
                                    deadline = time.monotonic() + config["episode_wall_timeout_seconds"] + 10
                                    heartbeat = time.monotonic()
                                    while process.poll() is None:
                                        keeper.check()
                                        if stopped.is_set() or time.monotonic() >= deadline:
                                            raise TimeoutError("Pilot stopped or worker deadline exceeded")
                                        if wall_budget is not None and time.monotonic() - started >= wall_budget:
                                            stopped.set()
                                            raise TimeoutError('Coordinator wall budget reached')
                                        if time.monotonic() - heartbeat >= 10:
                                            if guarded:
                                                volume = storage_check()
                                                if not volume['ready']:
                                                    stopped.set()
                                                    raise RuntimeError('Volume identity or water level changed')
                                            queue.heartbeat(job["job_id"], job["lease_token"])
                                            heartbeat = time.monotonic()
                                        stopped.wait(.1)
                                    if process.returncode != 0:
                                        failure_file = case / 'worker_result.json'
                                        if failure_file.is_file():
                                            failure = json.loads(failure_file.read_text()).get('failure')
                                            if failure is not None:
                                                raise WorkerFailure(failure)
                                        raise RuntimeError(f"Worker exited {process.returncode}; see {log.name}")
                                finally:
                                    stop_child(process)
                                    attempt_profile.add('worker_process_inclusive', time.monotonic() - worker_started)
                        phase = 'PACKAGING'
                        keeper.check()
                        attempt_profile.call('package_and_verify', package_episode, case, root, job['job_id'],
                                             job['asset_hash'], job_payload=job['payload'], formal=formal)
                    phase = 'REGISTERING'
                    keeper.check()
                    state = attempt_profile.call('register_and_verify', register_artifact, queue, job, root,
                                                 allow_development=not formal)
                    with counts_lock:
                        results.append({"job_id": job["job_id"], "state": state,
                                        'wall_profile': attempt_profile.snapshot()})
                        print(json.dumps(results[-1]), flush=True)
                except Exception as error:
                    if isinstance(error, sqlite3.DatabaseError):
                        stopped.set()
                        record_runtime_failure(root, slot, {'phase': phase, 'job_id': job['job_id'],
                            'error': f'{type(error).__name__}: {error}', 'kind': 'DATABASE_ERROR'})
                        raise
                    policy = failure_policy(error, 'RUNTIME_GUARD' if keeper.error else phase,
                                            interrupted=stopped.is_set() and keeper.error is None)
                    if policy['stop_coordinator']:
                        stopped.set()
                    try:
                        state = queue.fail(job["job_id"], job["lease_token"], str(error),
                                           retryable=policy['retryable'], attempt_limit=policy['attempt_limit'],
                                           details=policy)
                    except ValueError:
                        state = 'LEASE_LOST_REQUIRES_RECONCILIATION'
                    with counts_lock:
                        results.append({"job_id": job["job_id"], "state": state, "error": str(error),
                                        'failure_policy': policy})
                        print(json.dumps(results[-1]), flush=True)
                finally:
                    try:
                        keeper.close()
                    finally:
                        profile_dir = root / 'runtime/profiles'
                        try:
                            profile_dir.mkdir(parents=True, exist_ok=True)
                            with (profile_dir / f"{job['job_id']}_{job['attempts']:03d}.json").open('x') as stream:
                                json.dump({'job_id': job['job_id'], 'attempt': job['attempts'],
                                    'state': state, 'wall_seconds': time.monotonic() - attempt_started,
                                    'profile': attempt_profile.snapshot()}, stream, indent=2)
                                stream.flush()
                                os.fsync(stream.fileno())
                        finally:
                            queue.record_attempt_metrics(job['job_id'], job['attempts'],
                                                         time.monotonic() - attempt_started, state,
                                                         candidate_seconds=candidate_duration(case))
                    if guarded and not stopped.is_set():
                        periodic_integrity()
                    if guarded:
                        with counts_lock:
                            remaining = queue.quota_status(config['task_hours'], min_assets=config['minimum_independent_assets'],
                                count_failures=config['count_failed_episodes'], outcome_hours=config.get('task_outcome_hours'),
                                coverage_success_only=config.get('coverage_requires_success', True))
                            progress = queue.observe_progress(sum(remaining['remaining_task_hours'].values()),
                                remaining['remaining_assets'], attempt=True, limit=config.get('maximum_no_progress_attempts', 1000),
                                progress_hint=(sum(remaining['remaining_task_hours'].values()) < sum(quota['remaining_task_hours'].values()) - 1e-10
                                               or remaining['remaining_assets'] < quota['remaining_assets']))
                            if progress['stalled']:
                                stopped.set()
                                results.append({'state': 'QUOTA_STALLED', 'progress': progress})
                        write_worker_status(root, slot, {'job_id': job['job_id'], 'task': job['task'],
                            'phase': phase, 'state': 'ATTEMPT_ENDED', 'stopped': stopped.is_set(),
                            'lease_guard_error': str(keeper.error) if keeper.error else None})
                    admission.release(slot)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, slot) for slot in range(workers)]
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException:
                    stopped.set()
                    raise
        quota = queue.quota_status(config['task_hours'], min_assets=config['minimum_independent_assets'],
                                   count_failures=config['count_failed_episodes'], outcome_hours=config.get('task_outcome_hours'),
                                   coverage_success_only=config.get('coverage_requires_success', True))
        summary = {"development_pilot": not formal, "production_ready": False, "attempted_jobs": attempts,
                   'worker_plan': admission.plan,
                   'worker_threads_started': workers,
                   'development_canary': canary_mode,
                   "wall_seconds": time.monotonic() - started, "results": list(results),
                   "stopped": stopped.is_set(), "wall_budget_seconds": wall_budget,
                   'stop_signals': stop_signals, 'runtime_guard_failures': guard_failures,
                   'integrity_failures': integrity_failures,
                   "quota_complete": quota['complete'],
                   "termination": 'QUOTA_COMPLETE' if quota['complete'] else 'RUNTIME_GUARD_FAILED' if any(
                       e['kind'] == 'RUNTIME_GUARD' for e in guard_failures) else 'QUOTA_STALLED' if any(
                       r.get('state') == 'QUOTA_STALLED' for r in results) else 'STOP_OR_WALL_BUDGET' if stopped.is_set()
                   else 'NO_RUNNABLE_CANDIDATE' if guarded and (max_episodes is None or attempts < max_episodes)
                   else 'ATTEMPT_BUDGET_OR_EMPTY_QUEUE',
                   "inventory": quota}
        if quota['complete']:
            from source.data.production_finalization import finalize_dataset

            acceptance = finalize_dataset(config, root, queue.path)
            summary['final_file_acceptance'] = acceptance
            summary['termination'] = {
                'FILE_ACCEPTED': 'QUOTA_COMPLETE_FILE_VERIFIED',
                'FILE_REJECTED': 'QUOTA_COMPLETE_FILE_REJECTED',
                'REPORT_FAILED': 'FINAL_REPORT_FAILED'}[acceptance['state']]
            # Include the automatic full-file verification in session timing.
            summary['wall_seconds'] = time.monotonic() - started
        if integrity_failures:
            summary['termination'] = 'INTEGRITY_AUDIT_FAILED'
        from source.data.production_statistics import production_statistics

        finish_session(queue, runtime_session)
        summary['runtime_statistics'] = production_statistics({**config, 'queue_database': str(queue.path)})
        runs = root / "runs"
        runs.mkdir(exist_ok=True)
        if (root / "summary.json").exists() and not list(runs.glob('*.json')):
            with (runs / 'legacy_latest_summary.json').open('x') as stream:
                json.dump(json.loads((root / 'summary.json').read_text()), stream, indent=2)
        with (runs / f"run_{time.time_ns()}.json").open('x') as stream:
            json.dump(summary, stream, indent=2)
        temporary = root / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(root / "summary.json")
        return summary
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        signal.signal(signal.SIGTERM, previous_term)
        lock_stream.close()
