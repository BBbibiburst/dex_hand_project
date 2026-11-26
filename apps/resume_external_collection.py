"""Resume a frozen external dataset using its original server queue and runtime.

No source/config upgrade, no historical-counter reset, and no file-only adoption.
New bounded recovery probes are explicit PENDING jobs, never rewrites of old jobs.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

POLICY = 'external-resume-20261125-v1'
MAX_PROBES_PER_TEMPLATE = 3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(project, root):
    """Read-only report. Never create an empty replacement queue."""
    from source.data.five_task_dataset import canonical_hash
    from source.data.production_bootstrap import intent_path
    from source.data.production_index import active_database
    from source.data.production_preflight import mount_info
    config = read(root / 'config.json')
    contract = read(root / 'run_contract.json')
    intent = read(intent_path(root))
    complete = read(root / 'bootstrap_complete.json')
    errors = []
    if (intent['configuration'] != config or intent['run_contract'] != contract
            or complete['intent_hash'] != canonical_hash(intent)):
        errors.append('Frozen bootstrap/config/contract proofs disagree')
    if str(root.resolve()) != intent['dataset_root'] or str(root.resolve()) != config['dataset_root']:
        errors.append('Mount at the ORIGINAL registered server path; metadata is not relocated automatically')
    dbpath = (project / config['queue_database']).resolve()
    if str(dbpath) != intent['database']:
        errors.append('Project/control path differs from original server path')
    selected = active_database(root, dbpath)
    if not selected.is_file():
        errors.append('Original server queue is missing; refusing to create or infer progress from summary')
    frozen = contract['execution']
    if sys.version != frozen['python']:
        errors.append('Python build differs from frozen server runtime')
    packages = {}
    for name, expected in frozen['packages'].items():
        try:
            packages[name] = version(name)
        except Exception:
            packages[name] = None
        if packages[name] != expected:
            errors.append(f'{name}: need {expected}, found {packages[name]}')
    for field in ('source_sha256', 'shared_resource_sha256'):
        for name, expected in frozen[field].items():
            path = project / name
            if not path.is_file() or digest(path) != expected:
                errors.append('Frozen file differs/missing: ' + name)
    actual_sources = {str(p.relative_to(project)) for p in (project / 'source').rglob('*.py')}
    expected_sources = {n for n in frozen['source_sha256'] if n.startswith('source/')}
    if actual_sources != expected_sources:
        errors.append('Frozen source file set changed (added or removed source module)')
    resources = {str(p.relative_to(project))
                 for name in ('robots', 'grippers', 'arenas', 'bases', 'textures')
                 for p in (project / 'assets' / name).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts}
    manifest = project / 'assets/maniskill/manifest.json'
    if manifest.is_file():
        resources.add(str(manifest.relative_to(project)))
    if resources != set(frozen['shared_resource_sha256']):
        errors.append('Frozen shared resource file set changed')
    for field, key in [('template_sha256', 'template_library'), ('registry_sha256', 'object_registry')]:
        path = project / config[key]
        if not path.is_file() or digest(path) != contract[field]:
            errors.append('Frozen catalogue/registry differs: ' + key)
    info = mount_info(root)
    usage = os.statvfs(root)
    total, free = usage.f_blocks * usage.f_frsize, usage.f_bavail * usage.f_frsize
    if info['mountpoint'] == '/' or os.stat(root).st_dev == os.stat('/').st_dev or total < 19_000_000_000_000:
        errors.append('Dataset is not on the mounted 20TB external disk')
    if free - total * config['disk_reserve_fraction'] < 6_000_000_000:
        errors.append('Insufficient disk space above frozen reserve')
    queue_report = None
    if selected.is_file():
        with sqlite3.connect(selected.as_uri() + '?mode=ro', uri=True) as db:
            if [r[0] for r in db.execute('PRAGMA quick_check')] != ['ok']:
                errors.append('Queue quick_check failed; do not resume')
            queue_report = {'episodes': db.execute('SELECT count(*) FROM episodes').fetchone()[0],
                            'states': dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state'))}
        historical = read(root / 'summary.json').get('inventory', {}).get('episodes', 0)
        if queue_report['episodes'] < historical:
            errors.append('Queue has fewer registered episodes than last summary; reconcile before resuming')
    infrastructure = None
    if not errors:
        from apps.five_task_production import load_config
        from source.data.production_preflight import preflight
        effective = load_config(root / 'config.json')
        if effective != config:
            errors.append('Effective configuration differs from frozen config')
        else:
            infrastructure = preflight(effective)
            if not infrastructure['infrastructure_ready']:
                errors.append('Original production infrastructure preflight failed')
    return {'ready': not errors, 'errors': errors, 'infrastructure': infrastructure,
            'dataset_root': str(root), 'database': str(selected),
            'packages': packages, 'mount': info, 'free_bytes': free, 'queue': queue_report,
            'sampling_seed': contract['sampling_seed'], 'policy': POLICY,
            'historical_episode_files_untouched': True}


def seed_recovery(queue, templates, config, seed):
    """Atomically add at most one next-index probe per previously successful blocked template.

    A lifetime cap persists in queue_meta. Historical terminal rows, attempts,
    consecutive streaks, episode identity and quota totals are never reset.
    """
    from source.data.five_task_dataset import canonical_hash
    from source.data.production_templates import candidate
    from source.data.production_capture import capture_from_config
    quota = queue.quota_status(config['task_hours'], min_assets=config['minimum_independent_assets'],
        count_failures=config['count_failed_episodes'], outcome_hours=config.get('task_outcome_hours'),
        coverage_success_only=config.get('coverage_requires_success', True))
    added = []
    with queue.transaction() as db:
        # Existing work must be resumed/reconciled by the original coordinator first.
        active = db.execute("SELECT count(*) FROM jobs WHERE state IN ('PENDING','RUNNING','VALIDATING')").fetchone()[0]
        if active:
            return {'added': [], 'reason': 'EXISTING_WORK_RESUME_FIRST'}
        for template in templates:
            if template['task'] not in quota['needed_tasks']:
                continue
            row = db.execute('SELECT * FROM template_stats WHERE template_id=?', (template['template_id'],)).fetchone()
            if row is None or row['successful'] < 1 or row['active']:
                continue
            if (row['consecutive_failures'] < config.get('template_failure_limit', 3)
                    and row['consecutive_duplicates'] < config.get('template_duplicate_limit', 3)):
                continue
            key = POLICY + ':' + template['template_id']
            record = db.execute('SELECT value FROM queue_meta WHERE key=?', (key,)).fetchone()
            count = int(record['value']) if record else 0
            if count >= MAX_PROBES_PER_TEMPLATE:
                continue
            # last_candidate_index is preserved; new candidate never repeats an old job.
            payload = candidate(template, seed, row['last_candidate_index'] + 1, config.get('perturbation_contract'))
            capture = capture_from_config(config)
            if capture is not None:
                payload['rgb_capture'] = capture
            job_id = template['template_id'] + '_' + canonical_hash(payload)[:20]
            if db.execute('SELECT 1 FROM jobs WHERE job_id=?', (job_id,)).fetchone():
                raise ValueError('Next candidate collides with existing job; inspect stats before continuation')
            db.execute('INSERT INTO jobs(job_id,task,asset_hash,payload,max_attempts,created_at) VALUES(?,?,?,?,?,?)',
                (job_id, template['task'], template['asset_hash'], json.dumps(payload, sort_keys=True, allow_nan=False),
                 config['maximum_retries'], time.time()))
            queue._scheduled(db, payload)
            queue.event(db, job_id, 'PENDING', {'policy': POLICY})
            queue.event(db, job_id, 'BOUNDED_RECOVERY_PROBE', {'policy': POLICY, 'number': count + 1,
                'maximum': MAX_PROBES_PER_TEMPLATE, 'previous_consecutive_failures': row['consecutive_failures'],
                'previous_consecutive_duplicates': row['consecutive_duplicates'], 'historical_rows_reset': False})
            db.execute('INSERT INTO queue_meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                       (key, str(count + 1)))
            added.append(job_id)
    return {'added': added, 'reason': 'PROBES_QUEUED' if added else 'NO_ELIGIBLE_RECOVERY_PROBE'}


def prepare(project, root, output):
    from source.data.production_bootstrap import intent_path
    from source.data.production_index import live_dataset_processes
    from source.data.production_queue import ProductionQueue
    from source.data.production_templates import load_templates
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Control/backup output must remain outside the dataset')
    report = check(project, root)
    if not report['ready']:
        raise ValueError(json.dumps(report, ensure_ascii=False))
    config = read(root / 'config.json')
    control = (project / config['queue_database']).parent
    with ExitStack() as locks:
        for path in (control / 'supervisor.lock', intent_path(root).with_suffix('.lock'), root / 'coordinator.lock'):
            stream = locks.enter_context(path.open('a'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if live_dataset_processes(root):
            raise RuntimeError('Workers are alive; stop original collection before preparation')
        # Recheck while locked, then SQLite backup including any committed WAL content.
        report = check(project, root)
        if not report['ready']:
            raise ValueError(json.dumps(report))
        templates = load_templates(project / config['template_library'], project / config['object_registry'])
        output.mkdir(parents=True, exist_ok=False)
        dbpath = Path(report['database'])
        with sqlite3.connect(dbpath.as_uri() + '?mode=ro', uri=True) as old:
            with sqlite3.connect(output / 'queue_before.sqlite') as backup:
                old.backup(backup)
        queue = ProductionQueue(dbpath)
        report['recovery'] = seed_recovery(queue, templates, config, report['sampling_seed'])
        report['backup'] = str(output / 'queue_before.sqlite')
        report['episodes_after'] = queue.status(count_failures=config['count_failed_episodes'])['episodes']
        (output / 'preparation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'prepare', 'run'])
    parser.add_argument('--project', type=Path, default=Path('/home/ftcl/dex_hand_project'))
    parser.add_argument('--dataset-root', type=Path, default=Path('/media/ftcl/外置存储/dex_hand_dataset_batch2'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    project, root = args.project.resolve(), args.dataset_root.resolve()
    sys.path.insert(0, str(project)); os.chdir(project)
    if os.environ.get('DEX_DATASET_ROOT'):
        raise ValueError('Unset DEX_DATASET_ROOT; continue the original frozen path without overrides')
    if args.command == 'check':
        report = check(project, root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['ready'] else 2
    base = args.output or project / 'outputs' / ('external_resume_' + str(time.time_ns()))
    if args.command == 'prepare':
        print(json.dumps(prepare(project, root, base), ensure_ascii=False, indent=2))
        return 0
    # Never re-enter on SIGINT, runtime guard failure, disk guard or unknown failure.
    # At most three re-entry cycles. Each template also has a durable lifetime cap.
    base.mkdir(parents=True, exist_ok=False)
    stopped = False
    child = None
    run_report = {'policy': POLICY, 'sessions': [], 'collection_started': False}
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    prior = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for cycle in range(MAX_PROBES_PER_TEMPLATE):
            if stopped:
                break
            report = prepare(project, root, base / f'prepare_{cycle}')
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if stopped:
                break
            command = [sys.executable, str(project / 'apps/five_task_production.py'), 'serve',
                       '--config', str(root / 'config.json'), '--sampling-seed', str(report['sampling_seed']),
                       '--output', str(base / f'session_{cycle}')]
            with (base / f'session_{cycle}.log').open('xb') as log:
                child = subprocess.Popen(command, cwd=project, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONPATH': str(project), 'MUJOCO_GL': 'egl'})
                run_report['collection_started'] = True
                code = child.wait()
            if stopped or code:
                run_report['termination'] = 'USER_STOP' if stopped else 'CHILD_FAILED'
                break
            status = read(base / f'session_{cycle}' / 'supervisor.json')
            run_report['sessions'].append({'cycle': cycle, 'termination': status['termination'],
                'new_probe_jobs': report['recovery']['added']})
            run_report['termination'] = status['termination']
            print('Session termination:', status['termination'], flush=True)
            if status['termination'] != 'NO_RUNNABLE_CANDIDATE':
                break
            if not report['recovery']['added'] and cycle > 0:
                break
        return code if 'code' in locals() else 0
    finally:
        run_report.setdefault('termination', 'USER_STOP' if stopped else 'PREPARATION_NOT_COMPLETED')
        if run_report['termination'] == 'NO_RUNNABLE_CANDIDATE':
            run_report['action_required'] = 'Recovery budget exhausted or no eligible template; inspect failed worker logs. No quality gate was disabled.'
        (base / 'resume_result.json').write_text(json.dumps(run_report, ensure_ascii=False, indent=2))
        for s, handler in prior.items():
            signal.signal(s, handler)


if __name__ == '__main__':
    raise SystemExit(main())
