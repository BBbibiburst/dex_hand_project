"""Persistent, bounded external-dataset watchdog; frozen physics remains unchanged."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time

_spec = importlib.util.spec_from_file_location('external_collection_base', Path(__file__).with_name('resume_external_collection.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
POLICY = 'collection-autorecover-quality-v3'
DEFAULTS = {'probe_cooldown_seconds': 900, 'probes_without_success': 3, 'maximum_new_probes': 0,
            'restart_window_seconds': 3600, 'maximum_restarts_per_window': 6,
            'backoff_initial_seconds': 30, 'backoff_maximum_seconds': 300,
            'resource_wait_seconds': 3600, 'poll_seconds': 5, 'stall_seconds': 900,
            'shutdown_grace_seconds': 120, 'initial_cpu_limit': 0, 'minimum_cpu_limit': 2}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def options(path=None):
    result = {**DEFAULTS, **(base.read(path) if path else {})}
    if set(result) != set(DEFAULTS):
        raise ValueError('Unknown watchdog option')
    for key, value in result.items():
        if type(value) is not int or value < (0 if key in ('initial_cpu_limit','maximum_new_probes') else 1):
            raise ValueError('Invalid watchdog integer option (zero means automatic only for CPU/probe capacity): ' + key)
    if result['probes_without_success'] > 3 or result['maximum_new_probes'] > 128:
        raise ValueError('Recovery cannot exceed three consecutive probes or 128 simultaneous new jobs')
    if result['maximum_restarts_per_window'] > 10:
        raise ValueError('Restart budget exceeds ten')
    if result['initial_cpu_limit'] != 0 and not result['minimum_cpu_limit'] <= result['initial_cpu_limit'] <= 128:
        raise ValueError('Invalid CPU-affinity bounds')
    if result['backoff_initial_seconds'] > result['backoff_maximum_seconds']:
        raise ValueError('Backoff bounds reversed')
    return result


def classify_check(report):
    if report.get('ready'):
        return 'READY'
    errors = report.get('errors', [])
    storage = ['Insufficient disk space above frozen reserve']
    infrastructure = report.get('infrastructure') or {}
    # All non-resource identity/contract/database errors remain hard stops.
    hard = [e for e in errors if e not in storage and e != 'Original production infrastructure preflight failed']
    if hard:
        return 'STOP_CONTRACT_OR_DATABASE'
    if errors and all(e in storage for e in errors):
        return 'WAIT_STORAGE'
    if infrastructure:
        if not all(infrastructure.get('dependencies', {}).values()) or not infrastructure.get('encoder', {}).get('ready'):
            return 'STOP_DEPENDENCY'
        if not infrastructure.get('volume', {}).get('ready'):
            return 'WAIT_STORAGE'
        if not infrastructure.get('memory', {}).get('ready'):
            return 'WAIT_MEMORY'
    return 'STOP_PREFLIGHT'


def diagnose(project, root):
    from source.data.production_preflight import mount_info
    try:
        info = mount_info(root.parent)
        if info['mountpoint'] == '/' or not root.is_dir():
            return 'WAIT_STORAGE', {'ready': False, 'errors': ['External dataset unavailable; no directories created']}
        report = base.check(project, root)
        return classify_check(report), report
    except sqlite3.OperationalError as error:
        if str(error) in ('database is locked', 'database is busy'):
            return 'WAIT_DATABASE_BUSY', {'ready': False, 'errors': [str(error)]}
        return 'STOP_DATABASE', {'ready': False, 'errors': [str(error)]}
    except (OSError, ValueError, KeyError, sqlite3.DatabaseError) as error:
        return 'STOP_CHECK_ERROR', {'ready': False, 'errors': [f'{type(error).__name__}: {error}']}


def probe_capacity(config, policy, *, include_running=False):
    """Fill the configured resource-adaptive worker pool, not a fixed four-job batch."""
    if policy['maximum_new_probes']:
        return policy['maximum_new_probes']
    from source.data.production_parallel import worker_plan
    plan = worker_plan({'worker_policy':'resource-adaptive-v1', 'workers':'auto', **config})
    return max(1, plan['pool_workers'] if include_running else plan['initial_admitted_workers'])


def seed_probes(queue, templates, config, seed, policy, *, now=None, while_running=False):
    """New candidate indices only. Success replenishes budget; restart does not."""
    from source.data.production_templates import candidate
    from source.data.production_capture import capture_from_config
    from source.data.five_task_dataset import canonical_hash
    now = time.time() if now is None else now
    quota = queue.quota_status(config['task_hours'], min_assets=config['minimum_independent_assets'],
        count_failures=config['count_failed_episodes'], outcome_hours=config.get('task_outcome_hours'),
        coverage_success_only=config.get('coverage_requires_success', True))
    added, cooling, isolated = [], [], []
    normal_available = False
    capacity = probe_capacity(config, policy, include_running=while_running)
    with queue.transaction() as db:
        busy_tasks = {r['task'] for r in db.execute(
            "SELECT DISTINCT task FROM jobs WHERE state IN ('PENDING','RUNNING','VALIDATING')")}
        target_tasks = set(quota['needed_tasks'])
        if while_running:
            # Recover an idle task even while another task remains productive.
            target_tasks -= busy_tasks
        elif busy_tasks & target_tasks or db.execute(
                "SELECT count(*) FROM jobs WHERE state IN ('RUNNING','VALIDATING')").fetchone()[0]:
            return {'added': [], 'state': 'RESUME_EXISTING', 'next_retry_at': None, 'isolated': []}
        if not target_tasks:
            return {'added': [], 'state':'NO_STARVED_TASK', 'next_retry_at':None,
                    'isolated':[], 'starved_tasks':[]}
        eligible = []
        for t in templates:
            if t['task'] not in target_tasks:
                continue
            row = db.execute('SELECT * FROM template_stats WHERE template_id=?', (t['template_id'],)).fetchone()
            if row is None:
                normal_available=True;continue
            if row['active']:continue
            if (row['consecutive_failures'] < config.get('template_failure_limit', 3)
                    and row['consecutive_duplicates'] < config.get('template_duplicate_limit', 3)):
                normal_available=True;continue
            if row['successful']<1:continue
            key = POLICY + ':' + t['template_id']
            stored = db.execute('SELECT value FROM queue_meta WHERE key=?', (key,)).fetchone()
            record = json.loads(stored['value']) if stored else {'successful_at_probe': row['successful'], 'used': 0, 'last_probe': 0}
            # Only an actual new successful Episode earns another recovery budget.
            if row['successful'] > record['successful_at_probe']:
                record = {'successful_at_probe': row['successful'], 'used': 0, 'last_probe': record['last_probe']}
            if record['used'] >= policy['probes_without_success']:
                isolated.append(t['template_id']); continue
            due = record['last_probe'] + policy['probe_cooldown_seconds']
            if due > now:
                cooling.append(due); continue
            count = db.execute('SELECT count(*) FROM episodes WHERE task=? AND asset_hash=?', (t['task'],t['asset_hash'])).fetchone()[0]
            eligible.append((count, row['scheduled'], t['template_id'], t, dict(row), key, record))
        # Preserve task diversity within a small batch, then prefer sparse assets.
        taken_tasks = set()
        while eligible and len(added) < capacity:
            item = min(eligible, key=lambda x:(x[3]['task'] in taken_tasks,x[0],x[1],x[2]));eligible.remove(item)
            _,_,_,t,row,key,record=item;taken_tasks.add(t['task'])
            payload = candidate(t,seed,row['last_candidate_index']+1,config.get('perturbation_contract'))
            capture = capture_from_config(config)
            if capture is not None:payload['rgb_capture']=capture
            job=t['template_id']+'_'+canonical_hash(payload)[:20]
            if db.execute('SELECT 1 FROM jobs WHERE job_id=?',(job,)).fetchone():
                raise ValueError('Candidate-index collision; refusing historical job overwrite')
            db.execute('INSERT INTO jobs(job_id,task,asset_hash,payload,max_attempts,created_at) VALUES(?,?,?,?,?,?)',
                (job,t['task'],t['asset_hash'],json.dumps(payload,sort_keys=True,allow_nan=False),config['maximum_retries'],now))
            queue._scheduled(db,payload)
            queue.event(db,job,'PENDING',{'policy':POLICY})
            record={**record,'used':record['used']+1,'last_probe':now}
            queue.event(db,job,'AUTORECOVERY_PROBE',{'policy':POLICY,**record,'previous_failures':row['consecutive_failures'],
                'historical_counters_reset':False,'new_index':payload['candidate_index']})
            db.execute('INSERT INTO queue_meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,json.dumps(record)))
            added.append(job)
    return {'added':added,'state':'PROBES_QUEUED' if added else 'NORMAL_SCHEDULER' if normal_available else 'COOLDOWN' if cooling else 'NO_PROBE',
            'next_retry_at':min(cooling) if cooling else None,'isolated':isolated,
            'starved_tasks':sorted(target_tasks), 'capacity':capacity}



class LiveProbeFeeder:
    """Bounded task-specific recovery alongside the running coordinator.

    Only idle needed tasks are considered. SQLite BEGIN IMMEDIATE serializes
    reservation with the scheduler; no lease, counter or historical job is reset.
    """
    def __init__(self, project, root, seed, policy):
        from source.data.production_templates import load_templates
        self.project,self.root,self.seed,self.policy=project,root,seed,policy
        self.config=base.read(root/'config.json')
        self.contract=base.read(root/'run_contract.json')
        self._check_frozen_inputs()
        self.templates=load_templates(project/self.config['template_library'],
                                      project/self.config['object_registry'])

    def _check_frozen_inputs(self):
        if (base.read(self.root/'config.json') != self.config
                or base.read(self.root/'run_contract.json') != self.contract):
            raise ValueError('Frozen collection metadata changed during live recovery')
        for field,key in [('template_sha256','template_library'),('registry_sha256','object_registry')]:
            if base.digest(self.project/self.config[key]) != self.contract[field]:
                raise ValueError('Frozen catalogue changed during live recovery: '+key)

    def refill(self, database, *, now=None):
        from source.data.production_queue import ProductionQueue
        self._check_frozen_inputs()
        return seed_probes(ProductionQueue(Path(database)),self.templates,self.config,
                           self.seed,self.policy,now=now,while_running=True)

class PreflightChanged(RuntimeError):
    def __init__(self, report):
        super().__init__(json.dumps(report,ensure_ascii=False));self.report=report


def prepare(project,root,destination,policy):
    from source.data.production_bootstrap import intent_path
    from source.data.production_index import live_dataset_processes
    from source.data.production_queue import ProductionQueue
    from source.data.production_templates import load_templates
    report=base.check(project,root)
    if not report['ready']:raise PreflightChanged(report)
    config=base.read(root/'config.json'); control=(project/config['queue_database']).parent
    with ExitStack() as stack:
        for p in [control/'supervisor.lock',intent_path(root).with_suffix('.lock'),root/'coordinator.lock']:
            f=stack.enter_context(p.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if live_dataset_processes(root):raise BlockingIOError('Previous dataset workers remain alive')
        report=base.check(project,root)
        if not report['ready']:raise PreflightChanged(report)
        templates=load_templates(project/config['template_library'],project/config['object_registry'])
        destination.mkdir(parents=True,exist_ok=False)
        with sqlite3.connect(Path(report['database']).as_uri()+'?mode=ro',uri=True) as original:
            with sqlite3.connect(destination/'queue_before.sqlite') as backup:original.backup(backup)
        queue=ProductionQueue(Path(report['database']))
        report['recovery']=seed_probes(queue,templates,config,report['sampling_seed'],policy)
        atomic_json(destination/'preparation.json',report)
        return report


def budget(state, policy, now):
    timestamps=[t for t in state.get('restart_timestamps',[]) if now-t<policy['restart_window_seconds']]
    if len(timestamps)>=policy['maximum_restarts_per_window']:
        state['restart_timestamps']=timestamps;return None
    delay=min(policy['backoff_maximum_seconds'],policy['backoff_initial_seconds']*2**len(timestamps))
    state['restart_timestamps']=timestamps+[now];return delay


def decision(termination, returncode):
    if returncode in (75,-signal.SIGKILL,-signal.SIGSEGV,-signal.SIGABRT,-signal.SIGBUS):return 'RETRY_CRASH'
    if returncode:return 'STOP_CHILD_FAILURE'
    if termination in ('QUOTA_COMPLETE_FILE_VERIFIED','QUOTA_COMPLETE'):return 'COMPLETE'
    if termination=='NO_RUNNABLE_CANDIDATE':return 'RECOVER_CANDIDATES'
    if termination=='SUPERVISOR_RESTART_LIMIT':return 'RETRY_CRASH'
    if termination in ('SUPERVISOR_PREFLIGHT_REJECTED','RESOURCE_MEMORY_STOP','RESOURCE_STORAGE_STOP'):return 'RECHECK_RESOURCES'
    # A resource failure is retried only after diagnose has rechecked resources;
    # integrity, changed source, unknown lease failure and user stop are never ignored.
    return 'STOP_'+str(termination or 'MISSING_SUMMARY')


def resource_stop(summary):
    """Recognize only explicit recorded resource causes, never generic guard errors."""
    if summary.get('stop_signals') or summary.get('wall_budget_seconds') is not None or summary.get('integrity_failures'):
        return None
    errors=[e.get('error','') for e in summary.get('runtime_guard_failures',[])]
    states=[r.get('state') for r in summary.get('results',[])]
    if errors:
        if all(e.startswith('RuntimeError: System memory reserve violated:') for e in errors):return 'RESOURCE_MEMORY_STOP'
        return None
    if 'MEMORY_STOP' in states:return 'RESOURCE_MEMORY_STOP'
    volumes=[r.get('volume',{}) for r in summary.get('results',[]) if r.get('state')=='VOLUME_STOP']
    allowed=('Insufficient workspace above configured disk reserve','Available space after reserve is below the two-worker workspace budget')
    if volumes and all(v.get('errors') and all(e in allowed for e in v['errors']) for v in volumes):return 'RESOURCE_STORAGE_STOP'
    return None


def activity(database):
    with sqlite3.connect(Path(database).as_uri()+'?mode=ro',uri=True,timeout=1) as db:
        return (db.execute('SELECT max(id) FROM events').fetchone()[0],
                db.execute('SELECT max(rowid) FROM episodes').fetchone()[0])


def finalizing(database, root):
    """Long full-file acceptance legitimately produces no new Episode rows."""
    if not (root/'config.json').is_file():return False
    from source.data.production_statistics import ReadOnlyQueue
    c=base.read(root/'config.json')
    return ReadOnlyQueue(Path(database)).quota_status(c['task_hours'],min_assets=c['minimum_independent_assets'],
        count_failures=c['count_failed_episodes'],outcome_hours=c.get('task_outcome_hours'),
        coverage_success_only=c.get('coverage_requires_success',True))['complete']


class Watchdog:
    def __init__(self,project,root,state_dir,policy):
        self.project,self.root,self.directory,self.policy=project,root,state_dir,policy
        self.stopped=threading.Event();self.child=None
        self.path=state_dir/'state.json'
        identity={'policy':POLICY,'project':str(project),'dataset_root':str(root)}
        self.state=base.read(self.path) if self.path.exists() else {**identity,'restart_timestamps':[],'sessions':0}
        if any(self.state.get(k)!=v for k,v in identity.items()):raise ValueError('Watchdog state belongs to another dataset')
        legacy = {**DEFAULTS, 'initial_cpu_limit':8, 'maximum_new_probes':4}
        if self.state.get('options') == legacy and policy == DEFAULTS:
            # Only the shipped legacy defaults are migrated. Keep budgets, sessions and progress.
            backup = state_dir/'state.before-resource-adaptive-20261127.json'
            if not backup.exists():atomic_json(backup,self.state)
            old_limit = self.state.get('cpu_limit',8)
            self.state['cpu_limit'] = 0 if old_limit == 8 else old_limit
            self.state['options'] = policy
            self.state['resource_policy_migration'] = {'date':'2026-11-27','backup':str(backup),
                'old_cpu_limit':old_limit,'default_cpu_cap_removed':old_limit==8}
            atomic_json(self.path,self.state)
        if self.state.get('options') not in (None,policy):raise ValueError('Policy changed; use an explicit reviewed state migration')
        self.state['options']=policy
        self.state.setdefault('cpu_limit',policy['initial_cpu_limit'])
        self.state['cpu_limit_mode']='inherit-server-affinity' if self.state['cpu_limit']==0 else 'explicit-or-resource-reduced'

    def save(self,status,**values):
        previous=self.state.get('status')
        self.state['cpu_limit_mode']='inherit-server-affinity' if self.state['cpu_limit']==0 else 'explicit-or-resource-reduced'
        self.state.update(status=status,heartbeat_unix=time.time(),**values);atomic_json(self.path,self.state)
        if previous!=status:self.event('STATE_CHANGED',status=status)

    def event(self,kind,**values):
        with (self.directory/'events.jsonl').open('a') as f:
            value={'time':time.time(),'event':kind,**values}
            f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(value,ensure_ascii=False),flush=True)

    def wait(self,seconds,status):
        deadline=time.monotonic()+max(0,seconds)
        while not self.stopped.is_set() and time.monotonic()<deadline:
            self.save(status,next_check_unix=time.time()+max(0,deadline-time.monotonic()))
            self.stopped.wait(min(self.policy['poll_seconds'],max(0,deadline-time.monotonic())))

    def terminate_child(self):
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)
            try:self.child.wait(timeout=self.policy['shutdown_grace_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid,signal.SIGKILL);self.child.wait(timeout=10)
                self.event('CHILD_FORCED_KILL',pid=self.child.pid)

    def run_child(self,report,session):
        command=[sys.executable,str(self.project/'apps/five_task_production.py'),'serve','--config',str(self.root/'config.json'),
                 '--sampling-seed',str(report['sampling_seed']),'--maximum-restarts','3','--output',str(session/'supervisor')]
        session.mkdir(parents=True,exist_ok=True)
        last_activity=None;last_change=time.monotonic();stalled=False
        live_feeder=None;last_live_refill=None
        with (session/'console.log').open('xb') as log:
            affinity=os.sched_getaffinity(0) if hasattr(os,'sched_getaffinity') else None
            try:
                if affinity and self.state['cpu_limit']:os.sched_setaffinity(0,sorted(affinity)[:self.state['cpu_limit']])
                self.child=subprocess.Popen(command,cwd=self.project,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'PYTHONPATH':str(self.project),'MUJOCO_GL':'egl'})
            finally:
                if affinity:os.sched_setaffinity(0,affinity)
            self.event('CHILD_STARTED',pid=self.child.pid,command=command,cpu_limit=self.state['cpu_limit'],
                inherited_cpu_count=len(affinity) if affinity else os.cpu_count(),
                cpu_limit_mode=self.state['cpu_limit_mode'])
            try:
                while self.child.poll() is None and not self.stopped.is_set():
                    recovering=False
                    try:
                        if (self.root/'config.json').is_file():
                            from source.data.production_index import active_database
                            c=base.read(self.root/'config.json')
                            report['database']=str(active_database(self.root,self.project/c['queue_database']))
                        current=activity(report['database'])
                        if current!=last_activity:last_activity=current;last_change=time.monotonic()
                        if ((self.root/'config.json').is_file() and
                                (last_live_refill is None or time.monotonic()-last_live_refill>=60)):
                            if live_feeder is None:
                                live_feeder=LiveProbeFeeder(self.project,self.root,report['sampling_seed'],self.policy)
                            recovery=live_feeder.refill(report['database'])
                            previous_recovery=self.state.get('live_recovery',{})
                            self.state['live_recovery']={**recovery,'checked_unix':time.time()}
                            last_live_refill=time.monotonic()
                            if (recovery['added'] or recovery['state']!=previous_recovery.get('state')
                                    or recovery.get('starved_tasks')!=previous_recovery.get('starved_tasks')):
                                self.event('LIVE_RECOVERY_PROBES_QUEUED' if recovery['added'] else 'LIVE_RECOVERY_STATUS',**recovery)
                    except sqlite3.DatabaseError as e:
                        from source.data.production_index import is_database_corruption
                        if is_database_corruption(e):recovering=True
                        elif not (isinstance(e,sqlite3.OperationalError) and str(e) in ('database is locked','database is busy')):raise
                    self.save('COLLECTING',child_pid=self.child.pid,activity=last_activity,last_activity_age=time.monotonic()-last_change)
                    if recovering:
                        self.save('ORIGINAL_INDEX_RECOVERY_PENDING',child_pid=self.child.pid)
                    elif time.monotonic()-last_change>self.policy['stall_seconds']:
                        if finalizing(report['database'],self.root):
                            self.save('FINAL_FILE_ACCEPTANCE',child_pid=self.child.pid,
                                      note='No Episode growth expected; waiting for independent file acceptance')
                        else:
                            stalled=True;self.event('NO_DATABASE_PROGRESS',seconds=self.policy['stall_seconds']);break
                    self.stopped.wait(self.policy['poll_seconds'])
            finally:
                if self.stopped.is_set() or stalled:self.terminate_child()
            code=self.child.wait()
        if self.stopped.is_set():return 'USER_STOP',code
        if stalled:return 'WATCHDOG_STALLED',code
        p=session/'supervisor/supervisor.json'
        term=base.read(p).get('termination') if p.exists() else None
        if term in ('STOP_OR_WALL_BUDGET','RUNTIME_GUARD_FAILED'):
            summary=self.root/'summary.json'
            if summary.is_file():
                details=base.read(summary)
                # Preserve the evidence before the next coordinator replaces summary.json.
                snapshot=session/'child_stop_summary.json'
                atomic_json(snapshot,details)
                causes={k:details.get(k) for k in (
                    'termination','stop_signals','wall_budget_seconds',
                    'runtime_guard_failures','integrity_failures')}
                self.save('CHILD_STOP_DIAGNOSTICS',last_stop_details=causes,
                          last_stop_summary=str(snapshot))
                self.event('CHILD_STOP_DIAGNOSTICS',summary=str(snapshot),**causes)
                term=resource_stop(details) or term
        return term,code

    def run(self):
        previous={s:signal.signal(s,lambda *_:self.stopped.set()) for s in [signal.SIGINT,signal.SIGTERM]}
        self.event('WATCHDOG_STARTED',pid=os.getpid())
        try:
            while not self.stopped.is_set():
                recent=[t for t in self.state.get('restart_timestamps',[]) if time.time()-t<self.policy['restart_window_seconds']]
                if len(recent)>=self.policy['maximum_restarts_per_window']:
                    self.save('STOP_RESTART_BUDGET');return 78
                kind,report=diagnose(self.project,self.root)
                self.save(kind,check=report)
                if kind.startswith('WAIT_'):
                    since=self.state.setdefault('resource_wait_since',time.time())
                    if time.time()-since>=self.policy['resource_wait_seconds']:
                        self.save('STOP_RESOURCE_WAIT_LIMIT');return 78
                    self.wait(self.policy['backoff_initial_seconds'],kind);continue
                if kind!='READY':return 78
                index=self.state['sessions'];session=self.directory/'sessions'/f'{index:06d}'
                self.state['sessions']=index+1;self.save('PREPARING')
                try:prepared=prepare(self.project,self.root,session/'preparation',self.policy)
                except PreflightChanged as changed:
                    k=classify_check(changed.report)
                    self.save(k,check=changed.report)
                    if not k.startswith('WAIT_'):return 78
                    self.state.setdefault('resource_wait_since',time.time())
                    self.wait(self.policy['backoff_initial_seconds'],k);continue
                except BlockingIOError:
                    since=self.state.setdefault('owner_wait_since',time.time())
                    if time.time()-since>=self.policy['resource_wait_seconds']:
                        self.save('STOP_OTHER_OWNER_WAIT_LIMIT');return 78
                    self.wait(self.policy['backoff_initial_seconds'],'WAIT_PREVIOUS_OWNER');continue
                self.state.pop('resource_wait_since',None)
                self.state.pop('owner_wait_since',None)
                backups=self.state.get('managed_backups',[])+[str(session/'preparation/queue_before.sqlite')]
                while len(backups)>3:
                    obsolete=Path(backups.pop(0))
                    if not obsolete.resolve().is_relative_to((self.directory/'sessions').resolve()) or obsolete.is_symlink():
                        raise ValueError('Managed backup path escaped watchdog directory')
                    if obsolete.is_file():obsolete.unlink();self.event('OLD_MANAGED_BACKUP_PRUNED',path=str(obsolete))
                self.state['managed_backups']=backups
                recovery=prepared['recovery'];self.event('QUEUE_PREPARED',**recovery)
                if recovery['state']=='COOLDOWN':
                    self.wait(min(self.policy['probe_cooldown_seconds'],max(1,recovery['next_retry_at']-time.time())),'CANDIDATE_COOLDOWN');continue
                termination,code=self.run_child(prepared,session)
                self.event('CHILD_FINISHED',termination=termination,returncode=code)
                self.save('SESSION_FINISHED',last_termination=termination,last_returncode=code)
                if self.stopped.is_set() or termination=='USER_STOP':break
                if termination=='RESOURCE_MEMORY_STOP' or code==-signal.SIGKILL:
                    previous_limit=self.state['cpu_limit'] or (len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1))
                    self.state['cpu_limit']=max(self.policy['minimum_cpu_limit'],previous_limit//2)
                    self.event('RESOURCE_CPU_LIMIT_REDUCED',cpu_limit=self.state['cpu_limit'])
                choice=decision(termination,code)
                if termination=='WATCHDOG_STALLED':choice='RETRY_CRASH'
                if choice=='COMPLETE':self.save('COMPLETE');return 0
                if choice=='RECOVER_CANDIDATES':
                    if recovery['state']=='NO_PROBE':self.save('STOP_NO_ELIGIBLE_CANDIDATE');return 78
                    continue
                if choice=='RECHECK_RESOURCES':
                    k,_=diagnose(self.project,self.root)
                    if k.startswith('WAIT_'):continue
                    if k=='READY':
                        delay=budget(self.state,self.policy,time.time())
                        if delay is None:self.save('STOP_RESTART_BUDGET');return 78
                        self.wait(delay,'RESOURCE_RECOVERED_BACKOFF');continue
                    self.save('STOP_UNEXPLAINED_PREFLIGHT');return 78
                if choice!='RETRY_CRASH':self.save(choice);return 78
                delay=budget(self.state,self.policy,time.time())
                if delay is None:self.save('STOP_RESTART_BUDGET');return 78
                self.event('RECOVERY_BACKOFF',seconds=delay);self.save('RECOVERY_BACKOFF');self.wait(delay,'RECOVERY_BACKOFF')
            self.save('USER_STOP');return 0
        except Exception as error:
            self.terminate_child();self.save('STOP_EXCEPTION',error=f'{type(error).__name__}: {error}');self.event('EXCEPTION',error=str(error));return 78
        finally:
            self.terminate_child()
            for s,handler in previous.items():signal.signal(s,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run','status'])
    parser.add_argument('--project',type=Path,default=Path('/home/ftcl/dex_hand_project'))
    parser.add_argument('--dataset-root',type=Path,default=Path('/media/ftcl/外置存储/dex_hand_dataset_batch2'))
    parser.add_argument('--state-dir',type=Path)
    parser.add_argument('--policy',type=Path)
    args=parser.parse_args();project,root=args.project.resolve(),args.dataset_root.resolve()
    sys.path.insert(0,str(project));os.chdir(project)
    if os.environ.get('DEX_DATASET_ROOT'):raise ValueError('Unset DEX_DATASET_ROOT; frozen paths must not be overridden')
    state=args.state_dir or project/'outputs/five_task_speed_batch2_control/autorecovery_quality_v3'
    if args.command=='status':
        print(json.dumps(base.read(state/'state.json') if (state/'state.json').is_file() else {'status':'NOT_STARTED'},ensure_ascii=False,indent=2));return 0
    policy=options(args.policy)
    if args.command=='check':
        kind,report=diagnose(project,root);print(json.dumps({'classification':kind,**report,'watchdog_options':policy},ensure_ascii=False,indent=2));return 0 if kind=='READY' else 2
    if state.resolve().is_relative_to(root):raise ValueError('Watchdog state must remain outside dataset')
    state.mkdir(parents=True,exist_ok=True)
    control=project/'outputs/five_task_speed_batch2_control'
    control.mkdir(parents=True,exist_ok=True)
    with (control/'autorecovery.manager.lock').open('a') as manager, (state/'watchdog.lock').open('a') as lock:
        try:fcntl.flock(manager,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('Another external-dataset watchdog is running',file=sys.stderr);return 78
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('Another watchdog owns this state directory',file=sys.stderr);return 78
        try:return Watchdog(project,root,state,policy).run()
        except (ValueError,OSError) as error:
            print(f'Cannot load persistent watchdog state: {error}',file=sys.stderr);return 78


if __name__=='__main__':
    raise SystemExit(main())
