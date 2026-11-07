"""Durable work leases and quota accounting for the production coordinator."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid


TASKS = ("lift", "pick_place", "stack", "push", "nut_assembly")


def process_identity(pid):
    """Linux/WSL boot and process start ticks protect against PID reuse."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {"pid": pid, "boot": boot, "start_ticks": stat[19]}
    except FileNotFoundError:
        return None


class ProductionQueue:
    def __init__(self, path: Path, *, initializing=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables:
                required = {'jobs': {'job_id', 'payload', 'lease_token', 'attempts'},
                            'episodes': {'job_id', 'asset_hash', 'lineage_hash', 'content_hash', 'seconds'},
                            'events': {'job_id', 'state', 'details', 'timestamp'}}
                if initializing:
                    allowed = {'jobs', 'episodes', 'events', 'template_stats', 'queue_meta', 'attempt_metrics', 'template_efficiency'}
                    if tables - allowed:
                        raise ValueError('Unexpected tables in interrupted initialization')
                    if 'jobs' in tables and db.execute("SELECT count(*) FROM jobs WHERE attempts>0 OR state<>'PENDING'").fetchone()[0]:
                        raise ValueError('Interrupted initialization contains execution history')
                    if 'episodes' in tables and db.execute('SELECT count(*) FROM episodes').fetchone()[0]:
                        raise ValueError('Interrupted initialization contains registered data')
                elif not set(required).issubset(tables):
                    raise ValueError('Existing database is not this production queue; refusing changes')
                for table, columns in required.items():
                    if initializing and table not in tables:
                        continue
                    actual = {row['name'] for row in db.execute(f'PRAGMA table_info({table})')}
                    if not columns.issubset(actual):
                        raise ValueError('Existing inventory schema is incompatible; refusing changes')
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, task TEXT NOT NULL, asset_hash TEXT NOT NULL,
                    payload TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'PENDING',
                    attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL,
                    lease_token TEXT, lease_until REAL, error TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episodes (
                    job_id TEXT UNIQUE NOT NULL, task TEXT NOT NULL, asset_hash TEXT NOT NULL,
                    lineage_hash TEXT UNIQUE NOT NULL, content_hash TEXT UNIQUE NOT NULL,
                    relative_path TEXT UNIQUE NOT NULL, seconds REAL NOT NULL,
                    success INTEGER NOT NULL, verification TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, job_id TEXT NOT NULL, state TEXT NOT NULL,
                    details TEXT NOT NULL, timestamp REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS episode_asset_success ON episodes(asset_hash,success);
                CREATE TABLE IF NOT EXISTS template_stats (
                    template_id TEXT PRIMARY KEY, scheduled INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 0, successful INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0, duplicates INTEGER NOT NULL DEFAULT 0,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    consecutive_duplicates INTEGER NOT NULL DEFAULT 0,
                    last_candidate_index INTEGER NOT NULL DEFAULT -1
                );
                CREATE TABLE IF NOT EXISTS queue_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempt_metrics (
                    job_id TEXT NOT NULL, attempt INTEGER NOT NULL, wall_seconds REAL NOT NULL,
                    state TEXT NOT NULL, PRIMARY KEY(job_id,attempt),
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS template_efficiency (
                    template_id TEXT PRIMARY KEY, claimed INTEGER NOT NULL DEFAULT 0,
                    timed INTEGER NOT NULL DEFAULT 0, worker_seconds REAL NOT NULL DEFAULT 0,
                    success_seconds REAL NOT NULL DEFAULT 0, failure_seconds REAL NOT NULL DEFAULT 0
                );
            """)
            db.execute('BEGIN IMMEDIATE')
            try:
                if 'candidate_seconds' not in {row['name'] for row in db.execute('PRAGMA table_info(attempt_metrics)')}:
                    db.execute('ALTER TABLE attempt_metrics ADD COLUMN candidate_seconds REAL')
                if not db.execute("SELECT 1 FROM queue_meta WHERE key='template_stats_v1'").fetchone():
                    rows = db.execute('SELECT j.*,e.success FROM jobs j LEFT JOIN episodes e ON j.job_id=e.job_id').fetchall()
                    for row in rows:
                        self._scheduled(db, json.loads(row['payload']))
                        if row['state'] in ('EXPORTED', 'DUPLICATE', 'QUARANTINED'):
                            self._terminal(db, row, row['state'], bool(row['success']))
                    db.execute("INSERT INTO queue_meta VALUES('template_stats_v1','initialized')")
                if not db.execute("SELECT 1 FROM queue_meta WHERE key='template_efficiency_v1'").fetchone():
                    db.execute('UPDATE template_efficiency SET claimed=0,timed=0,worker_seconds=0,success_seconds=0,failure_seconds=0')
                    for row in db.execute('SELECT payload,attempts FROM jobs').fetchall():
                        self._efficiency(db, json.loads(row['payload']).get('template_id'), claimed=row['attempts'])
                    for row in db.execute('SELECT j.payload,m.wall_seconds,m.state,e.seconds,e.success '
                            'FROM attempt_metrics m JOIN jobs j ON j.job_id=m.job_id '
                            'LEFT JOIN episodes e ON e.job_id=m.job_id').fetchall():
                        credited = row['seconds'] if row['state'] == 'EXPORTED' and row['seconds'] is not None else 0.
                        self._efficiency(db, json.loads(row['payload']).get('template_id'), timed=1,
                            wall=row['wall_seconds'], success=credited if row['success'] else 0.,
                            failure=credited if not row['success'] else 0.)
                    db.execute("INSERT INTO queue_meta VALUES('template_efficiency_v1','initialized')")
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @staticmethod
    def _efficiency(db, template, *, claimed=0, timed=0, wall=0., success=0., failure=0.):
        if template:
            db.execute('INSERT INTO template_efficiency VALUES(?,?,?,?,?,?) ON CONFLICT(template_id) DO UPDATE SET '
                'claimed=claimed+excluded.claimed,timed=timed+excluded.timed,worker_seconds=worker_seconds+excluded.worker_seconds,'
                'success_seconds=success_seconds+excluded.success_seconds,failure_seconds=failure_seconds+excluded.failure_seconds',
                (template, claimed, timed, wall, success, failure))

    @staticmethod
    def _scheduled(db, payload):
        template = payload.get('template_id')
        if not template:
            return
        db.execute('INSERT INTO template_stats(template_id,scheduled,active,last_candidate_index) VALUES(?,1,1,?) '
                   'ON CONFLICT(template_id) DO UPDATE SET scheduled=scheduled+1,active=active+1,'
                   'last_candidate_index=max(last_candidate_index,excluded.last_candidate_index)',
                   (template, payload.get('candidate_index', -1)))

    @staticmethod
    def _terminal(db, row, state, success=False):
        template = json.loads(row['payload']).get('template_id')
        if not template:
            return
        good = state == 'EXPORTED' and success
        duplicate = state == 'DUPLICATE'
        failed = not good and not duplicate
        db.execute('UPDATE template_stats SET active=max(0,active-1),successful=successful+?,failed=failed+?,'
                   'duplicates=duplicates+?,consecutive_failures=CASE WHEN ? THEN consecutive_failures+1 ELSE 0 END,'
                   'consecutive_duplicates=CASE WHEN ? THEN consecutive_duplicates+1 ELSE 0 END WHERE template_id=?',
                   (int(good), int(failed), int(duplicate), int(failed), int(duplicate), template))

    @contextmanager
    def connect(self, *, timeout=30):
        db = sqlite3.connect(self.path, timeout=timeout, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @staticmethod
    def event(db, job_id, state, details):
        db.execute("INSERT INTO events(job_id,state,details,timestamp) VALUES(?,?,?,?)",
                   (job_id, state, json.dumps(details, sort_keys=True), time.time()))

    def enqueue(self, job_id, task, asset_hash, payload, max_attempts=3):
        if task not in TASKS or len(asset_hash) != 64 or max_attempts < 1 or any(
                char not in '0123456789abcdef' for char in asset_hash):
            raise ValueError("Invalid task, asset SHA-256 or retry budget")
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self.transaction() as db:
            old = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if old:
                if (old["task"], old["asset_hash"], old["payload"], old["max_attempts"]) != (
                        task, asset_hash, encoded, max_attempts):
                    raise ValueError("Existing job ID has a different immutable definition")
                return False
            db.execute("INSERT INTO jobs(job_id,task,asset_hash,payload,max_attempts,created_at) "
                       "VALUES(?,?,?,?,?,?)", (job_id, task, asset_hash, encoded, max_attempts, time.time()))
            self._scheduled(db, payload)
            self.event(db, job_id, "PENDING", {})
            return True

    def recover(self, now=None, *, abandoned=False):
        now = time.time() if now is None else now
        with self.transaction() as db:
            rows = db.execute("SELECT * FROM jobs WHERE state IN ('RUNNING','VALIDATING')").fetchall()
            recovered = 0
            for row in rows:
                owner_dead = False
                if abandoned:
                    event = db.execute("SELECT details FROM events WHERE job_id=? AND state='RUNNING' "
                                       "ORDER BY id DESC LIMIT 1", (row["job_id"],)).fetchone()
                    details = json.loads(event["details"]) if event else {}
                    owner = details.get("owner")
                    worker = details.get("worker")
                    worker_alive = worker is not None and process_identity(worker["pid"]) == worker
                    if worker_alive:
                        continue
                    owner_dead = owner is not None and process_identity(owner["pid"]) != owner
                if row["lease_until"] > now and not owner_dead:
                    continue
                state = "QUARANTINED" if row["attempts"] >= row["max_attempts"] else "PENDING"
                db.execute("UPDATE jobs SET state=?,lease_token=NULL,lease_until=NULL,"
                           "error='expired lease requires artifact reconciliation' WHERE job_id=?",
                           (state, row["job_id"]))
                self.event(db, row["job_id"], state, {"reason": "owner_dead" if owner_dead else "lease_expired"})
                if state == 'QUARANTINED':
                    self._terminal(db, row, state)
                recovered += 1
            return recovered

    def claim(self, allowed_tasks=TASKS, lease_seconds=300):
        tasks = tuple(allowed_tasks)
        if not tasks or any(task not in TASKS for task in tasks) or lease_seconds <= 0:
            raise ValueError("Invalid task filter or lease duration")
        with self.transaction() as db:
            marks = ",".join("?" for _ in tasks)
            priority = " ".join(f"WHEN ? THEN {index}" for index, _ in enumerate(tasks))
            row = db.execute(f"SELECT * FROM jobs WHERE state='PENDING' AND task IN ({marks}) "
                             f"ORDER BY CASE task {priority} END,created_at,job_id LIMIT 1",
                             (*tasks, *tasks)).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            db.execute("UPDATE jobs SET state='RUNNING',attempts=attempts+1,lease_token=?,"
                       "lease_until=? WHERE job_id=?", (token, time.time() + lease_seconds, row["job_id"]))
            self.event(db, row["job_id"], "RUNNING", {"lease_token": token, "owner": process_identity(os.getpid())})
            self._efficiency(db, json.loads(row['payload']).get('template_id'), claimed=1)
            return {**dict(row), "payload": json.loads(row["payload"]),
                    "lease_token": token, "attempts": row["attempts"] + 1, "state": "RUNNING"}

    def attach_worker(self, job_id, token, pid):
        with self.transaction() as db:
            self.owned(db, job_id, token)
            self.event(db, job_id, "RUNNING", {"lease_token": token,
                                             "owner": process_identity(os.getpid()),
                                             "worker": process_identity(pid)})

    @staticmethod
    def owned(db, job_id, token):
        row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None or row["lease_token"] != token or row["state"] not in ("RUNNING", "VALIDATING"):
            raise ValueError("Stale or invalid lease")
        if row["lease_until"] <= time.time():
            raise ValueError("Lease expired")
        return row

    def heartbeat(self, job_id, token, lease_seconds=300, *, cancelled=None):
        if lease_seconds <= 0:
            raise ValueError("Lease duration must be positive")
        deadline = time.monotonic() + 30
        while cancelled is None or not cancelled.is_set():
            try:
                with self.connect(timeout=.25) as db:
                    db.execute('BEGIN IMMEDIATE')
                    try:
                        if cancelled is not None and cancelled.is_set():
                            db.rollback()
                            return
                        self.owned(db, job_id, token)
                        db.execute('UPDATE jobs SET lease_until=? WHERE job_id=?',
                                   (time.time() + lease_seconds, job_id))
                        db.commit()
                    except BaseException:
                        db.rollback()
                        raise
                return
            except sqlite3.OperationalError as error:
                if 'locked' not in str(error).lower() or time.monotonic() >= deadline:
                    raise
                if cancelled is not None:
                    cancelled.wait(.05)
                else:
                    time.sleep(.05)

    def validating(self, job_id, token):
        with self.transaction() as db:
            self.owned(db, job_id, token)
            db.execute("UPDATE jobs SET state='VALIDATING' WHERE job_id=?", (job_id,))
            self.event(db, job_id, "VALIDATING", {})

    def fail(self, job_id, token, reason, retryable=True, *, attempt_limit=None, details=None):
        if attempt_limit is not None and (isinstance(attempt_limit, bool)
                                         or not isinstance(attempt_limit, int) or attempt_limit < 1):
            raise ValueError('Failure attempt limit must be a positive integer')
        with self.transaction() as db:
            row = self.owned(db, job_id, token)
            limit = min(row['max_attempts'], attempt_limit) if attempt_limit is not None else row['max_attempts']
            state = "PENDING" if retryable and row["attempts"] < limit else "QUARANTINED"
            db.execute("UPDATE jobs SET state=?,error=?,lease_token=NULL,lease_until=NULL WHERE job_id=?",
                       (state, reason, job_id))
            self.event(db, job_id, state, {"reason": reason, 'failure_policy': details,
                                         'effective_attempt_limit': limit})
            if state == 'QUARANTINED':
                self._terminal(db, row, state)
            return state

    def register(self, job_id, token, *, lineage_hash, content_hash, relative_path,
                 seconds, success, verification):
        import math

        if not math.isfinite(seconds) or seconds <= 0 or not isinstance(success, bool):
            raise ValueError("Invalid duration or success flag")
        if any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
               for value in (lineage_hash, content_hash)):
            raise ValueError("Expected SHA-256 content and lineage hashes")
        path = Path(relative_path)
        if path.is_absolute() or '..' in path.parts or str(path) == '.':
            raise ValueError("Episode path must stay within dataset root")
        if verification.get("passed") is not True:
            raise ValueError("Episode must pass external file verification before registration")
        with self.transaction() as db:
            row = self.owned(db, job_id, token)
            if row["state"] != "VALIDATING":
                raise ValueError("Episode must enter VALIDATING before registration")
            duplicate = db.execute("SELECT job_id FROM episodes WHERE lineage_hash=? OR content_hash=?",
                                   (lineage_hash, content_hash)).fetchone()
            if duplicate:
                state = "DUPLICATE"
                self.event(db, job_id, state, {"existing_job": duplicate["job_id"]})
            else:
                state = "EXPORTED"
                db.execute("INSERT INTO episodes VALUES(?,?,?,?,?,?,?,?,?)",
                           (job_id, row["task"], row["asset_hash"], lineage_hash, content_hash,
                            relative_path, seconds, int(success), json.dumps(verification)))
                self.event(db, job_id, state, {"seconds": seconds})
            db.execute("UPDATE jobs SET state=?,lease_token=NULL,lease_until=NULL WHERE job_id=?",
                       (state, job_id))
            self._terminal(db, row, state, success)
            return state

    def record_attempt_metrics(self, job_id, attempt, wall_seconds, state, *, candidate_seconds=None):
        import math

        if (isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1
                or not math.isfinite(wall_seconds) or wall_seconds < 0 or not isinstance(state, str)):
            raise ValueError('Invalid attempt metrics')
        if candidate_seconds is not None and (isinstance(candidate_seconds, bool)
                or not isinstance(candidate_seconds, (int, float))
                or not math.isfinite(candidate_seconds) or candidate_seconds < 0):
            raise ValueError('Invalid candidate duration')
        with self.transaction() as db:
            job = db.execute('SELECT attempts,payload FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if job is None or attempt > job['attempts']:
                raise ValueError('Metrics must refer to an actually claimed attempt')
            old = db.execute('SELECT wall_seconds,state,candidate_seconds FROM attempt_metrics WHERE job_id=? AND attempt=?',
                             (job_id, attempt)).fetchone()
            if old:
                if (old['wall_seconds'], old['state'], old['candidate_seconds']) != (wall_seconds, state, candidate_seconds):
                    raise ValueError('Attempt metrics are immutable')
                return
            db.execute('INSERT INTO attempt_metrics(job_id,attempt,wall_seconds,state,candidate_seconds) VALUES(?,?,?,?,?)',
                       (job_id, attempt, wall_seconds, state, candidate_seconds))
            episode = db.execute('SELECT seconds,success FROM episodes WHERE job_id=?', (job_id,)).fetchone()
            credited = episode['seconds'] if state == 'EXPORTED' and episode is not None else 0.
            self._efficiency(db, json.loads(job['payload']).get('template_id'), timed=1, wall=wall_seconds,
                             success=credited if episode is not None and episode['success'] else 0.,
                             failure=credited if episode is not None and not episode['success'] else 0.)

    def scheduling_efficiency(self):
        with self.connect() as db:
            if db.execute("SELECT 1 FROM queue_meta WHERE key='file_rebuild_history'").fetchone():
                return {}
            return {row['template_id']: dict(row) for row in db.execute('SELECT * FROM template_efficiency')}

    def scheduling_stats(self, *, count_failures=False):
        with self.connect() as db:
            templates = {row['template_id']: dict(row) for row in db.execute('SELECT * FROM template_stats')}
            covered = {row[0] for row in db.execute('SELECT DISTINCT asset_hash FROM episodes' +
                                                   ('' if count_failures else ' WHERE success=1'))}
        return templates, covered

    def observe_progress(self, remaining_hours, remaining_assets, *, attempt=False, limit=1000, progress_hint=False):
        """Persist stalled-attempt count across restarts without crediting excess data."""
        import math

        if (not math.isfinite(remaining_hours) or remaining_hours < 0 or remaining_assets < 0
                or isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError('Invalid progress watchdog values')
        with self.transaction() as db:
            row = db.execute("SELECT value FROM queue_meta WHERE key='quota_progress_v1'").fetchone()
            previous = json.loads(row['value']) if row else None
            improved = (progress_hint or previous is None or remaining_hours < previous['remaining_hours'] - 1e-10
                        or remaining_assets < previous['remaining_assets'])
            count = 0 if improved else previous['attempts_without_progress'] + int(attempt)
            state = {'remaining_hours': remaining_hours, 'remaining_assets': remaining_assets,
                     'attempts_without_progress': count}
            db.execute("INSERT INTO queue_meta(key,value) VALUES('quota_progress_v1',?) "
                       'ON CONFLICT(key) DO UPDATE SET value=excluded.value', (json.dumps(state),))
        return {**state, 'stalled': count >= limit, 'limit': limit}

    def status(self, *, count_failures=False):
        eligible = '1=1' if count_failures else 'success=1'
        with self.connect() as db:
            jobs = dict(db.execute("SELECT state,count(*) FROM jobs GROUP BY state").fetchall())
            rows = db.execute(f"""SELECT task,
                SUM(CASE WHEN success=1 THEN seconds ELSE 0 END)/3600.0 AS successful_hours,
                SUM(CASE WHEN success=0 THEN seconds ELSE 0 END)/3600.0 AS failed_hours,
                SUM(CASE WHEN {eligible} THEN seconds ELSE 0 END)/3600.0 AS eligible_hours,
                COUNT(DISTINCT CASE WHEN {eligible} THEN asset_hash END) AS independent_assets
                FROM episodes GROUP BY task""").fetchall()
            total = db.execute(f"SELECT COUNT(*) AS episodes,COUNT(DISTINCT CASE WHEN {eligible} "
                               "THEN asset_hash END) AS independent_assets FROM episodes").fetchone()
        by_task = {}
        for task in TASKS:
            by_task[task] = {
                "successful_hours": 0., "failed_hours": 0., "eligible_hours": 0., "independent_assets": 0,
            }
        for row in rows:
            by_task[row['task']] = {key: row[key] for key in by_task[row['task']]}
        return {"jobs": jobs, "tasks": by_task, "episodes": total['episodes'],
                "eligible_hours": sum(t["eligible_hours"] for t in by_task.values()),
                "independent_assets": total['independent_assets']}

    def quota_status(self, task_hours, *, min_assets=100, count_failures=False, outcome_hours=None,
                     coverage_success_only=False):
        import math
        from source.data.production_quotas import validate_outcome_hours

        if set(task_hours) != set(TASKS) or any(
                not math.isfinite(value) or value <= 0 for value in task_hours.values()):
            raise ValueError("All five tasks require positive finite hour quotas")
        if min_assets < 1:
            raise ValueError("Independent asset target must be positive")
        status = self.status(count_failures=count_failures)
        if coverage_success_only and count_failures:
            coverage = self.status(count_failures=False)
            status['independent_assets'] = coverage['independent_assets']
            for task in TASKS:
                status['tasks'][task]['independent_assets'] = coverage['tasks'][task]['independent_assets']
        targets = validate_outcome_hours(task_hours, outcome_hours, count_failures)
        remaining_outcomes = None
        if targets is not None:
            remaining_outcomes = {}
            status['raw_eligible_hours'] = status['eligible_hours']
            for task in TASKS:
                counts = status['tasks'][task]
                remaining_outcomes[task] = {
                    'success': max(0., targets[task]['success'] - counts['successful_hours']),
                    'failure': max(0., targets[task]['failure'] - counts['failed_hours'])}
                counts['eligible_hours'] = (min(targets[task]['success'], counts['successful_hours'])
                                            + min(targets[task]['failure'], counts['failed_hours']))
            status['eligible_hours'] = sum(row['eligible_hours'] for row in status['tasks'].values())
        remaining = {task: max(0., hours - status["tasks"][task]["eligible_hours"])
                     for task, hours in task_hours.items()}
        if remaining_outcomes is not None:
            remaining = {task: sum(values.values()) for task, values in remaining_outcomes.items()}
        needed = sorted((task for task in TASKS if remaining[task] > 0),
                        key=lambda task: -remaining[task] / task_hours[task])
        coverage_missing = status["independent_assets"] < min_assets
        if not needed and coverage_missing:
            needed = sorted(TASKS, key=lambda task: status["tasks"][task]["independent_assets"])
        return {**status, "remaining_task_hours": remaining, 'remaining_outcome_hours': remaining_outcomes,
                'coverage_success_only': coverage_success_only,
                "remaining_assets": max(0, min_assets - status["independent_assets"]),
                "needed_tasks": needed,
                "complete": not any(remaining.values()) and not coverage_missing}
