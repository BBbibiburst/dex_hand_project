from concurrent.futures import ThreadPoolExecutor
import time
import hashlib
import subprocess
import sys
import sqlite3

import pytest

from source.data.production_queue import ProductionQueue
from source.data.production_queue import TASKS


def add(q, key):
    return q.enqueue(key, "lift", "a" * 64, {"seed": 1}, max_attempts=2)


def register(q, job, content="c", success=True):
    q.validating(job["job_id"], job["lease_token"])
    return q.register(job["job_id"], job["lease_token"], lineage_hash="b" * 64,
                      content_hash=content * 64, relative_path="episodes/" + job["job_id"],
                      seconds=3600., success=success, verification={"passed": True})


def test_immutable_jobs_and_unique_parallel_claims(tmp_path):
    q = ProductionQueue(tmp_path / "queue.sqlite")
    for i in range(8):
        add(q, str(i))
    assert not add(q, "0")
    with pytest.raises(ValueError):
        q.enqueue("0", "lift", "a" * 64, {"seed": 2})
    with ThreadPoolExecutor(2) as pool:
        jobs = list(pool.map(lambda _: ProductionQueue(q.path).claim(), range(8)))
    assert len({j["job_id"] for j in jobs}) == 8


def test_recovery_rejects_old_worker_and_caps_retries(tmp_path):
    q = ProductionQueue(tmp_path / "queue.sqlite")
    add(q, "x")
    old = q.claim()
    assert q.recover(now=time.time() + 1000) == 1
    new = q.claim()
    with pytest.raises(ValueError):
        q.heartbeat("x", old["lease_token"])
    assert q.fail("x", new["lease_token"], "encoding failure") == "QUARANTINED"
    assert q.claim() is None


def test_dedup_and_success_quota_survive_reopen(tmp_path):
    q = ProductionQueue(tmp_path / "queue.sqlite")
    add(q, "x")
    assert register(q, q.claim(), success=False) == "EXPORTED"
    add(q, "y")
    assert register(q, q.claim()) == "DUPLICATE"
    reopened = ProductionQueue(q.path)
    assert reopened.status()["eligible_hours"] == 0
    assert reopened.status(count_failures=True)["eligible_hours"] == 1
    assert reopened.status(count_failures=True)["episodes"] == 1


def test_incomplete_output_cannot_register(tmp_path):
    q = ProductionQueue(tmp_path / "queue.sqlite")
    add(q, "x")
    job = q.claim()
    with pytest.raises(ValueError):
        q.register("x", job["lease_token"], lineage_hash="b" * 64, content_hash="c" * 64,
                   relative_path="episodes/x", seconds=1., success=True, verification={})
    assert q.status()["episodes"] == 0


def test_1000_hours_do_not_complete_missing_object_coverage(tmp_path):
    q = ProductionQueue(tmp_path / "queue.sqlite")
    for task in TASKS:
        q.enqueue(task, task, "a" * 64, {})
        job = q.claim([task])
        q.validating(task, job["lease_token"])
        digest = hashlib.sha256(task.encode()).hexdigest()
        q.register(task, job["lease_token"], lineage_hash=digest, content_hash=digest,
                   relative_path="episodes/" + task, seconds=200 * 3600, success=True,
                   verification={"passed": True})
    status = q.quota_status(dict.fromkeys(TASKS, 200.))
    assert status["eligible_hours"] == 1000
    assert status["remaining_assets"] == 99
    assert not status["complete"]
    assert q.quota_status(dict.fromkeys(TASKS, 200.), min_assets=1)["complete"]


def test_task_priority_filter_is_respected(tmp_path):
    q = ProductionQueue(tmp_path / "queue.sqlite")
    q.enqueue("old", "lift", "a" * 64, {})
    q.enqueue("new", "push", "a" * 64, {})
    assert q.claim(["push", "lift"])["job_id"] == "new"


def test_concurrent_duplicate_registration_and_independent_hash_checks(tmp_path):
    q = ProductionQueue(tmp_path / "queue.sqlite")
    for key in ("x", "y", "z"):
        add(q, key)
    jobs = [q.claim(), q.claim()]
    with ThreadPoolExecutor(2) as pool:
        states = list(pool.map(lambda job: register(ProductionQueue(q.path), job), jobs))
    assert sorted(states) == ["DUPLICATE", "EXPORTED"]
    job = q.claim()
    q.validating(job["job_id"], job["lease_token"])
    # Distinct lineage must still be rejected when the content is identical.
    assert q.register(job["job_id"], job["lease_token"], lineage_hash="e" * 64,
                      content_hash="c" * 64, relative_path="episodes/z", seconds=1.,
                      success=True, verification={"passed": True}) == "DUPLICATE"
    assert q.status()["episodes"] == 1


def test_live_worker_is_not_restarted_when_observation_lease_expires(tmp_path):
    q = ProductionQueue(tmp_path / 'queue.sqlite')
    add(q, 'job')
    job = q.claim()
    worker = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)
    try:
        q.attach_worker('job', job['lease_token'], worker.pid)
        assert q.recover(now=time.time() + 1000, abandoned=True) == 0
        assert q.claim() is None
    finally:
        worker.terminate()
        worker.wait(timeout=5)
    assert q.recover(now=time.time() + 1000, abandoned=True) == 1


def test_legacy_or_unrelated_inventory_is_not_modified(tmp_path):
    path = tmp_path / 'old_inventory.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE episodes(episode_id TEXT PRIMARY KEY)')
        db.execute("INSERT INTO episodes VALUES('keep')")
    with pytest.raises(ValueError, match='refusing changes'):
        ProductionQueue(path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT * FROM episodes').fetchall() == [('keep',)]
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [('episodes',)]
