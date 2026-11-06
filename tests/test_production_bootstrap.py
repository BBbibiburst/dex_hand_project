import json
from pathlib import Path
import subprocess
import sys

import pytest

from apps.five_task_production import load_config
from source.data.production_bootstrap import reserve_bootstrap, install_metadata, intent_path
from source.data.production_queue import ProductionQueue


TEMPLATES = [
    {'template_id': 'fixture_lift', 'task': 'lift', 'asset_hash': 'a' * 64,
     'base_seed': 0, 'perturbation_fields': ['dx_m']},
    {'template_id': 'fixture_push', 'task': 'push', 'asset_hash': 'b' * 64,
     'base_seed': 0, 'perturbation_fields': ['dx_m']}]


@pytest.mark.parametrize('stage', ['intent', 'metadata', 'empty_database', 'partial_schema', 'one_seed'])
def test_actual_process_exit_during_initialization_resumes_without_extra_jobs(tmp_path, monkeypatch, stage):
    import source.data.production_coordinator as coordinator

    root = tmp_path / 'dataset'
    config = load_config(Path('configs/five_task_production.json'))
    config_file = tmp_path / 'test_config.json'
    config_file.write_text(json.dumps(config))
    script = '''
import json,os,sqlite3,sys
from pathlib import Path
from source.data.production_bootstrap import reserve_bootstrap,install_metadata
from source.data.production_queue import ProductionQueue
from source.data.production_scheduler import enqueue_next
root,config_file,stage,templates=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],json.loads(sys.argv[4])
config=json.loads(config_file.read_text())
lock,record=reserve_bootstrap(root,root/'inventory.sqlite',config,None,42,2,resume=False)
if stage=='intent': os._exit(42)
root.mkdir()
install_metadata(root,record)
if stage=='metadata': os._exit(42)
if stage=='empty_database':
    (root/'inventory.sqlite').touch()
    os._exit(42)
if stage=='partial_schema':
    with sqlite3.connect(root/'inventory.sqlite') as db:
        db.execute("CREATE TABLE jobs(job_id TEXT PRIMARY KEY,task TEXT NOT NULL,asset_hash TEXT NOT NULL,payload TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'PENDING',attempts INTEGER NOT NULL DEFAULT 0,max_attempts INTEGER NOT NULL,lease_token TEXT,lease_until REAL,error TEXT,created_at REAL NOT NULL)")
    os._exit(42)
queue=ProductionQueue(root/'inventory.sqlite',initializing=True)
enqueue_next(queue,templates,config,42)
os._exit(42)
'''
    process = subprocess.run([sys.executable, '-c', script, str(root), str(config_file), stage,
                              json.dumps(TEMPLATES)], capture_output=True, text=True, timeout=20)
    assert process.returncode == 42, process.stderr
    monkeypatch.setattr(coordinator, 'load_templates', lambda *_: TEMPLATES)
    monkeypatch.setattr(coordinator.subprocess, 'Popen', lambda *a, **kw: pytest.fail('No simulation allowed'))
    result = coordinator._coordinate(config, root, 1, 42, resume=True, wall_budget=1e-20)
    assert result['attempted_jobs'] == 0
    assert (root / 'bootstrap_complete.json').is_file()
    with ProductionQueue(root / 'inventory.sqlite').connect() as db:
        assert db.execute('SELECT count(*),sum(attempts) FROM jobs').fetchone()[:] == (2, 0)


def test_interrupted_intent_rejects_changed_seed_and_unowned_root(tmp_path):
    root = tmp_path / 'dataset'
    lock, record = reserve_bootstrap(root, root / 'inventory.sqlite', {}, None, 42, 1, resume=False)
    lock.close()
    assert intent_path(root).exists()
    with pytest.raises(ValueError, match='differs'):
        reserve_bootstrap(root, root / 'inventory.sqlite', {}, None, 43, 1, resume=True)
    root.mkdir()
    install_metadata(root, record)
    other = tmp_path / 'unowned'
    other.mkdir()
    with pytest.raises(ValueError, match='Missing startup intent'):
        reserve_bootstrap(other, other / 'inventory.sqlite', {}, None, 42, 1, resume=True)
