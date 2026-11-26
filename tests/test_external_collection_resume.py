import json
from pathlib import Path

import pytest

from apps.resume_external_collection import seed_recovery, POLICY
from source.data.production_queue import ProductionQueue, TASKS
from source.data.production_scheduler import enqueue_next


def cfg():
    return {'task_hours': dict.fromkeys(TASKS, 200.), 'minimum_independent_assets': 100,
            'count_failed_episodes': True, 'coverage_requires_success': False,
            'maximum_retries': 2, 'template_failure_limit': 3, 'template_duplicate_limit': 3}


def template(name='lift_one', task='lift'):
    return {'template_id': name, 'task': task, 'asset_hash': 'a'*64,
            'base_seed': 0, 'perturbation_fields': ['dx_m', 'dy_m']}


def blocked(queue, row, *, successful=4, failures=3, duplicates=0):
    with queue.transaction() as db:
        db.execute('INSERT INTO template_stats(template_id,scheduled,active,successful,failed,duplicates,'
            'consecutive_failures,consecutive_duplicates,last_candidate_index) VALUES(?,10,0,?,6,?,?,?,9)',
            (row['template_id'], successful, duplicates, failures, duplicates))


def finish_failed(queue):
    job=queue.claim()
    queue.fail(job['job_id'], job['lease_token'], 'fixture failed', retryable=False)
    return job


def test_adds_next_candidate_without_resetting_old_history(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite'); t=template();blocked(q,t)
    assert enqueue_next(q,[t],cfg(),32000)['state']=='WAITING_OR_BLOCKED'
    out=seed_recovery(q,[t],cfg(),32000)
    assert len(out['added'])==1
    with q.connect() as db:
        s=dict(db.execute('SELECT * FROM template_stats').fetchone())
        job=dict(db.execute('SELECT * FROM jobs').fetchone())
        assert s['consecutive_failures']==3 and s['failed']==6 and s['successful']==4
        assert s['last_candidate_index']==10 and s['active']==1
        assert json.loads(job['payload'])['candidate_index']==10
        assert db.execute('SELECT count(*) FROM episodes').fetchone()[0]==0
        assert db.execute("SELECT count(*) FROM events WHERE state='BOUNDED_RECOVERY_PROBE'").fetchone()[0]==1


def test_existing_pending_prevents_duplicate_probe(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');t=template();blocked(q,t)
    seed_recovery(q,[t],cfg(),32000)
    assert seed_recovery(q,[t],cfg(),32000)['reason']=='EXISTING_WORK_RESUME_FIRST'


def test_budget_survives_restart_and_does_not_repeat_old_job(tmp_path):
    path=tmp_path/'q.sqlite';q=ProductionQueue(path);t=template();blocked(q,t)
    ids=[]
    for i in range(3):
        ids+=seed_recovery(q,[t],cfg(),32000)['added'];finish_failed(q);q=ProductionQueue(path)
    assert len(set(ids))==3
    assert not seed_recovery(q,[t],cfg(),32000)['added']
    with q.connect() as db:
        assert db.execute('SELECT value FROM queue_meta WHERE key=?',(POLICY+':'+t['template_id'],)).fetchone()[0]=='3'
        assert db.execute('SELECT max(attempts) FROM jobs').fetchone()[0]==1


def test_never_successful_templates_remain_isolated(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');t=template();blocked(q,t,successful=0)
    assert not seed_recovery(q,[t],cfg(),32000)['added']


def test_healthy_template_stays_with_original_scheduler(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');t=template();blocked(q,t,failures=0)
    assert not seed_recovery(q,[t],cfg(),32000)['added']
    assert enqueue_next(q,[t],cfg(),32000)['state']=='ENQUEUED'


def test_capture_contract_is_pinned_and_task_complete_not_requeued(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');t=template();blocked(q,t)
    c={**cfg(),'capture_profile':'speed-v1','rgb_width':640,'rgb_height':360,'rgb_fps':5}
    seed_recovery(q,[t],c,32000)
    with q.connect() as db:
        payload=json.loads(db.execute('SELECT payload FROM jobs').fetchone()[0])
        assert payload['rgb_capture']['fps']==5 and payload['rgb_capture']['control_fps']==20
    q2=ProductionQueue(tmp_path/'other.sqlite');blocked(q2,t)
    original=q2.quota_status
    q2.quota_status=lambda *a,**kw:{**original(*a,**kw),'needed_tasks':['nut_assembly']}
    assert not seed_recovery(q2,[t],cfg(),32000)['added']


def test_duplicate_block_is_bounded_without_clearing_counter(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');t=template();blocked(q,t,failures=0,duplicates=3)
    assert len(seed_recovery(q,[t],cfg(),32000)['added'])==1
    with q.connect() as db:
        assert db.execute('SELECT consecutive_duplicates FROM template_stats').fetchone()[0]==3


def test_atomic_rollback_if_later_candidate_is_invalid(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');t=template();u=template('second');blocked(q,t);blocked(q,u)
    u['base_seed']=object()
    with pytest.raises((ValueError,TypeError)):
        seed_recovery(q,[t,u],cfg(),32000)
    with q.connect() as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0]==0
        assert db.execute('SELECT count(*) FROM queue_meta WHERE key LIKE ?',(POLICY+'%',)).fetchone()[0]==0


def test_check_missing_database_does_not_create_it(tmp_path, monkeypatch):
    import apps.resume_external_collection as app
    from source.data.five_task_dataset import canonical_hash
    root=tmp_path/'data';root.mkdir();project=tmp_path/'project';project.mkdir()
    import sys
    config={'dataset_root':str(root),'queue_database':'missing/queue.sqlite', 'disk_reserve_fraction':.2,
            'template_library':'templates.json','object_registry':'registry.json'}
    for name in ['templates.json','registry.json']:(project/name).write_text('{}')
    contract={'sampling_seed':32000,'template_sha256':app.digest(project/'templates.json'),
        'registry_sha256':app.digest(project/'registry.json'),
        'execution':{'python':sys.version,'packages':{},'source_sha256':{},'shared_resource_sha256':{}}}
    intent={'dataset_root':str(root),'database':str(project/'missing/queue.sqlite'),
            'configuration':config,'run_contract':contract}
    for name,value in [('config.json',config),('run_contract.json',contract),
                       ('bootstrap_complete.json',{'intent_hash':canonical_hash(intent)})]:
        (root/name).write_text(json.dumps(value))
    (root.parent/('.'+root.name+'.bootstrap.json')).write_text(json.dumps(intent))
    report=app.check(project,root)
    assert not report['ready']
    assert any('queue is missing' in x for x in report['errors'])
    assert not (project/'missing').exists()


def test_prepare_backs_up_before_new_jobs_and_keeps_episode_files(tmp_path, monkeypatch):
    import apps.resume_external_collection as app
    import source.data.production_templates as templates_module
    import source.data.production_index as index_module
    import hashlib
    project=tmp_path/'project';project.mkdir();root=tmp_path/'disk'/'data';root.mkdir(parents=True)
    control=project/'control';control.mkdir();path=control/'queue.sqlite'
    q=ProductionQueue(path);t=template();blocked(q,t)
    c={**cfg(),'queue_database':'control/queue.sqlite','template_library':'templates.json','object_registry':'registry.json'}
    (root/'config.json').write_text(json.dumps(c));(root/'episode_sentinel.npz').write_bytes(b'unchanged fixture')
    before=hashlib.sha256((root/'episode_sentinel.npz').read_bytes()).hexdigest()
    monkeypatch.setattr(app,'check',lambda *a:{'ready':True,'database':str(path),'sampling_seed':32000})
    monkeypatch.setattr(templates_module,'load_templates',lambda *a:[t])
    monkeypatch.setattr(index_module,'live_dataset_processes',lambda *a:[])
    result=app.prepare(project,root,project/'backup')
    import sqlite3
    with sqlite3.connect(result['backup']) as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0]==0
        assert db.execute('SELECT consecutive_failures FROM template_stats').fetchone()[0]==3
    with q.connect() as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0]==1
    assert hashlib.sha256((root/'episode_sentinel.npz').read_bytes()).hexdigest()==before
    assert (project/'backup/preparation.json').is_file()


def test_preparation_refuses_failed_preflight_before_writes(tmp_path, monkeypatch):
    import apps.resume_external_collection as app
    root=tmp_path/'data';root.mkdir();project=tmp_path/'project';project.mkdir()
    monkeypatch.setattr(app,'check',lambda *a:{'ready':False,'errors':['runtime mismatch']})
    with pytest.raises(ValueError,match='runtime mismatch'):
        app.prepare(project,root,tmp_path/'backup')
    assert not (tmp_path/'backup').exists()
    assert not list(root.iterdir())
