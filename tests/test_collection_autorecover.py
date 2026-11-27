import copy
import json
import signal
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from apps import collection_autorecover as app
from source.data.production_queue import ProductionQueue, TASKS


def config():
    return {'task_hours':dict.fromkeys(TASKS,200.),'minimum_independent_assets':100,
            'count_failed_episodes':True,'coverage_requires_success':False,'maximum_retries':2,
            'template_failure_limit':3,'template_duplicate_limit':3}


def template(name='lift_a',task='lift',asset='a'):
    return {'template_id':name,'task':task,'asset_hash':asset*64,'base_seed':0,
            'perturbation_fields':['dx_m','dy_m']}


def blocked(q,t):
    with q.transaction() as db:
        db.execute('INSERT INTO template_stats(template_id,scheduled,active,successful,failed,'
            'consecutive_failures,last_candidate_index) VALUES(?,10,0,4,6,3,9)',(t['template_id'],))


def fail(q):
    j=q.claim();q.fail(j['job_id'],j['lease_token'],'fixture failure',retryable=False);return j


def test_probe_cooldown_and_budget_persist_without_counter_reset(tmp_path):
    path=tmp_path/'queue.sqlite';q=ProductionQueue(path);t=template();blocked(q,t)
    p=copy.deepcopy(app.DEFAULTS)
    ids=[]
    for i in range(3):
        now=1000+i*900
        result=app.seed_probes(q,[t],config(),32000,p,now=now);ids+=result['added'];fail(q)
        q=ProductionQueue(path)
        cooling=app.seed_probes(q,[t],config(),32000,p,now=now+1)
        if i<2:assert cooling['state']=='COOLDOWN' and cooling['next_retry_at']==now+900
    assert len(set(ids))==3
    assert app.seed_probes(q,[t],config(),32000,p,now=10000)['isolated']==[t['template_id']]
    with q.connect() as db:
        r=db.execute('SELECT * FROM template_stats').fetchone()
        assert r['consecutive_failures']==6 and r['successful']==4 and r['last_candidate_index']==12
        assert db.execute('SELECT count(*) FROM episodes').fetchone()[0]==0


def test_actual_new_success_replenishes_budget(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');t=template();blocked(q,t);p=app.DEFAULTS
    app.seed_probes(q,[t],config(),32000,p,now=1000)
    j=q.claim();q.validating(j['job_id'],j['lease_token'])
    q.register(j['job_id'],j['lease_token'],lineage_hash='b'*64,content_hash='c'*64,
               relative_path='episodes/'+j['job_id'],seconds=1,success=True,verification={'passed':True})
    with q.transaction() as db:db.execute('UPDATE template_stats SET consecutive_failures=3')
    assert app.seed_probes(q,[t],config(),32000,p,now=2000)['added']
    with q.connect() as db:
        record=json.loads(db.execute('SELECT value FROM queue_meta WHERE key=?',(app.POLICY+':lift_a',)).fetchone()[0])
        assert record['used']==1 and record['successful_at_probe']==5


def test_pending_prevents_duplicate_and_sparse_batch_is_small(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');rows=[template('lift_'+str(i),asset=chr(97+i)) for i in range(5)]
    rows+=[template('nut','nut_assembly','f')]
    for t in rows:blocked(q,t)
    result=app.seed_probes(q,rows,config(),32000,{**app.DEFAULTS,'maximum_new_probes':4},now=1000)
    assert len(result['added'])==4
    with q.connect() as db:
        tasks=[r[0] for r in db.execute('SELECT task FROM jobs ORDER BY created_at,rowid')]
        assert tasks[0]!=tasks[1]
    assert app.seed_probes(q,rows,config(),32000,app.DEFAULTS,now=2000)['state']=='RESUME_EXISTING'


def test_failed_only_template_cannot_be_unblocked(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');t=template();blocked(q,t)
    with q.transaction() as db:db.execute('UPDATE template_stats SET successful=0')
    assert not app.seed_probes(q,[t],config(),32000,app.DEFAULTS,now=1000)['added']


def test_transaction_rolls_back_when_second_payload_invalid(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');rows=[template('a'),template('b')]
    for t in rows:blocked(q,t)
    rows[1]['base_seed']=object()
    with pytest.raises((TypeError,ValueError)):app.seed_probes(q,rows,config(),32000,{**app.DEFAULTS,'maximum_new_probes':2},now=1000)
    with q.connect() as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0]==0
        assert not db.execute('SELECT value FROM queue_meta WHERE key LIKE ?',(app.POLICY+'%',)).fetchall()


@pytest.mark.parametrize('termination,code,expected',[
    ('SUPERVISOR_RESTART_LIMIT',0,'RETRY_CRASH'),(None,-signal.SIGKILL,'RETRY_CRASH'),
    ('NO_RUNNABLE_CANDIDATE',0,'RECOVER_CANDIDATES'),('INTEGRITY_AUDIT_FAILED',0,'STOP_INTEGRITY_AUDIT_FAILED'),
    ('RUNTIME_GUARD_FAILED',0,'STOP_RUNTIME_GUARD_FAILED'),('STOP_OR_WALL_BUDGET',0,'STOP_STOP_OR_WALL_BUDGET'),
    ('QUOTA_COMPLETE_FILE_REJECTED',0,'STOP_QUOTA_COMPLETE_FILE_REJECTED'),
    ('QUOTA_COMPLETE_FILE_VERIFIED',0,'COMPLETE'),(None,1,'STOP_CHILD_FAILURE')])
def test_exit_reason_policy(termination,code,expected):
    assert app.decision(termination,code)==expected


def test_classification_never_treats_contract_failure_as_memory_shortage():
    report={'ready':False,'errors':['Frozen file differs/missing: physics.py'],
            'infrastructure':{'memory':{'ready':False}}}
    assert app.classify_check(report)=='STOP_CONTRACT_OR_DATABASE'
    infra={'dependencies':{'numpy':True},'encoder':{'ready':True},'volume':{'ready':True},'memory':{'ready':False}}
    assert app.classify_check({'ready':False,'errors':['Original production infrastructure preflight failed'],'infrastructure':infra})=='WAIT_MEMORY'


def test_restart_budget_is_persistent_and_expires(tmp_path):
    d=tmp_path/'state';d.mkdir();p={**app.DEFAULTS,'maximum_restarts_per_window':2}
    w=app.Watchdog(tmp_path,tmp_path/'data',d,p)
    assert app.budget(w.state,p,1000)==30;w.save('TEST')
    w=app.Watchdog(tmp_path,tmp_path/'data',d,p)
    assert app.budget(w.state,p,1001)==60;w.save('TEST')
    w=app.Watchdog(tmp_path,tmp_path/'data',d,p)
    assert app.budget(w.state,p,1002) is None
    assert app.budget(w.state,p,5000)==30


def test_unknown_policy_keys_or_unbounded_limits_rejected(tmp_path):
    p=tmp_path/'options.json';p.write_text(json.dumps({'probes_without_success':999}))
    with pytest.raises(ValueError):app.options(p)
    p.write_text(json.dumps({'typo':1}))
    with pytest.raises(ValueError):app.options(p)


@pytest.fixture
def watchdog(tmp_path,monkeypatch):
    project=tmp_path/'project';project.mkdir();root=tmp_path/'data';root.mkdir();d=tmp_path/'manager';d.mkdir()
    p={**app.DEFAULTS,'poll_seconds':1,'stall_seconds':1,'shutdown_grace_seconds':2}
    w=app.Watchdog(project,root,d,p)
    monkeypatch.setattr(app,'diagnose',lambda *a:('READY',{'ready':True}))
    def prepare(project,root,destination,policy):
        destination.mkdir(parents=True);(destination/'queue_before.sqlite').write_bytes(b'fixture backup')
        return {'database':str(tmp_path/'q.sqlite'),'sampling_seed':32000,'recovery':{'state':'PROBES_QUEUED','added':['fixture'],'isolated':[],'next_retry_at':None}}
    monkeypatch.setattr(app,'prepare',prepare)
    return w


def test_watchdog_restarts_crash_and_then_completes(watchdog,monkeypatch):
    codes=iter([(None,-signal.SIGKILL),('QUOTA_COMPLETE_FILE_VERIFIED',0)]);waits=[]
    monkeypatch.setattr(watchdog,'run_child',lambda *a:next(codes))
    monkeypatch.setattr(watchdog,'wait',lambda n,status:waits.append((n,status)))
    assert watchdog.run()==0
    assert watchdog.state['sessions']==2 and waits==[(30,'RECOVERY_BACKOFF')]
    assert watchdog.state['status']=='COMPLETE'


def test_watchdog_resumes_after_memory_recovers(watchdog,monkeypatch):
    checks=iter([('WAIT_MEMORY',{'ready':False}),('READY',{'ready':True})]);waits=[]
    monkeypatch.setattr(app,'diagnose',lambda *a:next(checks))
    monkeypatch.setattr(watchdog,'wait',lambda n,status:waits.append(status))
    monkeypatch.setattr(watchdog,'run_child',lambda *a:('QUOTA_COMPLETE_FILE_VERIFIED',0))
    assert watchdog.run()==0 and waits==['WAIT_MEMORY']


def test_watchdog_user_stop_never_restarts(watchdog,monkeypatch):
    def child(*a):watchdog.stopped.set();return 'USER_STOP',-signal.SIGTERM
    monkeypatch.setattr(watchdog,'run_child',child)
    assert watchdog.run()==0 and watchdog.state['sessions']==1
    assert watchdog.state['status']=='USER_STOP'


def test_restart_storm_remains_stopped_after_manager_restart(watchdog,monkeypatch):
    watchdog.state['restart_timestamps']=[time.time()]*watchdog.policy['maximum_restarts_per_window'];watchdog.save('PREVIOUS_BUDGET_STOP')
    w=app.Watchdog(watchdog.project,watchdog.root,watchdog.directory,watchdog.policy)
    monkeypatch.setattr(w,'run_child',lambda *a:pytest.fail('Budget must prevent a new process'))
    assert w.run()==78 and w.state['status']=='STOP_RESTART_BUDGET'


def test_preparation_resource_race_rechecks_without_bypassing(watchdog,monkeypatch):
    original=app.prepare;calls=[]
    def prepare(*a):
        calls.append(1)
        if len(calls)==1:raise app.PreflightChanged({'ready':False,'errors':['Insufficient disk space above frozen reserve']})
        return original(*a)
    monkeypatch.setattr(app,'prepare',prepare)
    monkeypatch.setattr(watchdog,'wait',lambda *a:None)
    monkeypatch.setattr(watchdog,'run_child',lambda *a:('QUOTA_COMPLETE_FILE_VERIFIED',0))
    assert watchdog.run()==0 and len(calls)==2


def test_actual_child_is_terminated_on_stall_and_can_be_resumed(watchdog):
    apps=watchdog.project/'apps';apps.mkdir()
    (apps/'five_task_production.py').write_text('import time\ntime.sleep(30)\n')
    q=ProductionQueue(watchdog.directory/'q.sqlite')
    session=watchdog.directory/'sessions/actual'
    (session/'preparation').mkdir(parents=True)
    term,code=watchdog.run_child({'database':str(q.path),'sampling_seed':32000},session)
    assert term=='WATCHDOG_STALLED' and watchdog.child.poll() is not None
    assert code==-signal.SIGTERM
    assert app.decision(None,-signal.SIGKILL)=='RETRY_CRASH'


def test_actual_child_completion_reads_summary_from_existing_preparation(watchdog):
    apps=watchdog.project/'apps';apps.mkdir()
    (apps/'five_task_production.py').write_text("import pathlib,sys,json\np=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);p.mkdir(parents=True);(p/'supervisor.json').write_text(json.dumps({'termination':'QUOTA_COMPLETE_FILE_VERIFIED'}))\n")
    q=ProductionQueue(watchdog.directory/'q.sqlite');session=watchdog.directory/'sessions/actual'
    (session/'preparation').mkdir(parents=True)
    term,code=watchdog.run_child({'database':str(q.path),'sampling_seed':32000},session)
    assert term=='QUOTA_COMPLETE_FILE_VERIFIED' and code==0


def test_only_explicit_resource_stop_can_be_retried():
    memory={'stop_signals':[],'wall_budget_seconds':None,'integrity_failures':[],
            'runtime_guard_failures':[{'error':'RuntimeError: System memory reserve violated: {}'}]}
    assert app.resource_stop(memory)=='RESOURCE_MEMORY_STOP'
    memory['runtime_guard_failures']=[{'error':'RuntimeError: Executable source changed'}]
    assert app.resource_stop(memory) is None
    memory['runtime_guard_failures']=[];memory['results']=[{'state':'MEMORY_STOP'}]
    memory['stop_signals']=[{'signal':15}]
    assert app.resource_stop(memory) is None
    memory['stop_signals']=[];memory['integrity_failures']=[{'state':'FAILED'}]
    assert app.resource_stop(memory) is None


def test_resource_recovers_before_recheck_can_restart(watchdog,monkeypatch):
    results=iter([('RESOURCE_MEMORY_STOP',0),('QUOTA_COMPLETE_FILE_VERIFIED',0)]);waits=[]
    monkeypatch.setattr(watchdog,'run_child',lambda *a:next(results))
    monkeypatch.setattr(watchdog,'wait',lambda n,status:waits.append(status))
    assert watchdog.run()==0 and waits==['RESOURCE_RECOVERED_BACKOFF']


def test_wait_limit_is_not_reset_by_process_restart(watchdog,monkeypatch):
    watchdog.state['resource_wait_since']=time.time()-4000;watchdog.save('WAIT_MEMORY')
    w=app.Watchdog(watchdog.project,watchdog.root,watchdog.directory,watchdog.policy)
    monkeypatch.setattr(app,'diagnose',lambda *a:('WAIT_MEMORY',{'ready':False}))
    assert w.run()==78 and w.state['status']=='STOP_RESOURCE_WAIT_LIMIT'


def test_healthy_template_runs_while_another_template_cools(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');a=template('a');b=template('b')
    blocked(q,a);blocked(q,b)
    app.seed_probes(q,[a],config(),32000,app.DEFAULTS,now=1000);fail(q)
    with q.transaction() as db:db.execute("UPDATE template_stats SET consecutive_failures=0 WHERE template_id='b'")
    result=app.seed_probes(q,[a,b],config(),32000,app.DEFAULTS,now=1001)
    assert result['state']=='NORMAL_SCHEDULER' and not result['added']


def test_sigterm_stops_actual_child_and_does_not_restart(watchdog):
    import os,threading
    apps=watchdog.project/'apps';apps.mkdir();(apps/'five_task_production.py').write_text('import time\ntime.sleep(30)\n')
    q=ProductionQueue(watchdog.directory.parent/'q.sqlite')
    # Fixture preparation points to the same queue path.
    timer=threading.Timer(.3,lambda:os.kill(os.getpid(),signal.SIGTERM));timer.start()
    try:
        assert watchdog.run()==0
        assert watchdog.child is not None and watchdog.child.poll() is not None
        assert watchdog.state['sessions']==1 and watchdog.state['status']=='USER_STOP'
    finally:timer.cancel();timer.join()


def test_long_final_file_verification_is_not_a_stalled_collector(watchdog,monkeypatch):
    apps=watchdog.project/'apps';apps.mkdir()
    (apps/'five_task_production.py').write_text("import pathlib,sys,json,time\ntime.sleep(2.2)\np=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);p.mkdir(parents=True);(p/'supervisor.json').write_text(json.dumps({'termination':'QUOTA_COMPLETE_FILE_VERIFIED'}))\n")
    q=ProductionQueue(watchdog.directory/'q.sqlite');session=watchdog.directory/'sessions/finalizing'
    (session/'preparation').mkdir(parents=True)
    monkeypatch.setattr(app,'finalizing',lambda *a:True)
    term,code=watchdog.run_child({'database':str(q.path),'sampling_seed':32000},session)
    assert term=='QUOTA_COMPLETE_FILE_VERIFIED' and code==0
    assert 'FINAL_FILE_ACCEPTANCE' in (watchdog.directory/'events.jsonl').read_text()


def test_pending_of_completed_task_does_not_block_needed_task_probe(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');a=template();blocked(q,a)
    q.enqueue('old_push','push','b'*64,{'template_id':'push_old','candidate_index':1})
    original=q.quota_status
    q.quota_status=lambda *a,**kw:{**original(*a,**kw),'needed_tasks':['lift']}
    result=app.seed_probes(q,[a],config(),32000,app.DEFAULTS,now=1000)
    assert len(result['added'])==1
    with q.connect() as db:assert db.execute("SELECT state FROM jobs WHERE job_id='old_push'").fetchone()[0]=='PENDING'


@pytest.mark.parametrize('cpu_limit',[0,2])
def test_child_cpu_limit_is_inherited_and_parent_affinity_is_restored(watchdog,cpu_limit):
    import os
    if not hasattr(os,'sched_getaffinity'):pytest.skip('Linux affinity required')
    watchdog.state['cpu_limit']=cpu_limit
    before=os.sched_getaffinity(0);apps=watchdog.project/'apps';apps.mkdir();mark=watchdog.directory/'child_affinity.json'
    (apps/'five_task_production.py').write_text("import pathlib,sys,json,os\npathlib.Path("+repr(str(mark))+").write_text(json.dumps(sorted(os.sched_getaffinity(0))))\np=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);p.mkdir(parents=True);(p/'supervisor.json').write_text(json.dumps({'termination':'QUOTA_COMPLETE_FILE_VERIFIED'}))\n")
    q=ProductionQueue(watchdog.directory/'q.sqlite');session=watchdog.directory/'sessions/affinity'
    (session/'preparation').mkdir(parents=True)
    term,code=watchdog.run_child({'database':str(q.path),'sampling_seed':32000},session)
    expected=before if cpu_limit==0 else set(sorted(before)[:cpu_limit])
    assert code==0 and set(json.loads(mark.read_text()))==expected
    assert os.sched_getaffinity(0)==before


def test_legacy_defaults_migrate_without_resetting_recovery_or_sessions(tmp_path):
    p=tmp_path/'project';p.mkdir();d=tmp_path/'state';d.mkdir();root=tmp_path/'data'
    old={'policy':app.POLICY,'project':str(p),'dataset_root':str(root),
         'options':{**app.DEFAULTS,'initial_cpu_limit':8,'maximum_new_probes':4},
         'cpu_limit':8,'sessions':17,'restart_timestamps':[123,456],
         'managed_backups':['preserved'],'activity':[1234,107777]}
    (d/'state.json').write_text(json.dumps(old))
    w=app.Watchdog(p,root,d,app.options())
    assert w.state['cpu_limit']==0 and w.state['options']['maximum_new_probes']==0
    for key in ['sessions','restart_timestamps','managed_backups','activity']:assert w.state[key]==old[key]
    assert json.loads((d/'state.before-resource-adaptive-20261127.json').read_text())==old
    again=app.Watchdog(p,root,d,app.options())
    assert again.state==w.state


def test_migration_preserves_actual_memory_downscale(tmp_path):
    p=tmp_path/'project';p.mkdir();d=tmp_path/'state';d.mkdir();root=tmp_path/'data'
    old={'policy':app.POLICY,'project':str(p),'dataset_root':str(root),
         'options':{**app.DEFAULTS,'initial_cpu_limit':8,'maximum_new_probes':4},'cpu_limit':4}
    (d/'state.json').write_text(json.dumps(old))
    assert app.Watchdog(p,root,d,app.options()).state['cpu_limit']==4


def test_probe_capacity_follows_server_worker_plan(tmp_path,monkeypatch):
    import source.data.production_parallel as parallel
    monkeypatch.setattr(parallel,'worker_plan',lambda c:{'initial_admitted_workers':24})
    q=ProductionQueue(tmp_path/'q.sqlite');templates=[template('lift_'+str(i),asset=hex(i+1)[2:]) for i in range(30)]
    for t in templates:blocked(q,t)
    result=app.seed_probes(q,templates,config(),32000,app.options(),now=1000)
    assert len(result['added'])==24


def test_custom_policy_change_still_requires_review(tmp_path):
    p=tmp_path/'project';p.mkdir();d=tmp_path/'state';d.mkdir();root=tmp_path/'data'
    old={'policy':app.POLICY,'project':str(p),'dataset_root':str(root),
         'options':{**app.DEFAULTS,'initial_cpu_limit':16},'cpu_limit':16}
    (d/'state.json').write_text(json.dumps(old))
    with pytest.raises(ValueError,match='explicit reviewed'):app.Watchdog(p,root,d,app.options())


def test_live_recovery_seeds_idle_lift_while_nut_is_running(tmp_path,monkeypatch):
    import source.data.production_parallel as parallel
    monkeypatch.setattr(parallel,'worker_plan',lambda c:{'initial_admitted_workers':1,'pool_workers':24})
    q=ProductionQueue(tmp_path/'q.sqlite');lifts=[template('lift_'+str(i),asset=hex(i+1)[2:]) for i in range(30)]
    for t in lifts:blocked(q,t)
    q.enqueue('nut_active','nut_assembly','f'*64,{'template_id':'nut','candidate_index':0})
    nut=q.claim(['nut_assembly'])
    result=app.seed_probes(q,lifts,config(),32000,app.options(),now=1000,while_running=True)
    assert len(result['added'])==24 and 'lift' in result['starved_tasks']
    with q.connect() as db:
        row=db.execute("SELECT * FROM jobs WHERE job_id='nut_active'").fetchone()
        assert row['state']=='RUNNING' and row['lease_token']==nut['lease_token']
        assert db.execute("SELECT count(*) FROM jobs WHERE task='lift' AND state='PENDING'").fetchone()[0]==24
    # Pending recovery jobs prevent an unbounded second batch for the same task.
    assert not app.seed_probes(q,lifts,config(),32000,app.options(),now=1001,while_running=True)['added']


def test_live_recovery_retains_per_template_cooldown_and_budget_with_busy_nut(tmp_path):
    path=tmp_path/'q.sqlite';q=ProductionQueue(path);t=template();blocked(q,t)
    q.enqueue('nut_active','nut_assembly','f'*64,{'template_id':'nut','candidate_index':0})
    nut=q.claim(['nut_assembly']);policy={**app.DEFAULTS,'maximum_new_probes':4}
    ids=[]
    for attempt in range(3):
        now=1000+attempt*900
        result=app.seed_probes(q,[t],config(),32000,policy,now=now,while_running=True)
        assert len(result['added'])==1;ids+=result['added']
        job=q.claim(['lift']);q.fail(job['job_id'],job['lease_token'],'Strict failed',retryable=False)
        assert not app.seed_probes(q,[t],config(),32000,policy,now=now+1,while_running=True)['added']
        q=ProductionQueue(path)
    result=app.seed_probes(q,[t],config(),32000,policy,now=100000,while_running=True)
    assert not result['added'] and result['isolated']==[t['template_id']]
    assert len(set(ids))==3
    with q.connect() as db:
        row=db.execute("SELECT * FROM jobs WHERE job_id='nut_active'").fetchone()
        assert row['state']=='RUNNING' and row['lease_token']==nut['lease_token']
        assert db.execute('SELECT consecutive_failures FROM template_stats WHERE template_id=?',(t['template_id'],)).fetchone()[0]==6


def test_live_recovery_pending_other_task_does_not_block_idle_lift(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');t=template();blocked(q,t)
    q.enqueue('nut_pending','nut_assembly','f'*64,{'template_id':'nut','candidate_index':0})
    result=app.seed_probes(q,[t],config(),32000,app.DEFAULTS,now=1000,while_running=True)
    assert len(result['added'])==1


def setup_live_feeder(watchdog,monkeypatch):
    import source.data.production_templates as module
    q=ProductionQueue(watchdog.directory/'q.sqlite');t=template();blocked(q,t)
    q.enqueue('nut_active','nut_assembly','f'*64,{'template_id':'nut','candidate_index':0})
    nut=q.claim(['nut_assembly'])
    c={**config(),'queue_database':str(q.path),'template_library':'library.json','object_registry':'registry.json'}
    for name in ('library.json','registry.json'):(watchdog.project/name).write_text('{}')
    (watchdog.root/'config.json').write_text(json.dumps(c))
    (watchdog.root/'run_contract.json').write_text(json.dumps({'template_sha256':app.base.digest(watchdog.project/'library.json'),'registry_sha256':app.base.digest(watchdog.project/'registry.json')}))
    monkeypatch.setattr(module,'load_templates',lambda *a:[t])
    return q,t,nut


def test_running_child_observes_live_lift_refill_without_nut_restart(watchdog,monkeypatch):
    q,t,nut=setup_live_feeder(watchdog,monkeypatch)
    apps=watchdog.project/'apps';apps.mkdir();mark=watchdog.directory/'live_refill_observed.json'
    script="""import sys,pathlib,json,time,sqlite3,os
q=sqlite3.connect(DB)
end=time.monotonic()+10
while time.monotonic()<end:
    nut=q.execute("SELECT state FROM jobs WHERE job_id='nut_active'").fetchone()[0]
    count=q.execute("SELECT count(*) FROM jobs WHERE task='lift' AND state='PENDING'").fetchone()[0]
    if count:
        pathlib.Path(MARK).write_text(json.dumps({'nut_state':nut,'lift_pending':count,'pid':os.getpid()}))
        p=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);p.mkdir(parents=True,exist_ok=True)
        (p/'supervisor.json').write_text(json.dumps({'termination':'QUOTA_COMPLETE_FILE_VERIFIED'}))
        sys.exit(0)
    time.sleep(.02)
sys.exit(4)
""".replace('DB',repr(str(q.path))).replace('MARK',repr(str(mark)))
    (apps/'five_task_production.py').write_text(script)
    session=watchdog.directory/'sessions/live';(session/'preparation').mkdir(parents=True)
    term,code=watchdog.run_child({'database':str(q.path),'sampling_seed':32000},session)
    result=json.loads(mark.read_text())
    assert code==0 and term=='QUOTA_COMPLETE_FILE_VERIFIED'
    assert result['nut_state']=='RUNNING' and result['lift_pending']==1
    assert result['pid']==watchdog.child.pid
    assert watchdog.state['live_recovery']['added']
    assert 'LIVE_RECOVERY_PROBES_QUEUED' in (watchdog.directory/'events.jsonl').read_text()


def test_live_feeder_rejects_changed_frozen_catalogue(watchdog,monkeypatch):
    q,t,nut=setup_live_feeder(watchdog,monkeypatch)
    feeder=app.LiveProbeFeeder(watchdog.project,watchdog.root,32000,watchdog.policy)
    (watchdog.project/'library.json').write_text('{"tampered":true}')
    with pytest.raises(ValueError,match='Frozen catalogue'):feeder.refill(q.path,now=1000)
    with q.connect() as db:assert db.execute("SELECT count(*) FROM jobs WHERE task='lift'").fetchone()[0]==0


def test_runtime_guard_failure_preserves_cause_before_summary_replacement(watchdog):
    apps=watchdog.project/'apps';apps.mkdir()
    (apps/'five_task_production.py').write_text(
        "import pathlib,sys,json\n"
        "p=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);p.mkdir(parents=True)\n"
        "(p/'supervisor.json').write_text(json.dumps({'termination':'RUNTIME_GUARD_FAILED'}))\n")
    detail={'termination':'RUNTIME_GUARD_FAILED','runtime_guard_failures':[
        {'error':'RuntimeError: Executable source changed','record_path':'runtime/failures/old.json'}],
        'stop_signals':[],'wall_budget_seconds':None,'integrity_failures':[]}
    (watchdog.root/'summary.json').write_text(json.dumps(detail))
    q=ProductionQueue(watchdog.directory/'q.sqlite');session=watchdog.directory/'sessions/actual'
    (session/'preparation').mkdir(parents=True)
    term,code=watchdog.run_child({'database':str(q.path),'sampling_seed':32000},session)
    assert term=='RUNTIME_GUARD_FAILED' and code==0
    assert app.decision(term,code)=='STOP_RUNTIME_GUARD_FAILED'
    (watchdog.root/'summary.json').write_text('{}')
    assert json.loads((session/'child_stop_summary.json').read_text())==detail
    state=json.loads(watchdog.path.read_text())
    assert state['last_stop_details']['runtime_guard_failures']==detail['runtime_guard_failures']
