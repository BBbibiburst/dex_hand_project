import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import mujoco
import pytest
from source.grasping.probe_state import SimulationCheckpoint
from source.sensors.tactile.signal_processing import TactileSignalProcessor
from source.data.production_queue import ProductionQueue
from source.data.collection_catalogue import episode_record,training_view,require_single_profile


def environment():
    model=mujoco.MjModel.from_xml_string('<mujoco><option timestep=".001"/><worldbody><site name="ee"/><body pos="0 0 1"><joint name="j"/><geom size=".1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model)
    for _ in range(20):mujoco.mj_step(model,data)
    arm=SimpleNamespace(site_id=0,_prev_target_q=np.zeros(1),_filtered_velocity=np.ones(1),_prev_ee_target=np.zeros(7))
    processor=TactileSignalProcessor({'lowpass_alpha':.45,'noise_std':.01,'seed':123});processor._filtered=np.ones(2)
    return SimpleNamespace(model=model,data=data,elapsed_steps=20,controller=SimpleNamespace(arm_controller=arm,
        hand_controller=SimpleNamespace(ctrl_low=np.zeros(6),ctrl_high=np.ones(6))),
        tactile_sensor=SimpleNamespace(signal_processor=processor),pending={'a':np.ones(2)},prepared=None,
        _sensor_cache=('snapshot',np.ones(2)),task=SimpleNamespace(_body_pos=lambda *a:np.zeros(3)))


def test_checkpoint_restores_complete_solver_and_sensor_state():
    env=environment();checkpoint=SimulationCheckpoint(env);expected=checkpoint.state.copy();rng=env.tactile_sensor.signal_processor._rng.bit_generator.state
    env.data.qacc_warmstart[:]=456;env.data.qvel[:]=123;env.elapsed_steps=999
    env.pending['a'][:]=9;env._sensor_cache=None;env.tactile_sensor.signal_processor._filtered[:]=99
    env.tactile_sensor.signal_processor._rng.normal(size=20);checkpoint.restore()
    actual=np.empty_like(expected);mujoco.mj_getState(env.model,env.data,actual,checkpoint.spec)
    np.testing.assert_array_equal(actual,expected)
    assert env.elapsed_steps==20 and env._sensor_cache[0]=='snapshot'
    np.testing.assert_array_equal(env.pending['a'],np.ones(2));np.testing.assert_array_equal(env.tactile_sensor.signal_processor._filtered,np.ones(2))
    assert env.tactile_sensor.signal_processor._rng.bit_generator.state==rng


def test_preload_probe_restores_state_even_when_trial_raises(monkeypatch):
    from source.grasping import task_transfer as module
    env=environment();snapshot=SimulationCheckpoint(env)
    monkeypatch.setattr(module,'actuator_targets_from_fractions',lambda e,x:x)
    def fail(*a,**k):env.data.qacc_warmstart[:]=999;raise RuntimeError('fixture probe error')
    monkeypatch.setattr(module,'_run_pose_segment',fail)
    with pytest.raises(RuntimeError,match='fixture probe error'):module._select_transport_hand(env,np.ones(6)*.5)
    actual=np.empty_like(snapshot.state);mujoco.mj_getState(env.model,env.data,actual,snapshot.spec)
    np.testing.assert_array_equal(actual,snapshot.state)


def test_each_probe_starts_from_identical_solver_history(monkeypatch):
    from source.grasping import task_transfer as module
    env=environment();before=env.data.qacc_warmstart.copy();starts=[]
    class Recorder:
        def __init__(self,*a):pass
        def arrays(self):return {'robot_object_normal_force':np.array([1.])}
    monkeypatch.setattr(module,'_Recorder',Recorder);monkeypatch.setattr(module,'actuator_targets_from_fractions',lambda e,x:x)
    def trial(*a,**k):starts.append(env.data.qacc_warmstart.copy());env.data.qacc_warmstart[:]+=10
    monkeypatch.setattr(module,'_run_pose_segment',trial)
    module._select_transport_hand(env,np.ones(6)*.5)
    assert len(starts)==32
    for x in starts[::2]:np.testing.assert_array_equal(x,before)
    np.testing.assert_array_equal(env.data.qacc_warmstart,before)


def record(batch='a',name='same',content='a',lineage='b',profile='x'):
    return {'record_id':batch+'/'+name,'batch_id':batch,'job_id':name,'task':'lift','asset_hash':'asset',
            'success':True,'quality':'trajectory_verified','content_hash':content*64,'lineage_hash':lineage*64,'profile_id':profile,'split_group':'asset'}


def test_cross_batch_namespaced_view_keeps_different_content_and_deduplicates_aliases():
    a=record();b=record('b',content='c',lineage='d');c=record('c')
    result=training_view([a,b,c]);assert result['raw_records']==3 and result['deduplicated_records']==2
    assert next(r for r in result['records'] if r['batch_id']=='c')['duplicate_of']=='a/same'


def test_transitive_content_and_lineage_duplicates_have_one_canonical_record():
    result=training_view([record(),record('b',content='a',lineage='c'),record('c',content='d',lineage='c')])
    assert result['deduplicated_records']==1


def test_profile_mixing_is_rejected_and_views_partition_by_profile():
    a=record(profile='x');b=record('b',content='d',lineage='e',profile='y')
    with pytest.raises(ValueError,match='Mixed tactile'):require_single_profile([a,b])
    assert len(training_view([a,b])['partitions'])==2


def test_legacy_or_development_record_cannot_become_new_verified_expert():
    meta={'job_id':'j','task':'lift','asset_hash':'a'*64,'success':True,'content_hash':'b'*64,'lineage_hash':'c'*64,
          'sensor_provenance':{'signal_processor':{'normalize':True}},'admission':'production_verified'}
    assert episode_record('b',meta)['quality']=='legacy_review'
    meta['quality_contract']={'schema':'collection-quality-v2'}
    replay={'passed':True,'outcome_reproduced':True,'trajectory_reproduced':True,'max_qpos_component_error':0.}
    assert episode_record('b',meta,replay)['quality']=='trajectory_verified'
    meta['admission']='development_verified_not_formal_production'
    r=episode_record('b',meta,replay);assert r['quality']=='development_only'
    assert list(training_view([r])['partitions'])[0].endswith('/development_review')


def test_external_references_prevent_new_credit_without_importing_historical_hours(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite');q.import_external_references([{'batch_id':'legacy','job_id':'same','content_hash':'c'*64,'lineage_hash':'d'*64}])
    q.enqueue('new','lift','a'*64,{'template_id':'lift','candidate_index':1});j=q.claim();q.validating(j['job_id'],j['lease_token'])
    result=q.register(j['job_id'],j['lease_token'],lineage_hash='e'*64,content_hash='c'*64,relative_path='episodes/new',seconds=1.,success=True,verification={'passed':True})
    assert result=='DUPLICATE' and q.status()['episodes']==0
    with q.connect() as db:assert 'legacy/same' in db.execute("SELECT details FROM events WHERE state='DUPLICATE'").fetchone()[0]


def test_external_reference_namespaces_do_not_overwrite_same_job_id(tmp_path):
    q=ProductionQueue(tmp_path/'q.sqlite')
    q.import_external_references([{'batch_id':b,'job_id':'same','content_hash':c*64,'lineage_hash':l*64} for b,c,l in [('a','a','b'),('b','c','d')]])
    with q.connect() as db:assert db.execute('SELECT count(*) FROM cross_batch_refs').fetchone()[0]==2
    with pytest.raises(ValueError,match='refusing overwrite'):
        q.import_external_references([{'batch_id':'a','job_id':'same','content_hash':'e'*64,'lineage_hash':'f'*64}])


def test_replay_gate_cannot_be_loosened():
    from apps.replay_p32_task import replay
    for value in [float('nan'),float('inf'),1.,0.]:
        with pytest.raises(ValueError,match='precision gate'):replay(Path('not_read'),trajectory_tolerance=value)


def test_drifted_failure_is_reviewed_and_not_admitted_as_valid_failure():
    r=record();r['quality']='replay_drift_review';r['success']=False
    assert list(training_view([r])['partitions'])[0].endswith('/replay_drift_review')


def test_consistent_copy_is_preferred_over_legacy_drift_alias():
    a=record('a');a['quality']='replay_drift_review';b=record('b');b['quality']='legacy_review'
    view=training_view([a,b]);assert view['records'][0]['record_id']=='b/same'


def test_recorded_geometry_binding_ignores_recompiled_collision_cache():
    from apps.replay_p32_task import bind_recorded_objects
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="object_body"><freejoint name="object_joint"/><geom name="piece_0" size=".1"/><geom name="piece_1" size=".1" pos=".2 0 0"/><geom name="visual" size=".1" contype="0" conaffinity="0"/></body></worldbody></mujoco>')
    class Task:
        def __init__(self):self._objects=(SimpleNamespace(name='object',body_name='object_body',geom_names=('missing_cache_piece',)),)
        @property
        def objects(self):return self._objects
        def bind(self,m):self.bound=self.objects[0].geom_names
    task=Task();bind_recorded_objects(task,model)
    assert set(task.bound)=={'piece_0','piece_1'} and task.objects[0].name=='object'


def test_sparse_asset_is_preferred_over_fast_already_abundant_asset(tmp_path):
    from source.data.production_scheduler import enqueue_next
    from source.data.production_templates import candidate
    from source.data.five_task_dataset import canonical_hash
    from source.data.production_queue import TASKS
    q=ProductionQueue(tmp_path/'q.sqlite');rows=[]
    for name,asset,count,wall in [('abundant','a',5,1.),('rare','b',1,100.)]:
        t={'template_id':name,'task':'lift','asset_hash':asset*64,'base_seed':0,'perturbation_fields':['dx_m','dy_m']};rows.append(t)
        for i in range(count):
            payload=candidate(t,42,i);jid=name+'_'+canonical_hash(payload)[:20];q.enqueue(jid,'lift',asset*64,payload)
            j=q.claim();q.validating(jid,j['lease_token']);q.register(jid,j['lease_token'],lineage_hash=canonical_hash(jid),content_hash=canonical_hash(jid),relative_path='episodes/'+jid,seconds=1.,success=True,verification={'passed':True});q.record_attempt_metrics(jid,1,wall,'EXPORTED')
    c={'task_hours':dict.fromkeys(TASKS,200.),'minimum_independent_assets':100,'count_failed_episodes':False,'maximum_retries':1,'efficiency_minimum_attempts':1,'efficiency_exploration_interval':1000}
    result=enqueue_next(q,rows,c,42)
    assert result['job_id'].startswith('rare_') and result['selection_evidence']['registered_asset_episodes']==1
