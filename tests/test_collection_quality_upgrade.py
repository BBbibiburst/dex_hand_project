import hashlib,json
from pathlib import Path
import pytest
from apps import apply_collection_quality_patch as module
from source.data.production_queue import ProductionQueue


def setup_case(tmp_path,monkeypatch):
    project=tmp_path/'project';(project/'source').mkdir(parents=True);root=tmp_path/'disk/dataset';root.mkdir(parents=True)
    (project/'source/fix.py').write_bytes(b'old source')
    for name in ['templates.json','registry.json']:(project/name).write_text('{}')
    control=project/'control';control.mkdir();queue=ProductionQueue(control/'q.sqlite')
    queue.enqueue('historical','lift','a'*64,{'candidate_index':1});job=queue.claim();queue.validating(job['job_id'],job['lease_token'])
    queue.register(job['job_id'],job['lease_token'],lineage_hash='b'*64,content_hash='c'*64,relative_path='episodes/historical',seconds=1.,success=True,verification={'passed':True})
    def fingerprint(p):return {'python':'fixture','packages':{},'shared_resource_sha256':{},'source_sha256':{str(x.relative_to(p)):module.sha(x) for x in (p/'source').glob('*.py')}}
    import source.data.production_execution_contract as execution
    import source.data.production_index as index
    monkeypatch.setattr(execution,'execution_fingerprint',fingerprint);monkeypatch.setattr(index,'live_dataset_processes',lambda r:[])
    c={'dataset_root':str(root),'queue_database':'control/q.sqlite','count_failed_episodes':True,'template_library':'templates.json','object_registry':'registry.json'}
    contract={'execution':fingerprint(project),'template_sha256':module.sha(project/'templates.json'),'registry_sha256':module.sha(project/'registry.json')}
    intent={'dataset_root':str(root),'database':str(control/'q.sqlite'),'configuration':c,'run_contract':contract,'sampling_seed':32000,'initial_jobs':1}
    (root/'config.json').write_text(json.dumps(c));(root/'run_contract.json').write_text(json.dumps(contract));(root.parent/'.dataset.bootstrap.json').write_text(json.dumps(intent));(root/'bootstrap_complete.json').write_text(json.dumps({'intent_hash':module.canonical(intent)}))
    (root/'episode_sentinel').write_bytes(b'original raw bytes')
    kit=tmp_path/'kit';(kit/'patch/source').mkdir(parents=True)
    (kit/'patch/source/fix.py').write_bytes(b'new source');(kit/'patch/source/new.py').write_bytes(b'new module')
    manifest={'files':{'source/fix.py':{'before':module.sha(project/'source/fix.py'),'after':module.sha(kit/'patch/source/fix.py')},'source/new.py':{'before':None,'after':module.sha(kit/'patch/source/new.py')}}}
    (kit/'patch_manifest.json').write_text(json.dumps(manifest))
    return project,root,kit,queue,contract


def test_upgrade_preserves_raw_quota_and_is_idempotent(tmp_path,monkeypatch):
    p,r,k,q,old=setup_case(tmp_path,monkeypatch);before=q.status(count_failures=True)
    result=module.upgrade(p,r,k)
    assert result['upgraded'] and q.status(count_failures=True)==before
    assert (r/'episode_sentinel').read_bytes()==b'original raw bytes'
    assert (r/'config.json').read_text()==json.dumps(json.loads((r/'config.json').read_text()))
    assert module.read(r/'run_contract.json')['execution']['source_sha256']['source/fix.py']==module.sha(k/'patch/source/fix.py')
    assert not module.upgrade(p,r,k)['upgraded']


@pytest.mark.parametrize('fault',['after_source','after_metadata'])
def test_failed_upgrade_restores_code_and_bootstrap_proofs(tmp_path,monkeypatch,fault):
    p,r,k,q,old=setup_case(tmp_path,monkeypatch);before=q.status(count_failures=True)
    with pytest.raises(RuntimeError,match='Injected'):module.upgrade(p,r,k,fault=fault)
    assert (p/'source/fix.py').read_bytes()==b'old source' and not (p/'source/new.py').exists()
    assert module.read(r/'run_contract.json')==old
    intent=module.read(r.parent/'.dataset.bootstrap.json');assert module.read(r/'bootstrap_complete.json')['intent_hash']==module.canonical(intent)
    assert q.status(count_failures=True)==before and (r/'episode_sentinel').read_bytes()==b'original raw bytes'
    assert not (p/'control/quality_upgrade_pending.json').exists()


def test_upgrade_rejects_unknown_source_or_catalogue_change(tmp_path,monkeypatch):
    p,r,k,q,old=setup_case(tmp_path,monkeypatch)
    (p/'source/fix.py').write_bytes(b'user unreviewed change')
    with pytest.raises(ValueError,match='does not match'):module.upgrade(p,r,k)
    assert (p/'source/fix.py').read_bytes()==b'user unreviewed change'
