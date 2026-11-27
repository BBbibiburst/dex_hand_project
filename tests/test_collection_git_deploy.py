import json
from pathlib import Path
import shutil
import pytest
from apps import deploy_collection_update as module
from test_collection_quality_upgrade import setup_case


def prepared(tmp_path, monkeypatch):
    p,r,k,q,old=setup_case(tmp_path,monkeypatch)
    import source.data.production_preflight as preflight
    monkeypatch.setattr(preflight,'mount_info',lambda root:{'mountpoint':str(root.parent)})
    class Space:
        f_blocks=20_000_000_000_000
        f_frsize=1
    monkeypatch.setattr(module.os,'statvfs',lambda root:Space())
    manifest=module.read(k/'patch_manifest.json')
    for name in manifest['files']:
        shutil.copyfile(k/'patch'/name,p/name)
    new=tmp_path/'newdisk'/r.name
    new.parent.mkdir();shutil.move(str(r),new)
    shutil.move(str(r.parent/'.dataset.bootstrap.json'),new.parent/'.dataset.bootstrap.json')
    state=p/'control/autorecovery_quality_v3/state.json'
    state.parent.mkdir();state.write_text(json.dumps({'dataset_root':str(r),'probes_used':3,'restart_times':[1,2]}))
    return p,r,new,manifest,q,state


def test_git_deploy_relocates_without_reset(tmp_path,monkeypatch):
    p,old,r,m,q,state=prepared(tmp_path,monkeypatch)
    before=q.status(count_failures=True)
    module.deploy(p,r,m)
    assert q.status(count_failures=True)==before
    assert module.read(r/'config.json')['dataset_root']==str(r)
    assert module.read(state)=={'dataset_root':str(r),'probes_used':3,'restart_times':[1,2]}
    assert (r/'episode_sentinel').read_bytes()==b'original raw bytes'
    module.deploy(p,r,m)
    assert q.status(count_failures=True)==before


def test_failure_restores_metadata(tmp_path,monkeypatch):
    p,old,r,m,q,state=prepared(tmp_path,monkeypatch)
    before=(r/'config.json').read_text()
    with pytest.raises(RuntimeError):module.deploy(p,r,m,fault=True)
    assert module.read(r/'config.json')==json.loads(before)
    assert module.read(state)['dataset_root']==str(old)


def test_unknown_code_and_missing_queue_refused(tmp_path,monkeypatch):
    p,old,r,m,q,state=prepared(tmp_path,monkeypatch)
    (p/'source/fix.py').write_text('unapproved')
    with pytest.raises(ValueError,match='Unapproved'):module.deploy(p,r,m)
    shutil.copyfile(tmp_path/'kit/patch/source/fix.py',p/'source/fix.py')
    (p/'control/q.sqlite').unlink()
    with pytest.raises(ValueError,match='Original queue missing'):module.deploy(p,r,m)
    assert not (p/'control/q.sqlite').exists()


def test_pointer_relocation(tmp_path):
    old=Path('/old/dataset');root=tmp_path/'dataset';db=tmp_path/'q.sqlite'
    selected=tmp_path/'dataset_index_recovery/run/q.sqlite';selected.parent.mkdir(parents=True);selected.touch()
    value={'original_database':str(db),'dataset_root':str(old),'active_database':'/old/dataset_index_recovery/run/q.sqlite'}
    assert module.relocate_pointer(value,old,root,db)['active_database']==str(selected)
    with pytest.raises(ValueError):module.relocate_pointer({**value,'active_database':'/etc/passwd'},old,root,db)


def test_approved_scheduler_predecessor_is_pinned_and_environment_remains_frozen():
    old={'python':'original','packages':{'mujoco':'3.12.0'},'shared_resource_sha256':{'mesh':'fixed'},'source_sha256':{'scheduler':'quality-v1','other':'same'}}
    current={**old,'source_sha256':{'scheduler':'parallel-v2','other':'same'}}
    manifest={'files':{'scheduler':{'before':'original','after':'parallel-v2','approved_predecessors':['quality-v1']}}}
    module.approve_execution(old,current,manifest)
    with pytest.raises(ValueError):module.approve_execution(old,{**current,'packages':{'mujoco':'wrong'}},manifest)
    with pytest.raises(ValueError):module.approve_execution({**old,'source_sha256':{'scheduler':'unknown','other':'same'}},current,manifest)


def test_repository_deployment_manifest_matches_current_source_bytes():
    p=Path(__file__).resolve().parents[1];manifest=module.deployment_manifest(p)
    for name,item in manifest['files'].items():assert module.sha(p/name)==item['after']
