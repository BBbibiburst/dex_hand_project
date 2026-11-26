"""Apply a pinned quality patch while all collectors are stopped; retain old proofs."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sync_parent(path):
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_json(path,value):
    temporary=path.with_name(path.name+'.quality-tmp')
    with temporary.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path);sync_parent(path)


def install(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.quality-tmp')
    with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path);sync_parent(path)


def validate_paths(manifest):
    for name in manifest['files']:
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or not name.startswith(('source/','apps/')):
            raise ValueError('Unsafe patch path')


def rollback(project,root,journal):
    """Only restore our exact old/new versions; unknown concurrent edits stop recovery."""
    backup=Path(journal['backup']);manifest=journal['manifest']
    for name,entry in manifest['files'].items():
        p=project/name
        if p.is_file() and sha(p) not in [entry['before'],entry['after']]:
            raise ValueError('Unexpected source edit during interrupted upgrade: '+name)
    old=journal['old_metadata'];new=journal.get('new_metadata',{})
    paths={'contract':root/'run_contract.json','intent':root.parent/('.'+root.name+'.bootstrap.json'),'complete':root/'bootstrap_complete.json'}
    for name,p in paths.items():
        if read(p) not in [old[name],new.get(name)]:raise ValueError('Unexpected metadata change during upgrade')
    for name,entry in manifest['files'].items():
        p=project/name
        if entry['before'] is None:
            if p.exists():p.unlink()
        else:
            b=backup/'source_before'/name
            if sha(b)!=entry['before']:raise ValueError('Upgrade backup is corrupt')
            install(p,b.read_bytes())
    for name,p in paths.items():write_json(p,old[name])


def upgrade(project,root,kit,*,fault=None):
    from source.data.production_index import live_dataset_processes,active_database
    manifest=read(kit/'patch_manifest.json');validate_paths(manifest)
    config=read(root/'config.json');control=(project/config['queue_database']).parent
    if not control.is_dir():raise ValueError('Original server control directory is missing')
    paths={'contract':root/'run_contract.json','intent':root.parent/('.'+root.name+'.bootstrap.json'),'complete':root/'bootstrap_complete.json'}
    pending=control/'quality_upgrade_pending.json'
    with ExitStack() as stack:
        for p in [control/'autorecovery.manager.lock',control/'supervisor.lock',paths['intent'].with_suffix('.lock'),root/'coordinator.lock']:
            f=stack.enter_context(p.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if live_dataset_processes(root):raise ValueError('Stop all dataset workers before applying patch')
        if pending.exists():
            rollback(project,root,read(pending));pending.unlink()
        old={name:read(path) for name,path in paths.items()};intent=old['intent']
        if (intent['configuration']!=config or intent['run_contract']!=old['contract']
                or old['complete']['intent_hash']!=canonical(intent) or intent['dataset_root']!=str(root)
                or intent['database']!=str((project/config['queue_database']).resolve())):
            raise ValueError('Frozen initialization proofs or original server paths differ')
        for field,key in [('template_sha256','template_library'),('registry_sha256','object_registry')]:
            if sha(project/config[key])!=old['contract'][field]:
                raise ValueError('Frozen catalogue or registry changed')
        for name,entry in manifest['files'].items():
            if sha(kit/'patch'/name)!=entry['after']:raise ValueError('Delivered patch checksum failed: '+name)
        # An already-applied run must still satisfy its complete current fingerprint.
        from source.data.production_execution_contract import execution_fingerprint
        current=execution_fingerprint(project)
        installed=all((project/n).is_file() and sha(project/n)==e['after'] for n,e in manifest['files'].items())
        if installed:
            if old['contract']['execution']!=current:raise ValueError('Installed patch and frozen runtime disagree')
            return {'upgraded':False,'reason':'Already upgraded; no data or queue changed'}
        if current!=old['contract']['execution']:raise ValueError('Original frozen source/environment does not match; refusing arbitrary upgrade')
        for name,entry in manifest['files'].items():
            p=project/name
            if entry['before'] is None:
                if p.exists():raise ValueError('Added module already exists outside this upgrade: '+name)
            elif not p.is_file() or sha(p)!=entry['before']:raise ValueError('Patch base differs: '+name)
        database=active_database(root,project/config['queue_database'])
        if not database.is_file():raise ValueError('Original queue missing; will not create a replacement')
        backup=control/'quality_upgrades'/str(time.time_ns());backup.mkdir(parents=True)
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
            if db.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise ValueError('Original database failed quick_check')
            with sqlite3.connect(backup/'queue_before.sqlite') as target:db.backup(target)
        for name,entry in manifest['files'].items():
            if entry['before'] is not None:
                p=backup/'source_before'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/name,p)
        journal={'schema':'collection-quality-upgrade-v2','backup':str(backup),'manifest':manifest,'old_metadata':old}
        write_json(backup/'upgrade.json',journal);write_json(pending,journal)
        try:
            for name in manifest['files']:install(project/name,(kit/'patch'/name).read_bytes())
            if fault=='after_source':raise RuntimeError('Injected upgrade interruption')
            updated=execution_fingerprint(project)
            # Environment and geometry are never part of this authorized code-only patch.
            for field in ('python','packages','shared_resource_sha256'):
                if updated[field]!=current[field]:raise ValueError('Physical resources or runtime changed during upgrade')
            expected={**current['source_sha256'],**{n:e['after'] for n,e in manifest['files'].items()}}
            if updated['source_sha256']!=expected:raise ValueError('Unexpected executable source changes')
            new_contract={**old['contract'],'execution':updated}
            new_intent={**intent,'run_contract':new_contract}
            new={'contract':new_contract,'intent':new_intent,'complete':{**old['complete'],'intent_hash':canonical(new_intent)}}
            journal['new_metadata']=new;write_json(pending,journal);write_json(backup/'upgrade.json',journal)
            # Legacy reference hashes only: no historical hours are imported or removed.
            import importlib
            import source.data.production_queue as queue_module
            if "source/data/production_queue.py" in manifest["files"]:
                queue_module=importlib.reload(queue_module)
            from source.data.production_artifacts import verify_episode
            legacy=root.parent/'dex_hand_dataset';refs=[]
            if legacy.is_dir() and legacy!=root:
                for directory in sorted((legacy/'episodes').iterdir()):
                    if not directory.is_dir() or directory.is_symlink():raise ValueError('Unexpected legacy entry')
                    verified=verify_episode(directory);meta=read(directory/'episode.json')
                    refs.append({'batch_id':str(legacy.resolve()),'job_id':meta['job_id'],
                                 'content_hash':verified['content_hash'],'lineage_hash':verified['lineage_hash']})
            queue=queue_module.ProductionQueue(database);before=queue.status(count_failures=config['count_failed_episodes'])
            queue.import_external_references(refs)
            after=queue.status(count_failures=config['count_failed_episodes'])
            if before!=after:raise ValueError('Reference import changed historical quota statistics')
            for name in ('contract','intent','complete'):write_json(paths[name],new[name])
            if fault=='after_metadata':raise RuntimeError('Injected metadata interruption')
            result={'upgraded':True,'backup':str(backup),'external_references_imported':len(refs),
                    'episodes_unchanged':True,'quota_statistics_unchanged':True,'configuration_unchanged':True,
                    'new_collection_started':False,'changed_files':list(manifest['files'])}
            write_json(backup/'complete.json',result);pending.unlink();return result
        except BaseException:
            rollback(project,root,journal)
            if pending.exists():pending.unlink()
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path('/home/ftcl/dex_hand_project'))
    p.add_argument('--dataset-root',type=Path,default=Path('/media/ftcl/外置存储/dex_hand_dataset_batch2'))
    p.add_argument('--kit',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
    project=a.project.resolve();sys.path.insert(0,str(project));os.chdir(project)
    print(json.dumps(upgrade(project,a.dataset_root.resolve(),a.kit.resolve()),ensure_ascii=False,indent=2))
