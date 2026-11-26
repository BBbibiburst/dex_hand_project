"""Namespaced, profile-separated training views; raw episodes are never edited."""
import json
from source.data.five_task_dataset import canonical_hash


def episode_record(batch_id, metadata, replay=None, payload=None):
    sensor=metadata.get('sensor_provenance')
    if not isinstance(sensor,dict) or not sensor.get('signal_processor'):
        raise ValueError('Tactile processing identity is required; mixed UNKNOWN input is not admissible')
    replay=replay or {};payload=payload or {}
    new=metadata.get('quality_contract',{}).get('schema')=='collection-quality-v2'
    quality='trajectory_verified' if new and replay.get('trajectory_reproduced') is True and replay.get('passed') is True and replay.get('outcome_reproduced') is True else 'legacy_review'
    if replay.get('max_qpos_component_error',0)>1e-6:quality='replay_drift_review'
    if metadata.get('admission')!='production_verified':quality='development_only'
    return {'record_id':batch_id+'/'+metadata['job_id'],'batch_id':batch_id,'job_id':metadata['job_id'],
            'task':metadata['task'],'asset_hash':metadata['asset_hash'],'success':bool(metadata['success']),
            'content_hash':metadata['content_hash'],'lineage_hash':metadata['lineage_hash'],
            'profile_id':canonical_hash(sensor),'quality':quality,
            'source_reference':payload.get('reference_manifest_sha256'),
            'split_group':metadata['asset_hash']}


def training_view(records):
    """Deduplicate content/lineage, preserving aliases and their original IDs."""
    rows=sorted((dict(r) for r in records),key=lambda r:({'trajectory_verified':0,'legacy_review':1,'replay_drift_review':2,'development_only':3}.get(r['quality'],4),r['record_id']))
    ids=set();contents={};lineages={};partitions={};parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        if row['record_id'] in ids:raise ValueError('Repeated batch/job identity')
        ids.add(row['record_id'])
        for index,value in [(contents,row['content_hash']),(lineages,row['lineage_hash'])]:
            key=(row['profile_id'],value)
            if key in index:
                a,b=root(i),root(index[key]);parent[max(a,b)]=min(a,b)
            else:index[key]=i
    for i,row in enumerate(rows):
        canonical=root(i)
        row['duplicate_of']=rows[canonical]['record_id'] if canonical!=i else None
        row['split_group']=rows[canonical]['asset_hash']
        if canonical==i:
            role=('development_review' if row['quality']=='development_only' else 'replay_drift_review' if row['quality']=='replay_drift_review' else ('expert' if row['success'] else 'failure') if row['quality']=='trajectory_verified' else ('legacy_review' if row['success'] else 'legacy_failure_review'))
            key=row['profile_id']+'/'+role
            partitions.setdefault(key,[]).append(row['record_id'])
    return {'records':rows,'partitions':partitions,'raw_records':len(rows),
            'deduplicated_records':sum(not r['duplicate_of'] for r in rows),
            'split_rule':'Keep each asset split_group entirely on one side; no Frame/window random split',
            'note':'A digest or asset grouping is not certification of independent physical Roots'}


def require_single_profile(records):
    if len({r['profile_id'] for r in records})>1:
        raise ValueError('Mixed tactile profiles require separate partitions; no silent normalization')
