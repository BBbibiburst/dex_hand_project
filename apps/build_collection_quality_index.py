"""Build a separate deduplicated, profile-aware view; never mutate source data."""
import argparse
import json
from pathlib import Path
import sqlite3
from source.data.collection_catalogue import episode_record, training_view


def build(sources, output):
    if output.exists():raise FileExistsError('Use a fresh output; indices are retained for review')
    records=[]
    for root,database in sources:
        if output.resolve().is_relative_to(root.resolve()):raise ValueError('Index output must be outside raw dataset')
        if database:
            with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
                directories=[root/r[0] for r in db.execute('SELECT relative_path FROM episodes ORDER BY job_id')]
        else:directories=sorted((root/'episodes').iterdir())
        for p in directories:
            if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():raise ValueError('Dataset path escapes source root')
            metadata=json.loads((p/'episode.json').read_text())
            replay=json.loads((p/'independent_replay.json').read_text()) if (p/'independent_replay.json').is_file() else {}
            payload=json.loads((p/'job_payload.json').read_text()) if (p/'job_payload.json').is_file() else {}
            record=episode_record(str(root.resolve()),metadata,replay,payload)
            record['episode_path']=str(p.resolve());records.append(record)
    result=training_view(records);output.mkdir(parents=True)
    by_id={r['record_id']:r for r in result['records']}
    for i,(key,ids) in enumerate(sorted(result['partitions'].items())):
        (output/f'partition_{i:03d}.jsonl').write_text(''.join(json.dumps(by_id[k],ensure_ascii=False)+'\n' for k in ids))
    (output/'catalogue.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--database',type=Path,help='Optional primary-source original queue; remaining sources scanned read-only')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=build([(root,a.database if i==0 else None) for i,root in enumerate(a.source)],a.output)
    print(json.dumps({k:r[k] for k in ['raw_records','deduplicated_records','split_rule']},ensure_ascii=False))
