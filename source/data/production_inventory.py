"""Rebuild inventory evidence from independently verified Episode files."""

import json
from pathlib import Path
import sqlite3

from source.data.production_artifacts import verify_episode
from source.data.five_task_dataset import canonical_hash


def verify_inventory_row(root, row):
    path = root / row['relative_path']
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Inventory path escapes dataset root')
    actual = verify_episode(path)
    for key in ('task', 'asset_hash', 'lineage_hash', 'content_hash'):
        if actual[key] != row[key]:
            raise ValueError(f'Database and file disagree: {key}')
    if actual['success'] != bool(row['success']) or abs(actual['duration_seconds'] - row['seconds']) > 1e-7:
        raise ValueError('Database and file disagree on outcome or duration')
    meta = json.loads((path / 'episode.json').read_text())
    if meta['job_id'] != row['job_id'] or row['planned_payload'] is None:
        raise ValueError('Episode job identity differs from queue')
    if meta['job_payload_hash'] != canonical_hash(json.loads(row['planned_payload'])):
        raise ValueError('Episode payload differs from immutable queue plan')
    return {'job_id': row['job_id'], 'relative_path': row['relative_path'],
            'production_admission': meta['admission'], **actual}


def audit_inventory(root: Path, database: Path):
    db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute('SELECT e.*,j.payload AS planned_payload FROM episodes e '
                          'LEFT JOIN jobs j ON j.job_id=e.job_id').fetchall()
    finally:
        db.close()
    failures = []
    checked = []
    for row in rows:
        try:
            checked.append(verify_inventory_row(root, row))
        except Exception as error:
            failures.append({"job_id": row["job_id"], "error": f"{type(error).__name__}: {error}"})
    registered_paths = {row["relative_path"] for row in rows}
    orphans = [str(path.relative_to(root)) for path in (root / "episodes").glob('*')
               if path.is_dir() and str(path.relative_to(root)) not in registered_paths]
    staged = [str(path.relative_to(root)) for path in (root / ".transactions").glob('*')]
    return {"passed": not failures and not orphans and not staged,
            "registered_episodes": len(rows), "verified_episodes": len(checked),
            "errors": failures, "orphaned_paths": orphans, "staged_paths": staged,
            "preserved_duplicate_paths": [str(path.relative_to(root)) for path in (root / "duplicates").glob('*')],
            "unique_seconds_from_files": sum(row["duration_seconds"] for row in checked),
            "successful_seconds_from_files": sum(row["duration_seconds"] for row in checked if row["success"]),
            "independent_assets_from_files": len({row["asset_hash"] for row in checked}),
            "all_formally_admitted": bool(checked) and all(
                row["production_admission"] == "production_verified" for row in checked),
            "episodes": checked}
