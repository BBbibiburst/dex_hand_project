import copy

import pytest

from source.data.production_acceptance import acceptance_from_audit
from source.data.production_queue import TASKS
from source.data.production_report import footprint, save_report


def config():
    return {'task_hours': dict.fromkeys(TASKS, 200.), 'minimum_independent_assets': 100,
            'count_failed_episodes': False, 'coverage_requires_success': True}


def artificial_audit():
    # Artificial summary fixture: these hours are never generated or stored as real data.
    rows = []
    for i in range(100):
        rows.append({'job_id': str(i), 'task': TASKS[i % 5], 'asset_hash': f'{i:064x}',
                     'content_hash': f'{i + 1000:064x}', 'lineage_hash': f'{i + 2000:064x}',
                     'success': True, 'duration_seconds': 10 * 3600, 'passed': True,
                     'production_admission': 'development_verified_not_formal_production'})
    return {'passed': True, 'episodes': rows}


def test_data_hours_do_not_certify_software_or_admit_development_files():
    result = acceptance_from_audit(config(), artificial_audit())
    assert result['data_target_met']
    assert not result['formal_dataset_accepted']
    assert not result['software_delivery_certified']


def test_duplicate_semantic_content_is_not_credited_and_blocks_acceptance():
    audit = artificial_audit()
    for row in audit['episodes']:
        row['production_admission'] = 'production_verified'
    assert acceptance_from_audit(config(), audit)['formal_dataset_accepted']
    audit['episodes'].append(copy.deepcopy(audit['episodes'][0]))
    result = acceptance_from_audit(config(), audit)
    assert result['credited_hours'] == 1000
    assert len(result['duplicate_jobs_not_credited']) == 1
    assert not result['formal_dataset_accepted']


def test_file_report_honours_separate_outcome_deficits():
    settings = config()
    settings['count_failed_episodes'] = True
    settings['task_outcome_hours'] = {t: {'success': 180., 'failure': 20.} for t in TASKS}
    result = acceptance_from_audit(settings, artificial_audit())
    assert result['credited_hours'] == 900
    assert not result['data_target_met']
    assert all(r['remaining_outcome_hours']['failure'] == 20 for r in result['tasks'].values())


def test_nonfinite_physical_time_is_rejected():
    audit = artificial_audit()
    audit['episodes'][0]['duration_seconds'] = float('nan')
    with pytest.raises(ValueError, match='physical duration'):
        acceptance_from_audit(config(), audit)


def test_storage_counts_hardlinked_allocation_once_and_skips_external_symlink(tmp_path):
    import os

    (tmp_path / 'modalities.npz').write_bytes(b'x' * 100)
    os.link(tmp_path / 'modalities.npz', tmp_path / 'another.npz')
    (tmp_path / 'outside').symlink_to('/etc/hosts')
    result = footprint(tmp_path)
    assert result['logical_bytes'] == 200
    assert result['categories_logical_bytes']['arrays'] == 200
    assert result['allocated_unique_inode_bytes'] == (tmp_path / 'modalities.npz').stat().st_blocks * 512


def test_export_cannot_write_into_dataset_being_measured(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    with pytest.raises(ValueError, match='outside the measured dataset'):
        save_report({'configuration': {'dataset_root': str(dataset)}}, dataset / 'report')
    assert list(dataset.iterdir()) == []
