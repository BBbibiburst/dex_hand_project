import pytest

from apps.probe_production_compatibility import select_cases
from source.data.production_compatibility import compatibility_matrix
from source.data.production_queue import ProductionQueue


def test_zero_perturbation_selection_does_not_change_historical_library():
    row = {'task': 'push', 'object_id': 'fixture:a', 'template_id': 'one',
           'perturbation_fields': ['dx_m', 'yaw_deg'], 'dependencies': [{'sha256': 'frozen'}]}
    selected = select_cases([row], ['push=fixture:a'])
    assert selected[0]['perturbation_fields'] == []
    assert row['perturbation_fields'] == ['dx_m', 'yaw_deg']
    assert selected[0]['dependencies'] == row['dependencies']
    with pytest.raises(ValueError, match='Repeated'):
        select_cases([row], ['push=fixture:a', 'push=fixture:a'])
    with pytest.raises(ValueError, match='No frozen'):
        select_cases([row], ['push=fixture:missing'])
    preferred = {**row, 'template_id': 'z_current', 'validation_status': 'CURRENT_SINGLE_CONDITION_SUCCESS'}
    assert select_cases([row, preferred], ['push=fixture:a'])[0]['template_id'] == 'z_current'


def test_failure_or_more_windows_does_not_create_additional_compatible_assets(tmp_path):
    queue = ProductionQueue(tmp_path / 'queue.sqlite')
    for index, success in enumerate((True, False)):
        queue.enqueue(str(index), 'push', 'a' * 64, {'object_id': 'artificial:a', 'conditions': {'dx_m': 0.}})
        job = queue.claim()
        queue.validating(job['job_id'], job['lease_token'])
        queue.register(job['job_id'], job['lease_token'], lineage_hash=str(index) * 64,
                       content_hash=str(index + 2) * 64, relative_path=f'artificial/{index}',
                       seconds=1_000, success=success, verification={'passed': True, 'artificial_test': True})
    assert compatibility_matrix(queue.path)['current_success_assets'] == 0
    result = compatibility_matrix(queue.path, {'passed': True, 'episodes': [{'job_id': '0', 'passed': True}]})
    assert result['current_success_assets'] == 1
    assert not result['all_selected_cases_passed']
    assert result['rows'][1]['compatibility'] == 'CURRENT_TASK_FAILURE'
    assert all(row['zero_perturbation'] for row in result['rows'])
    reason = {'success': False, 'failure_category': 'PLACEMENT_CRITERION', 'failure_reason': 'reported reason'}
    audited = compatibility_matrix(queue.path, {'passed': True, 'episodes': [
        {'job_id': '1', 'passed': True, 'task_outcome': reason}]})
    assert audited['rows'][1]['task_outcome'] == reason
    assert audited['rows'][0]['task_outcome'] is None
