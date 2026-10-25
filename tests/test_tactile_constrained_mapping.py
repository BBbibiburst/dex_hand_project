import numpy as np

from apps.run_tactile_constrained_mapping import contract_violations, project_contract


def test_project_contract_enforces_all_feature_ranges():
    values = np.full((3, 127), -2.0, dtype=np.float32)
    values[:, 0::11] = 4.0
    values[:, 3:8] = [5.0, 4.0, 3.0, 2.0, 1.0]
    values[:, 121:127] = [-1.0, 2.0, 0.0, 4.0, -3.0, 1.0]
    projected = project_contract(values)
    result = contract_violations(projected)
    assert result["finite"]
    assert all(value == 0.0 for key, value in result.items() if key.endswith("violation_fraction"))
    assert np.allclose(projected[:, 121:127].sum(axis=1), 1.0)
