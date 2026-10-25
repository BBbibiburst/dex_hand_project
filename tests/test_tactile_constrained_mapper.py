import numpy as np
import torch

from apps.train_tactile_constrained_mapper import ConstrainedMapper
from apps.run_tactile_constrained_mapping import contract_violations


def test_learned_mapper_parameterization_is_always_valid():
    mapper = ConstrainedMapper(np.zeros(127), np.ones(127), hidden=16)
    result = mapper(torch.randn(5, 127), torch.eye(3)[[0, 1, 2, 0, 1]]).detach().numpy()
    violations = contract_violations(result)
    assert violations["finite"]
    assert all(
        value == 0.0 for key, value in violations.items() if key.endswith("violation_fraction")
    )
