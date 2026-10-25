import numpy as np

from apps.run_tactile_empirical_transport import EmpiricalTaskTransport


def test_empirical_transport_only_returns_fitted_real_rows():
    mapper = EmpiricalTaskTransport()
    mapper.sim_median = np.zeros(127, dtype=np.float32)
    mapper.sim_scale = np.ones(127, dtype=np.float32)
    mapper.models = []
    mapper._score = lambda values: values[:, 0]  # type: ignore[method-assign]
    for stage in ("CLOSE", "HOLD", "LIFT"):
        mapper.score_quantiles[stage] = np.linspace(0, 1, 101)
        mapper.real_prototypes[stage] = np.stack(
            (np.zeros(127, dtype=np.float32), np.ones(127, dtype=np.float32))
        )
    values = np.zeros((3, 127), dtype=np.float32)
    values[:, 0] = [0.0, 1.0, 0.0]
    output = mapper.transform(values, np.asarray(["CLOSE", "HOLD", "LIFT"]))
    assert set(np.unique(output)) <= {0.0, 1.0}
