import numpy as np
import pytest

from source.data.production_conditions import environment_conditions, sample_conditions, shifted_pose, validate_conditions, sampling_contract


def test_candidates_are_reproducible_and_different():
    a = sample_conditions("lift", 42, 0)
    assert a == sample_conditions("lift", 42, 0)
    assert a != sample_conditions("lift", 42, 1)


@pytest.mark.parametrize("values", [{"dx_m": float("nan")}, {"dx_m": .1}, {"mass": 2}, {"yaw_deg": 4}])
def test_physics_or_excessive_perturbations_rejected(values):
    with pytest.raises(ValueError):
        validate_conditions("lift", values)


def test_pose_rotation_keeps_normalized_quaternion():
    p, q = shifted_pose([.5, 0, .6], [1, 0, 0, 0], {"dx_m": .001, "yaw_deg": .5})
    assert p.tolist() == [.501, 0, .6]
    assert np.linalg.norm(q) == pytest.approx(1.)


def test_stack_goal_changes_before_factory_without_mutating_original_config():
    original = {"support_center": [.5, .1], "support_height": .055}
    changed = environment_conditions("stack", original, {"target_dx_m": .002})
    assert changed["support_center"] == (.502, .1)
    assert original["support_center"] == [.5, .1]
    assert changed["support_height"] == original["support_height"]


def test_default_sampling_preserves_original_rng_contract():
    rng = np.random.default_rng(np.random.SeedSequence([42, 7]))
    expected = {'dx_m': float(rng.uniform(-.002, .002)),
                'dy_m': float(rng.uniform(-.002, .002)), 'yaw_deg': float(rng.uniform(-.5, .5))}
    assert sample_conditions('lift', 42, 7) == expected


def test_configurable_ranges_are_deterministic_and_do_not_mutate_config():
    contract = sampling_contract()
    contract['tasks']['lift']['dx_m'] = [.001, .001]
    contract['tasks']['lift']['yaw_deg'] = [-.1, .1]
    for index in range(10):
        actual = sample_conditions('lift', 42, index, contract)
        assert actual['dx_m'] == .001
        assert -.1 <= actual['yaw_deg'] <= .1
        assert actual == sample_conditions('lift', 42, index, contract)
    assert contract['tasks']['lift']['dx_m'] == [.001, .001]


@pytest.mark.parametrize('bounds', [[.002, -.002], [0, float('nan')], [True, .001], [0, .1], [0]])
def test_sampling_contract_rejects_invalid_ranges(bounds):
    contract = sampling_contract()
    contract['tasks']['lift']['dx_m'] = bounds
    with pytest.raises(ValueError):
        sampling_contract(contract)


def test_sampling_contract_rejects_physics_fields_or_missing_task():
    contract = sampling_contract()
    contract['tasks']['push']['mass'] = [1., 2.]
    with pytest.raises(ValueError):
        sampling_contract(contract)
    contract = sampling_contract()
    del contract['tasks']['stack']
    with pytest.raises(ValueError):
        sampling_contract(contract)
