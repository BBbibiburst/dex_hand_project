import numpy as np
import pytest

from apps.generate_verified_lift_demonstrations import build_parser
from apps.query_p16_state_conditioned_expert import QUERY_STEP, _cartesian_segment
from apps.analyze_p16_randomness_and_queries import (
    ARM_BOUNDARY_LIMIT_RAD,
    HAND_BOUNDARY_LIMIT_M,
    RESTORE_LIMIT,
)


def test_p16_query_step_is_inside_preregistered_precontact_window():
    assert 5 <= QUERY_STEP <= 20


def test_generated_demo_close_steps_is_explicit_and_positive():
    args = build_parser().parse_args([])
    assert args.close_steps == 45
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--close-steps", "not-an-integer"])


def test_cartesian_segment_rejects_no_steps_via_empty_execution():
    class Adapter:
        def measured_pose(self):
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    result = _cartesian_segment(
        None,
        None,
        Adapter(),
        target_position=np.ones(3),
        target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        start_hand=np.zeros(6),
        target_hand=np.ones(6),
        steps=0,
        stage="approach",
    )
    assert result == (0, 0, False)


def test_p16_coverage_thresholds_are_inherited_and_strict():
    assert RESTORE_LIMIT == 1e-6
    assert ARM_BOUNDARY_LIMIT_RAD == 0.05
    assert HAND_BOUNDARY_LIMIT_M == 0.001
