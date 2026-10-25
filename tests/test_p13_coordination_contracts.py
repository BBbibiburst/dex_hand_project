import numpy as np

from apps.search_p13_close_coordination import (
    MAX_CANDIDATES_PER_ROOT,
    CoordinationCandidate,
    _failure_reason,
    refinement_candidates,
    select_validation_candidates,
    training_eligible,
    tuning_candidates,
)


def test_tuning_candidates_are_unique_and_within_budget():
    candidates = tuning_candidates()
    assert len(candidates) == 18
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert len(candidates) <= MAX_CANDIDATES_PER_ROOT
    assert all(candidate.ramp_steps <= 30 for candidate in candidates)
    refined = refinement_candidates()
    assert len(candidates) + len(refined) == MAX_CANDIDATES_PER_ROOT
    assert not {candidate.candidate_id for candidate in candidates}.intersection(
        candidate.candidate_id for candidate in refined
    )


def test_validation_selection_is_ranked_and_does_not_leak_roots():
    def row(name, rise, following, slip):
        candidate = CoordinationCandidate(name, (0, 0, 0), 0)
        return {
            "candidate": candidate.__dict__,
            "preparation_validated": rise >= 0.003 and following >= 0.5,
            "object_rise_m": rise,
            "object_following_ratio": following,
            "object_relative_translation_m": slip,
        }

    summary = {
        "results": [
            row("baseline_repeat_0", 0.01, 1.0, 0.0),
            row("first", 0.0031, 0.51, 0.02),
            row("second", 0.0029, 0.80, 0.01),
            row("third", 0.0032, 0.55, 0.01),
        ]
    }
    selected = select_validation_candidates(summary, maximum=2)
    assert [candidate.name for candidate in selected] == ["third", "first"]


def test_failure_reason_preserves_frozen_thresholds():
    base = {
        "preparation_validated": False,
        "ik_failures": 0,
        "joint_limit_violations": 0,
        "contact_fraction": 1.0,
        "opposition_fraction": 1.0,
        "object_rise_m": 0.00299,
        "object_following_ratio": 0.9,
    }
    assert _failure_reason(base) == "INSUFFICIENT_OBJECT_RISE"
    assert _failure_reason({**base, "object_rise_m": 0.0031, "object_following_ratio": 0.49}) == (
        "INSUFFICIENT_FOLLOWING"
    )
    assert _failure_reason({**base, "preparation_validated": True}) is None


def test_candidate_hash_is_stable_for_numpy_independent_values():
    first = CoordinationCandidate("same", tuple(np.zeros(3)), 0.5)
    second = CoordinationCandidate("same", (0.0, 0.0, 0.0), 0.5)
    assert first.candidate_id == second.candidate_id


def test_training_gate_counts_distinct_roots_not_derived_candidates():
    repeated = [
        {"root_id": 3, "preparation_validated": True},
        {"root_id": 3, "preparation_validated": True},
        {"root_id": 1, "preparation_validated": False},
    ]
    assert not training_eligible(repeated)
    assert training_eligible(repeated + [{"root_id": 1, "preparation_validated": True}])
