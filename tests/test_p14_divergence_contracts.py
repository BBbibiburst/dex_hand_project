import numpy as np

from apps.audit_p14_expert_bc_divergence import first_contact_per_digit, first_persistent
from apps.search_p14_local_demonstrations import (
    MAX_CANDIDATES_PER_ROOT,
    STARTS,
    candidate_family,
    select_validation_mechanisms,
)


def test_first_persistent_requires_complete_window():
    values = np.asarray([0.0, 0.6, 0.7, 0.2, 0.8, 0.9, 1.0])
    assert first_persistent(values, 0.5, steps=3) == 4
    assert first_persistent(values, 0.5, steps=4) is None


def test_first_contact_per_digit_preserves_missing_contacts():
    counts = np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 2]])
    assert first_contact_per_digit(counts) == [1, None, 2]


def test_local_demonstration_budget_and_candidate_ids():
    candidates = candidate_family()
    assert len(candidates) * len(STARTS) == 15
    assert len(candidates) * len(STARTS) <= MAX_CANDIDATES_PER_ROOT
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)


def test_validation_mechanisms_keep_start_and_action_contract():
    candidate = candidate_family()[0]
    rows = []
    for index, start in enumerate(STARTS):
        rows.append(
            {
                "start_name": start,
                "candidate_id": candidate.candidate_id,
                "candidate": candidate.__dict__,
                "preparation_validated": False,
                "object_rise_m": 0.001 + index * 0.001,
                "object_following_ratio": 0.1 + index * 0.1,
                "object_relative_translation_m": 0.01,
            }
        )
    selected = select_validation_mechanisms({"results": rows}, maximum=2)
    assert [name for name, _ in selected] == ["close_early", "preclose"]
