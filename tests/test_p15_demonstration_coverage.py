from __future__ import annotations

import numpy as np

from apps.analyze_p15_demonstration_coverage import (
    _first_contacts,
    _normalized_distance,
)


def test_first_contact_timing_is_per_digit_and_keeps_missing_contacts():
    counts = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 2],
        ]
    )
    assert _first_contacts(counts) == [1, None, 2]


def test_normalized_nearest_state_returns_distance_and_index():
    query = np.asarray([1.0, 2.0])
    candidates = np.asarray([[0.0, 0.0], [1.5, 1.5]])
    distance, index = _normalized_distance(query, candidates, np.asarray([1.0, 0.5]))
    assert index == 1
    assert np.isclose(distance, np.sqrt((0.5**2 + 1.0**2) / 2.0))
