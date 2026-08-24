from __future__ import annotations

import numpy as np

from source.ultradexgrasp.dexevolve_contacts import _farthest_point_indices


def test_farthest_contact_sampling_is_unique_and_spread() -> None:
    points = np.asarray([[float(index), 0.0, 0.0] for index in range(20)])
    indices = _farthest_point_indices(points, 4)
    assert len(np.unique(indices)) == 4
    assert 0 in indices
    assert 19 in indices
