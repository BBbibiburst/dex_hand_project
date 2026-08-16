import numpy as np

from source.grasping.search.planning import approach_direction_metadata


def test_approach_direction_metadata_separates_elevation_and_azimuth() -> None:
    front = approach_direction_metadata(np.array([1.0, 0.0, 0.0]))
    upper = approach_direction_metadata(np.array([1.0, 0.0, 1.0]))

    assert front["approach_bin"].endswith("_level")
    assert upper["approach_bin"].endswith("_upper")
    assert front["approach_azimuth"] == 0.0
    assert upper["approach_elevation"] > front["approach_elevation"]
