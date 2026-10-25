import numpy as np

from apps.diagnose_tactile_p19_gan import _auc, _family_indices
from apps.run_tactile_domain_baseline import feature_names


def test_auc_and_directionless_feature_groups():
    assert _auc(np.asarray([2.0, 3.0]), np.asarray([0.0, 1.0])) == 1.0
    groups = _family_indices(feature_names())
    assert len(groups["nonzero"]) == 11
    assert len(groups["temporal"]) == 33
    assert len(groups["activation_share"]) == 6
    assert len(groups["region:index_tip"]) == 11
