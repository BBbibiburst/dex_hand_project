import numpy as np

from apps.run_tactile_domain_baseline import Preprocessor, _regional_features, feature_names, metrics


def test_regional_features_ignore_within_patch_permutations():
    rng = np.random.default_rng(7)
    frames = rng.random((8, 712), dtype=np.float32)
    permuted = frames.copy()
    for start, stop in ((0, 56), (56, 88), (88, 120), (600, 712)):
        permuted[:, start:stop] = permuted[:, rng.permutation(np.arange(start, stop))]
    np.testing.assert_allclose(_regional_features(frames, "real"), _regional_features(permuted, "real"), atol=1e-6)


def test_feature_dimension_is_stable():
    features = _regional_features(np.ones((4, 712), dtype=np.float32), "sim")
    assert len(features) == len(feature_names()) == 127


def test_preprocessor_fits_training_data_only_and_transforms_finite():
    x = np.asarray([[0.0, 1.0], [1.0, 2.0], [10.0, 20.0], [11.0, 21.0]], dtype=np.float32)
    domains = np.asarray([0, 0, 1, 1])
    for kind in ("common_robust", "domain_robust", "domain_quantile"):
        processor = Preprocessor(kind)
        processor.fit(x, domains)
        transformed = processor.transform(x, domains)
        assert transformed.shape == x.shape
        assert np.isfinite(transformed).all()


def test_domain_confusion_definition():
    result = metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.9, 0.8, 0.2]))
    assert result["balanced_accuracy"] == 0.5
    assert result["domain_confusion"] == 1.0
