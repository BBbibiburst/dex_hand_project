import numpy as np
import torch

from apps.train_tactile_sim2real_gan import DomainNormalizer, Generator


def test_domain_normalizer_round_trip():
    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    normalizer = DomainNormalizer().fit(values)
    transformed = normalizer.transform(values)
    restored = normalizer.inverse_tensor(torch.from_numpy(transformed)).numpy()
    np.testing.assert_allclose(restored, values, atol=1e-5)


def test_generator_shape_and_finite():
    generator = Generator(dimension=127, hidden=16)
    output = generator(torch.zeros(5, 127))
    assert output.shape == (5, 127)
    assert torch.isfinite(output).all()
