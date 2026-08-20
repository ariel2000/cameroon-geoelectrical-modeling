import numpy as np

from scripts.uncertainty import add_gaussian_noise, normalized_rms, standard_deviation


def test_standard_deviation_combines_terms_in_quadrature():
    data = np.array([0.0, 10.0])
    got = standard_deviation(data, 0.1, 1.0)
    np.testing.assert_allclose(got, [1.0, np.sqrt(2.0)])


def test_noise_is_reproducible():
    data = np.ones(10)
    first, _ = add_gaussian_noise(data, 0.05, 0.01, seed=42)
    second, _ = add_gaussian_noise(data, 0.05, 0.01, seed=42)
    np.testing.assert_array_equal(first, second)


def test_normalized_rms():
    assert normalized_rms([1.0, -1.0], [1.0, 1.0]) == 1.0
