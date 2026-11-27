import numpy as np
import pytest

from romlab.metrics import (
    jacobian_spectral_norms,
    mean_relative_l2,
    noise_sensitivity,
    paired_bootstrap_ci,
    relative_l2_errors,
)


def test_relative_error_of_a_known_pair():
    truth = np.array([[3.0, 4.0], [0.0, 2.0]])
    pred = np.array([[3.0, 0.0], [0.0, 2.0]])
    assert relative_l2_errors(truth, pred) == pytest.approx([0.8, 0.0])
    assert mean_relative_l2(truth, pred) == pytest.approx(0.4)


def test_relative_error_accepts_single_fields():
    assert relative_l2_errors(np.array([1.0, 0.0]), np.array([0.0, 0.0])) == pytest.approx([1.0])


def test_jacobian_norm_of_a_linear_decoder_is_the_largest_singular_value():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(3, 12))  # decode(a) = a @ m, Jacobian is m.T
    codes = rng.normal(size=(5, 3))
    norms = jacobian_spectral_norms(lambda a: a @ m, codes)
    assert norms == pytest.approx(np.full(5, np.linalg.svd(m, compute_uv=False)[0]), rel=1e-6)


def test_jacobian_norm_of_a_quadratic_decoder_varies_with_the_point():
    def decode(a):
        return np.column_stack([a[:, 0] ** 2, a[:, 1]])

    codes = np.array([[1.0, 5.0], [3.0, -2.0], [0.0, 0.0]])
    # d decode / d a = [[2 a0, 0], [0, 1]] -> spectral norm max(2 |a0|, 1)
    assert jacobian_spectral_norms(decode, codes) == pytest.approx([2.0, 6.0, 1.0], rel=1e-6)


def test_noise_sensitivity_is_zero_without_noise_and_grows_with_it():
    m = np.random.default_rng(1).normal(size=(2, 8))
    codes = np.random.default_rng(2).normal(size=(50, 2))

    def decode(a):
        return 2.0 + a @ m

    none = noise_sensitivity(decode, codes, 0.0, np.random.default_rng(3))
    small = noise_sensitivity(decode, codes, 0.01, np.random.default_rng(3))
    large = noise_sensitivity(decode, codes, 0.1, np.random.default_rng(3))
    assert none == 0.0
    assert 0.0 < small < large
    assert large == pytest.approx(10 * small, rel=1e-6)


def test_noise_sensitivity_is_reproducible_for_a_seed():
    codes = np.random.default_rng(4).normal(size=(20, 2))

    def decode(a):
        return np.column_stack([a, a**2])

    a = noise_sensitivity(decode, codes, 0.05, np.random.default_rng(5))
    b = noise_sensitivity(decode, codes, 0.05, np.random.default_rng(5))
    assert a == b


def test_bootstrap_interval_of_constant_differences_is_a_point():
    mean, lo, hi = paired_bootstrap_ci(np.full(30, 0.7))
    assert mean == pytest.approx(0.7) and lo == pytest.approx(0.7) and hi == pytest.approx(0.7)


def test_bootstrap_interval_covers_the_mean_and_excludes_zero_for_a_clear_shift():
    diff = 1.0 + np.random.default_rng(6).normal(size=200)
    mean, lo, hi = paired_bootstrap_ci(diff, seed=1)
    assert lo < mean < hi
    assert lo > 0.0
    assert hi - lo == pytest.approx(2 * 1.96 / np.sqrt(200), rel=0.35)


def test_bootstrap_is_reproducible_for_a_seed():
    diff = np.random.default_rng(7).normal(size=40)
    assert paired_bootstrap_ci(diff, seed=3) == paired_bootstrap_ci(diff, seed=3)
