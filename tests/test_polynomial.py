import numpy as np
import pytest

from romlab.linear import POD
from romlab.metrics import mean_relative_l2
from romlab.polynomial import (
    PolynomialDecoder,
    PolynomialROM,
    legendre_features,
    n_polynomial_terms,
    total_degree_indices,
)


@pytest.mark.parametrize(("n", "degree"), [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)])
def test_exponent_vectors_match_the_term_count(n, degree):
    idx = total_degree_indices(n, degree)
    assert len(idx) == n_polynomial_terms(n, degree)
    assert len({tuple(row) for row in idx}) == len(idx)
    assert idx.sum(axis=1).max() == degree
    assert (idx.sum(axis=1) == 0).sum() == 1


def test_term_counts_for_small_cases():
    assert n_polynomial_terms(1, 3) == 4
    assert n_polynomial_terms(2, 2) == 6
    assert n_polynomial_terms(3, 2) == 10


def test_legendre_features_match_closed_forms():
    rng = np.random.default_rng(0)
    c = rng.uniform(-1, 1, size=(7, 2))
    idx = np.array([[0, 0], [1, 0], [2, 0], [0, 3], [1, 1]])
    x, y = c[:, 0], c[:, 1]
    expected = np.column_stack([np.ones(7), x, (3 * x**2 - 1) / 2, (5 * y**3 - 3 * y) / 2, x * y])
    assert np.allclose(legendre_features(c, idx), expected, atol=1e-14)


def test_decoder_recovers_an_exact_cubic_map():
    rng = np.random.default_rng(1)
    coef = rng.normal(size=(10, 6))

    def monomials(a):
        x, y = a[:, 0], a[:, 1]
        return np.column_stack([x**2, x * y, y**2, x**3, x**2 * y, x * y**2, y**3, 0 * x + 1, x, y])

    train = rng.uniform(-2, 2, size=(80, 2))
    test = rng.uniform(-2, 2, size=(30, 2))
    dec = PolynomialDecoder(3, ridge=1e-14).fit(train, monomials(train) @ coef)
    err = np.abs(dec(test) - monomials(test) @ coef).max()
    assert err < 1e-6 * np.abs(monomials(test) @ coef).max()


def test_decoder_rejects_a_degree_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        PolynomialDecoder(0)


def _quadratic_family(rng, n_grid=60):
    """Fields phi * a + w * (a^2) with phi and w orthogonal, so POD finds phi exactly."""
    q, _ = np.linalg.qr(rng.normal(size=(n_grid, 2)))
    phi, w = q[:, 0], 0.3 * q[:, 1]
    base = rng.normal(size=n_grid)

    def make(a):
        return base + np.outer(a, phi) + np.outer(a**2, w)

    return make


def test_quadratic_manifold_is_recovered_where_pod_fails():
    # A symmetric design makes a and a^2 uncorrelated, so the first POD mode is phi.
    make = _quadratic_family(np.random.default_rng(2))
    train = make(np.linspace(-1, 1, 41))
    test = make(np.random.default_rng(3).uniform(-1, 1, size=25))
    pod_error = mean_relative_l2(test, POD(1).fit(train).reconstruct(test))
    rom = PolynomialROM(1, degree=2).fit(train)
    quad_error = mean_relative_l2(test, rom.reconstruct(test))
    assert pod_error > 1e-3
    assert quad_error < 1e-8


def test_degree_one_gives_back_pod():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(30, 12)) @ np.diag(np.linspace(1, 0.1, 12))
    test = rng.normal(size=(5, 12)) @ np.diag(np.linspace(1, 0.1, 12))
    rom = PolynomialROM(3, degree=1).fit(train)
    assert np.allclose(rom.reconstruct(test), POD(3).fit(train).reconstruct(test), atol=1e-9)


def test_degree_is_picked_on_validation_and_ties_go_to_the_lowest():
    make = _quadratic_family(np.random.default_rng(5))
    train = make(np.linspace(-1, 1, 41))
    val = make(np.random.default_rng(6).uniform(-1, 1, size=25))
    rom = PolynomialROM(1, degree=None, max_degree=6).fit(train, val)
    assert rom.chosen_degree == 2


def test_validation_prefers_a_high_degree_when_the_manifold_needs_it():
    x = np.linspace(0.0, 1.0, 120)
    mu = np.linspace(0.2, 0.8, 60)
    fields = np.exp(-0.5 * ((x[None] - mu[:, None]) / 0.06) ** 2)
    train, val = fields[::2], fields[1::2]
    rom = PolynomialROM(1, degree=None, max_degree=12).fit(train, val)
    assert rom.chosen_degree > 2
    assert mean_relative_l2(val, rom.reconstruct(val)) < mean_relative_l2(
        val, POD(1).fit(train).reconstruct(val)
    )


def test_choosing_the_degree_needs_validation_fields():
    with pytest.raises(ValueError, match="validation"):
        PolynomialROM(1, degree=None).fit(np.random.default_rng(7).normal(size=(10, 5)))


def test_the_term_cap_skips_expensive_degrees():
    rng = np.random.default_rng(8)
    train = rng.normal(size=(40, 30))
    val = rng.normal(size=(10, 30))
    rom = PolynomialROM(8, degree=None, max_degree=6, max_terms=60).fit(train, val)
    assert rom.chosen_degree <= 2
    with pytest.raises(ValueError, match="max_terms"):
        PolynomialROM(8, degree=None, max_degree=6, max_terms=5).fit(train, val)


def test_decode_of_encode_has_the_input_shape():
    rng = np.random.default_rng(9)
    train = rng.normal(size=(30, 12))
    rom = PolynomialROM(2, degree=3).fit(train)
    assert rom.reconstruct(train[:4]).shape == (4, 12)
    assert rom.decode(np.zeros(2)).shape == (1, 12)
