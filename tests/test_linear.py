import numpy as np
import pytest

from romlab.linear import POD


def _low_rank_snapshots(rng, m=40, n=30, rank=5):
    return rng.normal(size=(m, rank)) @ rng.normal(size=(rank, n)) + 3.0


def test_basis_is_orthonormal():
    pod = POD(4).fit(_low_rank_snapshots(np.random.default_rng(0)))
    assert np.allclose(pod.basis.T @ pod.basis, np.eye(4), atol=1e-12)


def test_reconstruction_is_exact_at_the_rank_of_the_data():
    x = _low_rank_snapshots(np.random.default_rng(1), rank=5)
    pod = POD(5).fit(x)
    assert np.abs(pod.reconstruct(x) - x).max() < 1e-10


def test_error_is_the_tail_of_the_spectrum():
    # Eckart-Young: the squared Frobenius error is the sum of the discarded squared singular values.
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 30)) @ np.diag(np.linspace(1.0, 0.05, 30))
    for n in (1, 3, 10):
        pod = POD(n).fit(x)
        err = np.linalg.norm(x - pod.reconstruct(x)) / np.linalg.norm(x - x.mean(axis=0))
        assert err == pytest.approx(pod.tail_error(), rel=1e-10)
        assert err == pytest.approx(
            np.sqrt((pod.singular_values[n:] ** 2).sum() / (pod.singular_values**2).sum()),
            rel=1e-10,
        )


def test_no_other_orthonormal_basis_of_the_same_size_does_better():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 20)) @ np.diag(np.linspace(1.0, 0.1, 20))
    pod = POD(4).fit(x)
    best = np.linalg.norm(x - pod.reconstruct(x))
    centred = x - x.mean(axis=0)
    for _ in range(20):
        q, _ = np.linalg.qr(rng.normal(size=(20, 4)))
        other = np.linalg.norm(centred - centred @ q @ q.T)
        assert other >= best - 1e-12


def test_mean_field_encodes_to_zero_and_a_single_field_is_accepted():
    x = _low_rank_snapshots(np.random.default_rng(4))
    pod = POD(3).fit(x)
    assert np.allclose(pod.encode(pod.mean), 0.0, atol=1e-12)
    assert pod.encode(x[0]).shape == (1, 3)
    assert pod.decode(np.zeros(3)).shape == (1, 30)


def test_more_modes_never_increase_the_training_error():
    x = np.random.default_rng(5).normal(size=(30, 20))
    errors = [np.linalg.norm(x - POD(n).fit(x).reconstruct(x)) for n in range(1, 11)]
    assert all(a >= b - 1e-12 for a, b in zip(errors, errors[1:], strict=False))


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_dimension(bad):
    with pytest.raises(ValueError, match="at least 1"):
        POD(bad)


def test_rejects_a_dimension_above_the_rank_bound_and_non_matrix_input():
    with pytest.raises(ValueError, match="rank bound"):
        POD(9).fit(np.zeros((8, 20)))
    with pytest.raises(ValueError, match="2D array"):
        POD(1).fit(np.zeros(5))
