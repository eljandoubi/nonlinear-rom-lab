import numpy as np
import pytest
import torch

from romlab.linear import POD
from romlab.metrics import mean_relative_l2
from romlab.operators import (
    FixedLinearDecoder,
    FNO1d,
    LinearPlusMLPDecoder,
    SpectralConv1d,
    count_parameters,
    fit_fno,
    fit_latent,
    init_linear_from_pod,
    make_operator_dataset,
    random_initial_fields,
)


def test_initial_fields_are_band_limited_and_have_zero_mean():
    rng = np.random.default_rng(0)
    u = random_initial_fields(5, 3, 64, rng)
    spectrum = np.abs(np.fft.rfft(u, axis=1)) / 64
    assert spectrum[:, 0].max() < 1e-12
    assert spectrum[:, 4:].max() < 1e-12
    assert spectrum[:, 1:4].max() > 1e-3


def test_initial_field_coefficients_decay_like_one_over_k():
    u = random_initial_fields(4000, 2, 16, np.random.default_rng(1))
    coef = np.abs(np.fft.rfft(u, axis=1))
    rms = np.sqrt((coef**2).mean(axis=0))
    assert rms[2] / rms[1] == pytest.approx(0.5, rel=0.1)


def test_initial_fields_are_reproducible_for_a_seed():
    a = random_initial_fields(3, 2, 32, np.random.default_rng(4))
    assert np.array_equal(a, random_initial_fields(3, 2, 32, np.random.default_rng(4)))
    assert not np.array_equal(a, random_initial_fields(3, 2, 32, np.random.default_rng(5)))


def test_dataset_shapes_conservation_and_dissipation():
    data = make_operator_dataset(6, seed=0, modes=2, n_grid=64, dt=2e-3)
    assert data["u0"].shape == data["u_final"].shape == (6, 64)
    assert np.abs(data["u_final"].mean(axis=1)).max() < 1e-12
    assert np.all(np.linalg.norm(data["u_final"], axis=1) < np.linalg.norm(data["u0"], axis=1))


def test_dataset_input_is_the_subsampled_initial_field():
    data = make_operator_dataset(4, seed=3, modes=2, n_grid=32, oversample=2, dt=5e-3)
    u0 = random_initial_fields(4, 2, 64, np.random.default_rng(3))[:, ::2]
    assert np.array_equal(data["u0"], u0)


def test_dataset_does_not_depend_on_the_chunk_size():
    a = make_operator_dataset(5, seed=2, n_grid=32, dt=5e-3, chunk=2)
    b = make_operator_dataset(5, seed=2, n_grid=32, dt=5e-3, chunk=100)
    assert np.allclose(a["u_final"], b["u_final"], atol=1e-13)


def test_spectral_conv_with_identity_weights_is_a_low_pass_filter():
    conv = SpectralConv1d(width=2, modes=3).double()
    with torch.no_grad():
        conv.weights.zero_()
        for c in range(2):
            conv.weights[c, c, :, 0] = 1.0
    x = torch.randn(4, 2, 32, dtype=torch.float64)
    xf = torch.fft.rfft(x)
    xf[:, :, 3:] = 0
    assert torch.allclose(conv(x), torch.fft.irfft(xf, n=32), atol=1e-12)


def test_spectral_weights_hold_a_real_pair_per_complex_entry():
    conv = SpectralConv1d(2, 3)
    assert tuple(conv.weights.shape) == (2, 2, 3, 2)
    assert count_parameters(conv) == 2 * 2 * 3 * 2


def test_fno_shape_gradients_and_resolution_independence():
    torch.manual_seed(0)
    model = FNO1d(modes=4, width=6, layers=2).double()
    out = model(torch.randn(3, 32, dtype=torch.float64))
    assert out.shape == (3, 32)
    out.pow(2).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model(torch.randn(2, 64, dtype=torch.float64)).shape == (2, 64)


def test_fixed_decoder_has_no_parameters_and_applies_the_formula():
    basis = np.random.default_rng(0).normal(size=(2, 5))
    dec = FixedLinearDecoder(basis, np.ones(5), np.array([2.0, 3.0]))
    assert list(dec.parameters()) == []
    z = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    expected = (np.array([[2.0, 3.0]]) @ basis) + 1.0
    assert np.allclose(dec(z).numpy(), expected)


def test_correction_starts_at_zero_so_the_decoder_starts_linear():
    torch.manual_seed(0)
    dec = LinearPlusMLPDecoder(3, 7).double()
    z = torch.randn(4, 3, dtype=torch.float64)
    assert torch.equal(dec(z), dec.linear(z))


def test_linear_decoder_initialised_from_pod_matches_the_fixed_decoder():
    rng = np.random.default_rng(0)
    fields = rng.normal(size=(30, 9)) @ np.diag(np.linspace(1, 0.1, 9))
    pod = POD(3).fit(fields)
    spread = pod.encode(fields).std(axis=0)
    fixed = FixedLinearDecoder(pod.basis.T, pod.mean, spread)
    linear = torch.nn.Linear(3, 9).double()
    init_linear_from_pod(linear, pod, spread)
    z = torch.randn(5, 3, dtype=torch.float64)
    assert torch.allclose(linear(z), fixed(z), atol=1e-12)


def _linear_operator_data(n=80, n_grid=32, seed=0):
    rng = np.random.default_rng(seed)
    u0 = random_initial_fields(n, 2, n_grid, rng)
    return u0, np.roll(u0, 5, axis=1)


@pytest.mark.parametrize("kind", ["pod", "linear", "nonlinear"])
def test_latent_models_learn_a_linear_operator(kind):
    u0, u_final = _linear_operator_data()
    train, val, test = (
        (u0[:50], u_final[:50]),
        (u0[50:65], u_final[50:65]),
        (u0[65:], u_final[65:]),
    )
    fitted = fit_latent(kind, 4, 4, train, val, seed=0, epochs=800)
    assert mean_relative_l2(test[1], fitted.predict(test[0])) < 0.08
    assert fitted.predict(test[0][0]).shape == (1, 32)


def test_fixed_pod_decoder_has_fewer_parameters_than_learned_ones():
    u0, u_final = _linear_operator_data(40)
    train, val = (u0[:30], u_final[:30]), (u0[30:], u_final[30:])
    counts = {
        k: fit_latent(k, 4, 4, train, val, seed=0, epochs=3).n_params
        for k in ("pod", "linear", "nonlinear")
    }
    assert counts["pod"] < counts["linear"] < counts["nonlinear"]


def test_unknown_decoder_kind_is_rejected():
    u0, u_final = _linear_operator_data(20)
    with pytest.raises(ValueError, match="unknown decoder kind"):
        fit_latent("cubic", 2, 4, (u0, u_final), (u0, u_final), seed=0, epochs=1)


def test_latent_fit_is_reproducible_for_a_seed():
    u0, u_final = _linear_operator_data(30)
    train, val = (u0[:20], u_final[:20]), (u0[20:], u_final[20:])
    a = fit_latent("nonlinear", 3, 4, train, val, seed=1, epochs=20).predict(u0[:3])
    b = fit_latent("nonlinear", 3, 4, train, val, seed=1, epochs=20).predict(u0[:3])
    assert np.array_equal(a, b)


def test_fno_learns_the_identity_map():
    u0, _ = _linear_operator_data(60, n_grid=32)
    train, val, test = (u0[:40], u0[:40]), (u0[40:50], u0[40:50]), u0[50:]
    fitted = fit_fno(train, val, seed=0, steps=300, modes=4, width=8, layers=2, lr=3e-3)
    assert mean_relative_l2(test, fitted.predict(test)) < 0.25


def test_fno_fit_is_reproducible_and_reports_parameters():
    u0, u_final = _linear_operator_data(30, n_grid=16)
    train, val = (u0[:20], u_final[:20]), (u0[20:], u_final[20:])
    a = fit_fno(train, val, seed=2, steps=10, modes=3, width=4, layers=2)
    b = fit_fno(train, val, seed=2, steps=10, modes=3, width=4, layers=2)
    assert np.array_equal(a.predict(u0[:2]), b.predict(u0[:2]))
    assert a.n_params > 0
