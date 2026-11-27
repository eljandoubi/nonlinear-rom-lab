import numpy as np
import pytest

from romlab.families import FAMILY_NAMES, cole_hopf, get_family


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_samples_stay_in_the_box_and_fields_are_finite(name):
    fam = get_family(name)
    params = fam.sample(7, np.random.default_rng(0))
    assert params.shape == (7, fam.dim)
    assert np.all(params >= fam.low) and np.all(params <= fam.high)
    u = fam.solve(params)
    assert u.shape == (7, len(fam.grid))
    assert np.all(np.isfinite(u))


def test_sampling_is_reproducible_for_a_seed():
    fam = get_family("soliton")
    a = fam.sample(4, np.random.default_rng(3))
    b = fam.sample(4, np.random.default_rng(3))
    c = fam.sample(4, np.random.default_rng(4))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_solve_rejects_a_wrong_parameter_count():
    with pytest.raises(ValueError, match="takes 2 parameters"):
        get_family("soliton").solve([[1.0, 2.0, 3.0]])


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown family"):
        get_family("heat")


def test_pulse_peaks_at_its_centre():
    fam = get_family("pulse", n_grid=401)
    u = fam.solve([[0.35], [0.7]])
    assert fam.grid[np.argmax(u[0])] == pytest.approx(0.35, abs=1e-12)
    assert fam.grid[np.argmax(u[1])] == pytest.approx(0.7, abs=1e-12)
    assert u.max() == pytest.approx(1.0)


def test_front_goes_from_one_to_zero_and_is_one_half_at_its_position():
    fam = get_family("front", n_grid=401)
    mu = fam.grid[160]
    u = fam.solve([[mu]])[0]
    assert u[0] == pytest.approx(1.0, abs=1e-12)
    assert u[-1] == pytest.approx(0.0, abs=1e-12)
    assert u[160] == pytest.approx(0.5)
    assert np.all(np.diff(u) <= 0.0)


def test_soliton_solves_the_kdv_equation():
    # u(x, t) = F(x - x0 - c t): the time derivative is a shift of the position parameter.
    fam = get_family("soliton", n_grid=4001)
    dx = fam.grid[1] - fam.grid[0]
    c, x0, delta = 2.5, 9.0, 1e-4
    u = fam.solve([[c, x0]])[0]
    u_t = (fam.solve([[c, x0 + c * delta]])[0] - fam.solve([[c, x0 - c * delta]])[0]) / (2 * delta)
    u_x = (u[2:] - u[:-2]) / (2 * dx)
    u_xxx = (u[4:] - 2 * u[3:-1] + 2 * u[1:-3] - u[:-4]) / (2 * dx**3)
    residual = u_t[2:-2] + 6 * u[2:-2] * u_x[1:-1] + u_xxx
    assert np.abs(residual).max() < 1e-3 * np.abs(u_t).max()


def test_soliton_height_is_half_the_speed():
    fam = get_family("soliton", n_grid=2001)
    u = fam.solve([[3.0, 10.0]])[0]
    assert u.max() == pytest.approx(1.5, rel=1e-6)


def test_cole_hopf_solves_the_viscous_burgers_equation():
    n, nu, eps, t, dt = 512, 0.02, 0.95, 0.1, 1e-6
    x = np.arange(n) / n
    u = cole_hopf(x, t, nu, eps)
    u_t = (cole_hopf(x, t + dt, nu, eps) - cole_hopf(x, t - dt, nu, eps)) / (2 * dt)
    k = 2 * np.pi * np.arange(n // 2 + 1)
    u_hat = np.fft.rfft(u)
    u_x = np.fft.irfft(1j * k * u_hat, n)
    u_xx = np.fft.irfft(-(k**2) * u_hat, n)
    residual = u_t + u * u_x - nu * u_xx
    assert np.abs(residual).max() < 1e-5 * np.abs(u_t).max()


def test_cole_hopf_is_periodic_with_zero_mean():
    x = np.arange(256) / 256
    u = cole_hopf(x, 0.0, 0.02, 0.97)
    assert abs(u.mean()) < 1e-12
    assert cole_hopf(np.array([0.0]), 0.0, 0.02, 0.97)[0] == pytest.approx(
        cole_hopf(np.array([1.0]), 0.0, 0.02, 0.97)[0], abs=1e-12
    )


def test_burgers_front_shifts_with_its_second_parameter():
    fam = get_family("burgers", n_grid=200)
    a = fam.solve([[0.95, 0.0]])[0]
    b = fam.solve([[0.95, 0.1]])[0]
    assert np.allclose(np.roll(a, 20), b, atol=1e-12)
