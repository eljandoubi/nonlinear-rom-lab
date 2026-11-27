import numpy as np
import pytest
import torch

from romlab.families import cole_hopf
from romlab.solvers import burgers_solve


def _exact(n, nu, eps, t):
    x = np.arange(n) / n
    return cole_hopf(x, t, nu, eps)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _solve(u0, nu, t, dt):
    out = burgers_solve(torch.tensor(u0, dtype=torch.float64)[None], nu, t, dt)
    return out[0].numpy()


def test_matches_the_cole_hopf_solution():
    n, nu, eps, t = 128, 0.05, 0.9, 0.2
    u = _solve(_exact(n, nu, eps, 0.0), nu, t, 1e-3)
    assert _rel(u, _exact(n, nu, eps, t)) < 1e-6


def test_error_shrinks_faster_than_the_third_power_of_the_step():
    # Dividing the step by 4 divides a fourth-order error by 256 (measured: about 150) and a
    # third-order error by 64.
    n, nu, eps, t = 64, 0.05, 0.9, 0.2
    u0, exact = _exact(n, nu, eps, 0.0), _exact(n, nu, eps, t)
    coarse = _rel(_solve(u0, nu, t, 0.02), exact)
    fine = _rel(_solve(u0, nu, t, 0.005), exact)
    assert coarse / fine > 100.0


def test_final_time_is_reached_when_the_step_does_not_divide_it():
    # Whole steps of the requested size would stop at 0.09 or 0.12, which moves the solution
    # by 4 to 7 percent.
    n, nu, eps, t = 128, 0.05, 0.9, 0.1
    u = _solve(_exact(n, nu, eps, 0.0), nu, t, 0.03)
    assert _rel(u, _exact(n, nu, eps, t)) < 1e-3


def test_mean_is_conserved_and_energy_decreases():
    n = 256
    x = np.arange(n) / n
    u0 = 0.3 + np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    u = _solve(u0, 0.02, 0.4, 5e-4)
    assert u.mean() == pytest.approx(u0.mean(), abs=1e-12)
    assert np.sum(u**2) < np.sum(u0**2)
    assert np.all(np.isfinite(u))


def test_batches_are_independent():
    n = 128
    x = np.arange(n) / n
    batch = torch.tensor(
        np.stack([np.sin(2 * np.pi * x), 0.5 * np.cos(2 * np.pi * x + 1.0)]), dtype=torch.float64
    )
    both = burgers_solve(batch, 0.05, 0.2, 1e-3)
    alone = burgers_solve(batch[1:], 0.05, 0.2, 1e-3)
    assert torch.allclose(both[1], alone[0], atol=1e-13)


def test_resolution_and_step_refinement_change_the_front_little():
    rng = np.random.default_rng(0)
    coef = rng.normal(size=4)

    def make(n):
        x = np.arange(n) / n
        return (
            coef[0] * np.sin(2 * np.pi * x)
            + coef[1] * np.cos(2 * np.pi * x)
            + coef[2] * np.sin(4 * np.pi * x) / 2
            + coef[3] * np.cos(4 * np.pi * x) / 2
        )

    coarse = _solve(make(256), 0.02, 0.4, 5e-4)
    fine = _solve(make(512), 0.02, 0.4, 2.5e-4)[::2]
    assert _rel(coarse, fine) < 1e-3


def test_a_zero_field_stays_zero():
    u = burgers_solve(torch.zeros(2, 64, dtype=torch.float64), 0.02, 0.1)
    assert torch.count_nonzero(u) == 0


def test_modes_above_the_two_thirds_cutoff_stay_zero():
    # Products of modes 40 and 35 land above n/2 and would alias without the cutoff.
    n = 128
    x = np.arange(n) / n
    u0 = np.cos(2 * np.pi * 40 * x) + 0.5 * np.sin(2 * np.pi * 35 * x)
    u = _solve(u0, 0.001, 0.05, 1e-3)
    assert np.abs(np.fft.rfft(u))[n // 3 + 1 :].max() < 1e-10
    assert np.abs(np.fft.rfft(u))[: n // 3 + 1].max() > 1.0
