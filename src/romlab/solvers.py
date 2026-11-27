"""Pseudo-spectral solver for the viscous Burgers equation on a periodic interval."""

from __future__ import annotations

import math

import torch


def burgers_solve(
    u0: torch.Tensor, nu: float, t_final: float, dt: float = 5e-4, length: float = 1.0
) -> torch.Tensor:
    """Integrate u_t + u u_x = nu u_xx from ``u0`` of shape (batch, n) to ``t_final``.

    Fourier collocation with the 2/3 rule against aliasing, the diffusion term integrated exactly
    and the nonlinear term advanced by classical fourth-order Runge-Kutta (integrating factor).
    The time step is shortened so that ``t_final`` is a whole number of steps.
    """
    n = u0.shape[-1]
    n_steps = max(1, math.ceil(t_final / dt))
    h = t_final / n_steps
    modes = torch.arange(n // 2 + 1, device=u0.device)
    k = (2.0 * math.pi / length) * modes.to(u0.dtype)
    keep = (modes <= n // 3).to(u0.dtype)
    half = torch.exp(-nu * k**2 * h / 2.0)
    full = half * half
    ik = -0.5j * k

    def nonlinear(uh: torch.Tensor) -> torch.Tensor:
        u = torch.fft.irfft(uh * keep, n=n)
        return ik * torch.fft.rfft(u * u) * keep

    uh = torch.fft.rfft(u0) * keep
    for _ in range(n_steps):
        a = h * nonlinear(uh)
        b = h * nonlinear(half * (uh + a / 2.0))
        c = h * nonlinear(half * uh + b / 2.0)
        d = h * nonlinear(full * uh + half * c)
        uh = full * uh + (full * a + 2.0 * half * (b + c) + d) / 6.0
    return torch.fft.irfft(uh, n=n)
