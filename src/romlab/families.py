"""Parametric families of one-dimensional PDE solutions with closed-form expressions.

Every family maps a parameter vector to a field sampled on a fixed grid. The fields are exact, so
the error of a reduced model is the error of the reduction and nothing else.

- ``pulse``: a Gaussian bump advected at constant speed (linear transport), one parameter.
- ``front``: a hyperbolic-tangent front, the profile of a viscous shock, one parameter.
- ``soliton``: the one-soliton solution of the Korteweg-de Vries equation, two parameters.
- ``burgers``: the Cole-Hopf solution of the viscous Burgers equation, shifted, two parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Family:
    """A parametric family u(x; mu) on a fixed grid, with a box of admissible parameters."""

    name: str
    description: str
    grid: np.ndarray
    low: np.ndarray
    high: np.ndarray
    field: Callable[[np.ndarray, np.ndarray], np.ndarray]

    @property
    def dim(self) -> int:
        return len(self.low)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` parameter vectors uniformly in the box."""
        return rng.uniform(self.low, self.high, size=(n, self.dim))

    def solve(self, params: np.ndarray) -> np.ndarray:
        """Evaluate the family on the grid for an array of parameters of shape (n, dim)."""
        params = np.atleast_2d(np.asarray(params, dtype=float))
        if params.shape[1] != self.dim:
            raise ValueError(f"{self.name} takes {self.dim} parameters, got {params.shape[1]}")
        return self.field(params, self.grid)


def gaussian_pulse(params: np.ndarray, x: np.ndarray, width: float = 0.04) -> np.ndarray:
    """exp(-(x - mu)^2 / (2 width^2)) for each centre mu in ``params[:, 0]``."""
    return np.exp(-0.5 * ((x[None, :] - params[:, :1]) / width) ** 2)


def tanh_front(params: np.ndarray, x: np.ndarray, width: float = 0.01) -> np.ndarray:
    """0.5 (1 - tanh((x - mu) / width)): a front of thickness ``width`` centred at mu."""
    return 0.5 * (1.0 - np.tanh((x[None, :] - params[:, :1]) / width))


def kdv_soliton(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """One-soliton solution of u_t + 6 u u_x + u_xxx = 0 with speed c at position x0.

    ``params`` holds (c, x0). The amplitude is c / 2 and the width scales as 1 / sqrt(c), so the
    two parameters move the field in position, height and width at the same time.
    """
    c, x0 = params[:, :1], params[:, 1:2]
    return 0.5 * c / np.cosh(0.5 * np.sqrt(c) * (x[None, :] - x0)) ** 2


def cole_hopf(
    x: np.ndarray, t: float | np.ndarray, nu: float, eps: float | np.ndarray, k: float = TWO_PI
) -> np.ndarray:
    """Exact periodic solution of u_t + u u_x = nu u_xx, from phi = 1 + eps e^(-nu k^2 t) cos(kx).

    With u = -2 nu phi_x / phi, the heat equation for phi gives Burgers for u. The solution is
    2 nu eps k e^(-nu k^2 t) sin(kx) / (1 + eps e^(-nu k^2 t) cos(kx)), periodic when k = 2 pi m
    on the unit interval, with a viscous front at x = 1/2 whose steepness grows with eps < 1.
    """
    damp = np.asarray(eps) * np.exp(-nu * k**2 * np.asarray(t))
    return 2.0 * nu * k * damp * np.sin(k * x) / (1.0 + damp * np.cos(k * x))


def burgers_front(params: np.ndarray, x: np.ndarray, nu: float = 0.02) -> np.ndarray:
    """Cole-Hopf profile with amplitude eps = ``params[:, 0]`` shifted by ``params[:, 1]``."""
    return cole_hopf(x[None, :] - params[:, 1:2], 0.0, nu, params[:, :1])


def get_family(name: str, n_grid: int = 256) -> Family:
    """Build one of the four families on a grid of ``n_grid`` points."""
    unit = np.arange(n_grid) / n_grid
    if name == "pulse":
        return Family(
            "pulse",
            "Gaussian bump (width 0.04) advected on [0, 1]; parameter: centre in [0.2, 0.8]",
            np.linspace(0.0, 1.0, n_grid),
            np.array([0.2]),
            np.array([0.8]),
            gaussian_pulse,
        )
    if name == "front":
        return Family(
            "front",
            "tanh front (thickness 0.01) on [0, 1]; parameter: position in [0.2, 0.8]",
            np.linspace(0.0, 1.0, n_grid),
            np.array([0.2]),
            np.array([0.8]),
            tanh_front,
        )
    if name == "soliton":
        return Family(
            "soliton",
            "KdV one-soliton on [0, 20]; parameters: speed in [1, 4], position in [6, 14]",
            np.linspace(0.0, 20.0, n_grid),
            np.array([1.0, 6.0]),
            np.array([4.0, 14.0]),
            kdv_soliton,
        )
    if name == "burgers":
        return Family(
            "burgers",
            "Cole-Hopf solution of viscous Burgers (nu = 0.02) on the periodic unit interval; "
            "parameters: eps in [0.90, 0.995], shift in [-0.25, 0.25]",
            unit,
            np.array([0.90, -0.25]),
            np.array([0.995, 0.25]),
            burgers_front,
        )
    raise ValueError(f"unknown family {name!r}")


FAMILY_NAMES = ("pulse", "front", "soliton", "burgers")
