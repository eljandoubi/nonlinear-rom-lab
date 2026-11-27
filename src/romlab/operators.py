"""Operator learning for Burgers: a Fourier neural operator and models with a latent bottleneck.

The task maps an initial field u0 to the field u(T) of the viscous Burgers equation. The latent
models first reduce u0 to a few POD coordinates, map them to a latent vector of size ``n_z`` with
an MLP, and decode that vector to the whole output field. They differ only in the decoder: a
fixed POD basis of the training outputs, a learned linear map, or a learned linear map plus a
nonlinear correction. The Fourier neural operator reads the full input field.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from romlab.linear import POD
from romlab.solvers import burgers_solve
from romlab.training import History, fit, mlp, to_numpy, to_tensor

LATENT_KINDS = ("pod", "linear", "nonlinear")


def random_initial_fields(
    n: int, modes: int, n_points: int, rng: np.random.Generator, amplitude: float = 0.8
) -> np.ndarray:
    """Periodic fields with random Fourier coefficients of mean square (amplitude / k)^2.

    The field is sum_k a_k sin(2 pi k x) + b_k cos(2 pi k x) for k = 1..modes, with a_k and b_k
    drawn from N(0, (amplitude / k)^2).
    """
    x = np.arange(n_points) / n_points
    k = np.arange(1, modes + 1)
    a = rng.normal(size=(n, modes)) * amplitude / k
    b = rng.normal(size=(n, modes)) * amplitude / k
    return a @ np.sin(2 * np.pi * k[:, None] * x) + b @ np.cos(2 * np.pi * k[:, None] * x)


def make_operator_dataset(
    n: int,
    seed: int,
    modes: int = 2,
    nu: float = 0.02,
    t_final: float = 0.4,
    n_grid: int = 256,
    oversample: int = 2,
    dt: float = 5e-4,
    device: str = "cpu",
    chunk: int = 256,
) -> dict[str, np.ndarray]:
    """Initial fields and Burgers solutions at ``t_final`` on ``n_grid`` points.

    The solver runs on a grid ``oversample`` times finer and the result is sampled every
    ``oversample`` points, so the solver error stays far below the errors measured later.
    """
    rng = np.random.default_rng(seed)
    n_fine = n_grid * oversample
    u0 = random_initial_fields(n, modes, n_fine, rng)
    out = np.empty_like(u0)
    for start in range(0, n, chunk):
        batch = to_tensor(u0[start : start + chunk], device)
        out[start : start + chunk] = to_numpy(burgers_solve(batch, nu, t_final, dt))
    return {"u0": u0[:, ::oversample], "u_final": out[:, ::oversample]}


class SpectralConv1d(nn.Module):
    """Multiplication of the lowest Fourier modes by a learned complex matrix per mode.

    The complex weights are stored as pairs of reals (last axis: real and imaginary part), so a
    cast of the module to another float type keeps both parts.
    """

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (width * width)
        self.weights = nn.Parameter(scale * torch.randn(width, width, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        xf = torch.fft.rfft(x)
        out = torch.zeros_like(xf)
        weights = torch.view_as_complex(self.weights)
        out[:, :, : self.modes] = torch.einsum("bim,iom->bom", xf[:, :, : self.modes], weights)
        return torch.fft.irfft(out, n=n)


class FNO1d(nn.Module):
    """Fourier neural operator on a periodic grid; the input channels are u0 and the coordinate."""

    def __init__(self, modes: int = 16, width: int = 32, layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.spectral = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        batch, n = u0.shape
        grid = torch.linspace(0.0, 1.0, n, dtype=u0.dtype, device=u0.device).expand(batch, n)
        h = self.lift(torch.stack([u0, grid], dim=-1)).permute(0, 2, 1)
        last = len(self.spectral) - 1
        for i, (spec, point) in enumerate(zip(self.spectral, self.pointwise, strict=True)):
            h = spec(h) + point(h)
            if i < last:
                h = torch.nn.functional.gelu(h)
        return self.project(h.permute(0, 2, 1)).squeeze(-1)


class FixedLinearDecoder(nn.Module):
    """Decoder z -> (z * spread) @ basis + mean with no trainable parameter."""

    def __init__(self, basis: np.ndarray, mean: np.ndarray, spread: np.ndarray):
        super().__init__()
        self.register_buffer("basis", to_tensor(basis))
        self.register_buffer("mean", to_tensor(mean))
        self.register_buffer("spread", to_tensor(spread))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (z * self.spread) @ self.basis + self.mean


class LinearPlusMLPDecoder(nn.Module):
    """A learned linear decoder plus an MLP correction that starts at zero."""

    def __init__(self, n_z: int, n_out: int, hidden: tuple[int, ...] = (128, 256)):
        super().__init__()
        self.linear = nn.Linear(n_z, n_out)
        self.correction = mlp([n_z, *hidden, n_out], zero_last=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z) + self.correction(z)


def init_linear_from_pod(linear: nn.Linear, pod: POD, spread: np.ndarray) -> None:
    """Set ``linear`` to the map z -> mean + basis (z * spread) of a fitted POD."""
    with torch.no_grad():
        linear.weight.copy_(to_tensor(pod.basis * spread[None, :]))
        linear.bias.copy_(to_tensor(pod.mean))


@dataclass
class FittedOperator:
    """A trained model with a ``predict`` that takes and returns physical fields."""

    predict: Callable[[np.ndarray], np.ndarray]
    n_params: int
    history: History
    seconds: float


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fit_latent(
    kind: str,
    n_z: int,
    n_in: int,
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    seed: int,
    epochs: int,
    device: str = "cpu",
    hidden: tuple[int, ...] = (128, 128),
) -> FittedOperator:
    """Train an input-POD, MLP, latent-decoder model of the given decoder ``kind``.

    The three kinds share the encoder and the latent size. ``pod`` decodes with the POD basis of
    the training outputs and trains only the MLP. ``linear`` and ``nonlinear`` start from that
    same decoder and train it too; ``nonlinear`` adds an MLP correction that starts at zero.
    """
    if kind not in LATENT_KINDS:
        raise ValueError(f"unknown decoder kind {kind!r}")
    start = time.perf_counter()
    u0, u_final = train
    encoder = POD(n_in).fit(u0)
    c_spread = np.maximum(encoder.encode(u0).std(axis=0), 1e-12)
    out_mean, out_scale = u_final.mean(axis=0), max(float(u_final.std()), 1e-12)
    n_out = u_final.shape[1]

    def features(fields: np.ndarray) -> torch.Tensor:
        return to_tensor(encoder.encode(fields) / c_spread, device)

    target = (u_final - out_mean) / out_scale
    torch.manual_seed(seed)
    net = mlp([n_in, *hidden, n_z])
    pod = POD(n_z).fit(target)
    spread = pod.encode(target).std(axis=0)
    if kind == "pod":
        decoder: nn.Module = FixedLinearDecoder(pod.basis.T, pod.mean, spread)
    elif kind == "linear":
        decoder = nn.Linear(n_z, n_out)
        init_linear_from_pod(decoder, pod, spread)
    else:
        decoder = LinearPlusMLPDecoder(n_z, n_out)
        init_linear_from_pod(decoder.linear, pod, spread)
    model = nn.Sequential(net, decoder).to(device, torch.float64)
    hist = fit(
        model,
        features(u0),
        to_tensor(target, device),
        features(val[0]),
        to_tensor((val[1] - out_mean) / out_scale, device),
        epochs=epochs,
        seed=seed,
    )

    def predict(fields: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return to_numpy(model(features(np.atleast_2d(fields)))) * out_scale + out_mean

    return FittedOperator(predict, count_parameters(model), hist, time.perf_counter() - start)


def fit_fno(
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    seed: int,
    steps: int,
    device: str = "cpu",
    modes: int = 16,
    width: int = 32,
    layers: int = 4,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> FittedOperator:
    """Train a Fourier neural operator for about ``steps`` optimiser steps."""
    start = time.perf_counter()
    u0, u_final = train
    s_in, s_out = max(float(u0.std()), 1e-12), max(float(u_final.std()), 1e-12)
    per_epoch = math.ceil(len(u0) / batch_size)
    epochs = math.ceil(steps / per_epoch)
    torch.manual_seed(seed)
    model = FNO1d(modes, width, layers).to(device, torch.float64)
    hist = fit(
        model,
        to_tensor(u0 / s_in, device),
        to_tensor(u_final / s_out, device),
        to_tensor(val[0] / s_in, device),
        to_tensor(val[1] / s_out, device),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        eval_every=max(1, round(25 / per_epoch)),
        seed=seed,
    )

    def predict(fields: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(to_tensor(np.atleast_2d(fields) / s_in, device))
        return to_numpy(out) * s_out

    return FittedOperator(predict, count_parameters(model), hist, time.perf_counter() - start)
