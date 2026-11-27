"""Metrics shared by the studies: reconstruction error, decoder sensitivity, paired bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def relative_l2_errors(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-field relative L2 error ||truth - pred|| / ||truth|| for rows of two arrays."""
    truth = np.atleast_2d(truth)
    pred = np.atleast_2d(pred)
    return np.linalg.norm(truth - pred, axis=1) / np.linalg.norm(truth, axis=1)


def mean_relative_l2(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(relative_l2_errors(truth, pred).mean())


def jacobian_spectral_norms(
    decode: Callable[[np.ndarray], np.ndarray], codes: np.ndarray, step: float = 1e-5
) -> np.ndarray:
    """Largest singular value of the Jacobian of ``decode`` at each row of ``codes``.

    Central differences along every latent axis, in one batched call to ``decode``. The step is
    relative to the spread of each latent coordinate. The maximum over codes is a lower bound on
    the Lipschitz constant of the decoder over the region the codes cover.
    """
    codes = np.atleast_2d(codes)
    m, n = codes.shape
    h = step * np.maximum(codes.std(axis=0), 1e-12)
    plus = np.repeat(codes[:, None, :], n, axis=1)
    minus = plus.copy()
    idx = np.arange(n)
    plus[:, idx, idx] += h
    minus[:, idx, idx] -= h
    both = decode(np.concatenate([plus.reshape(m * n, n), minus.reshape(m * n, n)]))
    both = both.reshape(2, m, n, -1)
    jac = (both[0] - both[1]) / (2.0 * h[None, :, None])  # (m, n, N): column j is d decode / d a_j
    return np.linalg.norm(jac, ord=2, axis=(1, 2))


def noise_sensitivity(
    decode: Callable[[np.ndarray], np.ndarray],
    codes: np.ndarray,
    level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Relative change of each decoded field when the codes get Gaussian noise.

    The noise on each latent coordinate has standard deviation ``level`` times the spread of that
    coordinate over ``codes``. Returns one value per row of ``codes``.
    """
    codes = np.atleast_2d(codes)
    noisy = codes + level * codes.std(axis=0) * rng.standard_normal(codes.shape)
    return relative_l2_errors(decode(codes), decode(noisy))


def paired_bootstrap_ci(
    diff: np.ndarray, n_boot: int = 4000, level: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of paired differences with a percentile bootstrap interval over the pairs."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[picks].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [tail, 1.0 - tail])
    return float(diff.mean()), float(lo), float(hi)
