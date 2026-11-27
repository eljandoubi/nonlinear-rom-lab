"""Proper orthogonal decomposition (POD), the linear reduced representation."""

from __future__ import annotations

import numpy as np


class POD:
    """Mean field plus the first ``n`` left singular directions of the centred snapshots.

    ``encode`` is the orthogonal projection onto the basis and ``decode`` the linear map back, so
    ``decode(encode(x))`` is the best rank-``n`` affine approximation of the training set in the
    least-squares sense (Eckart-Young).
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n

    def fit(self, snapshots: np.ndarray) -> POD:
        x = np.asarray(snapshots, dtype=float)
        if x.ndim != 2:
            raise ValueError("snapshots must be a 2D array with one field per row")
        if self.n > min(x.shape):
            raise ValueError(f"n={self.n} exceeds the rank bound {min(x.shape)} of the snapshots")
        self.mean = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.singular_values = s
        self.basis = vt[: self.n].T.copy()
        return self

    def encode(self, fields: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(fields) - self.mean) @ self.basis

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return self.mean + np.atleast_2d(codes) @ self.basis.T

    def reconstruct(self, fields: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(fields))

    def tail_error(self, n: int | None = None) -> float:
        """Relative Frobenius error of the training set at dimension ``n``, from the spectrum."""
        n = self.n if n is None else n
        energy = self.singular_values**2
        return float(np.sqrt(energy[n:].sum() / energy.sum()))
