"""Polynomial decoders on POD coordinates: a linear encoder and a nonlinear decoder.

The decoder adds to the linear reconstruction a polynomial of the POD coordinates fitted to the
part of the snapshots that the basis misses. Degree 2 is the usual quadratic manifold, degree 1
fits nothing beyond POD. Higher degrees are chosen on validation data, together with a ridge
penalty, because the number of coefficients grows quickly with the latent dimension.
"""

from __future__ import annotations

from collections.abc import Iterator
from math import comb

import numpy as np

from romlab.linear import POD
from romlab.metrics import mean_relative_l2


def total_degree_indices(n: int, degree: int) -> np.ndarray:
    """Exponent vectors in N^n with total degree at most ``degree``, as rows."""

    def compositions(left: int, parts: int) -> Iterator[tuple[int, ...]]:
        if parts == 1:
            yield from ((d,) for d in range(left + 1))
            return
        for d in range(left + 1):
            for rest in compositions(left - d, parts - 1):
                yield (d, *rest)

    rows = list(compositions(degree, n))
    return np.array(rows, dtype=int).reshape(len(rows), n)


def n_polynomial_terms(n: int, degree: int) -> int:
    """Number of terms of total degree at most ``degree`` in n variables."""
    return comb(n + degree, degree)


def legendre_features(coords: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Products of Legendre polynomials of each coordinate, one column per exponent vector."""
    coords = np.atleast_2d(coords)
    degree = int(indices.max())
    vander = [np.polynomial.legendre.legvander(c, degree) for c in coords.T]
    out = np.ones((coords.shape[0], len(indices)))
    for col, alpha in enumerate(indices):
        for i, d in enumerate(alpha):
            if d:
                out[:, col] *= vander[i][:, d]
    return out


class PolynomialDecoder:
    """Ridge regression of target fields on Legendre features of the latent codes."""

    def __init__(self, degree: int, ridge: float = 1e-10):
        if degree < 1:
            raise ValueError("degree must be at least 1")
        self.degree = degree
        self.ridge = ridge

    def fit(self, codes: np.ndarray, targets: np.ndarray) -> PolynomialDecoder:
        codes = np.atleast_2d(codes)
        self.scale = np.maximum(np.abs(codes).max(axis=0), 1e-12)
        self.indices = total_degree_indices(codes.shape[1], self.degree)
        feats = legendre_features(codes / self.scale, self.indices)
        n_terms = feats.shape[1]
        penalty = np.sqrt(self.ridge * np.trace(feats.T @ feats) / n_terms)
        lhs = np.vstack([feats, penalty * np.eye(n_terms)])
        rhs = np.vstack([targets, np.zeros((n_terms, targets.shape[1]))])
        self.coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        return self

    def __call__(self, codes: np.ndarray) -> np.ndarray:
        codes = np.atleast_2d(codes)
        return legendre_features(codes / self.scale, self.indices) @ self.coef


class PolynomialROM:
    """POD encoder with a polynomial decoder of the part of the fields outside the basis.

    Without validation fields the decoder has the fixed ``degree`` (2 by default) and the first
    ridge penalty. With validation fields, every candidate is scored by mean relative L2 error:
    the ridge penalties for a fixed ``degree``, or degrees 1 to ``max_degree`` times the ridge
    penalties for ``degree=None``. The first candidate (lowest degree, then smallest ridge)
    within 1 percent of the best score is kept, and scores below 1e-8 count as equal. Degrees
    that would need more than ``max_terms`` coefficients are skipped.
    """

    def __init__(
        self,
        n: int,
        degree: int | None = 2,
        max_degree: int = 8,
        ridges: tuple[float, ...] = (1e-10, 1e-6, 1e-3),
        max_terms: int = 2500,
    ):
        self.n = n
        self.degree = degree
        self.max_degree = max_degree
        self.ridges = ridges
        self.max_terms = max_terms

    def fit(self, train: np.ndarray, val: np.ndarray | None = None) -> PolynomialROM:
        self.pod = POD(self.n).fit(train)
        codes = self.pod.encode(train)
        outside = train - self.pod.decode(codes)
        if self.degree is not None and val is None:
            self.decoder = PolynomialDecoder(self.degree, self.ridges[0]).fit(codes, outside)
            return self
        if val is None:
            raise ValueError("validation fields are needed to choose the degree")
        val_codes = self.pod.encode(val)
        linear_part = self.pod.decode(val_codes)
        degrees = [self.degree] if self.degree is not None else range(1, self.max_degree + 1)
        scored = []
        for degree in degrees:
            if n_polynomial_terms(self.n, degree) > self.max_terms:
                break
            for ridge in self.ridges:
                decoder = PolynomialDecoder(degree, ridge).fit(codes, outside)
                err = mean_relative_l2(val, linear_part + decoder(val_codes))
                if np.isfinite(err):
                    scored.append((err, decoder))
        if not scored:
            raise ValueError("no polynomial degree fits within max_terms")
        cut = max(1.01 * min(err for err, _ in scored), 1e-8)
        self.decoder = next(decoder for err, decoder in scored if err <= cut)
        return self

    @property
    def chosen_degree(self) -> int:
        return self.decoder.degree

    @property
    def chosen_ridge(self) -> float:
        return self.decoder.ridge

    def encode(self, fields: np.ndarray) -> np.ndarray:
        return self.pod.encode(fields)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.atleast_2d(codes)
        return self.pod.decode(codes) + self.decoder(codes)

    def reconstruct(self, fields: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(fields))
