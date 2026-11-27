"""Study A: reconstruction error against latent dimension for linear and nonlinear reducers.

For each family of exact solutions, fields at random parameters are split into training,
validation and test sets. Each reducer is fitted on the training fields (validation fields choose
the polynomial degree and the epoch of the networks) and scored on test fields from parameters it
never saw. The same fits also give the sensitivity of each decoder to noise in its latent codes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from romlab.families import FAMILY_NAMES, Family, get_family
from romlab.linear import POD
from romlab.metrics import (
    jacobian_spectral_norms,
    mean_relative_l2,
    noise_sensitivity,
)
from romlab.neural import Autoencoder, NeuralDecoderROM
from romlab.polynomial import PolynomialROM

METHODS = ("pod", "quadratic", "polynomial", "pca_mlp", "autoencoder")
NEURAL = ("pca_mlp", "autoencoder")


@dataclass(frozen=True)
class ReductionConfig:
    families: tuple[str, ...] = FAMILY_NAMES
    dims: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
    methods: tuple[str, ...] = METHODS
    seeds: tuple[int, ...] = (0, 1, 2)
    n_train: int = 128
    n_val: int = 64
    n_test: int = 256
    n_grid: int = 256
    epochs: int = 2000
    device: str = "cpu"


def make_split(family: Family, seed: int, n_train: int, n_val: int, n_test: int) -> dict:
    """Fields at three independent uniform draws of the parameters."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, size in (("train", n_train), ("val", n_val), ("test", n_test)):
        params = family.sample(size, rng)
        out[name] = family.solve(params)
    return out


def fit_reducer(method: str, n: int, train, val, seed: int, epochs: int, device: str):
    """Fit one reducer. Every returned object has ``encode``, ``decode`` and ``reconstruct``."""
    if method == "pod":
        return POD(n).fit(train)
    if method == "quadratic":
        return PolynomialROM(n, degree=2).fit(train)
    if method == "polynomial":
        return PolynomialROM(n, degree=None).fit(train, val)
    if method == "pca_mlp":
        return NeuralDecoderROM(n, epochs=epochs, seed=seed, device=device).fit(train, val)
    if method == "autoencoder":
        return Autoencoder(n, epochs=epochs, seed=seed, device=device).fit(train, val)
    raise ValueError(f"unknown method {method!r}")


def decoder_stability(model, fields: np.ndarray, rng: np.random.Generator) -> dict:
    """Sensitivity of the decoder in latent coordinates scaled to unit spread.

    ``jac_max`` and ``jac_mean`` are the largest and the mean spectral norm of the Jacobian over
    the codes of ``fields``, divided by the mean field norm. ``noise_5pct`` is the mean relative
    change of the decoded field when each code gets noise of 5 percent of its spread.
    """
    codes = model.encode(fields)
    mu, sd = codes.mean(axis=0), np.maximum(codes.std(axis=0), 1e-12)

    def decode_unit(z: np.ndarray) -> np.ndarray:
        return model.decode(mu + z * sd)

    norms = jacobian_spectral_norms(decode_unit, (codes - mu) / sd)
    scale = float(np.linalg.norm(fields, axis=1).mean())
    return {
        "jac_max": float(norms.max() / scale),
        "jac_mean": float(norms.mean() / scale),
        "noise_5pct": noise_sensitivity(model.decode, codes, 0.05, rng),
    }


def run_reduction_study(
    cfg: ReductionConfig, progress: Callable[[str], None] | None = None
) -> dict:
    rows = []
    spectra = {}
    descriptions = {}
    for name in cfg.families:
        family = get_family(name, cfg.n_grid)
        descriptions[name] = {"description": family.description, "dim": family.dim}
        for seed in cfg.seeds:
            data = make_split(family, seed, cfg.n_train, cfg.n_val, cfg.n_test)
            if seed == cfg.seeds[0]:
                sv = POD(min(cfg.n_train, cfg.n_grid) - 1).fit(data["train"]).singular_values
                spectra[name] = (sv / sv[0]).tolist()
            for n in cfg.dims:
                for method in cfg.methods:
                    start = time.perf_counter()
                    model = fit_reducer(
                        method, n, data["train"], data["val"], seed, cfg.epochs, cfg.device
                    )
                    row = {
                        "family": name,
                        "seed": seed,
                        "n": n,
                        "method": method,
                        "train_error": mean_relative_l2(
                            data["train"], model.reconstruct(data["train"])
                        ),
                        "val_error": mean_relative_l2(data["val"], model.reconstruct(data["val"])),
                        "test_error": mean_relative_l2(
                            data["test"], model.reconstruct(data["test"])
                        ),
                        **decoder_stability(model, data["test"], np.random.default_rng(seed)),
                    }
                    if method in ("quadratic", "polynomial"):
                        row["degree"] = model.chosen_degree
                    if method in NEURAL:
                        row["best_epoch"] = model.history.best_epoch
                    row["fit_seconds"] = time.perf_counter() - start
                    rows.append(row)
            if progress:
                progress(f"{name} seed {seed} done")
    return {
        "config": {**cfg.__dict__, "device": cfg.device},
        "families": descriptions,
        "spectra": spectra,
        "rows": rows,
    }
