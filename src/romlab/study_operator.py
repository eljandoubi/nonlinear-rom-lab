"""Study B: operator learning for Burgers, latent bottleneck models against an FNO.

The initial fields are random trigonometric polynomials with ``modes`` modes, so they live in a
linear space of dimension 2 * modes, and the outputs are Burgers solutions with shocks. Training
sets are nested: the set of size m is the first m fields of one pool, so every method sees the
same data at a given size. Seeds change the network initialisation and the mini-batch order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from romlab.metrics import paired_bootstrap_ci, relative_l2_errors
from romlab.operators import LATENT_KINDS, fit_fno, fit_latent, make_operator_dataset

COMPARISONS = (
    ("nonlinear", "linear"),
    ("nonlinear", "pod"),
    ("nonlinear", "fno"),
    ("fno", "pod"),
)


@dataclass(frozen=True)
class OperatorConfig:
    modes: int = 2
    train_sizes: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
    latent_dims: tuple[int, ...] | None = None
    methods: tuple[str, ...] = ("fno", *LATENT_KINDS)
    seeds: tuple[int, ...] = (0, 1, 2)
    n_val: int = 128
    n_test: int = 256
    nu: float = 0.02
    t_final: float = 0.4
    n_grid: int = 256
    dt: float = 5e-4
    latent_epochs: int = 3000
    fno_steps: int = 3000
    fno_modes: int = 16
    fno_width: int = 32
    fno_layers: int = 4
    device: str = "cpu"

    @property
    def n_in(self) -> int:
        return 2 * self.modes

    @property
    def dims(self) -> tuple[int, ...]:
        if self.latent_dims is not None:
            return self.latent_dims
        return tuple(sorted({max(1, self.n_in // 2), self.n_in, 2 * self.n_in}))


def _selected_dim(rows: list[dict], m: int, seed: int, method: str) -> int:
    """Latent size with the lowest validation error, the smaller one among equals."""
    cand = [r for r in rows if r["m"] == m and r["seed"] == seed and r["method"] == method]
    return min(cand, key=lambda r: (r["val_error"], r["n_z"]))["n_z"]


def run_operator_study(cfg: OperatorConfig, progress: Callable[[str], None] | None = None) -> dict:
    pool = max(cfg.train_sizes)
    data = make_operator_dataset(
        pool + cfg.n_val + cfg.n_test,
        seed=0,
        modes=cfg.modes,
        nu=cfg.nu,
        t_final=cfg.t_final,
        n_grid=cfg.n_grid,
        dt=cfg.dt,
        device=cfg.device,
    )
    u0, u_final = data["u0"], data["u_final"]
    val = (u0[pool : pool + cfg.n_val], u_final[pool : pool + cfg.n_val])
    test = (u0[pool + cfg.n_val :], u_final[pool + cfg.n_val :])
    rows: list[dict] = []
    errors: dict[tuple, np.ndarray] = {}

    def record(method, n_z, m, seed, fitted):
        val_err = float(relative_l2_errors(val[1], fitted.predict(val[0])).mean())
        test_errs = relative_l2_errors(test[1], fitted.predict(test[0]))
        errors[(m, seed, method, n_z)] = test_errs
        rows.append(
            {
                "method": method,
                "n_z": n_z,
                "m": m,
                "seed": seed,
                "val_error": val_err,
                "test_error": float(test_errs.mean()),
                "n_params": fitted.n_params,
                "best_epoch": fitted.history.best_epoch,
                "fit_seconds": fitted.seconds,
            }
        )

    for m in cfg.train_sizes:
        train = (u0[:m], u_final[:m])
        for seed in cfg.seeds:
            for method in cfg.methods:
                if method == "fno":
                    fitted = fit_fno(
                        train,
                        val,
                        seed,
                        cfg.fno_steps,
                        cfg.device,
                        cfg.fno_modes,
                        cfg.fno_width,
                        cfg.fno_layers,
                    )
                    record("fno", None, m, seed, fitted)
                    continue
                for n_z in cfg.dims:
                    fitted = fit_latent(
                        method, n_z, cfg.n_in, train, val, seed, cfg.latent_epochs, cfg.device
                    )
                    record(method, n_z, m, seed, fitted)
            if progress:
                progress(f"m={m} seed={seed} done")

    latent_methods = [k for k in cfg.methods if k != "fno"]
    selected = []
    for m in cfg.train_sizes:
        for seed in cfg.seeds:
            for method in latent_methods:
                n_z = _selected_dim(rows, m, seed, method)
                selected.append(
                    {
                        "method": method,
                        "m": m,
                        "seed": seed,
                        "n_z": n_z,
                        "test_error": float(errors[(m, seed, method, n_z)].mean()),
                    }
                )

    def per_sample(m, seed, method):
        if method == "fno":
            return errors[(m, seed, "fno", None)]
        return errors[(m, seed, method, _selected_dim(rows, m, seed, method))]

    comparisons = []
    for m in cfg.train_sizes:
        for a, b in COMPARISONS:
            if a not in cfg.methods or b not in cfg.methods:
                continue
            diff = np.concatenate([per_sample(m, s, a) - per_sample(m, s, b) for s in cfg.seeds])
            mean, lo, hi = paired_bootstrap_ci(diff)
            comparisons.append({"m": m, "a": a, "b": b, "mean_diff": mean, "lo": lo, "hi": hi})

    energy = np.linalg.norm(u_final, axis=1) / np.linalg.norm(u0, axis=1)
    return {
        "config": {**cfg.__dict__, "dims": list(cfg.dims)},
        "dataset": {
            "n_in": cfg.n_in,
            "mean_output_to_input_norm": float(energy.mean()),
            "max_abs_mean_output": float(np.abs(u_final.mean(axis=1)).max()),
        },
        "rows": rows,
        "selected": selected,
        "comparisons": comparisons,
    }
