"""Ceiling of the latent decoders: reconstruct the true outputs of the operator task.

A latent operator model predicts n_z numbers from its input and decodes them to the output field.
Even with a perfect predictor its error cannot go below what the decoder family loses when it
reconstructs the true output from n_z numbers. This study fits the reducers of the reduction
study on the training outputs of the operator task and scores them on the test outputs, which are
the same fields as in the operator study for the same configuration.
"""

from __future__ import annotations

from collections.abc import Callable

from romlab.metrics import mean_relative_l2
from romlab.operators import make_operator_dataset
from romlab.study_operator import OperatorConfig
from romlab.study_reduction import NEURAL, fit_reducer

CEILING_METHODS = ("pod", "pca_mlp", "autoencoder")


def run_ceiling_study(
    cfg: OperatorConfig,
    train_size: int,
    methods: tuple[str, ...] = CEILING_METHODS,
    seeds: tuple[int, ...] = (0,),
    epochs: int = 4000,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Test error of each reducer on the output fields, for each latent size of ``cfg``."""
    data = make_operator_dataset(
        train_size + cfg.n_val + cfg.n_test,
        seed=0,
        modes=cfg.modes,
        nu=cfg.nu,
        t_final=cfg.t_final,
        n_grid=cfg.n_grid,
        dt=cfg.dt,
        device=cfg.device,
    )
    fields = data["u_final"]
    train = fields[:train_size]
    val = fields[train_size : train_size + cfg.n_val]
    test = fields[train_size + cfg.n_val :]
    rows = []
    for n_z in cfg.dims:
        for method in methods:
            for seed in seeds if method in NEURAL else seeds[:1]:
                model = fit_reducer(method, n_z, train, val, seed, epochs, cfg.device)
                rows.append(
                    {
                        "method": method,
                        "n_z": n_z,
                        "seed": seed,
                        "test_error": mean_relative_l2(test, model.reconstruct(test)),
                    }
                )
        if progress:
            progress(f"n_z={n_z} done")
    return {
        "config": {
            "modes": cfg.modes,
            "nu": cfg.nu,
            "t_final": cfg.t_final,
            "n_grid": cfg.n_grid,
            "train_size": train_size,
            "dims": list(cfg.dims),
            "methods": list(methods),
            "seeds": list(seeds),
            "epochs": epochs,
            "device": cfg.device,
        },
        "rows": rows,
    }
