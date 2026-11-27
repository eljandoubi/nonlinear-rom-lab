import json

import numpy as np
import pytest

from romlab.study_operator import COMPARISONS, OperatorConfig, _selected_dim, run_operator_study

TINY = OperatorConfig(
    modes=1,
    train_sizes=(8, 12),
    seeds=(0, 1),
    n_val=4,
    n_test=6,
    n_grid=32,
    dt=5e-3,
    latent_epochs=5,
    fno_steps=4,
    fno_modes=4,
    fno_width=4,
    fno_layers=2,
)


def test_default_latent_sizes_follow_the_input_dimension():
    assert OperatorConfig(modes=1).dims == (1, 2, 4)
    assert OperatorConfig(modes=2).dims == (2, 4, 8)
    assert OperatorConfig(modes=6).dims == (6, 12, 24)
    assert OperatorConfig(modes=2, latent_dims=(3, 5)).dims == (3, 5)


def test_selected_dimension_is_the_lowest_validation_error_and_ties_go_small():
    rows = [
        {"m": 8, "seed": 0, "method": "pod", "n_z": 4, "val_error": 0.2},
        {"m": 8, "seed": 0, "method": "pod", "n_z": 2, "val_error": 0.3},
        {"m": 8, "seed": 0, "method": "pod", "n_z": 8, "val_error": 0.2},
        {"m": 8, "seed": 1, "method": "pod", "n_z": 2, "val_error": 0.1},
        {"m": 8, "seed": 0, "method": "linear", "n_z": 2, "val_error": 0.0},
    ]
    assert _selected_dim(rows, 8, 0, "pod") == 4
    assert _selected_dim(rows, 8, 1, "pod") == 2
    assert _selected_dim(rows, 8, 0, "linear") == 2


@pytest.fixture(scope="module")
def tiny_run():
    seen = []
    out = run_operator_study(TINY, progress=seen.append)
    return out, seen


def test_one_row_per_method_size_seed_and_latent_dimension(tiny_run):
    out, _ = tiny_run
    assert len(out["rows"]) == 2 * 2 * (1 + 3 * 3)
    fno = [r for r in out["rows"] if r["method"] == "fno"]
    assert len(fno) == 4 and all(r["n_z"] is None for r in fno)
    for r in out["rows"]:
        assert np.isfinite([r["val_error"], r["test_error"], r["fit_seconds"]]).all()
        assert r["n_params"] > 0


def test_selected_rows_match_the_validation_choice(tiny_run):
    out, _ = tiny_run
    assert len(out["selected"]) == 2 * 2 * 3
    for sel in out["selected"]:
        assert sel["n_z"] == _selected_dim(out["rows"], sel["m"], sel["seed"], sel["method"])
        match = [
            r
            for r in out["rows"]
            if (r["m"], r["seed"], r["method"], r["n_z"])
            == (sel["m"], sel["seed"], sel["method"], sel["n_z"])
        ]
        assert match[0]["test_error"] == sel["test_error"]


def test_comparisons_have_ordered_intervals(tiny_run):
    out, _ = tiny_run
    assert len(out["comparisons"]) == 2 * len(COMPARISONS)
    for c in out["comparisons"]:
        assert c["lo"] <= c["mean_diff"] <= c["hi"]
        assert (c["a"], c["b"]) in COMPARISONS


def test_result_is_json_and_reports_the_dataset(tiny_run):
    out, seen = tiny_run
    json.dumps(out)
    assert out["dataset"]["n_in"] == 2
    assert out["dataset"]["max_abs_mean_output"] < 1e-10
    assert out["dataset"]["mean_output_to_input_norm"] < 1.0
    assert seen == ["m=8 seed=0 done", "m=8 seed=1 done", "m=12 seed=0 done", "m=12 seed=1 done"]


def test_comparisons_skip_pairs_with_a_missing_method():
    cfg = OperatorConfig(
        modes=1,
        train_sizes=(8,),
        methods=("pod", "nonlinear"),
        seeds=(0,),
        n_val=4,
        n_test=6,
        n_grid=32,
        dt=5e-3,
        latent_epochs=3,
    )
    out = run_operator_study(cfg)
    assert [(c["a"], c["b"]) for c in out["comparisons"]] == [("nonlinear", "pod")]
    assert all(r["method"] != "fno" for r in out["rows"])


def test_checkpoint_receives_the_rows_so_far():
    cfg = OperatorConfig(
        modes=1,
        train_sizes=(8,),
        methods=("pod",),
        latent_dims=(1, 2),
        seeds=(0, 1),
        n_val=4,
        n_test=6,
        n_grid=32,
        dt=5e-3,
        latent_epochs=2,
    )
    sizes = []
    run_operator_study(cfg, checkpoint=lambda rows: sizes.append(len(rows)))
    assert sizes == [2, 4]
