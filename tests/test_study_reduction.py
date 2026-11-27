import json

import numpy as np
import pytest

from romlab.families import get_family
from romlab.study_reduction import (
    METHODS,
    ReductionConfig,
    decoder_stability,
    fit_reducer,
    make_split,
    run_reduction_study,
)

TINY = ReductionConfig(
    families=("pulse", "soliton"),
    dims=(1, 2),
    seeds=(0,),
    n_train=24,
    n_val=8,
    n_test=10,
    n_grid=48,
    epochs=10,
)


def test_split_sizes_and_independent_draws():
    fam = get_family("pulse", 40)
    a = make_split(fam, 0, 6, 4, 5)
    assert a["train"].shape == (6, 40) and a["val"].shape == (4, 40) and a["test"].shape == (5, 40)
    assert not np.allclose(a["train"][:4], a["val"])
    b = make_split(fam, 0, 6, 4, 5)
    assert np.array_equal(a["test"], b["test"])
    assert not np.array_equal(a["test"], make_split(fam, 1, 6, 4, 5)["test"])


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_a_working_reducer(method):
    fam = get_family("pulse", 40)
    data = make_split(fam, 0, 20, 6, 5)
    model = fit_reducer(method, 2, data["train"], data["val"], 0, 5, "cpu")
    assert model.reconstruct(data["test"]).shape == (5, 40)
    assert model.encode(data["test"]).shape == (5, 2)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        fit_reducer("svd", 2, np.zeros((5, 5)), None, 0, 1, "cpu")


def test_linear_decoder_has_unit_jacobian_in_pod_coordinates():
    fam = get_family("pulse", 40)
    data = make_split(fam, 0, 20, 6, 30)
    pod = fit_reducer("pod", 2, data["train"], data["val"], 0, 1, "cpu")
    stab = decoder_stability(pod, data["test"], np.random.default_rng(0))
    # In unit-spread coordinates the Jacobian of a POD decoder is basis * std: its largest
    # singular value is the largest standard deviation of the codes, the same at every point.
    codes = pod.encode(data["test"])
    scale = np.linalg.norm(data["test"], axis=1).mean()
    assert stab["jac_max"] == pytest.approx(codes.std(axis=0).max() / scale, rel=1e-5)
    assert stab["jac_median"] == pytest.approx(stab["jac_max"], rel=1e-5)


def test_run_produces_one_row_per_cell_with_finite_numbers():
    out = run_reduction_study(TINY)
    assert len(out["rows"]) == 2 * 2 * len(METHODS)
    keys = {"family", "seed", "n", "method", "train_error", "val_error", "test_error"}
    for row in out["rows"]:
        assert keys <= row.keys()
        assert np.isfinite([row["train_error"], row["test_error"], row["noise_median"]]).all()
    assert set(out["spectra"]) == {"pulse", "soliton"}
    assert out["spectra"]["pulse"][0] == 1.0
    json.dumps(out)


def test_pod_error_falls_with_dimension_in_the_run():
    out = run_reduction_study(
        ReductionConfig(
            families=("front",),
            dims=(1, 3, 6),
            methods=("pod",),
            seeds=(0,),
            n_train=30,
            n_val=8,
            n_test=20,
            n_grid=64,
        )
    )
    errors = [r["test_error"] for r in out["rows"]]
    assert errors[0] > errors[1] > errors[2]


def test_progress_callback_is_called_once_per_seed():
    seen = []
    run_reduction_study(
        ReductionConfig(
            families=("pulse",),
            dims=(1,),
            methods=("pod",),
            seeds=(0, 1),
            n_train=10,
            n_val=4,
            n_test=4,
            n_grid=32,
        ),
        progress=seen.append,
    )
    assert seen == ["pulse seed 0 done", "pulse seed 1 done"]
