import json

import numpy as np
import pytest

from romlab.cli import main
from romlab.operators import make_operator_dataset
from romlab.study_ceiling import run_ceiling_study
from romlab.study_operator import OperatorConfig

TINY = OperatorConfig(
    modes=1,
    n_val=4,
    n_test=6,
    n_grid=32,
    dt=5e-3,
    latent_dims=(1, 2),
)


def test_one_row_per_size_and_method_with_seeds_only_for_networks():
    out = run_ceiling_study(TINY, train_size=8, seeds=(0, 1), epochs=3)
    rows = out["rows"]
    assert len(rows) == 2 * (1 + 2 + 2)
    assert {r["seed"] for r in rows if r["method"] == "pod"} == {0}
    assert {r["seed"] for r in rows if r["method"] == "autoencoder"} == {0, 1}
    assert all(np.isfinite(r["test_error"]) for r in rows)
    json.dumps(out)


def test_pod_row_is_the_reconstruction_of_the_test_outputs():
    from romlab.linear import POD
    from romlab.metrics import mean_relative_l2

    out = run_ceiling_study(TINY, train_size=8, methods=("pod",), epochs=1)
    data = make_operator_dataset(8 + 4 + 6, seed=0, modes=1, n_grid=32, dt=5e-3)
    fields = data["u_final"]
    pod = POD(1).fit(fields[:8])
    expected = mean_relative_l2(fields[12:], pod.reconstruct(fields[12:]))
    first = next(r for r in out["rows"] if r["n_z"] == 1)
    assert first["test_error"] == pytest.approx(expected)


def test_error_falls_with_the_latent_size_for_pod():
    cfg = OperatorConfig(modes=2, n_val=4, n_test=8, n_grid=32, dt=5e-3, latent_dims=(1, 2, 4, 8))
    out = run_ceiling_study(cfg, train_size=16, methods=("pod",), epochs=1)
    errors = [r["test_error"] for r in out["rows"]]
    assert errors == sorted(errors, reverse=True)


def test_command_line_writes_the_file_and_the_environment(tmp_path):
    path = tmp_path / "ceiling_k1.json"
    code = main(
        [
            "ceiling",
            "--modes",
            "1",
            "--train-size",
            "8",
            "--latent-dims",
            "1,2",
            "--methods",
            "pod,autoencoder",
            "--n-val",
            "4",
            "--n-test",
            "6",
            "--epochs",
            "2",
            "--device",
            "cpu",
            "--out",
            str(path),
        ]
    )
    data = json.loads(path.read_text())
    assert code == 0
    assert data["config"]["train_size"] == 8 and data["config"]["dims"] == [1, 2]
    assert "torch" in data["environment"]
