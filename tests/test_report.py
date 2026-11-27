import json

import numpy as np
import pytest

from romlab.report import (
    build_report,
    ceiling_table,
    fmt,
    md_table,
    operator_comparison_table,
    operator_error_table,
    operator_params_table,
    pod_decay_table,
    reduction_curve,
    reduction_error_table,
    smallest_dimension_table,
    stability_table,
)


def fake_reduction():
    rows = []
    for family in ("a", "b"):
        for seed in (0, 1):
            for n in (1, 2, 4):
                for method in ("pod", "autoencoder"):
                    err = 0.4 / n if method == "pod" else 0.005 + 0.001 * seed
                    rows.append(
                        {
                            "family": family,
                            "seed": seed,
                            "n": n,
                            "method": method,
                            "test_error": err,
                            "noise_median": 0.1,
                            "noise_p95": 0.2,
                            "jac_p95": 2.0,
                        }
                    )
    return {
        "config": {
            "families": ["a", "b"],
            "dims": [1, 2, 4],
            "methods": ["pod", "autoencoder"],
            "seeds": [0, 1],
            "n_train": 10,
            "n_val": 5,
            "n_test": 7,
            "n_grid": 32,
        },
        "families": {"a": {"description": "family a"}, "b": {"description": "family b"}},
        "spectra": {"a": [1.0, 0.5, 0.25], "b": [1.0, 0.1, 0.01]},
        "rows": rows,
    }


def fake_operator():
    rows, selected, comps = [], [], []
    for m in (8, 16):
        for seed in (0, 1):
            rows.append(
                {
                    "method": "fno",
                    "n_z": None,
                    "m": m,
                    "seed": seed,
                    "test_error": 0.2 / m,
                    "n_params": 139745,
                }
            )
            for kind, scale, n_params in (("pod", 3.0, 17668), ("nonlinear", 1.0, 118404)):
                selected.append(
                    {"method": kind, "m": m, "seed": seed, "n_z": 4, "test_error": scale * 0.2 / m}
                )
                rows.append(
                    {
                        "method": kind,
                        "n_z": 4,
                        "m": m,
                        "seed": seed,
                        "test_error": scale * 0.2 / m,
                        "n_params": n_params,
                    }
                )
        comps.append(
            {"m": m, "a": "nonlinear", "b": "pod", "mean_diff": -0.01, "lo": -0.02, "hi": 0.0}
        )
    return {
        "config": {
            "modes": 2,
            "nu": 0.02,
            "t_final": 0.4,
            "n_grid": 32,
            "seeds": [0, 1],
            "dims": [2, 4],
            "train_sizes": [8, 16],
            "latent_epochs": 4000,
            "methods": ["fno", "pod", "nonlinear"],
        },
        "rows": rows,
        "selected": selected,
        "comparisons": comps,
    }


def fake_ceiling():
    rows = []
    for n_z, pod, ae in ((4, 0.2, 0.04), (8, 0.1, 0.02)):
        rows.append({"method": "pod", "n_z": n_z, "seed": 0, "test_error": pod})
        for seed in (0, 1):
            rows.append({"method": "autoencoder", "n_z": n_z, "seed": seed, "test_error": ae})
    return {
        "config": {
            "modes": 2,
            "nu": 0.02,
            "t_final": 0.4,
            "n_grid": 32,
            "train_size": 16,
            "dims": [4, 8],
            "methods": ["pod", "autoencoder"],
            "seeds": [0, 1],
            "epochs": 10,
        },
        "rows": rows,
    }


def test_ceiling_table_lists_each_latent_size():
    lines = ceiling_table(fake_ceiling()).splitlines()
    assert lines == [
        "| latent size | POD | autoencoder |",
        "|---|---|---|",
        "| 4 | 0.2 | 0.04 |",
        "| 8 | 0.1 | 0.02 |",
    ]


def test_report_includes_the_ceiling_section(tmp_path):
    (tmp_path / "ceiling_k2.json").write_text(json.dumps(fake_ceiling()))
    paths = build_report(tmp_path)
    text = (tmp_path / "summary.md").read_text()
    assert [p.name for p in paths] == ["summary.md"]
    assert "## Output reconstruction, 4-dimensional inputs" in text
    assert "| 8 | 0.1 | 0.02 |" in text


def test_number_format():
    assert fmt(0.0123) == "0.012"
    assert fmt(0.5) == "0.5"
    assert fmt(1.234e-4) == "1.2e-04"
    assert fmt(0.0) == "0"


def test_markdown_table_has_a_separator_line():
    text = md_table(["x", "y"], [["1", "2"]])
    assert text.splitlines() == ["| x | y |", "|---|---|", "| 1 | 2 |"]


def test_reduction_curve_gives_mean_and_range_over_seeds():
    dims, mean, lo, hi = reduction_curve(fake_reduction(), "a", "autoencoder")
    assert list(dims) == [1, 2, 4]
    assert mean == pytest.approx([0.0055] * 3)
    assert lo == pytest.approx([0.005] * 3) and hi == pytest.approx([0.006] * 3)


def test_error_table_lists_one_row_per_dimension():
    lines = reduction_error_table(fake_reduction(), "a").splitlines()
    assert lines[0] == "| n | POD | autoencoder |"
    assert lines[2] == "| 1 | 0.4 | 0.0055 |"
    assert lines[4] == "| 4 | 0.1 | 0.0055 |"


def test_smallest_dimension_uses_a_strict_threshold_and_marks_misses():
    text = smallest_dimension_table(fake_reduction(), threshold=0.01)
    assert "| a | >4 | 1 |" in text
    assert "| b | >4 | 1 |" in text
    assert "| a | 4 | 1 |" in smallest_dimension_table(fake_reduction(), threshold=0.15)


def test_pod_decay_slope_is_recovered():
    text = pod_decay_table(fake_reduction())
    assert "| a | -1.00 |" in text


def test_pod_decay_needs_two_dimensions_from_the_minimum():
    data = fake_reduction()
    data["config"]["dims"] = [1, 2]
    data["rows"] = [r for r in data["rows"] if r["n"] in (1, 2)]
    assert "| a | n/a | n/a | n/a |" in pod_decay_table(data)


def test_stability_table_reports_each_decoder():
    text = stability_table(fake_reduction(), 4)
    assert "| a | POD | 0.1 | 0.1 | 0.2 | 2 |" in text
    assert "| b | autoencoder | 0.0055 | 0.1 | 0.2 | 2 |" in text


def test_operator_tables():
    data = fake_operator()
    errors = operator_error_table(data).splitlines()
    assert errors[0] == "| training fields | FNO | fixed POD decoder | linear + MLP decoder |"
    assert errors[2] == (
        "| 8 | 0.025 (0.025 to 0.025) | 0.075 (0.075 to 0.075) | 0.025 (0.025 to 0.025) |"
    )
    comps = operator_comparison_table(data)
    assert "| 8 | -0.01 [-0.02, 0] |" in comps
    assert "linear + MLP decoder minus fixed POD decoder" in comps


def test_a_file_without_an_fno_is_summarised_but_not_plotted(tmp_path):
    data = fake_operator()
    data["config"]["methods"] = ["pod", "nonlinear"]
    data["config"]["latent_epochs"] = 20000
    data["rows"] = [r for r in data["rows"] if r["method"] != "fno"]
    (tmp_path / "operator_k2_long.json").write_text(json.dumps(data))
    paths = build_report(tmp_path)
    assert [p.name for p in paths] == ["summary.md"]
    assert "4-dimensional inputs, 20000 latent epochs" in (tmp_path / "summary.md").read_text()


def test_parameter_table_lists_each_model_and_latent_size():
    lines = operator_params_table(fake_operator()).splitlines()
    assert lines == [
        "| model | latent size | parameters |",
        "|---|---|---|",
        "| FNO | none | 139,745 |",
        "| fixed POD decoder | 4 | 17,668 |",
        "| linear + MLP decoder | 4 | 118,404 |",
    ]


def test_build_report_writes_tables_and_figures(tmp_path):
    (tmp_path / "reduction.json").write_text(json.dumps(fake_reduction()))
    (tmp_path / "operator_k2.json").write_text(json.dumps(fake_operator()))
    paths = build_report(tmp_path)
    names = {p.name for p in paths}
    assert names == {
        "reduction_error.png",
        "pod_spectra.png",
        "decoder_noise.png",
        "operator_error.png",
        "summary.md",
    }
    assert all(p.stat().st_size > 0 for p in paths)
    text = (tmp_path / "summary.md").read_text()
    assert (
        "## Reduction study" in text
        and "### a" in text
        and "## Operator study, 4-dimensional inputs, 4000 latent epochs" in text
    )
    assert np.isfinite(len(text))


def test_report_without_results_still_writes_a_summary(tmp_path):
    paths = build_report(tmp_path)
    assert [p.name for p in paths] == ["summary.md"]
