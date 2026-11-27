import json

import pytest

from romlab.cli import build_parser, environment_info, main, resolve_device


def test_device_names():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")


def test_environment_records_the_gpu_name_only_on_a_gpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda index: "Fake T4")
    assert environment_info("cuda")["gpu"] == "Fake T4"
    assert "gpu" not in environment_info("cpu")


def test_parser_reads_integer_lists_and_defaults():
    args = build_parser().parse_args(["reduction", "--dims", "1,2,5", "--seeds", "3"])
    assert args.dims == (1, 2, 5) and args.seeds == (3,)
    assert "pulse" in args.families
    args = build_parser().parse_args(["operator", "--train-sizes", "8,16"])
    assert args.train_sizes == (8, 16) and args.modes == 2


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_reduction_then_operator_then_report(tmp_path, capsys):
    red = tmp_path / "reduction.json"
    ops = tmp_path / "operator_k1.json"
    assert (
        main(
            [
                "reduction",
                "--families",
                "pulse,front",
                "--methods",
                "pod,quadratic",
                "--dims",
                "1,2",
                "--seeds",
                "0",
                "--n-train",
                "20",
                "--n-val",
                "6",
                "--n-test",
                "8",
                "--device",
                "cpu",
                "--out",
                str(red),
            ]
        )
        == 0
    )
    assert json.loads(red.read_text())["config"]["dims"] == [1, 2]
    env = json.loads(red.read_text())["environment"]
    assert env["device"] == "cpu" and {"python", "numpy", "torch"} <= env.keys()
    assert (
        main(
            [
                "operator",
                "--modes",
                "1",
                "--methods",
                "pod,nonlinear",
                "--train-sizes",
                "8",
                "--seeds",
                "0",
                "--n-val",
                "4",
                "--n-test",
                "6",
                "--latent-epochs",
                "3",
                "--device",
                "cpu",
                "--out",
                str(ops),
            ]
        )
        == 0
    )
    assert json.loads(ops.read_text())["config"]["train_sizes"] == [8]
    assert main(["report", "--results", str(tmp_path)]) == 0
    assert (tmp_path / "summary.md").exists() and (tmp_path / "operator_error.png").exists()
    assert "wrote" in capsys.readouterr().err
