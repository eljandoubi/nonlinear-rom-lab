"""Command line: run the two studies and build the report from their JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from romlab.families import FAMILY_NAMES
from romlab.operators import LATENT_KINDS
from romlab.study_operator import OperatorConfig, run_operator_study
from romlab.study_reduction import METHODS, ReductionConfig, run_reduction_study


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def environment_info(device: str) -> dict:
    """Versions and hardware recorded next to the results."""
    import platform

    import numpy
    import torch

    info = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(t) for t in text.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="romlab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    red = sub.add_parser("reduction", help="error against latent dimension on exact families")
    red.add_argument("--families", default=",".join(FAMILY_NAMES))
    red.add_argument("--methods", default=",".join(METHODS))
    red.add_argument("--dims", type=_ints, default=ReductionConfig.dims)
    red.add_argument("--seeds", type=_ints, default=ReductionConfig.seeds)
    red.add_argument("--n-train", type=int, default=ReductionConfig.n_train)
    red.add_argument("--n-val", type=int, default=ReductionConfig.n_val)
    red.add_argument("--n-test", type=int, default=ReductionConfig.n_test)
    red.add_argument("--epochs", type=int, default=ReductionConfig.epochs)
    red.add_argument("--device", default="auto")
    red.add_argument("--out", type=Path, default=Path("results/reduction.json"))

    op = sub.add_parser("operator", help="latent decoders against an FNO on Burgers")
    op.add_argument("--modes", type=int, default=OperatorConfig.modes)
    op.add_argument("--methods", default=",".join(("fno", *LATENT_KINDS)))
    op.add_argument("--train-sizes", type=_ints, default=OperatorConfig.train_sizes)
    op.add_argument("--latent-dims", type=_ints, default=None)
    op.add_argument("--seeds", type=_ints, default=OperatorConfig.seeds)
    op.add_argument("--n-val", type=int, default=OperatorConfig.n_val)
    op.add_argument("--n-test", type=int, default=OperatorConfig.n_test)
    op.add_argument("--latent-epochs", type=int, default=OperatorConfig.latent_epochs)
    op.add_argument("--fno-steps", type=int, default=OperatorConfig.fno_steps)
    op.add_argument("--device", default="auto")
    op.add_argument("--out", type=Path, default=Path("results/operator.json"))

    rep = sub.add_parser("report", help="tables and figures from the JSON results")
    rep.add_argument("--results", type=Path, default=Path("results"))
    return parser


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def say(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    if args.command == "reduction":
        cfg = ReductionConfig(
            families=tuple(args.families.split(",")),
            dims=args.dims,
            methods=tuple(args.methods.split(",")),
            seeds=args.seeds,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            epochs=args.epochs,
            device=resolve_device(args.device),
        )
        _write(
            args.out,
            {**run_reduction_study(cfg, progress=say), "environment": environment_info(cfg.device)},
        )
        say(f"wrote {args.out}")
    elif args.command == "operator":
        cfg = OperatorConfig(
            modes=args.modes,
            train_sizes=args.train_sizes,
            latent_dims=args.latent_dims,
            methods=tuple(args.methods.split(",")),
            seeds=args.seeds,
            n_val=args.n_val,
            n_test=args.n_test,
            latent_epochs=args.latent_epochs,
            fno_steps=args.fno_steps,
            device=resolve_device(args.device),
        )
        _write(
            args.out,
            {**run_operator_study(cfg, progress=say), "environment": environment_info(cfg.device)},
        )
        say(f"wrote {args.out}")
    else:
        from romlab.report import build_report

        for path in build_report(args.results):
            say(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
