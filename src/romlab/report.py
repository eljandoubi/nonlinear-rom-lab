"""Tables and figures built from the JSON results of the two studies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REDUCTION_LABELS = {
    "pod": "POD",
    "quadratic": "quadratic",
    "polynomial": "polynomial",
    "pca_mlp": "POD + MLP",
    "autoencoder": "autoencoder",
}
OPERATOR_LABELS = {
    "fno": "FNO",
    "pod": "fixed POD decoder",
    "linear": "trained linear decoder",
    "nonlinear": "linear + MLP decoder",
}
COLORS = {
    "pod": "#555555",
    "quadratic": "#d98c00",
    "polynomial": "#b2182b",
    "pca_mlp": "#2166ac",
    "autoencoder": "#1b7837",
    "fno": "#7b3294",
    "linear": "#d98c00",
    "nonlinear": "#2166ac",
}


def fmt(x: float) -> str:
    """Two significant digits, scientific notation below 1e-3."""
    return f"{x:.2g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.1e}"


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _values(rows: list[dict], key: str, **match) -> np.ndarray:
    return np.array([r[key] for r in rows if all(r[k] == v for k, v in match.items())])


def reduction_curve(data: dict, family: str, method: str, key: str = "test_error"):
    """Latent dimensions with the mean, minimum and maximum of ``key`` over seeds."""
    dims = data["config"]["dims"]
    stats = [_values(data["rows"], key, family=family, method=method, n=n) for n in dims]
    mean = np.array([s.mean() for s in stats])
    return (
        np.array(dims),
        mean,
        np.array([s.min() for s in stats]),
        np.array([s.max() for s in stats]),
    )


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def reduction_error_table(data: dict, family: str) -> str:
    methods = data["config"]["methods"]
    rows = []
    for n in data["config"]["dims"]:
        cells = [str(n)]
        for m in methods:
            cells.append(
                fmt(_values(data["rows"], "test_error", family=family, method=m, n=n).mean())
            )
        rows.append(cells)
    return md_table(["n"] + [REDUCTION_LABELS[m] for m in methods], rows)


def smallest_dimension_table(data: dict, threshold: float = 0.01) -> str:
    """Smallest latent dimension whose mean test error is below ``threshold``."""
    methods = data["config"]["methods"]
    top = max(data["config"]["dims"])
    rows = []
    for family in data["config"]["families"]:
        cells = [family]
        for m in methods:
            dims, mean, _, _ = reduction_curve(data, family, m)
            ok = dims[mean < threshold]
            cells.append(str(int(ok.min())) if len(ok) else f">{top}")
        rows.append(cells)
    return md_table(["family"] + [REDUCTION_LABELS[m] for m in methods], rows)


def pod_decay_table(data: dict, n_min: int = 2) -> str:
    """Slope of log(mean POD test error) against log(n) for n >= ``n_min``."""
    rows = []
    for family in data["config"]["families"]:
        dims, mean, _, _ = reduction_curve(data, family, "pod")
        keep = dims >= n_min
        if keep.sum() < 2:
            rows.append([family, "n/a", "n/a", "n/a"])
            continue
        slope = np.polyfit(np.log(dims[keep]), np.log(mean[keep]), 1)[0]
        rows.append([family, f"{slope:.2f}", fmt(mean[keep][0]), fmt(mean[keep][-1])])
    last = int(max(data["config"]["dims"]))
    return md_table(["family", "slope", f"error at n={n_min}", f"error at n={last}"], rows)


def stability_table(data: dict, n: int) -> str:
    methods = data["config"]["methods"]
    rows = []
    for family in data["config"]["families"]:
        for m in methods:
            sel = dict(family=family, method=m, n=n)
            rows.append(
                [
                    family,
                    REDUCTION_LABELS[m],
                    fmt(_values(data["rows"], "test_error", **sel).mean()),
                    fmt(_values(data["rows"], "noise_median", **sel).mean()),
                    fmt(_values(data["rows"], "noise_p95", **sel).mean()),
                    fmt(_values(data["rows"], "jac_p95", **sel).mean()),
                ]
            )
    header = ["family", "decoder", "test error", "noise median", "noise p95", "Jacobian p95"]
    return md_table(header, rows)


def operator_error_table(data: dict) -> str:
    methods = data["config"]["methods"]
    rows = []
    for m_size in data["config"]["train_sizes"]:
        cells = [str(m_size)]
        for m in methods:
            if m == "fno":
                vals = _values(data["rows"], "test_error", method="fno", m=m_size)
            else:
                vals = _values(data["selected"], "test_error", method=m, m=m_size)
            cells.append(f"{fmt(vals.mean())} ({fmt(vals.min())} to {fmt(vals.max())})")
        rows.append(cells)
    return md_table(["training fields"] + [OPERATOR_LABELS[m] for m in methods], rows)


def operator_params_table(data: dict) -> str:
    """Number of trainable parameters of each operator model and latent size."""
    counts = {(r["method"], r["n_z"]): r["n_params"] for r in data["rows"]}
    order = {m: i for i, m in enumerate(data["config"]["methods"])}
    keys = sorted(counts, key=lambda k: (order[k[0]], k[1] or 0))
    rows = [
        [OPERATOR_LABELS[m], "none" if z is None else str(z), f"{counts[(m, z)]:,}"]
        for m, z in keys
    ]
    return md_table(["model", "latent size", "parameters"], rows)


def operator_comparison_table(data: dict) -> str:
    pairs = sorted({(c["a"], c["b"]) for c in data["comparisons"]})
    rows = []
    for m_size in data["config"]["train_sizes"]:
        cells = [str(m_size)]
        for a, b in pairs:
            c = next(c for c in data["comparisons"] if (c["m"], c["a"], c["b"]) == (m_size, a, b))
            cells.append(f"{fmt(c['mean_diff'])} [{fmt(c['lo'])}, {fmt(c['hi'])}]")
        rows.append(cells)
    header = ["training fields"] + [
        f"{OPERATOR_LABELS[a]} minus {OPERATOR_LABELS[b]}" for a, b in pairs
    ]
    return md_table(header, rows)


def plot_reduction(data: dict, path: Path) -> None:
    families = data["config"]["families"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for ax, family in zip(axes.flat, families, strict=False):
        for m in data["config"]["methods"]:
            dims, mean, lo, hi = reduction_curve(data, family, m)
            ax.plot(dims, mean, "-o", ms=3, color=COLORS[m], label=REDUCTION_LABELS[m])
            ax.fill_between(dims, lo, hi, color=COLORS[m], alpha=0.15, lw=0)
        ax.set(xscale="log", yscale="log", title=family, xlabel="latent dimension n")
        ax.set_xticks(data["config"]["dims"], [str(d) for d in data["config"]["dims"]])
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("test relative L2 error")
    axes[1, 0].set_ylabel("test relative L2 error")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spectra(data: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for family, sv in data["spectra"].items():
        sv = np.array(sv[:64])
        ax.semilogy(np.arange(1, len(sv) + 1), sv, label=family)
    ax.set(
        xlabel="index k", ylabel="singular value / largest", title="POD spectra (training fields)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_stability(data: dict, path: Path) -> None:
    families = data["config"]["families"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for ax, family in zip(axes.flat, families, strict=False):
        for m in data["config"]["methods"]:
            dims, mean, lo, hi = reduction_curve(data, family, m, "noise_p95")
            ax.plot(dims, mean, "-o", ms=3, color=COLORS[m], label=REDUCTION_LABELS[m])
            ax.fill_between(dims, lo, hi, color=COLORS[m], alpha=0.15, lw=0)
        ax.set(xscale="log", yscale="log", title=family, xlabel="latent dimension n")
        ax.set_xticks(data["config"]["dims"], [str(d) for d in data["config"]["dims"]])
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("95th percentile of relative change")
    axes[1, 0].set_ylabel("95th percentile of relative change")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Decoded field change under 5% noise on the latent codes", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_operator(datasets: list[tuple[str, dict]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4), squeeze=False)
    for ax, (title, data) in zip(axes[0], datasets, strict=True):
        sizes = data["config"]["train_sizes"]
        for m in data["config"]["methods"]:
            stats = []
            for size in sizes:
                if m == "fno":
                    stats.append(_values(data["rows"], "test_error", method="fno", m=size))
                else:
                    stats.append(_values(data["selected"], "test_error", method=m, m=size))
            mean = np.array([s.mean() for s in stats])
            ax.plot(sizes, mean, "-o", ms=3, color=COLORS[m], label=OPERATOR_LABELS[m])
            ax.fill_between(
                sizes,
                [s.min() for s in stats],
                [s.max() for s in stats],
                color=COLORS[m],
                alpha=0.15,
                lw=0,
            )
        ax.set(xscale="log", yscale="log", title=title, xlabel="training fields")
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("test relative L2 error")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_report(results: Path) -> list[Path]:
    """Write ``summary.md`` and the figures next to the JSON files; return the paths."""
    results = Path(results)
    written: list[Path] = []
    parts: list[str] = []
    reduction = results / "reduction.json"
    if reduction.exists():
        data = load(reduction)
        cfg = data["config"]
        parts.append(
            f"## Reduction study\n\n{cfg['n_train']} training, {cfg['n_val']} validation and "
            f"{cfg['n_test']} test fields per family, seeds {cfg['seeds']}, grid of "
            f"{cfg['n_grid']} points. Test error is the mean relative L2 error, averaged over "
            "seeds."
        )
        for family in cfg["families"]:
            info = data["families"][family]
            parts.append(
                f"### {family}\n\n{info['description']}\n\n{reduction_error_table(data, family)}"
            )
        parts.append(
            "### Smallest latent dimension with mean test error below 1%\n\n"
            + smallest_dimension_table(data)
        )
        parts.append(
            "### Decay of the POD error\n\nSlope of log error against log n, n from 2.\n\n"
            + pod_decay_table(data)
        )
        for n in (4,):
            if n in cfg["dims"]:
                parts.append(
                    f"### Decoder sensitivity at n = {n}\n\nNoise: change of the decoded field "
                    "relative to its norm when each latent code gets noise of 5% of its spread "
                    "(median and 95th percentile over test fields). Jacobian: 95th percentile of "
                    "the spectral norm of the decoder Jacobian in unit-spread latent coordinates, "
                    "divided by the mean field norm.\n\n" + stability_table(data, n)
                )
        for name, plot in (
            ("reduction_error.png", plot_reduction),
            ("pod_spectra.png", plot_spectra),
            ("decoder_noise.png", plot_stability),
        ):
            plot(data, results / name)
            written.append(results / name)
    operator_sets = []
    for path in sorted(results.glob("operator*.json")):
        data = load(path)
        cfg = data["config"]
        title = f"{2 * cfg['modes']}-dimensional inputs, {cfg['latent_epochs']} latent epochs"
        operator_sets.append((title, data))
        parts.append(
            f"## Operator study, {title}\n\nBurgers, nu = {cfg['nu']}, T = {cfg['t_final']}, "
            f"{cfg['n_grid']} grid points, seeds {cfg['seeds']}, latent sizes {cfg['dims']} "
            "(chosen per fit on validation fields). Test error: mean over seeds with the range "
            "over seeds in brackets.\n\n"
            + operator_error_table(data)
            + "\n\nTrainable parameters.\n\n"
            + operator_params_table(data)
            + "\n\nPaired differences of test error (first minus second, negative means the first "
            "is better), mean with a 95% bootstrap interval over test fields pooled over seeds.\n\n"
            + operator_comparison_table(data)
        )
    if operator_sets:
        plot_operator(operator_sets, results / "operator_error.png")
        written.append(results / "operator_error.png")
    summary = results / "summary.md"
    summary.write_text(
        "# Results\n\nGenerated by `romlab report` from the JSON files.\n\n"
        + "\n\n".join(parts)
        + "\n"
    )
    written.append(summary)
    return written
