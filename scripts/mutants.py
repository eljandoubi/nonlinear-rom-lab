"""Hand-made faults: change one line of the source, run the matching tests, expect a failure.

Usage: python scripts/mutants.py [results/mutants.txt]

Each fault is applied to a copy of the file text, the matching test file runs, and the original
text is written back whatever happens. A fault counts as caught when the tests fail.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "romlab"

FAULTS = [
    # (description, source file, old text, new text, test file)
    ("solver step ignores the final time", "solvers.py", "h = t_final / n_steps", "h = dt", "test_solvers.py"),
    ("solver dealiases at n/2", "solvers.py", "keep = (modes <= n // 3).to(u0.dtype)", "keep = (modes <= n // 2).to(u0.dtype)", "test_solvers.py"),
    ("solver half factor equals the full factor", "solvers.py", "half = torch.exp(-nu * k**2 * h / 2.0)", "half = torch.exp(-nu * k**2 * h)", "test_solvers.py"),
    ("solver nonlinear coefficient 1 instead of 1/2", "solvers.py", "ik = -0.5j * k", "ik = -1.0j * k", "test_solvers.py"),
    ("solver RK4 weights 1 instead of 2", "solvers.py", "(full * a + 2.0 * half * (b + c) + d) / 6.0", "(full * a + half * (b + c) + d) / 6.0", "test_solvers.py"),
    ("soliton height c instead of c/2", "families.py", "return 0.5 * c / np.cosh(", "return c / np.cosh(", "test_families.py"),
    ("Cole-Hopf factor 2 dropped", "families.py", "return 2.0 * nu * k * damp * np.sin(k * x)", "return nu * k * damp * np.sin(k * x)", "test_families.py"),
    ("Gaussian pulse without the square", "families.py", "/ width) ** 2)", "/ width))", "test_families.py"),
    ("POD mean not removed", "linear.py", "self.mean = x.mean(axis=0)", "self.mean = np.zeros(x.shape[1])", "test_linear.py"),
    ("POD skips the first mode", "linear.py", "self.basis = vt[: self.n].T.copy()", "self.basis = vt[1 : self.n + 1].T.copy()", "test_linear.py"),
    ("POD tail error off by one", "linear.py", "energy[n:].sum() / energy.sum()", "energy[n + 1 :].sum() / energy.sum()", "test_linear.py"),
    ("exponent vectors miss the top degree", "polynomial.py", "yield from ((d,) for d in range(left + 1))", "yield from ((d,) for d in range(left))", "test_polynomial.py"),
    ("Legendre degree shifted by one", "polynomial.py", "out[:, col] *= vander[i][:, d]", "out[:, col] *= vander[i][:, d - 1]", "test_polynomial.py"),
    ("polynomial tie tolerance removed", "polynomial.py", "cut = max(1.01 * min(err for err, _ in scored), 1e-8)", "cut = min(err for err, _ in scored)", "test_polynomial.py"),
    ("polynomial ridge removed", "polynomial.py", "penalty = np.sqrt(self.ridge * np.trace(feats.T @ feats) / n_terms)", "penalty = 0.0", "test_polynomial.py"),
    ("polynomial ROM drops the POD part", "polynomial.py", "return self.pod.decode(codes) + self.decoder(codes)", "return self.decoder(codes)", "test_polynomial.py"),
    ("relative error divides by the prediction", "metrics.py", "np.linalg.norm(truth - pred, axis=1) / np.linalg.norm(truth, axis=1)", "np.linalg.norm(truth - pred, axis=1) / np.linalg.norm(pred, axis=1)", "test_metrics.py"),
    ("Jacobian step factor 2 dropped", "metrics.py", "jac = (both[0] - both[1]) / (2.0 * h[None, :, None])", "jac = (both[0] - both[1]) / h[None, :, None]", "test_metrics.py"),
    ("bootstrap quantiles swapped", "metrics.py", "lo, hi = np.quantile(means, [tail, 1.0 - tail])", "lo, hi = np.quantile(means, [1.0 - tail, tail])", "test_metrics.py"),
    ("bootstrap resamples one index", "metrics.py", "picks = rng.integers(0, len(diff), size=(n_boot, len(diff)))", "picks = np.zeros((n_boot, len(diff)), dtype=int)", "test_metrics.py"),
    ("training keeps the worst validation weights", "training.py", "if val < best_val:", "if val > best_val:", "test_training.py"),
    ("training does not restore the best weights", "training.py", "model.load_state_dict(best_state)", "pass", "test_training.py"),
    ("last epoch not evaluated", "training.py", "if epoch % eval_every == 0 or epoch == epochs:", "if epoch % eval_every == 0:", "test_training.py"),
    ("neural decoder not zero-initialised", "neural.py", "train.shape[1]], zero_last=True)", "train.shape[1]], zero_last=False)", "test_neural.py"),
    ("neural decoder drops the POD part", "neural.py", "return self.pod.decode(codes) + self.scale * to_numpy(out)", "return self.scale * to_numpy(out)", "test_neural.py"),
    ("autoencoder decode forgets the scale", "neural.py", "return self.mean + self.scale * to_numpy(out)", "return self.mean + to_numpy(out)", "test_neural.py"),
    ("initial fields decay like 1/k^2", "operators.py", "a = rng.normal(size=(n, modes)) * amplitude / k", "a = rng.normal(size=(n, modes)) * amplitude / k**2", "test_operators.py"),
    ("spectral layer keeps one mode too many", "operators.py", "out[:, :, : self.modes] = torch.einsum(", "out[:, :, : self.modes + 1] = torch.einsum(", "test_operators.py"),
    ("fixed decoder ignores the spread", "operators.py", "return (z * self.spread) @ self.basis + self.mean", "return z @ self.basis + self.mean", "test_operators.py"),
    ("linear decoder initialisation ignores the spread", "operators.py", "linear.weight.copy_(to_tensor(pod.basis * spread[None, :]))", "linear.weight.copy_(to_tensor(pod.basis))", "test_operators.py"),
    ("dataset output sampled on the wrong points", "operators.py", "out[:, ::oversample]", "out[:, :n_grid]", "test_operators.py"),
    ("selection prefers the highest validation error", "study_operator.py", 'key=lambda r: (r["val_error"], r["n_z"])', 'key=lambda r: (-r["val_error"], r["n_z"])', "test_study_operator.py"),
    ("selection tie goes to the larger dimension", "study_operator.py", 'key=lambda r: (r["val_error"], r["n_z"])', 'key=lambda r: (r["val_error"], -r["n_z"])', "test_study_operator.py"),
    ("comparison sign reversed", "study_operator.py", "per_sample(m, s, a) - per_sample(m, s, b)", "per_sample(m, s, b) - per_sample(m, s, a)", "test_study_operator.py"),
    ("validation fields taken from the training pool", "study_operator.py", "val = (u0[pool : pool + cfg.n_val], u_final[pool : pool + cfg.n_val])", "val = (u0[: cfg.n_val], u_final[: cfg.n_val])", "test_study_operator.py"),
    ("same parameter draw for every split", "study_reduction.py", "params = family.sample(size, rng)", "params = family.sample(size, np.random.default_rng(seed))", "test_study_reduction.py"),
    ("sensitivity in raw instead of unit-spread coordinates", "study_reduction.py", "mu, sd = codes.mean(axis=0), np.maximum(codes.std(axis=0), 1e-12)", "mu, sd = codes.mean(axis=0), np.ones(codes.shape[1])", "test_study_reduction.py"),
    ("smallest dimension reports the largest", "report.py", "str(int(ok.min())) if len(ok)", "str(int(ok.max())) if len(ok)", "test_report.py"),
    ("number format switches at 1e-2", "report.py", "abs(x) >= 1e-3 or x == 0", "abs(x) >= 1e-2 or x == 0", "test_report.py"),
]  # fmt: skip


def run(fault) -> str:
    name, fname, old, new, tests = fault
    path = SRC / fname
    original = path.read_text()
    if original.count(old) != 1:
        return f"INVALID  {name}: the old text was found {original.count(old)} times"
    path.write_text(original.replace(old, new))
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"}
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-x",
                "-q",
                "-p",
                "no:cacheprovider",
                f"tests/{tests}",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        caught = res.returncode != 0
    except subprocess.TimeoutExpired:
        caught = True
    finally:
        path.write_text(original)
    return ("caught   " if caught else "SURVIVED ") + name


def main() -> int:
    lines = [run(f) for f in FAULTS]
    bad = [line for line in lines if not line.startswith("caught")]
    summary = f"{len(lines) - len(bad)} of {len(lines)} hand-made faults caught"
    text = "\n".join([*lines, "", summary]) + "\n"
    print(text, end="")
    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
