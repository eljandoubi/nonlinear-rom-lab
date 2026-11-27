import numpy as np
import pytest
import torch

from romlab.training import fit, mlp, to_numpy, to_tensor


def _line_data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g, dtype=torch.float64) * 2 - 1
    return x, torch.sin(3 * x)


def test_mlp_has_the_requested_shape_and_activations():
    net = mlp([3, 8, 8, 2])
    kinds = [type(m).__name__ for m in net]
    assert kinds == ["Linear", "GELU", "Linear", "GELU", "Linear"]
    assert net(torch.zeros(5, 3)).shape == (5, 2)


def test_zero_last_layer_gives_a_zero_network():
    net = mlp([3, 8, 2], zero_last=True)
    assert torch.count_nonzero(net(torch.randn(4, 3))) == 0


def test_array_roundtrip_keeps_float64():
    a = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    t = to_tensor(a)
    assert t.dtype == torch.float64
    assert np.array_equal(to_numpy(t), a.astype(np.float64))


def test_fit_lowers_the_loss():
    torch.manual_seed(0)
    net = mlp([1, 32, 1]).double()
    x, y = _line_data()
    hist = fit(net, x, y, epochs=400, lr=1e-2, eval_every=1)
    assert hist.train_loss[-1] < 1e-3 * hist.train_loss[0]
    assert hist.epochs[-1] == 400 and len(hist.epochs) == 400


def test_fit_is_reproducible_with_minibatches():
    x, y = _line_data(48)

    def run():
        torch.manual_seed(1)
        net = mlp([1, 16, 1]).double()
        fit(net, x, y, epochs=60, lr=5e-3, batch_size=16, seed=7)
        return torch.cat([p.flatten() for p in net.parameters()])

    assert torch.equal(run(), run())


def test_minibatch_seed_changes_the_result():
    x, y = _line_data(48)

    def run(seed):
        torch.manual_seed(1)
        net = mlp([1, 16, 1]).double()
        fit(net, x, y, epochs=30, lr=5e-3, batch_size=16, seed=seed)
        return torch.cat([p.flatten() for p in net.parameters()])

    assert not torch.equal(run(1), run(2))


def test_best_validation_weights_are_restored():
    # Eight noisy points are easy to overfit, so validation loss turns up well before the end.
    g = torch.Generator().manual_seed(3)
    x = torch.rand(8, 1, generator=g, dtype=torch.float64) * 2 - 1
    y = torch.sin(3 * x) + 0.3 * torch.randn(8, 1, generator=g, dtype=torch.float64)
    xv = torch.linspace(-1, 1, 50, dtype=torch.float64)[:, None]
    yv = torch.sin(3 * xv)
    torch.manual_seed(0)
    net = mlp([1, 64, 64, 1]).double()
    hist = fit(net, x, y, xv, yv, epochs=1500, lr=1e-2, eval_every=10)
    with torch.no_grad():
        kept = float(torch.mean((net(xv) - yv) ** 2))
    assert kept == pytest.approx(hist.best_val, rel=1e-12)
    assert hist.best_val == pytest.approx(min(hist.val_loss), rel=1e-12)
    assert hist.best_epoch < 1500
    assert hist.val_loss[-1] > hist.best_val


def test_without_validation_the_last_weights_are_kept():
    torch.manual_seed(0)
    net = mlp([1, 8, 1]).double()
    x, y = _line_data()
    hist = fit(net, x, y, epochs=20, eval_every=10)
    assert hist.best_epoch == 20
    assert hist.val_loss == []
    assert np.isnan(hist.best_val)
