"""Small helpers to train networks on tensors: MLP builder, training loop, array conversion."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


def mlp(sizes: list[int] | tuple[int, ...], zero_last: bool = False) -> nn.Sequential:
    """Fully connected network with GELU between layers and a linear last layer."""
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    net = nn.Sequential(*layers)
    if zero_last:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


def to_tensor(array: np.ndarray, device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=torch.float64, device=device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


@dataclass
class History:
    """Losses recorded during training and the epoch of the kept weights."""

    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("nan")


def fit(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    *,
    epochs: int = 2000,
    lr: float = 2e-3,
    final_lr_factor: float = 1e-2,
    batch_size: int | None = None,
    eval_every: int = 50,
    seed: int = 0,
) -> History:
    """Minimise the mean squared error ``model(x) ~ y`` with Adam and a cosine schedule.

    Batches are the whole set unless ``batch_size`` is given. When validation tensors are given,
    the weights with the lowest validation loss (checked every ``eval_every`` epochs and at the
    end) are restored when training stops.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * final_lr_factor
    )
    gen = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    size = n if batch_size is None else min(batch_size, n)
    has_val = x_val is not None and y_val is not None
    hist = History()
    best_state = None
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n, generator=gen).to(x.device) if size < n else None
        total = torch.zeros((), dtype=x.dtype, device=x.device)
        for start in range(0, n, size):
            idx = slice(None) if order is None else order[start : start + size]
            loss = torch.mean((model(x[idx]) - y[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach() * (n if order is None else len(order[start : start + size]))
        sched.step()
        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            hist.epochs.append(epoch)
            hist.train_loss.append(float(total) / n)
            if has_val:
                with torch.no_grad():
                    val = float(torch.mean((model(x_val) - y_val) ** 2))
                hist.val_loss.append(val)
                if val < best_val:
                    best_val, hist.best_epoch = val, epoch
                    best_state = copy.deepcopy(model.state_dict())
    if has_val and best_state is not None:
        model.load_state_dict(best_state)
        hist.best_val = best_val
    else:
        hist.best_epoch = epochs
    model.eval()
    return hist
