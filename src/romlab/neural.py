"""Neural reduced representations: a neural decoder on POD coordinates and an autoencoder."""

from __future__ import annotations

import numpy as np
import torch

from romlab.linear import POD
from romlab.training import History, fit, mlp, to_numpy, to_tensor


class NeuralDecoderROM:
    """POD encoder with a neural decoder of the part of the fields outside the basis.

    The decoder is ``mean + basis a + scale * net(a / code_scale)`` with the last layer of the
    network initialised at zero, so training starts from the POD reconstruction and the network
    only has to learn what the basis misses.
    """

    def __init__(
        self,
        n: int,
        hidden: tuple[int, ...] = (128, 256),
        epochs: int = 3000,
        lr: float = 2e-3,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.n, self.hidden, self.epochs, self.lr = n, hidden, epochs, lr
        self.seed, self.device = seed, device
        self.history: History | None = None

    def fit(self, train: np.ndarray, val: np.ndarray | None = None) -> NeuralDecoderROM:
        self.pod = POD(self.n).fit(train)
        codes = self.pod.encode(train)
        self.code_scale = np.maximum(np.abs(codes).max(axis=0), 1e-12)
        self.scale = max(float(train.std()), 1e-12)
        outside = train - self.pod.decode(codes)
        torch.manual_seed(self.seed)
        self.net = mlp([self.n, *self.hidden, train.shape[1]], zero_last=True)
        self.net.to(self.device, torch.float64)
        x, y = (
            to_tensor(codes / self.code_scale, self.device),
            to_tensor(outside / self.scale, self.device),
        )
        xv = yv = None
        if val is not None:
            val_codes = self.pod.encode(val)
            xv = to_tensor(val_codes / self.code_scale, self.device)
            yv = to_tensor((val - self.pod.decode(val_codes)) / self.scale, self.device)
        self.history = fit(self.net, x, y, xv, yv, epochs=self.epochs, lr=self.lr, seed=self.seed)
        return self

    def encode(self, fields: np.ndarray) -> np.ndarray:
        return self.pod.encode(fields)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.atleast_2d(codes)
        with torch.no_grad():
            out = self.net(to_tensor(codes / self.code_scale, self.device))
        return self.pod.decode(codes) + self.scale * to_numpy(out)

    def reconstruct(self, fields: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(fields))


class Autoencoder:
    """Fully connected autoencoder with a latent layer of size ``n`` and no linear shortcut."""

    def __init__(
        self,
        n: int,
        hidden: tuple[int, ...] = (256, 128),
        epochs: int = 3000,
        lr: float = 2e-3,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.n, self.hidden, self.epochs, self.lr = n, hidden, epochs, lr
        self.seed, self.device = seed, device
        self.history: History | None = None

    def fit(self, train: np.ndarray, val: np.ndarray | None = None) -> Autoencoder:
        self.mean = train.mean(axis=0)
        self.scale = max(float(train.std()), 1e-12)
        n_grid = train.shape[1]
        torch.manual_seed(self.seed)
        self.encoder = mlp([n_grid, *self.hidden, self.n]).to(self.device, torch.float64)
        self.decoder = mlp([self.n, *reversed(self.hidden), n_grid]).to(self.device, torch.float64)
        model = torch.nn.Sequential(self.encoder, self.decoder)
        x = to_tensor((train - self.mean) / self.scale, self.device)
        xv = None if val is None else to_tensor((val - self.mean) / self.scale, self.device)
        self.history = fit(model, x, x, xv, xv, epochs=self.epochs, lr=self.lr, seed=self.seed)
        return self

    def encode(self, fields: np.ndarray) -> np.ndarray:
        fields = np.atleast_2d(fields)
        with torch.no_grad():
            return to_numpy(self.encoder(to_tensor((fields - self.mean) / self.scale, self.device)))

    def decode(self, codes: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.decoder(to_tensor(np.atleast_2d(codes), self.device))
        return self.mean + self.scale * to_numpy(out)

    def reconstruct(self, fields: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(fields))
