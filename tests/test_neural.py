import numpy as np
import pytest

from romlab.families import get_family
from romlab.linear import POD
from romlab.metrics import mean_relative_l2
from romlab.neural import Autoencoder, NeuralDecoderROM


def _pulse_split(n_train=60, n_val=20, n_test=30, n_grid=64):
    fam = get_family("pulse", n_grid)
    rng = np.random.default_rng(0)
    train = fam.solve(fam.sample(n_train, rng))
    val = fam.solve(fam.sample(n_val, rng))
    test = fam.solve(fam.sample(n_test, rng))
    return train, val, test


def test_neural_decoder_starts_from_pod():
    train, val, test = _pulse_split()
    rom = NeuralDecoderROM(2, epochs=1, lr=1e-12).fit(train, val)
    assert np.allclose(rom.reconstruct(test), POD(2).fit(train).reconstruct(test), atol=1e-8)


def test_neural_decoder_beats_pod_on_a_transport_family():
    train, val, test = _pulse_split()
    pod_err = mean_relative_l2(test, POD(1).fit(train).reconstruct(test))
    rom = NeuralDecoderROM(1, epochs=600, seed=0).fit(train, val)
    assert mean_relative_l2(test, rom.reconstruct(test)) < 0.7 * pod_err


def test_autoencoder_learns_something_better_than_the_mean_field():
    train, val, test = _pulse_split()
    mean_err = mean_relative_l2(test, np.tile(train.mean(axis=0), (len(test), 1)))
    ae = Autoencoder(2, epochs=600, seed=0).fit(train, val)
    assert mean_relative_l2(test, ae.reconstruct(test)) < 0.5 * mean_err


@pytest.mark.parametrize("cls", [NeuralDecoderROM, Autoencoder])
def test_shapes_and_single_field_input(cls):
    train, val, test = _pulse_split(n_train=20, n_val=6, n_test=5)
    model = cls(3, epochs=5).fit(train, val)
    assert model.encode(test).shape == (5, 3)
    assert model.decode(model.encode(test)).shape == (5, 64)
    assert model.encode(test[0]).shape == (1, 3)
    assert model.decode(np.zeros(3)).shape == (1, 64)


@pytest.mark.parametrize("cls", [NeuralDecoderROM, Autoencoder])
def test_same_seed_same_model_and_other_seed_other_model(cls):
    train, val, test = _pulse_split(n_train=20, n_val=6, n_test=5)
    a = cls(2, epochs=20, seed=1).fit(train, val).reconstruct(test)
    b = cls(2, epochs=20, seed=1).fit(train, val).reconstruct(test)
    c = cls(2, epochs=20, seed=2).fit(train, val).reconstruct(test)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


@pytest.mark.parametrize("cls", [NeuralDecoderROM, Autoencoder])
def test_history_records_the_selected_epoch(cls):
    train, val, _ = _pulse_split(n_train=20, n_val=6)
    model = cls(2, epochs=60).fit(train, val)
    assert 1 <= model.history.best_epoch <= 60
    assert model.history.best_val == pytest.approx(min(model.history.val_loss))
