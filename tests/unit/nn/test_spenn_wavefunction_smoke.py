"""Smoke test: a tiny real SpENNWaveFunction maps a pair batch to an output."""

from __future__ import annotations

import torch
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from spenn.data.batch import WavefunctionOutput
from tests.helpers.hooke_models import build_tiny_spenn, tiny_pair_batch


def test_forward_returns_finite_wavefunction_output() -> None:
    model = build_tiny_spenn()
    batch = tiny_pair_batch(n_walkers=4)

    output = model(batch)

    assert isinstance(output, WavefunctionOutput)
    assert output.logabs.shape == (4,)
    assert output.sign.shape == (4,)
    assert torch.isfinite(output.logabs).all()
    assert torch.isfinite(output.sign).all()
    assert torch.all((output.sign == 1) | (output.sign == -1) | (output.sign == 0))


def test_tiny_spenn_initializes_stock_parameters_before_first_forward() -> None:
    model = build_tiny_spenn()
    batch = tiny_pair_batch(n_walkers=4)

    for _name, parameter in model.named_parameters():
        assert not isinstance(parameter, UninitializedParameter)
    for _name, buffer in model.named_buffers():
        assert not isinstance(buffer, UninitializedBuffer)
    state_keys_before = tuple(model.state_dict().keys())

    model(batch)

    assert tuple(model.state_dict().keys()) == state_keys_before
