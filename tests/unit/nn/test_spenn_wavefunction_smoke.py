"""Smoke test: a tiny real SpENNWaveFunction maps a pair batch to an output."""

from __future__ import annotations

import torch
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from tests.helpers.hooke_models import build_tiny_spenn, tiny_pair_batch


def _snapshot_state_dict_metadata(
    model: torch.nn.Module,
) -> dict[str, tuple[tuple[int, ...], torch.dtype, torch.device]]:
    return {
        name: (tuple(tensor.shape), tensor.dtype, tensor.device)
        for name, tensor in model.state_dict().items()
    }


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
    batch_n2 = tiny_pair_batch(n_walkers=4)
    batch_n4 = ElectronBatch(
        positions=torch.randn(4, 4, 3, generator=torch.Generator().manual_seed(123), dtype=torch.float64),
        spins=torch.tensor([[1.0, 1.0, -1.0, -1.0]] * 4, dtype=torch.float64),
    )

    for _name, parameter in model.named_parameters():
        assert not isinstance(parameter, UninitializedParameter)
    for _name, buffer in model.named_buffers():
        assert not isinstance(buffer, UninitializedBuffer)
    before = _snapshot_state_dict_metadata(model)

    model(batch_n2)
    after_n2 = _snapshot_state_dict_metadata(model)

    model(batch_n4)
    after_n4 = _snapshot_state_dict_metadata(model)

    assert after_n2 == before
    assert after_n4 == before
