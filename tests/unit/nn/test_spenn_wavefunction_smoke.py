"""Smoke test: a tiny real TPENWaveFunction maps a pair batch to an output."""

from __future__ import annotations

import torch
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.real import Feature
from spenn.equivariance import EquivariantMap
from spenn.nn import AdditiveEnvelope, Embedding, TPENForwardContext, TPENWaveFunction
from tests.helpers.hooke_models import build_tiny_spenn, tiny_pair_batch


def _snapshot_state_dict_metadata(
    model: torch.nn.Module,
) -> dict[str, tuple[tuple[int, ...], torch.dtype, torch.device]]:
    return {
        name: (tuple(tensor.shape), tensor.dtype, tensor.device)
        for name, tensor in model.state_dict().items()
    }


class SliceTupleInputs(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[..., :1]


class FillOneBody(EquivariantMap):
    def __init__(self, value: float, calls: list[str], label: str) -> None:
        super().__init__()
        self.value = float(value)
        self.calls = calls
        self.label = label

    def forward_impl(self, features: Feature) -> Feature:
        self.calls.append(self.label)
        return Feature(
            [
                features.blocks[0].clone(),
                torch.full_like(features.blocks[1], self.value),
            ]
        )


class RecordingEnvelope(EquivariantMap):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls

    def forward_impl(self, features: Feature, context: TPENForwardContext) -> Feature:
        assert context.batch is not None
        self.calls.append("embedding_envelope")
        return features


class RecordingLayer(torch.nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls

    def forward(self, features: Feature) -> Feature:
        self.calls.append("layer")
        torch.testing.assert_close(features.blocks[1], torch.full_like(features.blocks[1], 3.0))
        return Feature(
            [
                features.blocks[0].clone(),
                torch.full_like(features.blocks[1], 5.0),
            ]
        )


class RecordingReadout(torch.nn.Module):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls

    def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
        self.calls.append("readout")
        torch.testing.assert_close(features.blocks[1], torch.full_like(features.blocks[1], 5.0))
        logabs = features.blocks[1].sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


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


def test_wavefunction_passes_context_to_embedding_and_readout_sees_layer_output() -> None:
    calls: list[str] = []
    batch = ElectronBatch(positions=torch.tensor([[[1.0], [2.0]]], dtype=torch.float64))
    model = TPENWaveFunction(
        embedding=Embedding(
            max_order=1,
            spatial_dim=1,
            mlps={1: SliceTupleInputs()},
            include_spins=False,
            embedding_normalization=FillOneBody(3.0, calls, "embedding_normalization"),
            embedding_envelope=RecordingEnvelope(calls),
        ),
        layers=[RecordingLayer(calls)],
        readout=RecordingReadout(calls),
        envelope=AdditiveEnvelope(),
    )

    output = model(batch)

    assert calls == ["embedding_normalization", "embedding_envelope", "layer", "readout"]
    torch.testing.assert_close(output.logabs, torch.tensor([10.0], dtype=torch.float64))
