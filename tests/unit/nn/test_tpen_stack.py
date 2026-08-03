"""Tests for the TPENStack container (MIG-TPEN-000 section 2.2, slice c).

The stack owns the ordered TPEN layers of a wavefunction and dispatches the
forward context to :class:`SpENNLayer` members only; plain feature-to-feature
modules are called without it. Equivariance of a stack of real layers is
checked exhaustively over small-n permutations (gate T2).

Logged times in this suite use UTC per repository convention.
"""

from __future__ import annotations

import torch

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.real import RealFeature, RealInteraction, RealUpdate, zero_block
from spenn.equivariance import EquivariantMap
from spenn.nn import (
    AdditiveEnvelope,
    EquivariantMixing,
    PathAggregation,
    ResidualUpdate,
    SpENNForwardContext,
    SpENNLayer,
    SpENNWaveFunction,
    TPENStack,
    TorchInitializer,
)
from tests.helpers.equivariance import assert_equivariant_all
from tests.helpers.hooke_models import build_tiny_spenn


class IdentityMixing(EquivariantMap):
    """Wrap each feature block as a single-path real interaction."""

    def forward_impl(self, x: RealFeature) -> RealInteraction:
        return RealInteraction([tensor.unsqueeze(2) for tensor in x.blocks])


class SumPathAggregation(EquivariantMap):
    """Contract the path axis by summation; stub for the learned module."""

    def forward_impl(self, x: RealInteraction) -> RealUpdate:
        return RealUpdate([tensor.sum(dim=2) for tensor in x.blocks])


class RecordingScale(EquivariantMap):
    """Scale the order-1 block and record the call order."""

    def __init__(self, label: str, factor: float, calls: list[str]) -> None:
        super().__init__()
        self.label = label
        self.factor = float(factor)
        self.calls = calls

    def forward_impl(self, x: RealFeature) -> RealFeature:
        self.calls.append(self.label)
        return RealFeature([x.blocks[0].clone(), self.factor * x.blocks[1]])


class RecordingContextEnvelope(EquivariantMap):
    """Context-requiring per-layer envelope stub that records the call."""

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls

    def forward_impl(self, x: RealFeature, context: SpENNForwardContext) -> RealFeature:
        assert context.batch is not None
        self.calls.append("layer_envelope")
        return x


class EmptyEncoder(torch.nn.Module):
    def forward(self, batch: ElectronBatch, *, context=None) -> RealFeature:
        return RealFeature()


class ConstantReadout(torch.nn.Module):
    def forward(self, features: RealFeature, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def _feature() -> RealFeature:
    return RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64),
        ]
    )


def test_stack_applies_layers_in_declaration_order() -> None:
    calls: list[str] = []
    stack = TPENStack(
        [
            RecordingScale("first", 2.0, calls),
            RecordingScale("second", 5.0, calls),
        ]
    )

    output = stack(_feature())

    assert calls == ["first", "second"]
    torch.testing.assert_close(output.blocks[1], 10.0 * _feature().blocks[1])


def test_stack_dispatches_context_to_spenn_layers_only() -> None:
    calls: list[str] = []
    batch = ElectronBatch(positions=torch.tensor([[[0.0], [1.0], [2.0]]], dtype=torch.float64))
    context = SpENNForwardContext(batch=batch)
    stack = TPENStack(
        [
            RecordingScale("plain", 1.0, calls),
            SpENNLayer(
                mixing=IdentityMixing(),
                path_aggregation=SumPathAggregation(),
                update=ResidualUpdate(),
                feature_envelope=RecordingContextEnvelope(calls),
            ),
        ]
    )

    stack(_feature(), context)

    assert calls == ["plain", "layer_envelope"]


def test_stack_of_real_layers_passes_forced_runtime_equivariance_check() -> None:
    # T2 at stack level: two real TPEN layers (mixing with owned Gamma,
    # aggregation with owned Gamma_c, residual update) composed by the stack.
    generator = torch.Generator().manual_seed(24680)
    feature = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 2, 3, generator=generator, dtype=torch.float64),
        ]
    )

    def real_layer(seed: int) -> SpENNLayer:
        return SpENNLayer(
            mixing=EquivariantMixing(
                max_order=1,
                max_virtual_order=1,
                implementation="vectorized",
                channels=2,
                initial_weight=0.5,
                activation=torch.nn.SiLU(),
            ),
            path_aggregation=PathAggregation(
                max_order=1,
                max_virtual_order=1,
                channels=2,
                activation=torch.nn.SiLU(),
                initializer=TorchInitializer(seed=seed),
            ),
            update=ResidualUpdate(),
        )

    stack = TPENStack([real_layer(24680), real_layer(13579)]).to(dtype=torch.float64)

    output = stack(feature)

    assert output.validate() is output
    assert output.blocks[1].shape == feature.blocks[1].shape
    assert_equivariant_all(stack, feature)


def test_wavefunction_wraps_layers_into_stack() -> None:
    model = build_tiny_spenn()

    assert isinstance(model.stack, TPENStack)
    assert len(model.stack.layers) == 1
    assert not hasattr(model, "layers")


def test_wavefunction_accepts_prebuilt_stack_without_rewrapping() -> None:
    stack = TPENStack([torch.nn.Identity()])
    model = SpENNWaveFunction(
        embedding=EmptyEncoder(),
        layers=stack,
        readout=ConstantReadout(),
        envelope=AdditiveEnvelope(),
    )

    assert model.stack is stack
