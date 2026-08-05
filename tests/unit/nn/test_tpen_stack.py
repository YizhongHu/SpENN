"""Tests for the TPENStack container (MIG-TPEN-000 section 2.2, slice c).

The stack owns the ordered TPEN layers of a wavefunction and dispatches the
forward context to :class:`TPENLayer` members only; plain feature-to-feature
modules are called without it. Equivariance of a stack of real layers is
checked exhaustively over small-n permutations (gate T2).

Logged times in this suite use UTC per repository convention.
"""

from __future__ import annotations

import pytest
import torch

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.real import Feature, Interaction, Update, zero_block
from spenn.equivariance import EquivariantMap
from spenn.nn import (
    AdditiveEnvelope,
    EquivariantMixing,
    PathAggregation,
    ResidualUpdater,
    TPENForwardContext,
    TPENLayer,
    TPENWaveFunction,
    TPENStack,
    TorchInitializer,
)
from tests.helpers.equivariance import assert_equivariant_all
from tests.helpers.hooke_models import build_tiny_spenn


class IdentityMixing(EquivariantMap):
    """Wrap each feature block as a single-path real interaction."""

    def forward_impl(self, x: Feature) -> Interaction:
        return Interaction([tensor.unsqueeze(2) for tensor in x.blocks])


class SumPathAggregation(EquivariantMap):
    """Contract the path axis by summation; stub for the learned module."""

    def forward_impl(self, x: Interaction) -> Update:
        return Update([tensor.sum(dim=2) for tensor in x.blocks])


class RecordingScale(EquivariantMap):
    """Scale the order-1 block and record the call order."""

    def __init__(self, label: str, factor: float, calls: list[str]) -> None:
        super().__init__()
        self.label = label
        self.factor = float(factor)
        self.calls = calls

    def forward_impl(self, x: Feature) -> Feature:
        self.calls.append(self.label)
        return Feature([x.blocks[0].clone(), self.factor * x.blocks[1]])


class RecordingContextEnvelope(EquivariantMap):
    """Context-requiring per-layer envelope stub that records the call."""

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.calls = calls

    def forward_impl(self, x: Feature, context: TPENForwardContext) -> Feature:
        assert context.batch is not None
        self.calls.append("layer_envelope")
        return x


class EmptyEncoder(torch.nn.Module):
    def forward(self, batch: ElectronBatch, *, context=None) -> Feature:
        return Feature()


class ConstantReadout(torch.nn.Module):
    def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def _feature() -> Feature:
    return Feature(
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
    context = TPENForwardContext(batch=batch)
    stack = TPENStack(
        [
            RecordingScale("plain", 1.0, calls),
            TPENLayer(
                mixing=IdentityMixing(),
                path_aggregation=SumPathAggregation(),
                update=ResidualUpdater(),
                feature_envelope=RecordingContextEnvelope(calls),
            ),
        ]
    )

    stack(_feature(), context)

    assert calls == ["plain", "layer_envelope"]


@pytest.mark.parametrize("n_particles", [2, 3, 4])
def test_stack_of_real_layers_passes_forced_runtime_equivariance_check(n_particles: int) -> None:
    # T2 at stack level over n in {2, 3, 4} (MIG-TPEN-000 T2): two real TPEN
    # layers (mixing with owned Gamma, aggregation with owned Gamma_c,
    # residual update) with an explicit order-2 path axis, composed by the
    # stack. Exhaustive over every permutation for each particle count.
    generator = torch.Generator().manual_seed(24680 + n_particles)
    feature = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 2, n_particles, generator=generator, dtype=torch.float64),
            torch.randn(1, 2, n_particles, n_particles, generator=generator, dtype=torch.float64),
        ]
    )

    def real_layer(seed: int) -> TPENLayer:
        return TPENLayer(
            mixing=EquivariantMixing(
                max_order=2,
                max_virtual_order=2,
                implementation="slow",
                channels=2,
                initial_weight=0.5,
                activation=torch.nn.SiLU(),
            ),
            path_aggregation=PathAggregation(
                max_order=2,
                max_virtual_order=2,
                channels=2,
                activation=torch.nn.SiLU(),
                initializer=TorchInitializer(seed=seed),
            ),
            update=ResidualUpdater(),
        )

    stack = TPENStack([real_layer(24680), real_layer(13579)]).to(dtype=torch.float64)

    output = stack(feature)

    assert output.validate() is output
    assert output.blocks[1].shape == feature.blocks[1].shape
    assert output.blocks[2].shape == feature.blocks[2].shape
    assert_equivariant_all(stack, feature)


def test_wavefunction_wraps_layers_into_stack() -> None:
    model = build_tiny_spenn()

    assert isinstance(model.stack, TPENStack)
    assert len(model.stack.layers) == 1
    assert not hasattr(model, "layers")


def test_wavefunction_accepts_prebuilt_stack_without_rewrapping() -> None:
    stack = TPENStack([torch.nn.Identity()])
    model = TPENWaveFunction(
        embedding=EmptyEncoder(),
        layers=stack,
        readout=ConstantReadout(),
        envelope=AdditiveEnvelope(),
    )

    assert model.stack is stack


def test_stack_gradients_flow_end_to_end_through_wavefunction() -> None:
    # T12 through the new stack boundary: backward from full-model logabs must
    # reach every mixing and path-aggregation parameter in model.stack with a
    # finite, not-identically-zero gradient. An odd-n batch is used so the
    # readout's order-1 border exercises the order-1 paths too, so every stack
    # parameter contributes; a detach at the stack boundary would zero these.
    model = build_tiny_spenn()
    generator = torch.Generator().manual_seed(11)
    positions = torch.randn(4, 3, 3, generator=generator, dtype=torch.float64)
    spins = torch.tensor([[1.0, 1.0, -1.0]] * 4, dtype=torch.float64)
    batch = ElectronBatch(positions=positions, spins=spins)

    model(batch).logabs.sum().backward()

    stack_params = dict(model.stack.named_parameters())
    assert stack_params, "TPENStack exposes no parameters"
    for name, parameter in stack_params.items():
        assert parameter.grad is not None, f"no gradient for stack parameter {name}"
        assert torch.all(torch.isfinite(parameter.grad)), f"non-finite gradient for {name}"
        assert parameter.grad.abs().sum() > 0, f"identically-zero gradient for stack parameter {name}"
