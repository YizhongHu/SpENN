"""Tests for TPENLayer composition and runtime checks (TPEN layer contract).

The layer under test composes ``mixing -> path_aggregation -> update`` in
real space with optional normalization/envelope hooks (MIG-TPEN-000 §2.2).
There is no Fourier round-trip and no standalone activation stage: both
compute stages own their activations.

Logged times in this suite use UTC per repository convention.
"""

from __future__ import annotations

import pytest
import torch

from tpen.data.batch import ElectronBatch
from tpen.data.permutation import Permutation
from tpen.data.real import Feature, Interaction, Update, zero_block
from tpen.equivariance import EquivariantMap
from tpen.nn import (
    EquivariantMixing,
    GaussianCoordinateEnvelope,
    PathAggregation,
    RMSNorm,
    ResidualUpdater,
    TPENForwardContext,
    TPENLayer,
    TorchInitializer,
)
from tests.helpers.equivariance import assert_equivariant_all


class IdentityMixing(EquivariantMap):
    """Wrap each feature block as a single-path real interaction."""

    def forward_impl(self, x: Feature) -> Interaction:
        return Interaction([tensor.unsqueeze(2) for tensor in x.blocks])


class TwoPathMixing(EquivariantMap):
    """Produce a two-path order-1 interaction ``[x, 2x]``."""

    def forward_impl(self, x: Feature) -> Interaction:
        return Interaction(
            [
                x.blocks[0].unsqueeze(2),
                torch.stack([x.blocks[1], 2.0 * x.blocks[1]], dim=2),
            ]
        )


class SumPathAggregation(EquivariantMap):
    """Contract the path axis by summation; stub for the learned module."""

    def forward_impl(self, x: Interaction) -> Update:
        return Update([tensor.sum(dim=2) for tensor in x.blocks])


class RecordingRealMap(EquivariantMap):
    def __init__(self, label: str, calls: list[str]) -> None:
        super().__init__()
        self.label = label
        self.calls = calls

    def forward_impl(self, x: Feature) -> Feature:
        self.calls.append(self.label)
        return x


class RecordingRealEnvelope(EquivariantMap):
    def __init__(self, label: str, calls: list[str]) -> None:
        super().__init__()
        self.label = label
        self.calls = calls

    def forward_impl(self, x: Feature, context: TPENForwardContext) -> Feature:
        assert context.batch is not None
        self.calls.append(self.label)
        return x


def test_spenn_layer_scaffold_passes_runtime_equivariance_check() -> None:
    feature = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
        ]
    )
    layer = TPENLayer(
        mixing=IdentityMixing(),
        path_aggregation=SumPathAggregation(),
        update=ResidualUpdater(),
    ).to(dtype=torch.float64)

    output = layer(feature)

    # Single-path identity mixing summed over paths gives u = x, so the
    # residual update yields exactly 2x.
    torch.testing.assert_close(output.blocks[1], 2.0 * feature.blocks[1])
    assert_equivariant_all(layer, feature)


def test_spenn_layer_applies_optional_real_controls_in_declared_order() -> None:
    calls: list[str] = []
    feature = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64),
        ]
    )
    context = TPENForwardContext(batch=object())  # type: ignore[arg-type]
    layer = TPENLayer(
        mixing=TwoPathMixing(),
        path_aggregation=SumPathAggregation(),
        update_normalization=RecordingRealMap("update_normalization", calls),
        update_envelope=RecordingRealEnvelope("update_envelope", calls),
        feature_normalization=RecordingRealMap("feature_normalization", calls),
        feature_envelope=RecordingRealEnvelope("feature_envelope", calls),
        update=ResidualUpdater(),
    )

    layer(feature, context)

    assert calls == [
        "update_normalization",
        "update_envelope",
        "feature_normalization",
        "feature_envelope",
    ]


def test_spenn_layer_envelopes_require_context() -> None:
    feature = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64),
        ]
    )
    layer = TPENLayer(
        mixing=TwoPathMixing(),
        path_aggregation=SumPathAggregation(),
        update_envelope=RecordingRealEnvelope("update_envelope", []),
        update=ResidualUpdater(),
    )

    with pytest.raises(ValueError, match="update_envelope"):
        layer(feature)


def test_spenn_layer_controls_are_equivariant_with_context() -> None:
    feature = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64),
        ]
    )
    batch = ElectronBatch(
        positions=torch.tensor(
            [[[0.0], [1.0], [2.0]]],
            dtype=torch.float64,
        )
    )
    permutation = Permutation((2, 0, 1))
    layer = TPENLayer(
        mixing=TwoPathMixing(),
        path_aggregation=SumPathAggregation(),
        update_normalization=RMSNorm(eps=1.0e-8),
        update_envelope=GaussianCoordinateEnvelope(sigma=2.0),
        feature_normalization=RMSNorm(eps=1.0e-8),
        feature_envelope=GaussianCoordinateEnvelope(sigma=2.0),
        update=ResidualUpdater(),
    )

    output = layer(feature, TPENForwardContext(batch=batch))
    permuted_batch = batch.permute(permutation)
    lhs = layer(feature.permute(permutation), TPENForwardContext(batch=permuted_batch))
    rhs = output.permute(permutation)

    close, comparison = lhs.compare(rhs)
    assert close, comparison


def test_spenn_layer_real_components_pass_forced_runtime_equivariance_check() -> None:
    generator = torch.Generator().manual_seed(24680)
    feature = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 2, 3, generator=generator, dtype=torch.float64),
        ]
    )
    # Real TPEN stack: mixing owns Gamma, aggregation owns Gamma_c; the layer
    # itself adds no standalone activation stage.
    layer = TPENLayer(
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
            initializer=TorchInitializer(seed=24680),
        ),
        update=ResidualUpdater(),
    ).to(dtype=torch.float64)

    output = layer(feature)

    assert output.validate() is output
    assert output.blocks[1].shape == feature.blocks[1].shape
    assert_equivariant_all(layer, feature)


def test_spenn_layer_matches_slow_tpen_reference_layer() -> None:
    # T1 at layer level: the composed module (real mixing with owned Gamma,
    # real aggregation with owned Gamma_c, residual update) must reproduce
    # the slow reference layer exactly, with the same weights. This also pins
    # stage/activation ordering through the composed layer numerically.
    from tests.helpers.tpen_reference import slow_tpen_layer

    generator = torch.Generator().manual_seed(13579)
    feature = Feature(
        [
            zero_block(batch_size=2, dtype=torch.float64),
            torch.randn(2, 2, 3, generator=generator, dtype=torch.float64),
            torch.randn(2, 2, 3, 3, generator=generator, dtype=torch.float64),
        ]
    )
    # The reference applies Gamma post-hoc to an unactivated mixing, so the
    # module-owned Gamma must equal the same function applied afterwards.
    plain_mixing = EquivariantMixing(
        max_order=2,
        implementation="slow",
        channels=2,
        initial_weight=0.5,
    ).to(dtype=torch.float64)
    owned_mixing = EquivariantMixing(
        max_order=2,
        implementation="slow",
        channels=2,
        initial_weight=0.5,
        activation=torch.nn.SiLU(),
    ).to(dtype=torch.float64)
    aggregation = PathAggregation(
        max_order=2,
        channels=2,
        activation=torch.nn.Tanh(),
        initializer=TorchInitializer(seed=13579),
    ).to(dtype=torch.float64)

    layer = TPENLayer(
        mixing=owned_mixing,
        path_aggregation=aggregation,
        update=ResidualUpdater(),
    )

    # Copy the module's per-order aggregation weights into the reference list.
    path_weights: list[torch.Tensor | None] = [None]
    for order in (1, 2):
        path_weights.append(aggregation.weights[f"o{order}"].detach().clone())

    reference = slow_tpen_layer(
        feature,
        mixing=plain_mixing,
        mixing_activation=torch.nn.functional.silu,
        path_weights=path_weights,
        aggregation_activation=torch.tanh,
    )

    matches, stats = layer(feature).compare(reference, atol=1e-12, rtol=1e-12)
    assert matches, f"TPENLayer diverged from slow_tpen_layer: {stats}"
