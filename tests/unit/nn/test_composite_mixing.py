"""Tests for static linear/TP interaction composition."""

from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from tpen.data.paths import (
    LinearPathMetadata,
    NormalizedChannels,
    NormalizedOrders,
    PathMetadata,
    compose_path_layout,
)
from tpen.data.real import Feature, Interaction, zero_block
from tpen.nn import CompositeMixing, EquivariantMixing, LinearEquivariantMixing


def _layout():
    linear = LinearPathMetadata.generate(max_order=1)
    tensor_product = PathMetadata.generate(max_order=1, max_virtual_order=1, output_embedding="canonical")
    return compose_path_layout(
        linear=linear,
        tensor_product=tensor_product,
        input_orders=NormalizedOrders((1,)),
        output_orders=NormalizedOrders((1,)),
        input_channels=NormalizedChannels(((1, 1),)),
        output_channels=NormalizedChannels(((1, 1),)),
    ), linear, tensor_product


def _feature() -> Feature:
    values = torch.tensor([[[1.0, 2.0, 4.0]]], dtype=torch.float64)
    return Feature([zero_block(batch_size=1, dtype=values.dtype), values])


def test_hybrid_composition_has_hand_computed_union_values() -> None:
    layout, linear_metadata, tp_metadata = _layout()
    linear = LinearEquivariantMixing(max_order=1, channels=1, metadata=linear_metadata).to(dtype=torch.float64)
    tensor_product = EquivariantMixing(
        max_order=1, channels=1, paths=tp_metadata, aggregation="sum", activation=None
    ).to(dtype=torch.float64)
    composite = CompositeMixing(layout=layout, producers=(linear, tensor_product), activation=lambda value: value + 1)

    actual = composite(_feature()).blocks[1]
    # Linear identity, linear completion-mean, then TP x*x; common Gamma is +1.
    expected = torch.tensor(
        [[[[2.0, 3.0, 5.0], [4.0, 3.5, 2.5], [2.0, 5.0, 17.0]]]], dtype=torch.float64
    )
    torch.testing.assert_close(actual, expected)


def test_composite_slices_match_standalone_raw_producers() -> None:
    layout, linear_metadata, tp_metadata = _layout()
    linear = LinearEquivariantMixing(max_order=1, channels=1, metadata=linear_metadata)
    tensor_product = EquivariantMixing(max_order=1, channels=1, paths=tp_metadata, activation=None)
    composite = CompositeMixing(layout=layout, producers=(linear, tensor_product))
    standalone_linear = linear(_feature()).blocks[1]
    standalone_tp = tensor_product.forward_pre_activation(_feature()).blocks[1]
    actual = composite(_feature()).blocks[1]
    torch.testing.assert_close(actual[:, :, :2], standalone_linear)
    torch.testing.assert_close(actual[:, :, 2:], standalone_tp)


def test_composite_rejects_tp_owned_activation() -> None:
    layout, linear_metadata, tp_metadata = _layout()
    linear = LinearEquivariantMixing(max_order=1, channels=1, metadata=linear_metadata)
    tensor_product = EquivariantMixing(max_order=1, channels=1, paths=tp_metadata, activation=torch.tanh)
    with pytest.raises(ValueError, match="pre-Gamma"):
        CompositeMixing(layout=layout, producers=(linear, tensor_product), activation=torch.tanh)


def test_composite_uses_one_pre_activation_interface() -> None:
    layout, linear_metadata, tp_metadata = _layout()
    class CountingLinear(LinearEquivariantMixing):
        calls = 0

        def forward_pre_activation(self, x: Feature) -> Interaction:
            self.calls += 1
            return super().forward_pre_activation(x)

    class CountingTP(EquivariantMixing):
        calls = 0

        def forward_pre_activation(self, x: Feature) -> Interaction:
            self.calls += 1
            return super().forward_pre_activation(x)

    linear = CountingLinear(max_order=1, channels=1, metadata=linear_metadata)
    tensor_product = CountingTP(max_order=1, channels=1, paths=tp_metadata, activation=None)
    CompositeMixing(layout=layout, producers=(linear, tensor_product), activation=torch.tanh)(_feature())
    assert linear.calls == 1
    assert tensor_product.calls == 1


def test_composite_rejects_disagreeing_order_zero_blocks() -> None:
    layout, linear_metadata, tp_metadata = _layout()

    class DivergentLinear(LinearEquivariantMixing):
        def forward_pre_activation(self, x: Feature) -> Interaction:
            result = super().forward_pre_activation(x)
            # A valid Interaction's reserved order-0 block has zero elements;
            # use a producer-returned block container to test value divergence
            # without changing dtype, shape, or device.
            blocks = [torch.zeros((1, 1), dtype=torch.float64), *result.blocks[1:]]
            return SimpleNamespace(blocks=blocks)

    class DivergentTP(EquivariantMixing):
        def forward_pre_activation(self, x: Feature) -> Interaction:
            result = super().forward_pre_activation(x)
            blocks = [torch.ones((1, 1), dtype=torch.float64), *result.blocks[1:]]
            return SimpleNamespace(blocks=blocks)

    linear = DivergentLinear(max_order=1, channels=1, metadata=linear_metadata)
    tensor_product = DivergentTP(max_order=1, channels=1, paths=tp_metadata, activation=None)
    composite = CompositeMixing(layout=layout, producers=(linear, tensor_product))
    with pytest.raises(ValueError, match="order-0 blocks"):
        composite(_feature())


def test_composite_rejects_dtype_mismatch_independently() -> None:
    layout, linear_metadata, tp_metadata = _layout()

    class DtypeLinear(LinearEquivariantMixing):
        def forward_pre_activation(self, x: Feature) -> Interaction:
            result = super().forward_pre_activation(x)
            blocks = [result.blocks[0].to(dtype=torch.float32), *result.blocks[1:]]
            return SimpleNamespace(blocks=blocks)

    linear = DtypeLinear(max_order=1, channels=1, metadata=linear_metadata)
    tensor_product = EquivariantMixing(max_order=1, channels=1, paths=tp_metadata, activation=None)
    composite = CompositeMixing(layout=layout, producers=(linear, tensor_product))
    with pytest.raises(ValueError, match="dtypes disagree"):
        composite(_feature())


def test_tp_only_composite_keeps_legacy_state_keys_and_numerics() -> None:
    _, _, tp_metadata = _layout()
    layout = compose_path_layout(
        linear=None,
        tensor_product=tp_metadata,
        input_orders=NormalizedOrders((1,)),
        output_orders=NormalizedOrders((1,)),
        input_channels=NormalizedChannels(((1, 1),)),
        output_channels=NormalizedChannels(((1, 1),)),
    )
    legacy = EquivariantMixing(max_order=1, channels=1, paths=tp_metadata, activation=torch.tanh)
    raw = EquivariantMixing(max_order=1, channels=1, paths=tp_metadata, activation=None)
    composite = CompositeMixing(layout=layout, producers=(raw,), activation=torch.tanh)
    torch.testing.assert_close(legacy(_feature()).blocks[1], composite(_feature()).blocks[1])
    assert tuple(composite.state_dict()) == tuple(legacy.state_dict())
