"""Tests for SpENNLayer scaffold composition and runtime checks."""

from __future__ import annotations

import torch

from spenn.equivariance import EquivariantMap
from spenn.data.irrep import IrrepFeature, IrrepInteraction
from spenn.data.partition import Partition
from spenn.data.real import RealFeature, RealInteraction, RealUpdate, zero_block
from spenn.nn import (
    ActivationByType,
    ChannelMappedUpdate,
    EquivariantMixing,
    PathAggregation,
    ReplaceUpdate,
    SpENNLayer,
)
from spenn.reps import FourierTransform, InverseFourierTransform
from tests.helpers.equivariance import assert_equivariant_all


class IdentityMixing(EquivariantMap):
    def forward_impl(self, x: RealFeature) -> RealInteraction:
        return RealInteraction([tensor.unsqueeze(2) for tensor in x.blocks])


class TwoPathMixing(EquivariantMap):
    def forward_impl(self, x: RealFeature) -> RealInteraction:
        return RealInteraction(
            [
                x.blocks[0].unsqueeze(2),
                torch.stack([x.blocks[1], 2.0 * x.blocks[1]], dim=2),
            ]
        )


class IdentityFourier(EquivariantMap):
    def forward_impl(self, x: RealInteraction) -> IrrepInteraction:
        partition = Partition((1,))
        return IrrepInteraction({partition: x.blocks[1].unsqueeze(-1).unsqueeze(-1)})


class IdentityActivation(EquivariantMap):
    def forward_impl(self, x: IrrepInteraction) -> IrrepInteraction:
        return x.clone()


class SquareActivation(EquivariantMap):
    def forward_impl(self, x: IrrepInteraction) -> IrrepInteraction:
        return IrrepInteraction({partition: tensor.square() for partition, tensor in x.items()})


class SumPathAggregation(EquivariantMap):
    def forward_impl(self, x: IrrepInteraction) -> IrrepFeature:
        return IrrepFeature({partition: tensor.sum(dim=2) for partition, tensor in x.items()})


class IdentityInverseFourier(EquivariantMap):
    def forward_impl(self, x: IrrepFeature) -> RealUpdate:
        tensor = next(iter(x.blocks.values())).squeeze(-1).squeeze(-1)
        return RealUpdate(
            [
                zero_block(batch_size=tensor.shape[0], device=tensor.device, dtype=tensor.dtype),
                tensor,
            ]
        )


def test_spenn_layer_scaffold_passes_runtime_equivariance_check() -> None:
    feature = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
        ]
    )
    layer = SpENNLayer(
        mixing=IdentityMixing(),
        fourier=IdentityFourier(),
        activation=IdentityActivation(),
        path_aggregation=SumPathAggregation(),
        inverse_fourier=IdentityInverseFourier(),
        update=ChannelMappedUpdate(max_order=1, channels=2),
    )

    output = layer(feature)

    torch.testing.assert_close(output.blocks[1], 2.0 * feature.blocks[1])
    assert_equivariant_all(layer, feature)


def test_spenn_layer_applies_activation_before_path_aggregation() -> None:
    feature = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64),
        ]
    )
    layer = SpENNLayer(
        mixing=TwoPathMixing(),
        fourier=IdentityFourier(),
        activation=SquareActivation(),
        path_aggregation=SumPathAggregation(),
        inverse_fourier=IdentityInverseFourier(),
        update=ReplaceUpdate(),
    )

    output = layer(feature)

    torch.testing.assert_close(output.blocks[1], 5.0 * feature.blocks[1].square())


def test_spenn_layer_real_components_pass_forced_runtime_equivariance_check() -> None:
    generator = torch.Generator().manual_seed(24680)
    feature = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 2, 3, generator=generator, dtype=torch.float64),
        ]
    )
    partition = Partition((1,))
    torch.manual_seed(24680)
    layer = SpENNLayer(
        mixing=EquivariantMixing(
            max_order=1,
            max_virtual_order=1,
            implementation="vectorized",
            channels=2,
            initial_weight=0.5,
        ),
        fourier=FourierTransform(partitions=(partition,)),
        activation=ActivationByType(symmetric_activation=torch.nn.Tanh()),
        path_aggregation=PathAggregation(
            max_order=1,
            channels=2,
            channel_out_by_order=2,
            path_counts_by_order={1: 1},
            partitions=(partition,),
        ),
        inverse_fourier=InverseFourierTransform(partitions=(partition,)),
        update=ChannelMappedUpdate(max_order=1, channels=2),
    )

    output = layer(feature)

    assert output.validate() is output
    assert output.blocks[1].shape == feature.blocks[1].shape
    assert_equivariant_all(layer, feature)
