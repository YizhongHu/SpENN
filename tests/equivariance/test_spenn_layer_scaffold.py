"""Tests for SpENNLayer scaffold composition and runtime checks."""

from __future__ import annotations

import torch

from spenn.data import IrrepFeature, IrrepInteraction, Partition, RealFeature, RealInteraction, RealUpdate, zero_block
from spenn.nn import EquivariantMap, SpENNLayer, Update


class IdentityMixing(EquivariantMap):
    def forward_impl(self, x: RealFeature) -> RealInteraction:
        return RealInteraction([tensor.unsqueeze(2) for tensor in x.blocks])


class IdentityFourier(EquivariantMap):
    def forward_impl(self, x: RealInteraction) -> IrrepInteraction:
        partition = Partition((1,))
        return IrrepInteraction({partition: x.blocks[1].unsqueeze(-1).unsqueeze(-1)})


class IdentityActivation(EquivariantMap):
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
        inverse_fourier=IdentityInverseFourier(),
        update=Update(),
        equivariance_check=True,
        check_probability=1.0,
    )

    output = layer(feature)

    torch.testing.assert_close(output.blocks[1], 2.0 * feature.blocks[1])
