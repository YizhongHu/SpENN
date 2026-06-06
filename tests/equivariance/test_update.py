"""Tests for real-feature update equivariance."""

from __future__ import annotations

import pytest
import torch

from spenn.data import EquivariantMap
from spenn.data.real import RealFeature, RealUpdate, zero_block
from spenn.nn import (
    ChannelMappedUpdate,
    NormGatedUpdate,
    ReplaceUpdate,
    ResidualUpdate,
    Update,
)


def test_update_passes_runtime_equivariance_check() -> None:
    feature = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
        ]
    )
    update = RealUpdate(
        [
            zero_block(dtype=torch.float64),
            torch.ones(1, 3, 3, dtype=torch.float64),
        ]
    )
    module = ChannelMappedUpdate(initial_weight=0.25, equivariance_check=True, check_probability=1.0)

    output = module(feature, update)

    assert output.blocks[1].shape == feature.blocks[1].shape
    assert module.channel_maps["1"].shape == (2, 3)


def _feature_and_update() -> tuple[RealFeature, RealUpdate]:
    feature = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
            torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3),
        ]
    )
    update = RealUpdate(
        [
            zero_block(dtype=torch.float64),
            torch.ones(1, 2, 3, dtype=torch.float64),
            torch.full((1, 2, 3, 3), 0.5, dtype=torch.float64),
        ]
    )
    return feature, update


@pytest.mark.parametrize("module_cls", [ReplaceUpdate, ResidualUpdate, NormGatedUpdate, ChannelMappedUpdate])
def test_update_strategy_scaffolds_are_equivariant_maps(module_cls) -> None:
    feature, update = _feature_and_update()
    module = module_cls(equivariance_check=True, check_probability=1.0)

    output = module(feature, update)

    assert isinstance(module, EquivariantMap)
    assert isinstance(output, RealFeature)
    assert [tuple(block.shape) for block in output.blocks] == [tuple(block.shape) for block in feature.blocks]


def test_update_reuses_channel_mapped_strategy() -> None:
    assert issubclass(ChannelMappedUpdate, Update)
    assert issubclass(NormGatedUpdate, Update)
    assert issubclass(ReplaceUpdate, Update)
    assert issubclass(ResidualUpdate, Update)


def test_update_strategies_keep_real_space_shapes() -> None:
    feature, update = _feature_and_update()

    replaced = ReplaceUpdate()(feature, update)
    residual = ResidualUpdate(step=0.25)(feature, update)
    gated = NormGatedUpdate(step=0.25)(feature, update)

    torch.testing.assert_close(replaced.blocks[1], update.blocks[1])
    torch.testing.assert_close(residual.blocks[1], feature.blocks[1] + 0.25 * update.blocks[1])
    assert tuple(gated.blocks[1].shape) == tuple(feature.blocks[1].shape)
    assert tuple(gated.blocks[2].shape) == tuple(feature.blocks[2].shape)


def test_channel_mapped_update_starts_as_identity_when_channels_match() -> None:
    feature, update = _feature_and_update()

    output = ChannelMappedUpdate()(feature, update)

    torch.testing.assert_close(output.blocks[1], feature.blocks[1] + update.blocks[1])
    torch.testing.assert_close(output.blocks[2], feature.blocks[2] + update.blocks[2])


def test_channel_mapped_update_allows_unequal_channels_with_shared_tuple_map() -> None:
    feature = RealFeature(
        [
            zero_block(dtype=torch.float64),
            torch.zeros(1, 3, 2, dtype=torch.float64),
        ]
    )
    update = RealUpdate(
        [
            zero_block(dtype=torch.float64),
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64),
        ]
    )
    module = ChannelMappedUpdate(step=2.0, initial_weight=0.5, identity_init=False)

    output = module(feature, update)

    expected_mapped = torch.einsum("oc,bc...->bo...", module.channel_maps["1"], update.blocks[1])
    torch.testing.assert_close(output.blocks[1], 2.0 * expected_mapped)
    assert tuple(output.blocks[1].shape) == (1, 3, 2)
