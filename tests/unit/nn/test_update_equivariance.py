"""Tests for baseline real-feature update equivariance."""

from __future__ import annotations

import torch

from tpen.equivariance import EquivariantMap
from tpen.data.real import Feature, Update, zero_block
from tpen.nn.update import ResidualUpdater, Updater
from tests.helpers.equivariance import assert_equivariant_all


def _feature_and_update() -> tuple[Feature, Update]:
    feature = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
            torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3),
        ]
    )
    update = Update(
        [
            zero_block(dtype=torch.float64),
            torch.ones(1, 2, 3, dtype=torch.float64),
            torch.full((1, 2, 3, 3), 0.5, dtype=torch.float64),
        ]
    )
    return feature, update


def test_residual_update_is_baseline_update_map() -> None:
    assert issubclass(ResidualUpdater, Updater)


def test_residual_update_passes_runtime_equivariance_check() -> None:
    feature, update = _feature_and_update()
    module = ResidualUpdater(step=0.25)

    output = module(feature, update)

    assert isinstance(module, EquivariantMap)
    assert isinstance(output, Feature)
    assert [tuple(block.shape) for block in output.blocks] == [tuple(block.shape) for block in feature.blocks]
    assert_equivariant_all(module, (feature, update))


def test_residual_update_keeps_real_space_shapes() -> None:
    feature, update = _feature_and_update()

    output = ResidualUpdater(step=0.25)(feature, update)

    torch.testing.assert_close(output.blocks[1], feature.blocks[1] + 0.25 * update.blocks[1])
    torch.testing.assert_close(output.blocks[2], feature.blocks[2] + 0.25 * update.blocks[2])
