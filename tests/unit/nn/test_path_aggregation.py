"""Tests for learned path aggregation."""

from __future__ import annotations

import pytest
import torch

from spenn.data.irrep import IrrepInteraction
from spenn.data.partition import Partition
from spenn.nn import PathAggregation


def test_path_aggregation_removes_path_axis_and_selects_learned_path() -> None:
    partition = Partition((1,))
    tensor = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=torch.float64).reshape(
        1,
        1,
        2,
        3,
        1,
        1,
    )
    interaction = IrrepInteraction({partition: tensor})
    module = PathAggregation(channel_out_by_order={1: 1})
    module(interaction)

    with torch.no_grad():
        module.weights[module.key(partition)].zero_()
        module.weights[module.key(partition)][0, 0, 0, 0, 0] = 1.0
    path_zero = module(interaction)[partition]

    with torch.no_grad():
        module.weights[module.key(partition)].zero_()
        module.weights[module.key(partition)][0, 0, 0, 1, 0] = 1.0
    path_one = module(interaction)[partition]

    assert path_zero.shape == (1, 1, 3, 1, 1)
    torch.testing.assert_close(path_zero, tensor[:, :, 0])
    torch.testing.assert_close(path_one, tensor[:, :, 1])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(path_zero, path_one)


def test_path_aggregation_mixes_channels_paths_and_beta_without_alpha_mixing() -> None:
    partition = Partition((2, 1))
    tensor = torch.arange(1, 1 + 1 * 2 * 2 * 2 * 2 * 2 * 2 * 2, dtype=torch.float64).reshape(
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    )
    interaction = IrrepInteraction({partition: tensor})
    module = PathAggregation(channel_out_by_order={3: 3})
    module(interaction)
    weight = torch.arange(1, 1 + 3 * 2 * 2 * 2 * 2, dtype=torch.float64).reshape(3, 2, 2, 2, 2)

    with torch.no_grad():
        module.weights[module.key(partition)].copy_(weight)
    output = module(interaction)[partition]

    expected = torch.einsum("bcp...ad,oecpd->bo...ae", tensor, weight)
    torch.testing.assert_close(output, expected)


def test_path_aggregation_requires_mapped_output_channels_for_seen_orders() -> None:
    partition = Partition((1,))
    tensor = torch.ones(1, 1, 1, 2, 1, 1, dtype=torch.float64)
    module = PathAggregation(channel_out_by_order={2: 1})

    with pytest.raises(ValueError, match="missing order 1"):
        module(IrrepInteraction({partition: tensor}))


def test_path_aggregation_default_channels_reject_mismatched_same_order_partitions() -> None:
    interaction = IrrepInteraction(
        {
            Partition((2,)): torch.ones(1, 1, 1, 2, 2, 1, 1, dtype=torch.float64),
            Partition((1, 1)): torch.ones(1, 2, 1, 2, 2, 1, 1, dtype=torch.float64),
        }
    )

    with pytest.raises(ValueError, match="channel_out_by_order=None"):
        PathAggregation()(interaction)


def test_path_aggregation_rejects_changed_block_signature_after_initialization() -> None:
    partition = Partition((1,))
    module = PathAggregation(channel_out_by_order={1: 1})
    module(IrrepInteraction({partition: torch.ones(1, 1, 2, 3, 1, 1, dtype=torch.float64)}))

    with pytest.raises(ValueError, match="expected"):
        module(IrrepInteraction({partition: torch.ones(1, 1, 3, 3, 1, 1, dtype=torch.float64)}))
