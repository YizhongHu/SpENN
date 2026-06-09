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
    module = PathAggregation(
        max_order=1,
        channels=1,
        channel_out_by_order={1: 1},
        path_counts_by_order={1: 2},
        partitions=(partition,),
    )

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
    module = PathAggregation(
        max_order=3,
        channels=2,
        channel_out_by_order=3,
        path_counts_by_order={1: 0, 2: 0, 3: 2},
        partitions=(partition,),
    )
    weight = torch.arange(1, 1 + 3 * 2 * 2 * 2 * 2, dtype=torch.float64).reshape(3, 2, 2, 2, 2)

    with torch.no_grad():
        module.weights[module.key(partition)].copy_(weight)
    output = module(interaction)[partition]

    expected = torch.einsum("bcp...ad,oecpd->bo...ae", tensor, weight)
    torch.testing.assert_close(output, expected)


def test_path_aggregation_requires_configured_output_channels_for_all_orders() -> None:
    with pytest.raises(ValueError, match="missing orders"):
        PathAggregation(
            max_order=2,
            channels=1,
            channel_out_by_order={2: 1},
            path_counts_by_order={1: 1, 2: 1},
        )


def test_path_aggregation_rejects_input_channels_that_disagree_with_config() -> None:
    interaction = IrrepInteraction(
        {
            Partition((2,)): torch.ones(1, 1, 1, 2, 2, 1, 1, dtype=torch.float64),
            Partition((1, 1)): torch.ones(1, 2, 1, 2, 2, 1, 1, dtype=torch.float64),
        }
    )
    module = PathAggregation(
        max_order=2,
        channels=1,
        channel_out_by_order=1,
        path_counts_by_order={1: 0, 2: 1},
        partitions=(Partition((2,)), Partition((1, 1))),
    )

    with pytest.raises(ValueError, match="input channels"):
        module(interaction)


def test_path_aggregation_rejects_changed_block_signature_after_initialization() -> None:
    partition = Partition((1,))
    module = PathAggregation(
        max_order=1,
        channels=1,
        channel_out_by_order={1: 1},
        path_counts_by_order={1: 2},
        partitions=(partition,),
    )

    with pytest.raises(ValueError, match="expected"):
        module(IrrepInteraction({partition: torch.ones(1, 1, 3, 3, 1, 1, dtype=torch.float64)}))


def test_path_aggregation_eagerly_creates_zero_path_weights() -> None:
    partition = Partition((1,))
    tensor = torch.empty(1, 1, 0, 3, 1, 1, dtype=torch.float64)
    module = PathAggregation(
        max_order=1,
        channels=1,
        channel_out_by_order=2,
        path_counts_by_order={1: 0},
        partitions=(partition,),
    )

    output = module(IrrepInteraction({partition: tensor}))[partition]

    assert module.weights[module.key(partition)].shape == (2, 1, 1, 0, 1)
    torch.testing.assert_close(output, torch.zeros(1, 2, 3, 1, 1, dtype=torch.float64))
