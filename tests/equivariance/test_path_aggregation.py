"""Runtime equivariance tests for path aggregation."""

from __future__ import annotations

import torch

from spenn.data.irrep import IrrepInteraction
from spenn.data.partition import Partition
from spenn.data.permutation import Permutation
from spenn.nn import PathAggregation
from spenn.reps import specht_irrep


def test_path_aggregation_passes_forced_runtime_equivariance_check() -> None:
    torch.manual_seed(13579)
    symmetric = Partition((2,))
    sign = Partition((1, 1))
    interaction = IrrepInteraction(
        {
            symmetric: torch.arange(1, 1 + 1 * 2 * 3 * 3 * 3, dtype=torch.float64).reshape(
                1,
                2,
                3,
                3,
                3,
                1,
                1,
            ),
            sign: torch.linspace(-2.0, 2.0, 1 * 2 * 3 * 3 * 3, dtype=torch.float64).reshape(
                1,
                2,
                3,
                3,
                3,
                1,
                1,
            ),
        }
    )
    aggregation = PathAggregation(
        channel_out_by_order={2: 2},
        equivariance_check=True,
        check_probability=1.0,
        tensor_validation_check=True,
    )

    output = aggregation(interaction)

    assert output.validate() is output


def test_path_aggregation_preserves_orthogonal_coordinate_action() -> None:
    partition = Partition((2, 1))
    tensor = torch.randn(
        1,
        2,
        3,
        2,
        2,
        2,
        2,
        2,
        generator=torch.Generator().manual_seed(97531),
        dtype=torch.float64,
    )
    permutation = Permutation((1, 2, 0))
    representation = specht_irrep(partition).representation(permutation)
    aggregation = PathAggregation(channel_out_by_order={3: 2})
    aggregation(IrrepInteraction({partition: tensor}))

    transformed_input = torch.einsum("ab,...bc->...ac", representation, tensor)
    transformed_output = aggregation(IrrepInteraction({partition: transformed_input}))[partition]
    expected_output = torch.einsum(
        "ab,...bc->...ac",
        representation,
        aggregation(IrrepInteraction({partition: tensor}))[partition],
    )

    torch.testing.assert_close(transformed_output, expected_output)
