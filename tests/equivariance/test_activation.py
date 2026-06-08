"""Runtime equivariance tests for irrep activation modules."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.irrep import IrrepInteraction
from spenn.data.partition import Partition
from spenn.data.permutation import Permutation
from spenn.nn import ActivationByType
from spenn.reps import specht_irrep
from spenn.testing.equivariance import assert_equivariant_all


class Cube(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an odd scalar activation."""

        return x**3


def test_activation_by_type_passes_forced_runtime_equivariance_check_on_interactions() -> None:
    symmetric = Partition((2,))
    sign = Partition((1, 1))
    vector = Partition((2, 1))
    interaction = IrrepInteraction(
        {
            symmetric: torch.arange(1, 1 + 1 * 2 * 2 * 3 * 3, dtype=torch.float64).reshape(
                1,
                2,
                2,
                3,
                3,
                1,
                1,
            ),
            sign: torch.linspace(-2.0, 2.0, 1 * 2 * 2 * 3 * 3, dtype=torch.float64).reshape(
                1,
                2,
                2,
                3,
                3,
                1,
                1,
            ),
            vector: torch.arange(1, 1 + 1 * 2 * 2 * 3 * 3 * 3 * 2 * 2, dtype=torch.float64).reshape(
                1,
                2,
                2,
                3,
                3,
                3,
                2,
                2,
            ),
        }
    )
    activation = ActivationByType(
        symmetric_activation=nn.SiLU(),
        antisymmetric_activation=Cube(),
        tensor_activation=nn.Sigmoid(),
    )

    output = activation(interaction)

    assert isinstance(output, IrrepInteraction)
    assert output.validate() is output
    assert_equivariant_all(activation, interaction)


def test_activation_by_type_preserves_orthogonal_coordinate_action_with_paths() -> None:
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
        generator=torch.Generator().manual_seed(2468),
        dtype=torch.float64,
    )
    permutation = Permutation((1, 2, 0))
    representation = specht_irrep(partition).representation(permutation)
    activation = ActivationByType(tensor_activation=nn.Sigmoid())

    transformed_input = torch.einsum("ab,...bc->...ac", representation, tensor)
    transformed_output = activation(IrrepInteraction({partition: transformed_input}))[partition]
    expected_output = torch.einsum(
        "ab,...bc->...ac",
        representation,
        activation(IrrepInteraction({partition: tensor}))[partition],
    )

    torch.testing.assert_close(transformed_output, expected_output)
