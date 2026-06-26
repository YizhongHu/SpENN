"""Tests for irrep activation modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spenn.data.irrep import IrrepInteraction
from spenn.data.partition import Partition
from spenn.nn import Activation, ActivationByIrrep, ActivationByType


class Cube(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**3


def test_activation_modules_inherit_activation_template() -> None:
    assert issubclass(ActivationByType, Activation)
    assert issubclass(ActivationByIrrep, Activation)


def test_activation_by_type_preserves_path_resolved_interaction_type() -> None:
    symmetric = Partition((2,))
    sign = Partition((1, 1))
    interaction = IrrepInteraction(
        {
            symmetric: torch.tensor([-2.0, 1.0], dtype=torch.float64).reshape(1, 1, 2, 1, 1, 1, 1),
            sign: torch.tensor([-3.0, 2.0], dtype=torch.float64).reshape(1, 1, 2, 1, 1, 1, 1),
        }
    )
    activation = ActivationByType(symmetric_activation=nn.ReLU(), antisymmetric_activation=Cube())

    output = activation(interaction)

    assert isinstance(output, IrrepInteraction)
    assert output[symmetric].shape == interaction[symmetric].shape
    assert output[sign].shape == interaction[sign].shape
    torch.testing.assert_close(
        output[symmetric],
        torch.tensor([0.0, 1.0], dtype=torch.float64).reshape(1, 1, 2, 1, 1, 1, 1),
    )
    torch.testing.assert_close(
        output[sign],
        torch.tensor([-27.0, 8.0], dtype=torch.float64).reshape(1, 1, 2, 1, 1, 1, 1),
    )


def test_activation_by_type_rejects_non_odd_antisymmetric_activation() -> None:
    sign = Partition((1, 1))
    interaction = IrrepInteraction(
        {
            sign: torch.tensor([-3.0, 2.0], dtype=torch.float64).reshape(1, 1, 2, 1, 1, 1, 1),
        }
    )
    activation = ActivationByType(antisymmetric_activation=nn.ReLU())

    with pytest.raises(ValueError, match="must be odd"):
        activation(interaction)


def test_activation_by_type_broadcasts_tensor_gate_over_alpha_with_paths() -> None:
    partition = Partition((2, 1))
    tensor = torch.arange(1, 1 + 1 * 1 * 2 * 1 * 1 * 1 * 2 * 2, dtype=torch.float64).reshape(
        1,
        1,
        2,
        1,
        1,
        1,
        2,
        2,
    )
    interaction = IrrepInteraction({partition: tensor})
    activation = ActivationByType(tensor_activation=nn.Sigmoid())

    output = activation(interaction)[partition]
    gate = output / tensor

    torch.testing.assert_close(gate[..., 0, :], gate[..., 1, :])
