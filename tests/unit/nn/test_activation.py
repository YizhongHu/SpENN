"""Tests for baseline irrep activation modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spenn.data.irrep import IrrepInteraction
from spenn.data.partition import Partition
from spenn.data.permutation import Permutation
from spenn.nn.activation import Activation, GaussianActivation, GatedNormActivation
from spenn.reps import specht_irrep


def _interaction() -> tuple[IrrepInteraction, Partition, torch.Tensor]:
    partition = Partition((2, 1))
    tensor = torch.arange(1, 1 + 1 * 2 * 3 * 1 * 1 * 1 * 2 * 2, dtype=torch.float64).reshape(
        1,
        2,
        3,
        1,
        1,
        1,
        2,
        2,
    )
    return IrrepInteraction({partition: tensor}), partition, tensor


def test_activation_modules_inherit_activation_template() -> None:
    assert issubclass(GatedNormActivation, Activation)


def test_gaussian_activation_matches_a_exp_ax_squared() -> None:
    activation = GaussianActivation(amplitude=2.0, quadratic_coefficient=-0.5).to(dtype=torch.float64)
    x = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)

    torch.testing.assert_close(activation(x), 2.0 * torch.exp(-0.5 * x.square()))


def test_gaussian_activation_can_make_coefficients_trainable() -> None:
    activation = GaussianActivation(amplitude=2.0, quadratic_coefficient=-0.5, trainable=True)

    assert isinstance(activation.amplitude, nn.Parameter)
    assert isinstance(activation.quadratic_coefficient, nn.Parameter)


def test_gated_norm_activation_uses_configured_gate_module_and_preserves_shape() -> None:
    interaction, partition, tensor = _interaction()
    activation = GatedNormActivation(gate=nn.Identity())

    output = activation(interaction)[partition]
    expected_gate = tensor.square().sum(dim=-2, keepdim=True)

    assert output.shape == tensor.shape
    torch.testing.assert_close(output, tensor * expected_gate)


class BadGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-2)


def test_gated_norm_activation_rejects_wrong_gate_shape() -> None:
    interaction, _partition, _tensor = _interaction()

    with pytest.raises(ValueError, match="gate must preserve"):
        GatedNormActivation(gate=BadGate())(interaction)


def test_gated_norm_activation_preserves_orthogonal_coordinate_action_with_paths() -> None:
    interaction, partition, tensor = _interaction()
    permutation = Permutation((1, 2, 0))
    representation = specht_irrep(partition).representation(permutation)
    activation = GatedNormActivation(gate=nn.Sigmoid())

    transformed_input = torch.einsum("ab,...bc->...ac", representation, tensor)
    transformed_output = activation(IrrepInteraction({partition: transformed_input}))[partition]
    expected_output = torch.einsum(
        "ab,...bc->...ac",
        representation,
        activation(interaction)[partition],
    )

    torch.testing.assert_close(transformed_output, expected_output)


def test_gated_norm_activation_forward_does_not_mutate_state_inventory() -> None:
    interaction, _partition, _tensor = _interaction()
    activation = GatedNormActivation(gate=nn.Linear(2, 2, bias=False)).to(dtype=torch.float64)
    before = {
        name: (tuple(tensor.shape), tensor.dtype, tensor.device)
        for name, tensor in activation.state_dict().items()
    }

    activation(interaction)

    after = {
        name: (tuple(tensor.shape), tensor.dtype, tensor.device)
        for name, tensor in activation.state_dict().items()
    }
    assert after == before


def test_gated_norm_activation_requires_gate_module() -> None:
    with pytest.raises(TypeError, match="gate"):
        GatedNormActivation()  # type: ignore[call-arg]
