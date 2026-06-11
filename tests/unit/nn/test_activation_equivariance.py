"""Runtime equivariance tests for baseline irrep activation modules."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.irrep import IrrepInteraction
from spenn.data.partition import Partition
from spenn.nn.activation import GatedNormActivation
from tests.helpers.equivariance import assert_equivariant_all


def test_gated_norm_activation_passes_forced_runtime_equivariance_check_on_interactions() -> None:
    partition = Partition((2, 1))
    interaction = IrrepInteraction(
        {
            partition: torch.randn(
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
            ),
        }
    )
    activation = GatedNormActivation(gate=nn.Sigmoid())

    output = activation(interaction)

    assert isinstance(output, IrrepInteraction)
    assert output.validate() is output
    assert_equivariant_all(activation, interaction)
