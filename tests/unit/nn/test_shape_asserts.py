"""Unit tests for neural-network shape assertions."""

from __future__ import annotations

import pytest
import torch

from spenn.data import Par
from spenn.data.batch import ElectronBatch
from spenn.nn.cusp import ElectronElectronCusp
from spenn.nn.encoding import ElectronPairEncoder
from spenn.nn.spechtmp.message_head import _project_product
from spenn.nn.spechtmp.update_head import _project_branch


def test_encoder_shape_asserts_catch_bad_tuple_mlp_output() -> None:
    class BadTupleMLP(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.new_zeros(*inputs.shape[:-1], 2, 1)

    encoder = ElectronPairEncoder(channels=[0, 2, 0])
    positions = torch.zeros(1, 3, 2)

    with pytest.raises(AssertionError):
        encoder.apply_tuple_mlp(BadTupleMLP(), positions)


def test_spechtmp_projection_asserts_catch_bad_linear_output_shape() -> None:
    class BadLinear(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.new_zeros(*inputs.shape[:-1], 2, 1)

    product = torch.zeros(1, 1, 1, 2, 2, 2, 2, 1, 1)
    branch = torch.zeros(1, 1, 1, 2, 2, 2, 2, 1, 1)

    with pytest.raises(AssertionError):
        _project_product(BadLinear(), Par("S"), Par("H"), Par("H"), product)
    with pytest.raises(AssertionError):
        _project_branch(BadLinear(), Par("S"), Par("S"), branch)


def test_cusp_shape_asserts_preserve_batch_shape() -> None:
    positions = torch.zeros(4, 3, 2, dtype=torch.float64)
    output = ElectronElectronCusp()(ElectronBatch(positions))

    assert output.shape == (4,)

