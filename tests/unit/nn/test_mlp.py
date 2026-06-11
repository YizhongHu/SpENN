"""Tests for reusable neural-network helper modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spenn.nn import MLP


def test_mlp_preserves_leading_axes_and_sets_output_channels() -> None:
    mlp = MLP(in_channels=4, out_channels=5, hidden_channels=7, num_hidden_layers=2, activation=nn.Tanh()).to(dtype=torch.float64)
    inputs = torch.ones(2, 3, 4, dtype=torch.float64)

    outputs = mlp(inputs)

    assert outputs.shape == (2, 3, 5)
    assert outputs.dtype == torch.float64


def test_mlp_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="out_channels"):
        MLP(in_channels=1, out_channels=0)
    with pytest.raises(ValueError, match="in_channels"):
        MLP(in_channels=0, out_channels=1)
    with pytest.raises(ValueError, match="hidden_channels"):
        MLP(in_channels=1, out_channels=1, hidden_channels=0)
    with pytest.raises(ValueError, match="num_hidden_layers"):
        MLP(in_channels=1, out_channels=1, num_hidden_layers=-1)
