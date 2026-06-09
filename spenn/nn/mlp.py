"""Reusable multilayer perceptron modules."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class MLP(nn.Module):
    """Eager-input multilayer perceptron.

    Parameters
    ----------
    in_channels : int
        Size of the input feature dimension.
    out_channels : int
        Size of the final feature dimension.
    hidden_channels : int, optional
        Width of each hidden layer.
    num_hidden_layers : int, optional
        Number of hidden layers before the output layer.
    activation : torch.nn.Module or None, optional
        Activation copied after each hidden layer. If ``None``, SiLU is used.
    bias : bool, optional
        Whether linear layers include bias terms.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_hidden_layers: int = 2,
        activation: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("MLP in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("MLP out_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("MLP hidden_channels must be positive")
        if num_hidden_layers < 0:
            raise ValueError("MLP num_hidden_layers must be nonnegative")

        layers: list[nn.Module] = []
        current_channels = int(in_channels)
        for _layer_idx in range(num_hidden_layers):
            layers.append(nn.Linear(current_channels, hidden_channels, bias=bias))
            layers.append(deepcopy(activation) if activation is not None else nn.SiLU())
            current_channels = int(hidden_channels)
        layers.append(nn.Linear(current_channels, out_channels, bias=bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to the final dimension of an input tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor whose final axis is the feature axis.

        Returns
        -------
        torch.Tensor
            Tensor with final axis `out_channels`.
        """

        return self.layers(inputs)


__all__ = ["MLP"]
