"""Reusable multilayer perceptron modules."""

from __future__ import annotations

from copy import deepcopy

from spenn.dependencies import require_torch, require_torch_nn
from spenn.nn.initialization import SeededLinear, TorchInitializer

torch = require_torch(feature="SpENN MLP modules")
nn = require_torch_nn(feature="SpENN MLP modules")


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
    initializer : TorchInitializer or None, optional
        Explicit side-effect-free initializer for generated linear layers. If
        ``None``, generated layers use the standard ``torch.nn.Linear``
        initializer, which follows PyTorch's global RNG behavior.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_hidden_layers: int = 2,
        activation: nn.Module | None = None,
        bias: bool = True,
        initializer: TorchInitializer | None = None,
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
        for layer_idx in range(num_hidden_layers):
            layers.append(_linear(current_channels, hidden_channels, bias=bias, initializer=initializer, name=f"hidden_{layer_idx}"))
            layers.append(deepcopy(activation) if activation is not None else nn.SiLU())
            current_channels = int(hidden_channels)
        layers.append(_linear(current_channels, out_channels, bias=bias, initializer=initializer, name="output"))
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


def _linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    initializer: TorchInitializer | None,
    name: str,
) -> nn.Module:
    if initializer is None:
        return nn.Linear(in_features, out_features, bias=bias)
    return SeededLinear(
        in_features,
        out_features,
        bias=bias,
        initializer=initializer.spawn(name),
    )


__all__ = ["MLP"]
