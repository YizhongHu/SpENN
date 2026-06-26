"""VMC training metrics."""

from __future__ import annotations

import torch
from torch.nn.parameter import UninitializedParameter


def parameter_norm(model) -> torch.Tensor:
    """Return the L2 norm of trainable initialized parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters are inspected.

    Returns
    -------
    torch.Tensor
        Scalar parameter norm. Returns zero when no initialized trainable
        parameters are present.
    """

    total = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if isinstance(param, UninitializedParameter):
            continue
        value = param.detach().pow(2).sum()
        total = value if total is None else total + value
    return torch.sqrt(total) if total is not None else torch.tensor(0.0)


def gradient_norm(model) -> torch.Tensor:
    """Return the L2 norm of available gradients.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameter gradients are inspected.

    Returns
    -------
    torch.Tensor
        Scalar gradient norm. Returns zero when no gradients are present.
    """

    total = None
    for param in model.parameters():
        if param.grad is None:
            continue
        if isinstance(param, UninitializedParameter):
            continue
        value = param.grad.detach().pow(2).sum()
        total = value if total is None else total + value
    return torch.sqrt(total) if total is not None else torch.tensor(0.0)
