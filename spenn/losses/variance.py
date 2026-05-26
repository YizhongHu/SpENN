"""Variance diagnostics."""

from __future__ import annotations

import torch


def variance(local_values: torch.Tensor) -> torch.Tensor:
    """Return the population variance of a batched observable.

    Parameters
    ----------
    local_values : torch.Tensor
        Batched observable values.

    Returns
    -------
    torch.Tensor
        Population variance with ``unbiased=False``.
    """

    return local_values.var(unbiased=False)
