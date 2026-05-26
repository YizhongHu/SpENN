"""Energy helpers."""

from __future__ import annotations

import torch


def mean_energy(local_energy: torch.Tensor) -> torch.Tensor:
    """Return the mean local energy.

    Parameters
    ----------
    local_energy : torch.Tensor
        Batched local-energy values.

    Returns
    -------
    torch.Tensor
        Scalar mean energy.
    """

    return local_energy.mean()
