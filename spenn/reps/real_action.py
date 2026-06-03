"""Real-space permutation actions for ordered electron axes."""

from __future__ import annotations

import torch

from spenn.data.permutation import Permutation


def as_permutation(value: Permutation | tuple[int, ...] | list[int]) -> Permutation:
    """Convert a sequence-like value to :class:`Permutation`.

    Parameters
    ----------
    value : Permutation or sequence of int
        Permutation-like value.

    Returns
    -------
    Permutation
        Normalized permutation object.
    """

    return value if isinstance(value, Permutation) else Permutation(tuple(value))


def permute_axis(tensor: torch.Tensor, permutation: Permutation | tuple[int, ...] | list[int], axis: int) -> torch.Tensor:
    """Permute one tensor axis.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to permute.
    permutation : Permutation or sequence of int
        Axis permutation.
    axis : int
        Axis to permute.

    Returns
    -------
    torch.Tensor
        Tensor with `axis` re-indexed by `permutation`.
    """

    normalized = as_permutation(permutation)
    index = torch.tensor(normalized.inverse().image, dtype=torch.long, device=tensor.device)
    return torch.index_select(tensor, int(axis), index)


__all__ = ["Permutation", "as_permutation", "permute_axis"]
