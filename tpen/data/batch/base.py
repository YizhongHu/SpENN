"""Shared tensor helpers for batch state containers."""

from __future__ import annotations

from typing import Any

import torch


def _coerce_optional_tensor(value: Any | None, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    """Convert an optional tensor-like value to a tensor.

    Parameters
    ----------
    value : object or None
        Tensor-like value to convert. ``None`` is passed through unchanged.
    dtype : torch.dtype or None, optional
        Optional dtype applied to converted tensors.

    Returns
    -------
    torch.Tensor or None
        Converted tensor, or ``None``.
    """

    if value is None:
        return None
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


__all__: list[str] = []
