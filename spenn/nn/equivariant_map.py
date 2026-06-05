"""Runtime-checked equivariant module base class."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


class EquivariantMap(nn.Module):
    """Base class for modules that should commute with particle permutations.

    Parameters
    ----------
    equivariance_check : bool, optional
        Whether to run runtime equivariance checks in ``forward``.
    check_probability : float, optional
        Probability of checking a given forward call. Use ``1.0`` in tests for
        deterministic enforcement.
    check_atol, check_rtol : float, optional
        Tensor comparison tolerances.
    max_full_check_size : int, optional
        Check every particle permutation up to this size. Larger systems check
        adjacent transpositions and reversal.
    tensor_validation_check : bool, optional
        Whether to call ``validate`` on data tensors in the input and output
        trees during ``forward``.
    validation_probability : float, optional
        Probability of running tensor validation on a given forward call.
    """

    equivariance_check: bool
    check_probability: float
    check_atol: float
    check_rtol: float
    max_full_check_size: int
    tensor_validation_check: bool
    validation_probability: float

    def __init__(
        self,
        *,
        equivariance_check: bool = False,
        check_probability: float = 0.0,
        check_atol: float = 1.0e-6,
        check_rtol: float = 1.0e-5,
        max_full_check_size: int = 5,
        tensor_validation_check: bool = False,
        validation_probability: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= check_probability <= 1.0:
            raise ValueError(f"check_probability must be in [0, 1], got {check_probability}")
        if not 0.0 <= validation_probability <= 1.0:
            raise ValueError(f"validation_probability must be in [0, 1], got {validation_probability}")
        self.equivariance_check = bool(equivariance_check)
        self.check_probability = float(check_probability)
        self.check_atol = float(check_atol)
        self.check_rtol = float(check_rtol)
        self.max_full_check_size = int(max_full_check_size)
        self.tensor_validation_check = bool(tensor_validation_check)
        self.validation_probability = float(validation_probability)

    def forward(self, *args, **kwargs):
        """Apply ``forward_impl`` and optionally run runtime checks."""

        validate_tensors = self.should_validate_tensors()
        if validate_tensors:
            _validate_tree((args, kwargs))
        output = self.forward_impl(*args, **kwargs)
        if validate_tensors:
            _validate_tree(output)
        if self.should_check_equivariance():
            from spenn.testing.equivariance import assert_equivariant_all

            assert_equivariant_all(
                self,
                args,
                kwargs=kwargs,
                original_output=output,
                atol=self.check_atol,
                rtol=self.check_rtol,
                max_full_size=self.max_full_check_size,
            )
        return output

    def forward_impl(self, *args, **kwargs):
        """Implement the unchecked module computation in subclasses."""

        raise NotImplementedError(f"{type(self).__name__}.forward_impl is not implemented")

    def should_check_equivariance(self) -> bool:
        """Return whether this forward call should run runtime checks."""

        if not self.equivariance_check or self.check_probability <= 0.0:
            return False
        if self.check_probability >= 1.0:
            return True
        return bool(torch.rand(()) < self.check_probability)

    def should_validate_tensors(self) -> bool:
        """Return whether this forward call should validate data tensors."""

        if not self.tensor_validation_check or self.validation_probability <= 0.0:
            return False
        if self.validation_probability >= 1.0:
            return True
        return bool(torch.rand(()) < self.validation_probability)


def _validate_tree(obj: Any) -> None:
    validate = getattr(obj, "validate", None)
    if callable(validate):
        validate()
        return
    if isinstance(obj, Mapping):
        for value in obj.values():
            _validate_tree(value)
        return
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for value in obj:
            _validate_tree(value)


__all__ = ["EquivariantMap"]
