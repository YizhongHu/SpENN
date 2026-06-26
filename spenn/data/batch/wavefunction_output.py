"""Wavefunction output state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod
from typing import Any

import torch

from spenn.data.batch.base import _coerce_optional_tensor
from spenn.data.equivariant_state import EquivariantState, compare_tensor_blocks
from spenn.data.permutation import Permutation


@dataclass
class WavefunctionOutput(EquivariantState):
    """Store a fermionic scalar wavefunction value in signed-log form.

    This represents a fermionic scalar wavefunction output. Under particle
    permutation, ``logabs`` is invariant and ``sign`` transforms by the
    permutation parity (see :meth:`permute`). This sign-representation contract
    is specific to fermionic scalar outputs; a non-fermionic scalar output would
    need a different ``permute`` implementation.

    Parameters
    ----------
    logabs : torch.Tensor
        Log absolute wavefunction values with shape ``sample_shape``.
    sign : torch.Tensor
        Real wavefunction signs with the same shape as `logabs`.
    phase : torch.Tensor or None, optional
        Optional complex phase values with the same shape as `logabs`.
    aux : dict, optional
        Auxiliary readout or diagnostic values.

    Notes
    -----
    Exact zeros are represented by ``sign == 0`` and ``logabs == -inf``.
    Near-zero nonzero values should keep a finite `logabs` and nonzero `sign`.
    """

    logabs: torch.Tensor
    sign: torch.Tensor
    phase: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.logabs = self.logabs if isinstance(self.logabs, torch.Tensor) else torch.as_tensor(self.logabs)
        self.sign = self.sign if isinstance(self.sign, torch.Tensor) else torch.as_tensor(self.sign, dtype=self.logabs.dtype)
        self.phase = _coerce_optional_tensor(self.phase)
        if self.sign.shape != self.logabs.shape:
            raise ValueError("WavefunctionOutput.sign must have the same shape as logabs")
        if self.phase is not None and self.phase.shape != self.logabs.shape:
            raise ValueError("WavefunctionOutput.phase must have the same shape as logabs")
        zero_sign = self.sign == 0
        zero_logabs = torch.isneginf(self.logabs)
        if not torch.equal(zero_sign, zero_logabs):
            raise ValueError("WavefunctionOutput exact zeros require sign == 0 if and only if logabs == -inf")

    def validate(
        self,
        *,
        batch_size: int | None = None,
        sample_shape: tuple[int, ...] | None = None,
    ) -> "WavefunctionOutput":
        """Validate wavefunction output shape metadata.

        Parameters
        ----------
        batch_size : int or None, optional
            Expected flattened batch size.
        sample_shape : tuple of int or None, optional
            Expected output sample shape.

        Returns
        -------
        WavefunctionOutput
            This output, for fluent validation in runtime checks.

        Raises
        ------
        ValueError
            If this output does not match the expected sample shape or batch
            size.
        """

        if self.sign.shape != self.logabs.shape:
            raise ValueError("WavefunctionOutput.sign must have the same shape as logabs")
        if self.phase is not None and self.phase.shape != self.logabs.shape:
            raise ValueError("WavefunctionOutput.phase must have the same shape as logabs")
        zero_sign = self.sign == 0
        zero_logabs = torch.isneginf(self.logabs)
        if not torch.equal(zero_sign, zero_logabs):
            raise ValueError("WavefunctionOutput exact zeros require sign == 0 if and only if logabs == -inf")
        if sample_shape is not None:
            sample_shape = tuple(sample_shape)
            if tuple(self.logabs.shape) != sample_shape:
                raise ValueError(f"Expected sample shape {sample_shape}, got {tuple(self.logabs.shape)}")
            if batch_size is not None and prod(sample_shape) != batch_size:
                raise ValueError(f"Expected batch size {batch_size}, got {prod(sample_shape)} from sample_shape")
        elif batch_size is not None and prod(tuple(self.logabs.shape)) != batch_size:
            raise ValueError(f"Expected batch size {batch_size}, got shape {tuple(self.logabs.shape)}")
        return self

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "WavefunctionOutput":
        """Move tensor fields to a new device or dtype.

        Parameters
        ----------
        device : torch.device, str, or None, optional
            Target device. If ``None``, the current device is preserved.
        dtype : torch.dtype or None, optional
            Target floating-point dtype. If ``None``, the current dtype is
            preserved.

        Returns
        -------
        WavefunctionOutput
            Output with tensor fields moved to the requested device or dtype.
        """

        return replace(
            self,
            logabs=self.logabs.to(device=device, dtype=dtype),
            sign=self.sign.to(device=device, dtype=dtype),
            phase=None if self.phase is None else self.phase.to(device=device, dtype=dtype),
        )

    def permute(self, permutation: Permutation) -> "WavefunctionOutput":
        """Return the output under a particle permutation.

        Encodes the fermionic scalar-output contract: ``logabs`` is invariant
        and ``sign`` transforms by the permutation parity
        (``permutation.sign``). Scalar wavefunction outputs carry no tuple-index
        axes, so only the parity factor on ``sign`` is applied.
        """

        return replace(
            self,
            logabs=self.logabs.clone(),
            sign=self.sign * permutation.sign,
            phase=None if self.phase is None else self.phase.clone(),
            aux=dict(self.aux),
        )

    def compare(
        self,
        other: "WavefunctionOutput",
        *,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> tuple[bool, dict[str, float]]:
        """Compare ``logabs``/``sign``/``phase``; return ``(is_close, max_abs_error)``.

        ``aux`` is diagnostic and not compared.
        """

        if type(self) is not type(other) or (self.phase is None) != (other.phase is None):
            return False, {"max_abs_error": float("inf")}
        blocks_self = [self.logabs, self.sign] + ([] if self.phase is None else [self.phase])
        blocks_other = [other.logabs, other.sign] + ([] if other.phase is None else [other.phase])
        return compare_tensor_blocks(blocks_self, blocks_other, atol=atol, rtol=rtol)


__all__ = ["WavefunctionOutput"]
