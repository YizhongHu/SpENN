"""Core batch and wavefunction dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod
from typing import Any

import torch

def _coerce_optional_tensor(value: Any | None, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


@dataclass
class ElectronBatch:
    """Store batched electron coordinates and optional context.

    Parameters
    ----------
    positions : torch.Tensor
        Electron coordinates with shape
        ``[*sample_shape, n_electrons, spatial_dim]``.
    system : object or None, optional
        Optional system metadata, such as electron count and nuclear data.
    nuclear_positions : torch.Tensor or None, optional
        Nuclear coordinates with shape ``[n_nuclei, dim]`` or
        ``[*sample_shape, n_nuclei, dim]``.
    nuclear_charges : torch.Tensor or None, optional
        Nuclear charges with shape ``[n_nuclei]`` or
        ``[*sample_shape, n_nuclei]``.
    spins : torch.Tensor or None, optional
        Spin labels with shape ``[*sample_shape, n_electrons]`` and entries
        exactly equal to ``+1`` or ``-1``.
    aux : dict, optional
        Auxiliary metadata passed through model, sampler, or physics code.
    """

    positions: torch.Tensor
    system: Any | None = None
    nuclear_positions: torch.Tensor | None = None
    nuclear_charges: torch.Tensor | None = None
    spins: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = self.positions if isinstance(self.positions, torch.Tensor) else torch.as_tensor(self.positions)
        self.nuclear_positions = _coerce_optional_tensor(self.nuclear_positions, dtype=self.positions.dtype)
        self.nuclear_charges = _coerce_optional_tensor(self.nuclear_charges, dtype=self.positions.dtype)
        self.spins = _coerce_optional_tensor(self.spins, dtype=self.positions.dtype)
        if self.positions.ndim < 3:
            raise ValueError(
                "ElectronBatch.positions must have shape [*sample_shape, n_electrons, spatial_dim], "
                f"got {tuple(self.positions.shape)}"
            )
        if self.spins is not None:
            if tuple(self.spins.shape) != (*self.sample_shape, self.n_electrons):
                raise ValueError("ElectronBatch.spins must have shape [*sample_shape, n_electrons]")
            if not torch.all((self.spins == 1) | (self.spins == -1)):
                raise ValueError("ElectronBatch.spins entries must be exactly +1 or -1")
        if self.system is not None and hasattr(self.system, "n_electrons") and self.system.n_electrons is not None:
            if self.system.n_electrons != self.n_electrons:
                raise ValueError("ElectronBatch.positions disagree with system.n_electrons")
        if self.nuclear_positions is not None:
            nuclear_shape = tuple(self.nuclear_positions.shape)
            unbatched = self.nuclear_positions.ndim == 2 and nuclear_shape[-1] == self.spatial_dim
            sampled = (
                self.nuclear_positions.ndim == len(self.sample_shape) + 2
                and nuclear_shape[:-2] == self.sample_shape
                and nuclear_shape[-1] == self.spatial_dim
            )
            if not (unbatched or sampled):
                raise ValueError(
                    "ElectronBatch.nuclear_positions must have shape [n_nuclei, dim] "
                    "or [*sample_shape, n_nuclei, dim]"
                )
        if self.nuclear_charges is not None:
            charge_shape = tuple(self.nuclear_charges.shape)
            unbatched = self.nuclear_charges.ndim == 1
            sampled = self.nuclear_charges.ndim == len(self.sample_shape) + 1 and charge_shape[:-1] == self.sample_shape
            if not (unbatched or sampled):
                raise ValueError("ElectronBatch.nuclear_charges must have shape [n_nuclei] or [*sample_shape, n_nuclei]")
        if self.nuclear_positions is not None and self.nuclear_charges is not None:
            if self.nuclear_positions.shape[-2] != self.nuclear_charges.shape[-1]:
                raise ValueError("ElectronBatch.nuclear_positions and nuclear_charges must agree on n_nuclei")

    @property
    def device(self) -> torch.device:
        """Return the device of the position tensor.

        Returns
        -------
        torch.device
            Device on which `positions` is stored.
        """

        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the position tensor.

        Returns
        -------
        torch.dtype
            Data type used by `positions`.
        """

        return self.positions.dtype

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Return the sample axes before particle and coordinate axes.

        Returns
        -------
        tuple of int
            Leading sample shape of `positions`.
        """

        return tuple(self.positions.shape[:-2])

    @property
    def n_configurations(self) -> int:
        """Return the flattened number of electron configurations.

        Returns
        -------
        int
            Product of all sample axes.
        """

        return prod(self.sample_shape)

    @property
    def batch_size(self) -> int:
        """Return the flattened number of electron configurations.

        Returns
        -------
        int
            Product of all sample axes.
        """

        return self.n_configurations

    @property
    def n_electrons(self) -> int:
        """Return the number of electrons per configuration.

        Returns
        -------
        int
            Size of the electron axis.
        """

        return self.positions.shape[-2]

    @property
    def spatial_dim(self) -> int:
        """Return the spatial dimension of each electron coordinate.

        Returns
        -------
        int
            Size of the final coordinate axis.
        """

        return self.positions.shape[-1]

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "ElectronBatch":
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
        ElectronBatch
            Batch with tensor fields moved to the requested device or dtype.
        """

        return replace(
            self,
            positions=self.positions.to(device=device, dtype=dtype),
            nuclear_positions=None if self.nuclear_positions is None else self.nuclear_positions.to(device=device, dtype=dtype),
            nuclear_charges=None if self.nuclear_charges is None else self.nuclear_charges.to(device=device, dtype=dtype),
            spins=None if self.spins is None else self.spins.to(device=device, dtype=dtype),
        )

    def flatten_samples(self) -> "ElectronBatch":
        """Return a 3D batch by flattening all sample axes.

        Returns
        -------
        ElectronBatch
            Batch whose tensor fields use a single leading configuration axis.
        """

        positions = self.positions.reshape(self.n_configurations, self.n_electrons, self.spatial_dim)
        if self.nuclear_positions is None or self.nuclear_positions.ndim == 2:
            nuclear_positions = self.nuclear_positions
        else:
            nuclear_positions = self.nuclear_positions.reshape(self.n_configurations, *self.nuclear_positions.shape[-2:])
        if self.nuclear_charges is None or self.nuclear_charges.ndim == 1:
            nuclear_charges = self.nuclear_charges
        else:
            nuclear_charges = self.nuclear_charges.reshape(self.n_configurations, self.nuclear_charges.shape[-1])
        spins = None if self.spins is None else self.spins.reshape(self.n_configurations, self.n_electrons)
        return replace(
            self,
            positions=positions,
            nuclear_positions=nuclear_positions,
            nuclear_charges=nuclear_charges,
            spins=spins,
        )


@dataclass
class Walkers:
    """Store Monte Carlo walker state and cached model values.

    Parameters
    ----------
    positions : torch.Tensor
        Walker electron coordinates with shape
        ``[batch, n_electrons, spatial_dim]``.
    logabs : torch.Tensor or None, optional
        Cached log absolute wavefunction values with shape ``[batch]``.
    sign : torch.Tensor or None, optional
        Cached real wavefunction signs with shape ``[batch]``.
    spins : torch.Tensor or None, optional
        Fixed spin labels with shape ``[batch, n_electrons]`` and entries
        exactly equal to ``+1`` or ``-1``.
    aux : dict, optional
        Auxiliary sampler state and metadata.
    """

    positions: torch.Tensor
    logabs: torch.Tensor | None = None
    sign: torch.Tensor | None = None
    spins: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = self.positions if isinstance(self.positions, torch.Tensor) else torch.as_tensor(self.positions)
        self.logabs = _coerce_optional_tensor(self.logabs, dtype=self.positions.dtype)
        self.sign = _coerce_optional_tensor(self.sign, dtype=self.positions.dtype)
        self.spins = _coerce_optional_tensor(self.spins, dtype=self.positions.dtype)
        if self.positions.ndim != 3:
            raise ValueError("Walkers.positions must have shape [batch, n_electrons, spatial_dim]")
        if self.spins is not None:
            if tuple(self.spins.shape) != tuple(self.positions.shape[:2]):
                raise ValueError("Walkers.spins must have shape [batch, n_electrons]")
            if not torch.all((self.spins == 1) | (self.spins == -1)):
                raise ValueError("Walkers.spins entries must be exactly +1 or -1")
        if self.logabs is not None and self.logabs.shape[0] != self.positions.shape[0]:
            raise ValueError("Walkers.logabs must have shape [batch]")
        if self.sign is not None and self.sign.shape[0] != self.positions.shape[0]:
            raise ValueError("Walkers.sign must have shape [batch]")

    @property
    def device(self) -> torch.device:
        """Return the device of the walker positions.

        Returns
        -------
        torch.device
            Device on which `positions` is stored.
        """

        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the walker positions.

        Returns
        -------
        torch.dtype
            Data type used by `positions`.
        """

        return self.positions.dtype

    @property
    def batch_size(self) -> int:
        """Return the number of walkers.

        Returns
        -------
        int
            Size of the leading walker axis.
        """

        return self.positions.shape[0]

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "Walkers":
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
        Walkers
            Walker state with tensor fields moved to the requested device or
            dtype.
        """

        return replace(
            self,
            positions=self.positions.to(device=device, dtype=dtype),
            logabs=None if self.logabs is None else self.logabs.to(device=device, dtype=dtype),
            sign=None if self.sign is None else self.sign.to(device=device, dtype=dtype),
            spins=None if self.spins is None else self.spins.to(device=device, dtype=dtype),
        )


@dataclass
class WavefunctionOutput:
    """Store a wavefunction value in signed-log form.

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


def validate_output(
    output: WavefunctionOutput,
    *,
    batch_size: int | None = None,
    sample_shape: tuple[int, ...] | None = None,
) -> None:
    """Validate a wavefunction output shape.

    Parameters
    ----------
    output : WavefunctionOutput
        Output object to validate.
    batch_size : int or None, optional
        Legacy expected flattened batch size.
    sample_shape : tuple of int or None, optional
        Expected output sample shape.

    Raises
    ------
    ValueError
        If `output` does not match the expected sample shape or batch size.
    """

    if sample_shape is not None:
        sample_shape = tuple(sample_shape)
        if tuple(output.logabs.shape) != sample_shape:
            raise ValueError(f"Expected sample shape {sample_shape}, got {tuple(output.logabs.shape)}")
        if batch_size is not None and prod(sample_shape) != batch_size:
            raise ValueError(f"Expected batch size {batch_size}, got {prod(sample_shape)} from sample_shape")
    elif batch_size is not None and prod(tuple(output.logabs.shape)) != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got shape {tuple(output.logabs.shape)}")


def validate_batch(batch: ElectronBatch) -> None:
    """Validate an electron batch coordinate tensor.

    Parameters
    ----------
    batch : ElectronBatch
        Batch to validate.

    Raises
    ------
    ValueError
        If the position tensor does not satisfy the expected batch convention.
    """

    flat_batch = batch.flatten_samples()
    if flat_batch.positions.ndim != 3:
        raise ValueError("ElectronBatch positions must flatten to [batch, n_electrons, spatial_dim]")
    if flat_batch.positions.shape[0] != flat_batch.batch_size:
        raise ValueError("ElectronBatch flattened batch axis disagrees with batch_size")
