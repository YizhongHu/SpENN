"""Immutable fixed Born-Oppenheimer nuclear geometry value.

`AtomicConfiguration` is the sole typed representation of a fixed nuclear
geometry: nuclear positions $R_A$ and charges $Z_A$ for one system, held
constant across a whole training/evaluation run and passed once, at
construction, to whichever `HamiltonianTerm`, cusp, or decay/confinement
module needs it (see `main.typ`, "Electron-nucleus cusp (deferred)"). This
module owns only that value -- construction, validation, device/dtype
materialization, and equality/compare/identity. It does not own
`ElectronBatch`, sampler, Hamiltonian, wavefunction, config, or experiment
wiring.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

JsonScalar = int | float | bool | str | None


@dataclass(frozen=True, eq=False)
class AtomicConfiguration:
    """Store one fixed, unbatched Born-Oppenheimer nuclear geometry.

    Parameters
    ----------
    positions : torch.Tensor
        Nuclear coordinates with shape ``[n_nuclei, spatial_dim]``.
    charges : torch.Tensor
        Nuclear charges with shape ``[n_nuclei]``. Every entry must be
        finite and strictly positive.

    Notes
    -----
    Instances are immutable: `positions` and `charges` are cloned and
    detached from any input autograd graph at construction, so later
    mutation of the caller's original tensors cannot change the recorded
    configuration. There is exactly one instance per fixed molecule (e.g.
    one for helium, one for molecular hydrogen) -- it is data, not a
    wavefunction subclass or branch.
    """

    positions: torch.Tensor
    charges: torch.Tensor

    def __post_init__(self) -> None:
        positions = _as_owned_tensor(self.positions)
        charges = _as_owned_tensor(self.charges)

        if positions.ndim != 2:
            raise ValueError(
                "AtomicConfiguration.positions must have shape [n_nuclei, spatial_dim], "
                f"got {tuple(positions.shape)}"
            )
        n_nuclei, spatial_dim = positions.shape
        if n_nuclei < 1:
            raise ValueError("AtomicConfiguration requires at least one nucleus")
        if spatial_dim < 1:
            raise ValueError("AtomicConfiguration.positions spatial dimension must be at least 1")
        if charges.shape != (n_nuclei,):
            raise ValueError(
                f"AtomicConfiguration.charges must have shape [{n_nuclei}], got {tuple(charges.shape)}"
            )
        if not torch.isfinite(positions).all():
            raise ValueError("AtomicConfiguration.positions must be finite")
        if not torch.isfinite(charges).all():
            raise ValueError("AtomicConfiguration.charges must be finite")
        if not torch.all(charges > 0):
            raise ValueError("AtomicConfiguration.charges must be strictly positive")
        _validate_distinct_nuclei(positions)

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "charges", charges)

    @property
    def n_nuclei(self) -> int:
        """Return the number of nuclei.

        Returns
        -------
        int
            Size of the nucleus axis.
        """

        return self.positions.shape[0]

    @property
    def spatial_dim(self) -> int:
        """Return the spatial dimension of each nuclear coordinate.

        Returns
        -------
        int
            Size of the final coordinate axis.
        """

        return self.positions.shape[1]

    @property
    def device(self) -> torch.device:
        """Return the device holding this configuration's tensors.

        Returns
        -------
        torch.device
            Device of `positions` (and `charges`, which always match it).
        """

        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating dtype of `positions`.

        Returns
        -------
        torch.dtype
            Data type used by `positions`.
        """

        return self.positions.dtype

    def validate(self) -> "AtomicConfiguration":
        """Validate this configuration using the constructor invariants.

        Returns
        -------
        AtomicConfiguration
            This configuration, for fluent runtime validation.
        """

        AtomicConfiguration(positions=self.positions, charges=self.charges)
        return self

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "AtomicConfiguration":
        """Return a copy materialized on a new device or dtype.

        Parameters
        ----------
        device : torch.device, str, or None, optional
            Target device. If ``None``, the current device is preserved.
        dtype : torch.dtype or None, optional
            Target floating-point dtype. If ``None``, the current dtype is
            preserved. Applied to `positions`; `charges` follows the same
            dtype.

        Returns
        -------
        AtomicConfiguration
            New configuration with tensors moved to the requested device or
            dtype. The original instance is unchanged.
        """

        return AtomicConfiguration(
            positions=self.positions.to(device=device, dtype=dtype),
            charges=self.charges.to(device=device, dtype=dtype),
        )

    def compare(
        self,
        other: "AtomicConfiguration",
        *,
        atol: float = 1e-8,
        rtol: float = 1e-5,
    ) -> tuple[bool, Mapping[str, JsonScalar]]:
        """Return ``(is_close, metrics)`` versus another configuration.

        Parameters
        ----------
        other : AtomicConfiguration
            Configuration to compare against.
        atol : float, optional
            Absolute tolerance passed to `torch.allclose`.
        rtol : float, optional
            Relative tolerance passed to `torch.allclose`.

        Returns
        -------
        tuple of (bool, mapping)
            Whether `positions` and `charges` are close, and JSON-safe
            max-absolute-error metrics. Mismatched shapes compare as not
            close with infinite error rather than raising.
        """

        if self.positions.shape != other.positions.shape or self.charges.shape != other.charges.shape:
            return False, {"positions_max_abs_error": float("inf"), "charges_max_abs_error": float("inf")}

        other_positions = other.positions.to(device=self.positions.device, dtype=self.positions.dtype)
        other_charges = other.charges.to(device=self.charges.device, dtype=self.charges.dtype)

        positions_error = float((self.positions - other_positions).abs().max().item())
        charges_error = float((self.charges - other_charges).abs().max().item())
        is_close = bool(
            torch.allclose(self.positions, other_positions, atol=atol, rtol=rtol)
            and torch.allclose(self.charges, other_charges, atol=atol, rtol=rtol)
        )
        return is_close, {
            "positions_max_abs_error": positions_error,
            "charges_max_abs_error": charges_error,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicConfiguration):
            return NotImplemented
        return (
            self.positions.shape == other.positions.shape
            and self.charges.shape == other.charges.shape
            and torch.equal(self.positions, other.positions.to(device=self.positions.device, dtype=self.positions.dtype))
            and torch.equal(self.charges, other.charges.to(device=self.charges.device, dtype=self.charges.dtype))
        )

    def __hash__(self) -> int:
        return hash(self.content_id())

    def content_id(self) -> str:
        """Return a reproducible content identity for this configuration.

        Returns
        -------
        str
            Hex-digest fingerprint derived only from the numerical content
            of `positions` and `charges` (via a device/dtype-independent
            float64 CPU canonicalization) and from `n_nuclei`/`spatial_dim`.
            Two configurations with the same nuclear data produce the same
            id regardless of device, dtype, or process, unlike Python's
            randomized ``hash()``.
        """

        digest = hashlib.sha256()
        digest.update(str((self.n_nuclei, self.spatial_dim)).encode("utf-8"))
        digest.update(self.positions.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy().tobytes())
        digest.update(self.charges.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy().tobytes())
        return digest.hexdigest()


def strict_equal_atomic_configurations(
    left: AtomicConfiguration,
    right: AtomicConfiguration,
) -> bool:
    """Return exact, symmetric equality after common dtype promotion.

    Unlike :meth:`AtomicConfiguration.__eq__`, this comparison is intended for
    physics authority checks. Both operands are promoted to the same dtype and
    moved to CPU before comparison, so neither operand is narrowed and device
    placement cannot affect the result.
    """

    if left.positions.shape != right.positions.shape or left.charges.shape != right.charges.shape:
        return False

    positions_dtype = torch.promote_types(left.positions.dtype, right.positions.dtype)
    charges_dtype = torch.promote_types(left.charges.dtype, right.charges.dtype)
    left_positions = left.positions.to(device="cpu", dtype=positions_dtype)
    right_positions = right.positions.to(device="cpu", dtype=positions_dtype)
    left_charges = left.charges.to(device="cpu", dtype=charges_dtype)
    right_charges = right.charges.to(device="cpu", dtype=charges_dtype)
    return bool(
        torch.equal(left_positions, right_positions)
        and torch.equal(left_charges, right_charges)
    )


def _as_owned_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.clone().detach().requires_grad_(False)


def _validate_distinct_nuclei(positions: torch.Tensor, *, eps: float = 1e-9) -> None:
    n_nuclei = positions.shape[0]
    if n_nuclei < 2:
        return
    displacements = positions.unsqueeze(1) - positions.unsqueeze(0)
    pairwise_distances = displacements.norm(dim=-1)
    off_diagonal = ~torch.eye(n_nuclei, dtype=torch.bool, device=positions.device)
    if torch.any(pairwise_distances[off_diagonal] < eps):
        raise ValueError("AtomicConfiguration nuclei must occupy distinct positions")


__all__ = ["AtomicConfiguration"]
