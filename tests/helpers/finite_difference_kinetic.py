"""Optional Numdifftools finite-difference kinetic oracle for pytest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput


@dataclass(frozen=True)
class FiniteDifferenceKineticResult:
    """Results and diagnostics returned by the finite-difference oracle.

    All tensors are CPU ``float64`` tensors. Status masks are indexed by the
    flattened configuration axis and retain the explicit reason for an
    unavailable or untrusted finite-difference result.
    """

    center: WavefunctionOutput
    hessian_diagonal: torch.Tensor
    total_kinetic: torch.Tensor
    per_electron_kinetic: torch.Tensor
    error_estimate: torch.Tensor
    final_step: torch.Tensor
    center_node: torch.Tensor
    nonfinite_probe: torch.Tensor
    exceeded_tolerance: torch.Tensor

    @property
    def statuses(self) -> dict[str, torch.Tensor]:
        """Return the explicit TPEN status masks by canonical name."""

        return {
            "center_node": self.center_node,
            "nonfinite_probe": self.nonfinite_probe,
            "exceeded_tolerance": self.exceeded_tolerance,
        }


def finite_difference_kinetic(
    model: Any,
    batch: ElectronBatch,
    *,
    step: float | np.ndarray | None = None,
    tolerance: float = 1.0e-8,
    method: str = "central",
    order: int = 2,
) -> FiniteDifferenceKineticResult:
    """Estimate kinetic energy with ``numdifftools.Hessdiag``.

    The differentiated function is the direct signed amplitude normalized at
    each center, ``psi(q) / psi(q0) - 1``. The model is evaluated under
    ``torch.no_grad()`` on a typed CPU/float64 ``ElectronBatch`` probe.

    Notes
    -----
    This helper intentionally imports Numdifftools lazily. Install the
    ``finite-difference`` project extra to use it.
    """

    try:
        import numdifftools as nd
    except ImportError as exc:  # pragma: no cover - exercised by optional test
        raise ImportError(
            "finite_difference_kinetic requires the optional `finite-difference` "
            "extra; install with `uv sync --extra finite-difference`"
        ) from exc

    flat = batch.flatten_samples()
    if flat.positions.device.type != "cpu":
        raise ValueError("finite_difference_kinetic requires CPU positions")
    flat = flat.to(device="cpu", dtype=torch.float64)
    n_batch, n_electrons, spatial_dim = flat.batch_size, flat.n_electrons, flat.spatial_dim
    shape = (n_batch, n_electrons, spatial_dim)
    nan = torch.full(shape, float("nan"), dtype=torch.float64)
    center_node = torch.zeros(n_batch, dtype=torch.bool)
    nonfinite_probe = torch.zeros(n_batch, dtype=torch.bool)
    exceeded_tolerance = torch.zeros(n_batch, dtype=torch.bool)

    with torch.no_grad():
        center = model(_probe_batch(flat, flat.positions, 0, n_batch))
    if not isinstance(center, WavefunctionOutput):
        raise TypeError(f"wavefunction model must return WavefunctionOutput, got {type(center)!r}")
    center.validate(batch_size=n_batch)
    center = center.to(device="cpu", dtype=torch.float64)
    center_node = center.sign == 0

    hessian = nan.clone()
    errors = nan.clone()
    steps = nan.clone()
    valid = ~center_node
    for sample in range(n_batch):
        if not bool(valid[sample]):
            continue
        center_logabs = float(center.logabs[sample].item())
        center_sign = float(center.sign[sample].item())

        def normalized_amplitude(coordinates: np.ndarray) -> float:
            positions = torch.as_tensor(coordinates, dtype=torch.float64).reshape_as(flat.positions[sample])
            with torch.no_grad():
                output = model(_probe_batch(flat, positions.unsqueeze(0), sample, sample + 1))
            if not isinstance(output, WavefunctionOutput):
                raise TypeError(f"wavefunction model must return WavefunctionOutput, got {type(output)!r}")
            output.validate(batch_size=1)
            logabs = output.logabs[0]
            sign = output.sign[0]
            if not bool(torch.isfinite(logabs) & torch.isfinite(sign)):
                nonfinite_probe[sample] = True
                return float("nan")
            # This is psi/psi0 - 1 reconstructed from signed-log fields,
            # rather than a finite difference of logabs.
            ratio = float(sign.item()) / center_sign * np.exp(float(logabs.item()) - center_logabs)
            return ratio - 1.0

        differentiator = nd.Hessdiag(
            normalized_amplitude,
            step=step,
            method=method,
            order=order,
            full_output=True,
        )
        values, info = differentiator(flat.positions[sample].reshape(-1).numpy())
        if nonfinite_probe[sample] or not np.all(np.isfinite(values)):
            nonfinite_probe[sample] = True
            continue
        error = np.asarray(info.error_estimate, dtype=np.float64).reshape(-1)
        final_step = np.asarray(info.final_step, dtype=np.float64).reshape(-1)
        if error.size != n_electrons * spatial_dim or final_step.size != error.size:
            raise AssertionError("Numdifftools Hessdiag returned an unexpected diagnostic shape")
        hessian[sample] = torch.from_numpy(np.asarray(values, dtype=np.float64).reshape(shape[1:]))
        errors[sample] = torch.from_numpy(error.reshape(shape[1:]))
        steps[sample] = torch.from_numpy(final_step.reshape(shape[1:]))
        exceeded_tolerance[sample] = bool(np.any(error > tolerance))

    total = -0.5 * torch.nansum(hessian, dim=(1, 2))
    per_electron = -0.5 * torch.nansum(hessian, dim=2)
    invalid = center_node | nonfinite_probe
    total[invalid] = float("nan")
    per_electron[invalid] = float("nan")
    return FiniteDifferenceKineticResult(
        center=center,
        hessian_diagonal=hessian,
        total_kinetic=total,
        per_electron_kinetic=per_electron,
        error_estimate=errors,
        final_step=steps,
        center_node=center_node,
        nonfinite_probe=nonfinite_probe,
        exceeded_tolerance=exceeded_tolerance,
    )


def _probe_batch(batch: ElectronBatch, positions: torch.Tensor, start: int, end: int) -> ElectronBatch:
    """Construct a typed probe batch while changing only electron positions."""

    nuclear_positions = batch.nuclear_positions
    if nuclear_positions is not None and nuclear_positions.ndim == 3:
        nuclear_positions = nuclear_positions[start:end]
    nuclear_charges = batch.nuclear_charges
    if nuclear_charges is not None and nuclear_charges.ndim == 2:
        nuclear_charges = nuclear_charges[start:end]
    spins = None if batch.spins is None else batch.spins[start:end]
    return ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=nuclear_positions,
        nuclear_charges=nuclear_charges,
        atomic_configuration=batch.atomic_configuration,
        spins=spins,
        aux=dict(batch.aux),
    )


__all__ = ["FiniteDifferenceKineticResult", "finite_difference_kinetic"]
