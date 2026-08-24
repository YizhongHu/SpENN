"""Autograd-based kinetic-energy estimators."""

from __future__ import annotations

import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.physics.hamiltonian import LocalEnergyResult


def _extract_logabs(output: WavefunctionOutput) -> torch.Tensor:
    if not isinstance(output, WavefunctionOutput):
        raise TypeError(f"Wavefunction model must return WavefunctionOutput, got {type(output)!r}")
    return output.logabs


def autograd_laplacian(model, batch: ElectronBatch) -> torch.Tensor:
    """Return the Laplacian of ``log|psi|``.

    Parameters
    ----------
    model : callable
        Wavefunction model returning `WavefunctionOutput`.
    batch : ElectronBatch
        Electron batch with positions shaped ``[batch, n_electrons,
        spatial_dim]`` after flattening.

    Returns
    -------
    torch.Tensor
        Laplacian values with shape ``[batch]``.
    """

    batch = batch.flatten_samples()
    positions = batch.positions.detach().clone().requires_grad_(True)
    if positions.ndim != 3:
        raise ValueError("batch.positions must flatten to [batch, n_electrons, spatial_dim]")
    probe_batch = ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=batch.nuclear_positions,
        nuclear_charges=batch.nuclear_charges,
        spins=batch.spins,
        aux=dict(batch.aux),
    )
    output = model(probe_batch)
    logabs = _extract_logabs(output)
    if logabs.shape != (batch.batch_size,):
        raise ValueError(f"logabs must have shape {(batch.batch_size,)}, got {tuple(logabs.shape)}")
    grad = torch.autograd.grad(logabs.sum(), positions, create_graph=True)[0]
    if grad.shape != positions.shape:
        raise ValueError(f"logabs gradient must have shape {tuple(positions.shape)}, got {tuple(grad.shape)}")
    flat_grad = grad.reshape(grad.shape[0], -1)
    laplacian = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
    for idx in range(flat_grad.shape[1]):
        second = torch.autograd.grad(flat_grad[:, idx].sum(), positions, create_graph=True, retain_graph=True)[0]
        laplacian = laplacian + second.reshape(second.shape[0], -1)[:, idx]
    if laplacian.shape != (batch.batch_size,):
        raise ValueError(f"laplacian must have shape {(batch.batch_size,)}, got {tuple(laplacian.shape)}")
    return laplacian


def per_electron_kinetic_from_logabs_reference(model, batch: ElectronBatch) -> torch.Tensor:
    """Return slow per-electron kinetic attribution for diagnostic validation.

    The implementation deliberately evaluates one sample and one Cartesian
    coordinate at a time. It is the readable reference for the bounded
    vectorized atlas path and is not used by training or by `KineticEnergy`.

    Parameters
    ----------
    model : callable
        Wavefunction model returning `WavefunctionOutput`.
    batch : ElectronBatch
        Electron configurations to attribute.

    Returns
    -------
    torch.Tensor
        Per-electron kinetic values with shape ``[batch, n_electrons]``.
    """

    flat = batch.flatten_samples()
    rows: list[torch.Tensor] = []
    for sample_index in range(flat.batch_size):
        positions = flat.positions[sample_index : sample_index + 1].detach().clone().requires_grad_(True)
        probe_batch = _kinetic_probe_batch(flat, positions, sample_index, sample_index + 1)
        logabs = _extract_logabs(model(probe_batch))
        gradient = torch.autograd.grad(logabs.sum(), positions, create_graph=True)[0]
        electron_values: list[torch.Tensor] = []
        for electron in range(flat.n_electrons):
            laplacian = torch.zeros((), device=flat.device, dtype=flat.dtype)
            for coordinate in range(flat.spatial_dim):
                second = torch.autograd.grad(
                    gradient[0, electron, coordinate],
                    positions,
                    retain_graph=True,
                )[0]
                laplacian = laplacian + second[0, electron, coordinate]
            gradient_squared = gradient[0, electron].square().sum()
            electron_values.append(-0.5 * (laplacian + gradient_squared))
        # Detach only after every second derivative for this sample exists.
        rows.append(torch.stack(electron_values).detach())
    if not rows:
        return torch.empty(
            (0, flat.n_electrons), device=flat.device, dtype=flat.dtype
        )
    return torch.stack(rows)


def per_electron_kinetic_from_logabs(model, batch: ElectronBatch) -> torch.Tensor:
    """Return vectorized per-electron kinetic attribution for diagnostics.

    The first backward keeps ``create_graph=True``. Every diagonal Hessian
    entry is formed before the result is detached, so the path cannot silently
    drop the graph needed for curvature.

    Parameters
    ----------
    model : callable
        Wavefunction model returning `WavefunctionOutput`.
    batch : ElectronBatch
        Electron configurations to attribute.

    Returns
    -------
    torch.Tensor
        Per-electron kinetic values with shape ``[batch, n_electrons]``.
    """

    flat = batch.flatten_samples()
    positions = flat.positions.detach().clone().requires_grad_(True)
    probe_batch = _kinetic_probe_batch(flat, positions, 0, flat.batch_size)
    logabs = _extract_logabs(model(probe_batch))
    if tuple(logabs.shape) != (flat.batch_size,):
        raise ValueError(f"logabs must have shape {(flat.batch_size,)}, got {tuple(logabs.shape)}")
    gradient = torch.autograd.grad(logabs.sum(), positions, create_graph=True)[0]
    electron_laplacians: list[torch.Tensor] = []
    for electron in range(flat.n_electrons):
        coordinate_seconds: list[torch.Tensor] = []
        for coordinate in range(flat.spatial_dim):
            second = torch.autograd.grad(
                gradient[:, electron, coordinate].sum(),
                positions,
                retain_graph=True,
            )[0]
            coordinate_seconds.append(second[:, electron, coordinate])
        electron_laplacians.append(torch.stack(coordinate_seconds, dim=0).sum(dim=0))
    laplacian = torch.stack(electron_laplacians, dim=1)
    gradient_squared = gradient.square().sum(dim=-1)
    return (-0.5 * (laplacian + gradient_squared)).detach()


def _kinetic_probe_batch(
    batch: ElectronBatch,
    positions: torch.Tensor,
    start: int,
    end: int,
) -> ElectronBatch:
    """Build a graph-carrying kinetic probe without changing batch semantics."""

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


def kinetic_energy_from_logabs(model, batch: ElectronBatch) -> torch.Tensor:
    """Return local kinetic energy from the log-amplitude.

    Parameters
    ----------
    model : callable
        Wavefunction model returning `WavefunctionOutput`.
    batch : ElectronBatch
        Electron batch with positions shaped ``[batch, n_electrons,
        spatial_dim]`` after flattening.

    Returns
    -------
    torch.Tensor
        Kinetic local-energy contribution with shape ``[batch]``.
    """

    batch = batch.flatten_samples()
    positions = batch.positions.detach().clone().requires_grad_(True)
    if positions.ndim != 3:
        raise ValueError("batch.positions must flatten to [batch, n_electrons, spatial_dim]")
    probe_batch = ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=batch.nuclear_positions,
        nuclear_charges=batch.nuclear_charges,
        spins=batch.spins,
        aux=dict(batch.aux),
    )
    output = model(probe_batch)
    logabs = _extract_logabs(output)
    if logabs.shape != (batch.batch_size,):
        raise ValueError(f"logabs must have shape {(batch.batch_size,)}, got {tuple(logabs.shape)}")
    grad = torch.autograd.grad(logabs.sum(), positions, create_graph=True)[0]
    if grad.shape != positions.shape:
        raise ValueError(f"logabs gradient must have shape {tuple(positions.shape)}, got {tuple(grad.shape)}")
    flat_grad = grad.reshape(grad.shape[0], -1)
    laplacian = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
    for idx in range(flat_grad.shape[1]):
        second = torch.autograd.grad(flat_grad[:, idx].sum(), positions, create_graph=True, retain_graph=True)[0]
        laplacian = laplacian + second.reshape(second.shape[0], -1)[:, idx]
    output = -0.5 * (laplacian + flat_grad.pow(2).sum(dim=1))
    if output.shape != (batch.batch_size,):
        raise ValueError(f"kinetic local energy must have shape {(batch.batch_size,)}, got {tuple(output.shape)}")
    return output


class KineticEnergy:
    """Hamiltonian term for the quantum kinetic energy operator."""

    name = "kinetic"

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        value = kinetic_energy_from_logabs(wavefunction, batch)
        return LocalEnergyResult(total=value, terms={self.name: value})


__all__ = [
    "KineticEnergy",
    "autograd_laplacian",
    "kinetic_energy_from_logabs",
    "per_electron_kinetic_from_logabs",
    "per_electron_kinetic_from_logabs_reference",
]
