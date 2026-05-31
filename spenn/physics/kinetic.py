"""Autograd-based kinetic-energy estimators."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput


def _extract_logabs(output: WavefunctionOutput | torch.Tensor) -> torch.Tensor:
    if isinstance(output, WavefunctionOutput):
        return output.logabs
    return output


def autograd_laplacian(model, batch: ElectronBatch) -> torch.Tensor:
    """Return the Laplacian of ``log|psi|``.

    Parameters
    ----------
    model : callable
        Wavefunction model returning either `WavefunctionOutput` or a tensor of
        log absolute values.
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
    assert logabs.shape == (batch.batch_size,)
    grad = torch.autograd.grad(logabs.sum(), positions, create_graph=True)[0]
    assert grad.shape == positions.shape
    flat_grad = grad.reshape(grad.shape[0], -1)
    laplacian = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
    for idx in range(flat_grad.shape[1]):
        second = torch.autograd.grad(flat_grad[:, idx].sum(), positions, create_graph=True, retain_graph=True)[0]
        laplacian = laplacian + second.reshape(second.shape[0], -1)[:, idx]
    assert laplacian.shape == (batch.batch_size,)
    return laplacian


def kinetic_energy_from_logabs(model, batch: ElectronBatch) -> torch.Tensor:
    """Return local kinetic energy from the log-amplitude.

    Parameters
    ----------
    model : callable
        Wavefunction model returning either `WavefunctionOutput` or a tensor of
        log absolute values.
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
    assert logabs.shape == (batch.batch_size,)
    grad = torch.autograd.grad(logabs.sum(), positions, create_graph=True)[0]
    assert grad.shape == positions.shape
    flat_grad = grad.reshape(grad.shape[0], -1)
    laplacian = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
    for idx in range(flat_grad.shape[1]):
        second = torch.autograd.grad(flat_grad[:, idx].sum(), positions, create_graph=True, retain_graph=True)[0]
        laplacian = laplacian + second.reshape(second.shape[0], -1)[:, idx]
    output = -0.5 * (laplacian + flat_grad.pow(2).sum(dim=1))
    assert output.shape == (batch.batch_size,)
    return output


class LogAbsKineticEnergy(nn.Module):
    """Autograd kinetic-energy module for log-amplitude models."""

    def forward(self, model, batch: ElectronBatch) -> torch.Tensor:
        """Return local kinetic energy.

        Parameters
        ----------
        model : callable
            Wavefunction model returning log absolute values.
        batch : ElectronBatch
            Electron batch to evaluate.

        Returns
        -------
        torch.Tensor
            Kinetic local-energy contribution with shape ``[batch]``.
        """

        return kinetic_energy_from_logabs(model, batch)
