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
    """Return the Laplacian of ``log|psi|`` for a batched electron configuration."""

    positions = batch.positions.detach().clone().requires_grad_(True)
    assert positions.ndim == 3
    probe_batch = ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=batch.nuclear_positions,
        nuclear_charges=batch.nuclear_charges,
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
    """Return the local kinetic energy from the log-amplitude."""

    positions = batch.positions.detach().clone().requires_grad_(True)
    assert positions.ndim == 3
    probe_batch = ElectronBatch(
        positions=positions,
        system=batch.system,
        nuclear_positions=batch.nuclear_positions,
        nuclear_charges=batch.nuclear_charges,
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
    output = torch.nan_to_num(-0.5 * (laplacian + flat_grad.pow(2).sum(dim=1)))
    assert output.shape == (batch.batch_size,)
    return output


class LogAbsKineticEnergy(nn.Module):
    """Autograd kinetic-energy module for models returning ``log|psi|``."""

    def forward(self, model, batch: ElectronBatch) -> torch.Tensor:
        return kinetic_energy_from_logabs(model, batch)
