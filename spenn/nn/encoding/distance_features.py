"""Distance- and geometry-based feature helpers."""

from __future__ import annotations

import torch

from spenn.data_structures.batch import ElectronBatch
from spenn.utils.tensor_utils import pairwise_displacements, pairwise_distances


def one_body_raw_features(batch: ElectronBatch, include_spins: bool = True) -> torch.Tensor:
    """Build raw one-body features for each electron."""

    positions = batch.positions
    centered = positions - positions.mean(dim=1, keepdim=True)
    norm = torch.linalg.norm(positions, dim=-1, keepdim=True)
    centered_norm = torch.linalg.norm(centered, dim=-1, keepdim=True)
    ones = torch.ones(*positions.shape[:2], 1, device=positions.device, dtype=positions.dtype)
    parts = [positions, norm, centered_norm, ones]
    if include_spins:
        if batch.spins is None:
            spins = torch.zeros(*positions.shape[:2], 1, device=positions.device, dtype=positions.dtype)
        else:
            spins = batch.spins.unsqueeze(-1).to(dtype=positions.dtype)
        parts.append(spins)
    if batch.system is not None:
        n_electrons = torch.full_like(norm, float(getattr(batch.system, "n_electrons", positions.shape[1])))
        parts.append(n_electrons)
    return torch.cat(parts, dim=-1)


def pair_symmetric_raw_features(batch: ElectronBatch, include_spins: bool = True) -> torch.Tensor:
    """Build raw pairwise symmetric features."""

    positions = batch.positions
    disp = pairwise_displacements(positions)
    dist = pairwise_distances(positions)
    sum_pos = positions.unsqueeze(2) + positions.unsqueeze(1)
    sum_norm = torch.linalg.norm(sum_pos, dim=-1, keepdim=True)
    dot = (positions.unsqueeze(2) * positions.unsqueeze(1)).sum(dim=-1, keepdim=True)
    parts = [dist, dist.square(), sum_norm, dot]
    if include_spins:
        if batch.spins is None:
            spin_i = torch.zeros_like(dist)
            spin_j = torch.zeros_like(dist)
        else:
            spins = batch.spins.to(dtype=positions.dtype)
            spin_i = spins.unsqueeze(2).unsqueeze(-1)
            spin_j = spins.unsqueeze(1).unsqueeze(-1)
        parts.extend([spin_i + spin_j, spin_i * spin_j])
    return torch.cat(parts, dim=-1)


def pair_antisymmetric_raw_features(batch: ElectronBatch, include_spins: bool = True) -> torch.Tensor:
    """Build raw pairwise antisymmetric features."""

    positions = batch.positions
    disp = pairwise_displacements(positions)
    dist = pairwise_distances(positions)
    parts = [disp, disp * dist]
    if include_spins:
        if batch.spins is None:
            spin_diff = torch.zeros_like(dist)
        else:
            spins = batch.spins.to(dtype=positions.dtype)
            spin_diff = (spins.unsqueeze(2) - spins.unsqueeze(1)).unsqueeze(-1)
        parts.append(spin_diff)
    return torch.cat(parts, dim=-1)
