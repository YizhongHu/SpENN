"""Determinant readout."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data_structures.batch import ElectronBatch, WavefunctionOutput
from spenn.data_structures.feature_dict import FeatureDict


class DeterminantReadout(nn.Module):
    """Construct a square orbital matrix from order-1 features and take its determinant."""

    def __init__(self, num_orbitals: int | None = None, **_: object) -> None:
        super().__init__()
        self.num_orbitals = num_orbitals
        self._projection: nn.Module | None = None

    def _ensure_projection(self, n_electrons: int) -> None:
        if self._projection is None:
            output_dim = self.num_orbitals or n_electrons
            self._projection = nn.LazyLinear(output_dim, bias=False, dtype=torch.float64)
            self.add_module("projection", self._projection)

    def build_orbital_matrix(self, features: FeatureDict, batch: ElectronBatch) -> torch.Tensor:
        h = features.get(1, (1))
        if h is None:
            raise KeyError("DeterminantReadout requires one-body features at features[1][(1)]")
        self._ensure_projection(batch.n_electrons)
        self._projection = self._projection.to(device=h.device, dtype=h.dtype)
        matrix = self._projection(h)
        if matrix.shape[-1] != batch.n_electrons:
            raise ValueError(
                f"DeterminantReadout expects a square matrix with size {batch.n_electrons}, got {matrix.shape[-1]}"
            )
        return matrix

    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        matrix = self.build_orbital_matrix(features, batch)
        sign, logabs = torch.linalg.slogdet(matrix)
        return WavefunctionOutput(logabs=logabs, sign=sign, aux={"A": matrix})
