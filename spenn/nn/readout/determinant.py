"""Determinant readout."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.irrep_features import FeatureDict
from spenn.data.irrep_tensor import scalar_channels_last
from spenn.data.partitions import Par


class DeterminantReadout(nn.Module):
    """Construct a square orbital matrix and take its determinant.

    Parameters
    ----------
    num_orbitals : int or None, optional
        Number of projected orbital columns. If ``None``, use the electron
        count from the runtime batch.
    **_ : object
        Ignored compatibility keyword arguments.
    """

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
        """Build the determinant orbital matrix.

        Parameters
        ----------
        features : FeatureDict
            Feature dictionary containing one-body scalar features at
            ``Par("H")``.
        batch : ElectronBatch
            Electron batch used for shape checks.

        Returns
        -------
        torch.Tensor
            Square orbital matrix with shape ``[batch, n_electrons,
            n_electrons]``.
        """

        h = features.get(Par("H"))
        if h is None:
            raise KeyError("DeterminantReadout requires one-body features at features[1][(1)]")
        self._ensure_projection(batch.n_electrons)
        self._projection = self._projection.to(device=h.device, dtype=h.dtype)
        h_channels_last = scalar_channels_last(h)
        assert h_channels_last.shape[:2] == (batch.batch_size, batch.n_electrons)
        matrix = self._projection(h_channels_last)
        if matrix.shape[-1] != batch.n_electrons:
            raise ValueError(
                f"DeterminantReadout expects a square matrix with size {batch.n_electrons}, got {matrix.shape[-1]}"
            )
        assert matrix.shape == (batch.batch_size, batch.n_electrons, batch.n_electrons)
        return matrix

    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        """Return a signed-log determinant readout.

        Parameters
        ----------
        features : FeatureDict
            SpENN features used to build the orbital matrix.
        batch : ElectronBatch
            Electron batch with positions shaped ``[batch, n_electrons,
            spatial_dim]`` after sample flattening.

        Returns
        -------
        WavefunctionOutput
            Signed-log wavefunction output with shape ``[batch]``.
        """

        matrix = self.build_orbital_matrix(features, batch)
        sign, logabs = torch.linalg.slogdet(matrix)
        assert logabs.shape == (batch.batch_size,)
        assert sign.shape == (batch.batch_size,)
        return WavefunctionOutput(logabs=logabs, sign=sign, aux={"A": matrix})
