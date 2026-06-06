"""Determinant readout scaffold for real tuple features.

All readouts in the new SpENN core consume :class:`spenn.data.real.RealFeature`.
If determinant logic needs irrep coordinates, run the appropriate Fourier
transform before evaluating the readout itself.
"""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.data.real import RealFeature


class DeterminantReadout(nn.Module):
    """Build a one-body orbital matrix and return its determinant.

    Parameters
    ----------
    trainable : bool, optional
        Whether one-body channel-to-orbital projection weights are trainable.
        The fixed default uses deterministic identity-style channel selection.
    """

    def __init__(self, *, trainable: bool = False) -> None:
        super().__init__()
        self.trainable = bool(trainable)
        self.register_parameter("orbital_weights", None)
        self.register_buffer("orbital_weight_buffer", None, persistent=False)

    def build_orbital_matrix(self, features: RealFeature, batch: ElectronBatch) -> torch.Tensor:
        """Return the square orbital matrix consumed by ``slogdet``.

        Parameters
        ----------
        features : RealFeature
            Real feature state containing an order-1 block with shape
            ``[batch, channels, n_electrons]``.
        batch : ElectronBatch
            Electron batch used for batch-size and electron-count checks.

        Returns
        -------
        torch.Tensor
            Orbital matrix with shape ``[batch, n_electrons, n_electrons]``.
        """

        if 1 not in features:
            raise KeyError("DeterminantReadout requires an order-1 RealFeature block")
        one_body = features.blocks[1]
        if one_body.ndim != 3:
            raise ValueError(f"Order-1 block must have shape [batch, channels, n], got {tuple(one_body.shape)}")
        if one_body.shape[0] != batch.batch_size:
            raise ValueError("Feature batch size disagrees with ElectronBatch")
        if one_body.shape[-1] != batch.n_electrons:
            raise ValueError("Order-1 particle axis disagrees with ElectronBatch")
        weights = self._ensure_orbital_weights(
            n_electrons=batch.n_electrons,
            channels=one_body.shape[1],
            device=one_body.device,
            dtype=one_body.dtype,
        )
        matrix = torch.einsum("bcn,oc->bno", one_body, weights)
        assert matrix.shape == (batch.batch_size, batch.n_electrons, batch.n_electrons)
        return matrix

    def forward(self, features: RealFeature, batch: ElectronBatch) -> WavefunctionOutput:
        """Return a signed-log determinant readout."""

        matrix = self.build_orbital_matrix(features, batch)
        sign, logabs = torch.linalg.slogdet(matrix)
        assert logabs.shape == (batch.batch_size,)
        assert sign.shape == (batch.batch_size,)
        return WavefunctionOutput(logabs=logabs, sign=sign, aux={"A": matrix})

    def _ensure_orbital_weights(
        self,
        *,
        n_electrons: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        expected_shape = (n_electrons, channels)
        initial = _initial_orbital_weights(n_electrons, channels, device=device, dtype=dtype)
        if self.trainable:
            weights = self.orbital_weights
            if weights is None:
                self.orbital_weights = nn.Parameter(initial)
                weights = self.orbital_weights
            elif tuple(weights.shape) != expected_shape:
                raise ValueError(f"orbital_weights has shape {tuple(weights.shape)}, expected {expected_shape}")
            return weights.to(device=device, dtype=dtype)

        weights = self.orbital_weight_buffer
        if weights is None:
            self.orbital_weight_buffer = initial
            weights = self.orbital_weight_buffer
        elif tuple(weights.shape) != expected_shape:
            raise ValueError(f"orbital_weight_buffer has shape {tuple(weights.shape)}, expected {expected_shape}")
        return weights.to(device=device, dtype=dtype)


def _initial_orbital_weights(
    n_electrons: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return deterministic channel-to-orbital projection weights."""

    weights = torch.zeros(n_electrons, channels, device=device, dtype=dtype)
    for idx in range(min(n_electrons, channels)):
        weights[idx, idx] = 1.0
    return weights


__all__ = ["DeterminantReadout"]
