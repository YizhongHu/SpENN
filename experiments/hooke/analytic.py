"""Analytic wavefunctions and diagnostics for Hooke atom benchmarks."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput


class HookeExactWavefunction(nn.Module):
    """Analytic two-electron Hooke atom wavefunction.

    Parameters
    ----------
    sector : {"singlet", "triplet"}, optional
        Benchmark sector. The singlet is the opposite-spin symmetric spatial
        ground state at ``omega = 1 / 2``. The triplet is the same-spin
        antisymmetric spatial state at ``omega = 1 / 4``.
    node_axis : int, optional
        Cartesian axis used for the triplet nodal factor
        ``r_1[node_axis] - r_2[node_axis]``.
    eps : float, optional
        Distance floor used only for radial distances in diagnostics. Exact
        nodal zeros are still represented by ``sign == 0`` and
        ``logabs == -inf``.
    """

    def __init__(self, sector: str = "singlet", node_axis: int = 2, eps: float = 0.0) -> None:
        super().__init__()
        normalized = sector.lower().replace("-", "_")
        if normalized in {"singlet", "opposite_spin", "opposite_spin_singlet"}:
            self.sector = "singlet"
            self.cusp_slope = 0.5
            self.gaussian_coefficient = 0.25
            self.exact_energy = 2.0
        elif normalized in {"triplet", "same_spin", "same_spin_triplet"}:
            self.sector = "triplet"
            self.cusp_slope = 0.25
            self.gaussian_coefficient = 1.0 / 8.0
            self.exact_energy = 1.25
        else:
            raise ValueError(f"Unknown Hooke sector: {sector!r}")
        self.node_axis = int(node_axis)
        self.eps = float(eps)

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Evaluate the analytic wavefunction in signed-log form.

        Parameters
        ----------
        batch : ElectronBatch
            Electron positions with shape ``[batch, 2, 3]`` after sample
            flattening.

        Returns
        -------
        WavefunctionOutput
            Signed-log wavefunction values with shape ``[batch]``.
        """

        batch = batch.flatten_samples()
        _check_hooke_positions(batch.positions)
        positions = batch.positions
        r12 = torch.linalg.norm(positions[:, 0] - positions[:, 1], dim=-1).clamp_min(self.eps)
        radial_log = torch.log1p(self.cusp_slope * r12)
        gaussian_log = -self.gaussian_coefficient * positions.square().sum(dim=(1, 2))
        logabs = radial_log + gaussian_log
        sign = torch.ones_like(logabs)
        if self.sector == "triplet":
            node = positions[:, 0, self.node_axis] - positions[:, 1, self.node_axis]
            sign = torch.sign(node)
            logabs = logabs + torch.log(torch.abs(node))
        assert logabs.shape == (batch.batch_size,)
        assert sign.shape == (batch.batch_size,)
        return WavefunctionOutput(logabs=logabs, sign=sign)

    def factored_logabs(self, batch: ElectronBatch) -> torch.Tensor:
        """Return log-amplitude after removing the triplet node factor.

        Parameters
        ----------
        batch : ElectronBatch
            Electron positions with shape ``[batch, 2, 3]`` after sample
            flattening.

        Returns
        -------
        torch.Tensor
            For the singlet, this is ``log|psi|``. For the triplet, this is
            ``log|psi / (z_1 - z_2)|`` using `node_axis`.
        """

        batch = batch.flatten_samples()
        output = self(batch)
        if self.sector == "singlet":
            return output.logabs
        node = batch.positions[:, 0, self.node_axis] - batch.positions[:, 1, self.node_axis]
        log_node = torch.log(torch.abs(node))
        factored = output.logabs - log_node
        assert factored.shape == (batch.batch_size,)
        return factored


def hooke_spin_labels(
    sector: str,
    *,
    n_walkers: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return repeated spin labels for a Hooke benchmark sector.

    Parameters
    ----------
    sector : str
        Hooke sector name.
    n_walkers : int
        Number of walker rows to generate.
    device : torch.device, str, or None, optional
        Target device.
    dtype : torch.dtype, optional
        Target dtype.

    Returns
    -------
    torch.Tensor
        Spin labels with shape ``[n_walkers, 2]`` and entries ``+1`` or
        ``-1``.
    """

    normalized = sector.lower().replace("-", "_")
    if normalized in {"singlet", "opposite_spin", "opposite_spin_singlet"}:
        base = torch.tensor([1.0, -1.0], device=device, dtype=dtype)
    elif normalized in {"triplet", "same_spin", "same_spin_triplet"}:
        base = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown Hooke sector: {sector!r}")
    return base.unsqueeze(0).expand(n_walkers, -1).clone()


def _check_hooke_positions(positions: torch.Tensor) -> None:
    if positions.ndim != 3 or positions.shape[1:] != (2, 3):
        raise ValueError(f"Hooke positions must have shape [batch, 2, 3], got {tuple(positions.shape)}")
