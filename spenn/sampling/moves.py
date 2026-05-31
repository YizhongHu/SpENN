"""Proposal move helpers."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import Walkers


def gaussian_proposal(positions: torch.Tensor, step_size: float) -> torch.Tensor:
    """Return a Gaussian random-walk proposal.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions with shape ``[batch, n_electrons, spatial_dim]``.
    step_size : float
        Standard deviation of the isotropic Gaussian perturbation.

    Returns
    -------
    torch.Tensor
        Proposed positions with the same shape as `positions`.
    """

    if positions.ndim != 3:
        raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
    return positions + step_size * torch.randn_like(positions)


class GaussianMove(nn.Module):
    """Gaussian random-walk proposal module.

    Parameters
    ----------
    step_size : float, optional
        Standard deviation of each coordinate perturbation.
    move_all : bool, optional
        If ``True``, perturb every electron in every walker. If ``False``,
        perturb one randomly selected electron per walker.
    """

    def __init__(self, step_size: float = 0.05, move_all: bool = True) -> None:
        super().__init__()
        self.step_size = step_size
        self.move_all = move_all

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Return proposed positions.

        Parameters
        ----------
        positions : torch.Tensor
            Current positions with shape ``[batch, n_electrons, spatial_dim]``.

        Returns
        -------
        torch.Tensor
            Proposed positions with the same shape as `positions`.
        """

        if positions.ndim != 3:
            raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
        if self.move_all:
            return gaussian_proposal(positions, self.step_size)
        proposals = positions.clone()
        electron_idx = torch.randint(positions.shape[1], (positions.shape[0],), device=positions.device)
        walker_idx = torch.arange(positions.shape[0], device=positions.device)
        selected = proposals[walker_idx, electron_idx]
        proposals[walker_idx, electron_idx] = selected + self.step_size * torch.randn_like(selected)
        return proposals

    def propose(self, walkers: Walkers, model=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return proposed positions and proposal log-ratio.

        Parameters
        ----------
        walkers : Walkers
            Current walker state.
        model : callable or None, optional
            Unused compatibility argument for proposal kernels that need model
            information.

        Returns
        -------
        tuple of torch.Tensor
            Proposed positions with shape ``[batch, n_electrons, spatial_dim]``
            and ``log q(current | proposed) - log q(proposed | current)`` with
            shape ``[batch]``. The Gaussian random walk is symmetric, so the
            log-ratio is zero.
        """

        del model
        proposed = self.forward(walkers.positions)
        log_q_ratio = torch.zeros(walkers.batch_size, device=walkers.device, dtype=walkers.dtype)
        return proposed, log_q_ratio
