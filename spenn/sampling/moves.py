"""Proposal move helpers."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import Walkers


def gaussian_proposal(
    positions: torch.Tensor,
    step_size: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a Gaussian random-walk proposal.

    Parameters
    ----------
    positions : torch.Tensor
        Current positions with shape ``[batch, n_electrons, spatial_dim]``.
    step_size : float
        Standard deviation of the isotropic Gaussian perturbation.
    generator : torch.Generator or None, optional
        RNG stream consumed for the proposal noise. When ``None``, the default
        global RNG is used. Stateful samplers should pass their own generator so
        the Markov chain owns its randomness.

    Returns
    -------
    torch.Tensor
        Proposed positions with the same shape as `positions`.
    """

    if positions.ndim != 3:
        raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
    noise = torch.randn(
        positions.shape, device=positions.device, dtype=positions.dtype, generator=generator
    )
    return positions + step_size * noise


class GaussianMove(nn.Module):
    """Gaussian random-walk proposal module.

    The move owns the proposal shape/rules; it does not own an RNG. The sampler
    that drives it passes the generator for every proposal so that all
    Markov-chain randomness belongs to the sampler.

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

    def forward(
        self,
        positions: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return proposed positions.

        Parameters
        ----------
        positions : torch.Tensor
            Current positions with shape ``[batch, n_electrons, spatial_dim]``.
        generator : torch.Generator or None, optional
            RNG stream consumed for proposal noise and one-electron index
            selection. When ``None``, the default global RNG is used.

        Returns
        -------
        torch.Tensor
            Proposed positions with the same shape as `positions`.
        """

        if positions.ndim != 3:
            raise ValueError("positions must have shape [batch, n_electrons, spatial_dim]")
        if self.move_all:
            return gaussian_proposal(positions, self.step_size, generator=generator)
        proposals = positions.clone()
        electron_idx = torch.randint(
            positions.shape[1], (positions.shape[0],), device=positions.device, generator=generator
        )
        walker_idx = torch.arange(positions.shape[0], device=positions.device)
        selected = proposals[walker_idx, electron_idx]
        noise = torch.randn(
            selected.shape, device=selected.device, dtype=selected.dtype, generator=generator
        )
        proposals[walker_idx, electron_idx] = selected + self.step_size * noise
        return proposals

    def propose(
        self,
        walkers: Walkers,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return proposed positions and proposal log-ratio.

        Parameters
        ----------
        walkers : Walkers
            Current walker state.
        generator : torch.Generator or None, optional
            RNG stream consumed for the proposal, passed by the sampler.

        Returns
        -------
        tuple of torch.Tensor
            Proposed positions with shape ``[batch, n_electrons, spatial_dim]``
            and ``log q(current | proposed) - log q(proposed | current)`` with
            shape ``[batch]``. The Gaussian random walk is symmetric, so the
            log-ratio is zero.
        """

        proposed = self.forward(walkers.positions, generator=generator)
        log_q_ratio = torch.zeros(walkers.batch_size, device=walkers.device, dtype=walkers.dtype)
        return proposed, log_q_ratio
