"""Metropolis-adjusted Langevin sampler."""

from __future__ import annotations

import torch

from spenn.data.batch import Walkers, WavefunctionOutput
from spenn.sampling.metropolis import MetropolisSampler


class MALASampler(MetropolisSampler):
    """Metropolis-adjusted Langevin sampler for ``|psi|^2`` targets.

    The proposal is

    ``X' = X + step_size**2 * grad_X log|psi(X)| + step_size * Normal(0, I)``.

    Since the VMC target has log density ``2 log|psi|``, this is the standard
    MALA proposal with Gaussian standard deviation `step_size`.

    Parameters
    ----------
    name : str, optional
        Human-readable sampler name.
    n_walkers : int, optional
        Default number of walkers to initialize.
    warmup_steps : int, optional
        Suggested warmup length for callers.
    steps_per_iter : int, optional
        Default number of MCMC steps per training iteration.
    step_size : float, optional
        Gaussian proposal standard deviation.
    n_electrons : int, optional
        Default electron count used when no system is supplied.
    spatial_dim : int, optional
        Default spatial dimension used when no system is supplied.
    initial_scale : float, optional
        Standard deviation of normally initialized walker positions.
    dtype : torch.dtype or str, optional
        Floating-point dtype for initialized walkers.
    """

    def __init__(
        self,
        name: str = "mala",
        n_walkers: int = 1024,
        warmup_steps: int = 100,
        steps_per_iter: int = 10,
        step_size: float = 0.05,
        n_electrons: int = 2,
        spatial_dim: int = 3,
        initial_scale: float = 1.0,
        dtype: torch.dtype | str = torch.float64,
    ) -> None:
        if step_size <= 0.0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        super().__init__(
            name=name,
            n_walkers=n_walkers,
            warmup_steps=warmup_steps,
            steps_per_iter=steps_per_iter,
            step_size=step_size,
            n_electrons=n_electrons,
            spatial_dim=spatial_dim,
            initial_scale=initial_scale,
            dtype=dtype,
        )
        self.step_size = float(step_size)

    def _propose(self, model, walkers: Walkers) -> tuple[torch.Tensor, torch.Tensor]:
        current_positions = walkers.positions.detach()
        current_grad = self._logabs_gradient(model, walkers.with_positions(current_positions, invalidate_cache=True))
        drift_scale = self.step_size * self.step_size
        proposals = current_positions + drift_scale * current_grad + self.step_size * torch.randn_like(current_positions)
        proposal_walkers = walkers.with_positions(proposals.detach(), invalidate_cache=True)
        proposal_grad = self._logabs_gradient(model, proposal_walkers)
        log_q_ratio = self._proposal_log_ratio(
            current_positions=current_positions,
            proposed_positions=proposals,
            current_grad=current_grad,
            proposed_grad=proposal_grad,
        )
        return proposals.detach(), log_q_ratio.detach()

    def _logabs_gradient(self, model, walkers: Walkers) -> torch.Tensor:
        positions = walkers.positions.detach().clone().requires_grad_(True)
        gradient_walkers = Walkers(
            positions=positions,
            spins=walkers.spins,
            aux=dict(walkers.aux),
        )
        with torch.enable_grad():
            output = model(gradient_walkers.make_batch())
            if not isinstance(output, WavefunctionOutput):
                raise TypeError(f"Wavefunction model must return WavefunctionOutput, got {type(output)!r}")
            logabs = output.logabs
            if logabs.shape != (walkers.batch_size,):
                raise ValueError(f"Model logabs must have shape [{walkers.batch_size}], got {tuple(logabs.shape)}")
            grad = torch.autograd.grad(logabs.sum(), positions, create_graph=False, retain_graph=False)[0]
        if grad.shape != walkers.positions.shape:
            raise ValueError(f"MALA gradient must have shape {tuple(walkers.positions.shape)}, got {tuple(grad.shape)}")
        return grad.detach()

    def _proposal_log_ratio(
        self,
        *,
        current_positions: torch.Tensor,
        proposed_positions: torch.Tensor,
        current_grad: torch.Tensor,
        proposed_grad: torch.Tensor,
    ) -> torch.Tensor:
        drift_scale = self.step_size * self.step_size
        variance = drift_scale
        forward_mean = current_positions + drift_scale * current_grad
        reverse_mean = proposed_positions + drift_scale * proposed_grad
        reduce_dims = tuple(range(1, current_positions.ndim))
        forward_sq = (proposed_positions - forward_mean).square().sum(dim=reduce_dims)
        reverse_sq = (current_positions - reverse_mean).square().sum(dim=reduce_dims)
        return (forward_sq - reverse_sq) / (2.0 * variance)


__all__ = ["MALASampler"]
