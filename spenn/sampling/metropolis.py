"""Batched Metropolis sampler."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from spenn.physics.systems import ElectronicSystem
from spenn.sampling.moves import GaussianMove


class MetropolisSampler(nn.Module):
    """Batched Metropolis-Hastings sampler.

    Parameters
    ----------
    name : str, optional
        Human-readable sampler name.
    move : torch.nn.Module or None, optional
        Proposal kernel exposing ``propose(walkers)`` and returning proposed
        positions plus a proposal log-ratio.
    n_walkers : int, optional
        Default number of walkers to initialize.
    warmup_steps : int, optional
        Suggested warmup length for callers.
    steps_per_iter : int, optional
        Default number of MCMC steps per training iteration.
    step_size : float, optional
        Gaussian proposal step size used when `move` is ``None``.
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
        name: str = "metropolis",
        move: nn.Module | None = None,
        n_walkers: int = 1024,
        warmup_steps: int = 100,
        steps_per_iter: int = 10,
        step_size: float = 0.05,
        n_electrons: int = 2,
        spatial_dim: int = 3,
        initial_scale: float = 1.0,
        dtype: torch.dtype | str = torch.float64,
    ) -> None:
        super().__init__()
        self.name = name
        self.move = move or GaussianMove(step_size=step_size)
        self.n_walkers = n_walkers
        self.warmup_steps = warmup_steps
        self.steps_per_iter = steps_per_iter
        self.n_electrons = n_electrons
        self.spatial_dim = spatial_dim
        self.initial_scale = initial_scale
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.acceptance_rate = 0.0
        self.last_metrics: dict[str, float] = {}

    def initialize(self, system: ElectronicSystem | None = None, n_walkers: int | None = None, device=None) -> Walkers:
        """Initialize normally distributed walkers.

        Parameters
        ----------
        system : ElectronicSystem or None, optional
            System metadata. If absent, a default `ElectronicSystem` is built
            from sampler dimensions.
        n_walkers : int or None, optional
            Number of walkers to initialize. If ``None``, `self.n_walkers` is
            used.
        device : torch.device, str, or None, optional
            Target device. If ``None``, the system device is used.

        Returns
        -------
        Walkers
            Walker state with positions shaped ``[n_walkers, n_electrons,
            spatial_dim]`` and system metadata in ``aux``.
        """

        system = system or ElectronicSystem(n_electrons=self.n_electrons, spatial_dim=self.spatial_dim, dtype=self.dtype)
        self.system = system
        n_walkers = n_walkers or self.n_walkers
        dtype = getattr(torch, system.dtype) if isinstance(system.dtype, str) else (system.dtype or self.dtype)
        device = device or system.device
        positions = self.initial_scale * torch.randn(
            n_walkers, system.n_electrons, system.spatial_dim, device=device, dtype=dtype
        )
        spins = _default_spins(system, n_walkers=n_walkers, device=device, dtype=dtype)
        return Walkers(positions=positions, spins=spins, aux={"system": system})

    def _evaluate(self, model, walkers: Walkers) -> tuple[torch.Tensor, torch.Tensor]:
        batch = walkers.make_batch()
        with torch.no_grad():
            output = model(batch)
        if not isinstance(output, WavefunctionOutput):
            raise TypeError(f"Wavefunction model must return WavefunctionOutput, got {type(output)!r}")
        logabs = output.logabs
        sign = output.sign
        if logabs.shape != (walkers.batch_size,):
            raise ValueError(f"Model logabs must have shape [{walkers.batch_size}], got {tuple(logabs.shape)}")
        if sign.shape != (walkers.batch_size,):
            raise ValueError(f"Model sign must have shape [{walkers.batch_size}], got {tuple(sign.shape)}")
        return logabs, sign

    def _propose(self, model, walkers: Walkers) -> tuple[torch.Tensor, torch.Tensor]:
        del model
        if not hasattr(self.move, "propose"):
            raise TypeError("MetropolisSampler move must expose propose(walkers)")
        proposals, log_q_ratio = self.move.propose(walkers)
        if proposals.shape != walkers.positions.shape:
            raise ValueError(f"Proposal positions must have shape {tuple(walkers.positions.shape)}, got {tuple(proposals.shape)}")
        if log_q_ratio.shape != (walkers.batch_size,):
            raise ValueError(f"Proposal log-ratio must have shape [{walkers.batch_size}], got {tuple(log_q_ratio.shape)}")
        return proposals, log_q_ratio

    def step(self, model, walkers: Walkers) -> Walkers:
        """Run one Metropolis-Hastings step.

        Parameters
        ----------
        model : callable
            Wavefunction model returning `WavefunctionOutput`.
        walkers : Walkers
            Current walker state.

        Returns
        -------
        Walkers
            Updated walker state with cached wavefunction values and sampler
            diagnostics in ``aux``.
        """

        current_logabs = walkers.logabs
        current_sign = walkers.sign
        if current_logabs is None or current_sign is None:
            current_logabs, current_sign = self._evaluate(model, walkers)
        proposals, log_q_ratio = self._propose(model, walkers)
        proposal_walkers = Walkers(positions=proposals, spins=walkers.spins, aux=dict(walkers.aux))
        proposed_logabs, proposed_sign = self._evaluate(model, proposal_walkers)
        log_accept_ratio = torch.nan_to_num(2.0 * (proposed_logabs - current_logabs) + log_q_ratio, nan=-torch.inf)
        log_accept = torch.clamp(log_accept_ratio, max=0.0)
        accepted = torch.log(torch.rand_like(log_accept).clamp_min(1e-12)) < log_accept
        accepted_mask = accepted.view(-1, 1, 1)
        positions = torch.where(accepted_mask, proposals, walkers.positions)
        logabs = torch.where(accepted, proposed_logabs, current_logabs)
        sign = torch.where(accepted, proposed_sign, current_sign)
        self.acceptance_rate = accepted.to(dtype=torch.float32).mean().item()
        self.last_metrics = {
            "acceptance_rate": self.acceptance_rate,
            "mean_logabs": float(logabs.detach().mean().item()),
        }
        if hasattr(self.move, "step_size"):
            self.last_metrics["proposal_scale"] = float(self.move.step_size)
        return Walkers(
            positions=positions.detach(),
            logabs=logabs.detach(),
            sign=sign.detach(),
            spins=None if walkers.spins is None else walkers.spins.detach(),
            aux={
                **walkers.aux,
                "accepted": accepted.detach(),
                "log_accept_ratio": log_accept_ratio.detach(),
                "acceptance_rate": self.acceptance_rate,
            },
        )

    def sample(self, model, walkers: Walkers, n_steps: int | None = None) -> Walkers:
        """Run multiple Metropolis-Hastings steps.

        Parameters
        ----------
        model : callable
            Wavefunction model returning `WavefunctionOutput`.
        walkers : Walkers
            Current walker state.
        n_steps : int or None, optional
            Number of MCMC steps. If ``None``, `self.steps_per_iter` is used.

        Returns
        -------
        Walkers
            Walker state after sampling. ``self.acceptance_rate`` is the mean
            acceptance rate over all steps in this call.
        """

        total_steps = self.steps_per_iter if n_steps is None else n_steps
        if total_steps < 0:
            raise ValueError("n_steps must be non-negative")
        acceptance_sum = 0.0
        for _ in range(total_steps):
            walkers = self.step(model, walkers)
            acceptance_sum += float(walkers.aux["acceptance_rate"])
        if total_steps:
            self.acceptance_rate = acceptance_sum / total_steps
            self.last_metrics["acceptance_rate"] = self.acceptance_rate
        walkers.aux["sample_acceptance_rate"] = self.acceptance_rate
        return walkers


def _default_spins(
    system: ElectronicSystem,
    *,
    n_walkers: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return repeated spin labels from system metadata.

    Parameters
    ----------
    system : ElectronicSystem
        System whose ``n_up`` and ``n_down`` fields define the spin partition.
    n_walkers : int
        Number of walkers.
    device : torch.device, str, or None
        Target device for the spin tensor.
    dtype : torch.dtype
        Target dtype for the spin tensor.

    Returns
    -------
    torch.Tensor or None
        Spin labels with shape ``[n_walkers, n_electrons]`` when spin metadata
        is available, otherwise ``None``.
    """

    if system.n_up is None or system.n_down is None:
        return None
    spins = torch.tensor([1.0] * system.n_up + [-1.0] * system.n_down, device=device, dtype=dtype)
    if spins.numel() != system.n_electrons:
        raise ValueError("System spin partition must match n_electrons")
    return spins.unsqueeze(0).expand(n_walkers, -1).clone()
