"""Batched Metropolis sampler."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from spenn.data_structures.batch import ElectronBatch, Walkers
from spenn.physics.systems import ElectronicSystem
from spenn.sampling.moves import GaussianMove


class MetropolisSampler(nn.Module):
    """Simple batched Metropolis-Hastings sampler."""

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
        **_: object,
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

    def initialize(self, system: ElectronicSystem | None = None, n_walkers: int | None = None, device=None) -> Walkers:
        system = system or ElectronicSystem(n_electrons=self.n_electrons, spatial_dim=self.spatial_dim, dtype=self.dtype)
        self.system = system
        n_walkers = n_walkers or self.n_walkers
        dtype = getattr(torch, system.dtype) if isinstance(system.dtype, str) else (system.dtype or self.dtype)
        device = device or system.device
        positions = self.initial_scale * torch.randn(
            n_walkers, system.n_electrons, system.spatial_dim, device=device, dtype=dtype
        )
        spins = None
        return Walkers(positions=positions, spins=spins, aux={"system": system})

    def _evaluate(self, model, walkers: Walkers) -> tuple[torch.Tensor, torch.Tensor]:
        batch = ElectronBatch(
            positions=walkers.positions,
            spins=walkers.spins,
            system=walkers.aux.get("system"),
            aux=dict(walkers.aux),
        )
        with torch.no_grad():
            output = model(batch)
        logabs = output.logabs if hasattr(output, "logabs") else output
        sign = output.sign if hasattr(output, "sign") else torch.sign(logabs)
        return logabs, sign

    def step(self, model, walkers: Walkers) -> Walkers:
        current_logabs = walkers.logabs
        current_sign = walkers.sign
        if current_logabs is None or current_sign is None:
            current_logabs, current_sign = self._evaluate(model, walkers)
        proposals = self.move(walkers.positions)
        proposal_walkers = Walkers(positions=proposals, spins=walkers.spins, aux=dict(walkers.aux))
        proposed_logabs, proposed_sign = self._evaluate(model, proposal_walkers)
        log_accept_ratio = 2.0 * (proposed_logabs - current_logabs)
        accepted = torch.log(torch.rand_like(log_accept_ratio).clamp_min(1e-12)) < log_accept_ratio
        accepted_mask = accepted.view(-1, 1, 1)
        positions = torch.where(accepted_mask, proposals, walkers.positions)
        logabs = torch.where(accepted, proposed_logabs, current_logabs)
        sign = torch.where(accepted, proposed_sign, current_sign)
        self.acceptance_rate = accepted.to(dtype=torch.float32).mean().item()
        return Walkers(positions=positions, spins=walkers.spins, logabs=logabs, sign=sign, aux={**walkers.aux, "accepted": accepted})

    def sample(self, model, walkers: Walkers, n_steps: int) -> Walkers:
        for _ in range(n_steps):
            walkers = self.step(model, walkers)
        return walkers
