"""Rank-local random-walk sampler used to make resume state observable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tests.spikes.native_ddp.model_access import SemanticWavefunction
from tests.spikes.native_ddp.seed import SeedPartition


@dataclass
class RankLocalSampler:
    """Small persistent walker chain with an explicitly owned CPU generator."""

    seed_partition: SeedPartition
    n_walkers: int = 4
    proposal_scale: float = 0.05
    walkers: torch.Tensor | None = None
    cached_logabs: torch.Tensor | None = None
    proposal_count: int = 0

    def __post_init__(self) -> None:
        self.generator = self.seed_partition.make_generator()
        self.coordinate_forward_count = 0

    def initialize(self, model: SemanticWavefunction) -> None:
        """Initialize walkers and their cached raw-model values."""

        self.walkers = torch.randn(
            self.n_walkers, 2, dtype=torch.float64, generator=self.generator
        )
        self._refresh_cache(model)

    def advance(self, model: SemanticWavefunction, n_steps: int) -> None:
        """Advance the chain and refresh cached values on every proposal."""

        if n_steps < 0:
            raise ValueError("n_steps must be nonnegative")
        if self.walkers is None:
            self.initialize(model)
        assert self.walkers is not None
        for _ in range(n_steps):
            proposal = self.walkers + self.proposal_scale * torch.randn(
                self.walkers.shape, dtype=self.walkers.dtype, generator=self.generator
            )
            self.walkers = proposal
            self.proposal_count += 1
            self._refresh_cache(model)

    def _refresh_cache(self, model: SemanticWavefunction) -> None:
        assert self.walkers is not None
        with torch.no_grad():
            self.cached_logabs = model(self.walkers).detach()
        self.coordinate_forward_count += 1

    def state_dict(self) -> dict[str, Any]:
        """Return walkers, cached values, counters, and generator bytes."""

        return {
            "walkers": None if self.walkers is None else self.walkers.detach().clone(),
            "cached_logabs": (
                None if self.cached_logabs is None else self.cached_logabs.detach().clone()
            ),
            "proposal_count": self.proposal_count,
            "coordinate_forward_count": self.coordinate_forward_count,
            "generator_state": self.generator.get_state().clone(),
            "generator_device": "cpu",
            "seed_partition": {
                "base_seed": self.seed_partition.base_seed,
                "rank": self.seed_partition.rank,
                "world_size": self.seed_partition.world_size,
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore exact rank-local sampler state."""

        partition = state["seed_partition"]
        if partition != {
            "base_seed": self.seed_partition.base_seed,
            "rank": self.seed_partition.rank,
            "world_size": self.seed_partition.world_size,
        }:
            raise ValueError("sampler seed partition does not match this rank")
        walkers = state["walkers"]
        cached = state["cached_logabs"]
        self.walkers = None if walkers is None else walkers.detach().clone()
        self.cached_logabs = None if cached is None else cached.detach().clone()
        self.proposal_count = int(state["proposal_count"])
        self.coordinate_forward_count = int(state["coordinate_forward_count"])
        self.generator.set_state(state["generator_state"])


__all__ = ["RankLocalSampler"]
