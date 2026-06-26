"""MCMC-backed evaluation generators."""

from __future__ import annotations

from typing import Any

import torch

from spenn.evaluation.bundle import GeneratedConfigurations
from spenn.evaluation.protocols import EvaluationContext


class MCMCGenerator:
    """Generate evaluation configurations from an existing sampler."""

    name = "mcmc"

    def __init__(
        self,
        *,
        sampler: object,
        seed: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.sampler = sampler
        self.seed = None if seed is None else int(seed)
        self.max_samples = None if max_samples is None else int(max_samples)

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Collect sampler configurations and expose bookkeeping metadata."""

        if self.seed is not None:
            torch.manual_seed(self.seed)
        collect = getattr(self.sampler, "collect_samples", None)
        if not callable(collect):
            raise TypeError("MCMCGenerator sampler must expose collect_samples(model, device=...)")
        walkers, sampler_stats = collect(model, device=context.device)
        batch = walkers.make_batch().flatten_samples()
        if self.max_samples is not None and self.max_samples >= 0:
            batch = _slice_batch(batch, 0, min(self.max_samples, batch.batch_size))
        sample_index = torch.arange(batch.batch_size, device=batch.device)
        metadata: dict[str, Any] = {
            "sample_index": sample_index,
            "sampler_stats": dict(sampler_stats),
        }
        if "walker_index" not in metadata:
            metadata["walker_index"] = sample_index
        return GeneratedConfigurations(batch=batch, metadata=metadata)


def _slice_batch(batch, start: int, end: int):
    from spenn.evaluation.calculators.local_energy import slice_flat_batch

    return slice_flat_batch(batch.flatten_samples(), start, end)


__all__ = ["MCMCGenerator"]
