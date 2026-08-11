"""Mutable training-loop state shared with callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from tpen.data.batch import WavefunctionOutput
from tpen.events import DomainState
from tpen.sampling.stats import SamplerStats


@dataclass
class TrainerState(DomainState):
    """Snapshot of the VMC training loop at one step.

    This is the training domain's `DomainState`: the trainer passes it beside
    every typed occurrence it emits, and a `tpen.callback.StatefulCallback`
    declaring ``state_type = TrainerState`` receives it as a typed handler
    argument.

    The state is updated in place each step and handed to callbacks (notably
    `tpen.callback.Checkpoint`) through ``Event.state``. Fields beyond
    ``step``/``metrics`` carry the most recent loop artifacts for inspection.

    Parameters
    ----------
    step : int, optional
        Index of the most recently completed step (0-based; ``-1`` before the
        first step).
    metrics : dict, optional
        Scalar metrics logged for the most recent step.
    model : Any, optional
        Wavefunction model being optimized.
    optimizer : Any, optional
        Optimizer driving the model parameters.
    trainer : Any, optional
        Trainer object owning train-loop progress state.
    sampler : Any, optional
        Sampler producing walker configurations.
    samples : Any, optional
        Most recent walker state, when retained.
    batch : Any, optional
        Most recent electron batch.
    local_energy : Any, optional
        Most recent per-sample local energy (detached).
    loss : torch.Tensor or None, optional
        Most recent surrogate loss (detached).
    wavefunction_output : WavefunctionOutput or None, optional
        Most recent wavefunction output (signed-log form) for the batch.
    sampler_stats : SamplerStats or None, optional
        Most recent typed sampler diagnostics, or ``None`` before the first
        collection.
    """

    step: int = -1
    metrics: dict[str, Any] = field(default_factory=dict)
    model: Any = None
    optimizer: Any = None
    trainer: Any = None
    sampler: Any = None
    samples: Any = None
    batch: Any = None
    local_energy: Any = None
    loss: torch.Tensor | None = None
    wavefunction_output: WavefunctionOutput | None = None
    sampler_stats: SamplerStats | None = None


__all__ = ["TrainerState"]
