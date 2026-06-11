"""Shared diagnostic protocols and evaluation context objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import torch

from spenn.data.batch import ElectronBatch, WavefunctionOutput
from spenn.physics.hamiltonian import HamiltonianTerm

JsonScalar: TypeAlias = bool | int | float | str | None


@dataclass(frozen=True)
class EvaluationContext:
    """Shared state prepared once by `Evaluate` and consumed by diagnostics.

    Parameters
    ----------
    model : object
        Configured wavefunction model or exact reference.
    batch : ElectronBatch
        Sampled electron configurations.
    wavefunction_output : WavefunctionOutput
        Wavefunction output evaluated on ``batch``.
    local_energy : torch.Tensor
        Total local energy with shape ``[batch]``.
    local_energy_terms : Mapping[str, torch.Tensor] or None
        Optional per-term local-energy decomposition keyed by the configured
        Hamiltonian term names.
    sampler_stats : Mapping[str, JsonScalar]
        Sampler diagnostics gathered while collecting ``batch``.
    hamiltonian_terms : Mapping[str, HamiltonianTerm]
        Normalized Hamiltonian terms keyed by their public metric names.
    """

    model: object
    batch: ElectronBatch
    wavefunction_output: WavefunctionOutput
    local_energy: torch.Tensor
    local_energy_terms: Mapping[str, torch.Tensor] | None
    sampler_stats: Mapping[str, JsonScalar]
    hamiltonian_terms: Mapping[str, HamiltonianTerm]


class Diagnostic(Protocol):
    """Protocol for one evaluation diagnostic."""

    name: str

    def evaluate(self, context: EvaluationContext) -> Mapping[str, JsonScalar]:
        """Compute flat JSON-safe metrics from a prepared evaluation context."""
        ...
