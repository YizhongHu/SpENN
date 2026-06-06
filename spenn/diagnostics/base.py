"""Base types for reusable diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from spenn.data.batch import Walkers


@dataclass
class DiagnosticContext:
    """Runtime objects available to diagnostics.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Resolved run configuration.
    model : torch.nn.Module
        Evaluated wavefunction model.
    hamiltonian : object
        Hamiltonian with a ``local_energy`` method.
    system : object
        System metadata used by the sampler and Hamiltonian.
    sampler : object
        Sampler used for production blocks.
    walkers : Walkers
        Final production walker state.
    local_energy : torch.Tensor
        Concatenated production local energies with shape ``[samples]``.
    pair_distance : torch.Tensor
        Concatenated pair distances with shape ``[samples]``.
    dtype : torch.dtype
        Floating-point dtype used by the run.
    device : torch.device
        Device used by the run.
    """

    cfg: DictConfig
    model: torch.nn.Module
    hamiltonian: Any
    system: Any
    sampler: Any
    walkers: Walkers
    local_energy: torch.Tensor
    pair_distance: torch.Tensor
    dtype: torch.dtype
    device: torch.device


class Diagnostic:
    """Base diagnostic interface for future configured diagnostics."""

    name: str = "diagnostic"

    def run(self, context: Any, state: object | None = None) -> "DiagnosticResult":
        """Run this diagnostic."""

        raise NotImplementedError

    def __call__(self, context: Any, state: object | None = None) -> "DiagnosticResult":
        """Delegate callable diagnostics to :meth:`run`."""

        return self.run(context, state=state)


@dataclass
class DiagnosticResult:
    """Metrics and table rows produced by a diagnostic.

    Parameters
    ----------
    metrics : dict of str to float
        Scalar diagnostic metrics.
    tables : dict of str to list of dict
        Named CSV table rows. Table names become filenames.
    """

    metrics: dict[str, float | int | str] = field(default_factory=dict)
    tables: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    name: str = ""
    artifacts: dict[str, Path] = field(default_factory=dict)
    passed: bool | None = None
