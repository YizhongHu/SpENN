"""Shared diagnostic protocols and evaluation context objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def validate_diagnostics(diagnostics: Sequence[object] | None) -> tuple[Diagnostic, ...]:
    """Validate configured diagnostics without invoking them.

    Used by every phase that runs diagnostics (`Evaluate`, the `Validation`
    callback) so misconfigured diagnostics fail at construction time.
    """

    if diagnostics is None:
        raise ValueError(
            "at least one diagnostic is required. Configure EnergyEvaluation to report energy metrics."
        )

    validated: list[Diagnostic] = []
    for index, diagnostic in enumerate(diagnostics):
        if not callable(getattr(diagnostic, "evaluate", None)):
            raise TypeError(
                f"diagnostics[{index}] must be an instantiated diagnostic object with an evaluate(...) "
                f"method, got {type(diagnostic)!r}. This usually means the diagnostic config was not "
                "recursively instantiated by Hydra. Put diagnostics inside the owning runner/callback "
                "config or pass instantiated diagnostic objects."
            )
        name = getattr(diagnostic, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"diagnostics[{index}] must expose a non-empty string name")
        validated.append(diagnostic)
    if not validated:
        raise ValueError(
            "at least one diagnostic is required. Configure EnergyEvaluation to report energy metrics."
        )
    return tuple(validated)


def evaluate_diagnostics(
    diagnostics: Sequence[Diagnostic],
    context: EvaluationContext,
    *,
    emit=None,
    step: int = 0,
) -> dict[str, JsonScalar]:
    """Evaluate diagnostics and merge their flat metric mappings.

    Parameters
    ----------
    diagnostics : sequence of Diagnostic
        Already-validated diagnostics (see `validate_diagnostics`).
    context : EvaluationContext
        Shared evaluation state consumed by every diagnostic.
    emit : callable or None, optional
        Optional ``emit(name, payload=...)`` lifecycle hook receiving
        ``diagnostic_start``/``diagnostic_end``/``diagnostic_failed`` events.
    step : int, optional
        Step recorded in the emitted diagnostic event payloads.
    """

    metrics: dict[str, JsonScalar] = {}
    for diagnostic in diagnostics:
        payload = {"diagnostic_name": diagnostic.name, "step": step}
        if emit is not None:
            emit("diagnostic_start", payload=payload)
        try:
            result = diagnostic.evaluate(context)
        except Exception as exc:
            if emit is not None:
                emit("diagnostic_failed", payload={**payload, "exception": exc})
            raise
        if emit is not None:
            emit("diagnostic_end", payload=payload)
        if not isinstance(result, Mapping):
            raise TypeError(f"diagnostic {diagnostic.name!r} must return a mapping of metric names to scalars")
        for key, value in result.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"diagnostic {diagnostic.name!r} returned an empty metric name")
            if key in metrics:
                raise ValueError(f"diagnostic metric key collision for {key!r}")
            _validate_json_scalar(diagnostic.name, key, value)
            metrics[key] = value
    return metrics


def _validate_json_scalar(diagnostic_name: str, key: str, value: object) -> None:
    """Fail loudly when a diagnostic returns a non-scalar metric value."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return
    raise TypeError(
        f"diagnostic {diagnostic_name!r} metric {key!r} must be a JSON scalar, "
        f"got {type(value).__name__}"
    )
