"""Smoke test: one VMC trainer step over the real tiny TPEN stack."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tpen.artifacts import RunContext
from tpen.events import Ended, Occurrence, Started
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, HarmonicTrap
from tpen.training.events import (
    CollectSamples,
    TrainingIteration,
    TrainingIterationCompleted,
)
from tpen.training.trainer import VMCTrainer
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn


class _StubContext(RunContext):
    """Minimal RunContext subclass: satisfies typing, logs to a list."""

    def __init__(self) -> None:
        self.callbacks = []
        self.loggers = []
        self.metadata = SimpleNamespace(device="cpu", dtype="float64")
        self.records: list[tuple[str, dict]] = []
        self.occurrences: list[Occurrence[Any]] = []
        self.trace: list[tuple[str, object]] = []
        self._occurrence_counts = {}

    def log(self, metrics, *, step=None, namespace="run", event=None) -> None:
        self.records.append((namespace, dict(metrics)))

    def _dispatch_occurrence(self, occurrence: Occurrence[Any]) -> None:
        self.occurrences.append(occurrence)
        self.trace.append(("typed", occurrence.event))


_FORBIDDEN_METRICS = {
    "reference_energy",
    "energy_error",
    "energy_abs_error",
    "abs_energy_error",
    "exact_energy",
    "expected_energy",
}


def test_one_vmc_step_is_finite_and_vmc_native() -> None:
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, return_terms=True)

    state = trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda name, *, state=None, payload=None, step=None: None,
    )

    assert math.isfinite(float(state.loss))
    assert torch.isfinite(state.local_energy).all()
    # Native VMC metrics only -- no reference/exact comparison leaks in.
    assert _FORBIDDEN_METRICS.isdisjoint(state.metrics)
    for key in ("energy", "loss", "grad_norm"):
        assert key in state.metrics


def _fit_one_step(*, return_terms: bool, terms) -> object:
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, return_terms=return_terms)
    return trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=optimizer,
        context=_StubContext(),
        emit=lambda name, *, state=None, payload=None, step=None: None,
    )


def test_vmc_trainer_uses_canonical_objective_metrics() -> None:
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    state = _fit_one_step(return_terms=False, terms=terms)

    # Physical training estimator is logged as `energy`, never `energy_mean`.
    assert "energy" in state.metrics
    assert "energy_mean" not in state.metrics
    for key in (
        "loss",
        "energy_variance",
        "energy_std",
        "energy_stderr",
        "local_energy_n_finite",
        "local_energy_n_total",
        "local_energy_finite_fraction",
        "local_energy_nonfinite_count",
    ):
        assert key in state.metrics
    # No per-term metrics when return_terms is disabled.
    assert not any(key.startswith("energy_term_") for key in state.metrics)


def test_vmc_trainer_logs_term_metrics_when_return_terms_enabled() -> None:
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    state = _fit_one_step(return_terms=True, terms=terms)

    # A list of terms falls back to snake-case class names for the metric keys.
    expected_names = ("kinetic_energy", "harmonic_trap", "electron_electron_interaction")
    for name in expected_names:
        prefix = f"energy_term_{name}"
        assert prefix in state.metrics
        assert f"{prefix}_variance" in state.metrics


def test_vmc_trainer_uses_typed_sampling_and_keeps_other_phase_events() -> None:
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    phase_events: list[tuple[str, str]] = []
    context = _StubContext()

    def emit(name: str, *, state=None, payload=None, step=None) -> None:
        del state
        context.trace.append(("legacy", name))
        if name in {
            "step_start",
            "step_end",
            "train_phase_start",
            "train_phase_end",
        }:
            assert step == 0
            assert payload is None or "step" not in payload
        if name in {"train_phase_start", "train_phase_end"}:
            assert payload is not None
            phase_events.append((name, payload["phase"]))

    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=terms,
        optimizer=optimizer,
        context=context,
        emit=emit,
    )

    started = [phase for name, phase in phase_events if name == "train_phase_start"]
    ended = [phase for name, phase in phase_events if name == "train_phase_end"]
    assert started == [
        "batch_build",
        "local_energy",
        "forward",
        "objective",
        "backward",
        "optimizer_step",
        "post_step_metrics",
    ]
    # Phases are sequential and non-nested, so end order matches start order.
    assert ended == started
    assert len(context.occurrences) == 5
    assert isinstance(context.occurrences[0].event, Started)
    assert isinstance(context.occurrences[0].event.operation, TrainingIteration)
    assert isinstance(context.occurrences[1].event, Started)
    assert isinstance(context.occurrences[1].event.operation, CollectSamples)
    assert isinstance(context.occurrences[2].event, Ended)
    assert isinstance(context.occurrences[2].event.operation, CollectSamples)
    assert isinstance(context.occurrences[3].event, TrainingIterationCompleted)
    assert isinstance(context.occurrences[4].event, Ended)
    assert isinstance(context.occurrences[4].event.operation, TrainingIteration)
    assert [occurrence.count for occurrence in context.occurrences] == [1, 1, 1, 1, 1]
    step_end_index = context.trace.index(("legacy", "step_end"))
    completion_index = next(
        index
        for index, (kind, event) in enumerate(context.trace)
        if kind == "typed" and isinstance(event, TrainingIterationCompleted)
    )
    iteration_end_index = next(
        index
        for index, (kind, event) in enumerate(context.trace)
        if kind == "typed"
        and isinstance(event, Ended)
        and isinstance(event.operation, TrainingIteration)
    )
    assert step_end_index < completion_index < iteration_end_index


def test_vmc_trainer_step_end_failure_skips_completion_but_ends_iteration() -> None:
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1)
    context = _StubContext()

    def emit(name: str, *, state=None, payload=None, step=None) -> None:
        del state, payload, step
        context.trace.append(("legacy", name))
        if name == "step_end":
            raise RuntimeError("legacy step_end failed")

    with pytest.raises(RuntimeError, match="legacy step_end failed"):
        trainer.fit(
            model=model,
            sampler=sampler,
            hamiltonian_terms=terms,
            optimizer=optimizer,
            context=context,
            emit=emit,
        )

    assert not any(
        isinstance(occurrence.event, TrainingIterationCompleted)
        for occurrence in context.occurrences
    )
    assert isinstance(context.occurrences[-1].event, Ended)
    assert isinstance(context.occurrences[-1].event.operation, TrainingIteration)
