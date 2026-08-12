"""Delivery tests for the callbacks migrated onto `StatefulCallback`.

These exist because of a ruling recorded on item ``62593af4``. The dispatcher
in `tpen.artifacts.RunContext._dispatch_occurrence` SKIPS a `StatefulCallback`
whose domain does not match the state it was handed, and that same branch fires
when the emitter passes no state at all. Skipping is correct and specified -- a
run emitting several domains must not die on a callback with nothing to observe
-- but it means an emitter that forgets ``state=`` produces a callback that
observes nothing, silently, with no error anywhere.

That is the identical failure shape as defect ``933b5f78``, where `GradientStats`
reported ``passed: true`` while observing zero gradient tensors. In both cases
silence is indistinguishable from success, and in both cases the full suite was
blind to it. So rather than add a branch to the hottest dispatch path, every
migrated callback gets a test asserting it really is delivered its state at the
boundary it subscribes to. Each test below fails loudly if a ``state=`` is ever
dropped from the emitting site, which is the layer where that mistake is made.

The tests use the REAL dispatcher and, for the end-to-end case, the REAL
trainer. A `RunContext` stand-in would override the very method under test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from tpen.callback import DataIntegrity, GradientStats, RuntimeEquivariance, SamplerHealth
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.equivariance.checks import EquivarianceCheckResult
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, HarmonicTrap
from tpen.training.events import TrainingIteration, TrainingIterationCompleted
from tpen.training.state import TrainerState
from tpen.training.trainer import VMCTrainer
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn
from tests.helpers.run_context import RecordingLogger, make_run_context
from tests.unit.callback.support import make_sampler_stats, training_state


class _OneParamModel(nn.Module):
    """Model with exactly one gradient-carrying parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, dtype=torch.float64))


class _RecordingChecker:
    """Equivariance checker that records the state object it was handed.

    ``name`` is declared rather than derived so the log namespace this test
    asserts on cannot drift with the class name.
    """

    name = "recording"

    def __init__(self) -> None:
        self.states: list[Any] = []

    def run(self, state: Any) -> EquivarianceCheckResult:
        self.states.append(state)
        return EquivarianceCheckResult(
            passed=True, metrics={"max_abs_error": 0.0}, n_comparisons=1
        )


def _populated_state() -> TrainerState:
    """Return a `TrainerState` every migrated callback can read something from."""

    model = _OneParamModel()
    model.weight.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
    return training_state(
        model=model,
        batch=ElectronBatch(
            positions=torch.zeros(4, 2, 3, dtype=torch.float64),
            spins=torch.tensor([[1.0, -1.0]] * 4, dtype=torch.float64),
        ),
        local_energy=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        loss=torch.tensor(1.5, dtype=torch.float64),
        wavefunction_output=WavefunctionOutput(
            logabs=torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float64),
            sign=torch.ones(4, dtype=torch.float64),
        ),
        sampler_stats=make_sampler_stats(acceptance_rate=0.5),
    )


def _completed(step: int) -> TrainingIterationCompleted:
    return TrainingIterationCompleted(iteration=TrainingIteration(step=step))


# Factories, not instances: each case builds a fresh callback so one test's
# cadence counter and recorded states cannot leak into the next.
_MIGRATED: list[tuple[str, Any, str]] = [
    ("GradientStats", GradientStats, "checks/gradient"),
    ("DataIntegrity", DataIntegrity, "checks/data_integrity"),
    ("SamplerHealth", SamplerHealth, "checks/sampler"),
    (
        "RuntimeEquivariance",
        lambda: RuntimeEquivariance(checkers=[_RecordingChecker()], fail_fast=False),
        "checks/equivariance/recording",
    ),
]


@pytest.mark.parametrize(
    ("name", "build", "namespace"),
    _MIGRATED,
    ids=[entry[0] for entry in _MIGRATED],
)
def test_each_migrated_callback_receives_state_at_its_boundary(
    tmp_path: Path, name: str, build: Any, namespace: str
) -> None:
    """The dispatcher hands this callback its domain state, and it logs."""

    logger = RecordingLogger()
    context = make_run_context(tmp_path, callbacks=[build()], loggers=[logger])

    context.emit(_completed(step=3), state=_populated_state())

    assert namespace in logger.namespaces(), f"{name} observed nothing"
    assert logger.steps(namespace) == [3]


@pytest.mark.parametrize(
    ("name", "build", "namespace"),
    _MIGRATED,
    ids=[entry[0] for entry in _MIGRATED],
)
def test_a_boundary_emitted_without_state_delivers_nothing(
    tmp_path: Path, name: str, build: Any, namespace: str
) -> None:
    """Absent state is skipped, not raised -- which is why the tests above exist.

    This pins the hazard rather than the fix: it records that a forgotten
    ``state=`` really does produce total silence, so the positive tests above
    are the only thing standing between that mistake and an empty metric
    namespace nobody notices.
    """

    logger = RecordingLogger()
    context = make_run_context(tmp_path, callbacks=[build()], loggers=[logger])

    context.emit(_completed(step=3))

    assert logger.records == [], f"{name} logged despite receiving no state"


@pytest.mark.parametrize(
    ("name", "build", "namespace"),
    _MIGRATED,
    ids=[entry[0] for entry in _MIGRATED],
)
def test_no_migrated_callback_still_answers_a_legacy_trigger(
    name: str, build: Any, namespace: str
) -> None:
    """A stale ``triggers:`` key in a config cannot make these fire twice.

    Each migrated class dropped its ``on_step_end``, so the legacy dispatch in
    `_CallbackCore.handle` finds no method to call even when a config still
    carries the key. Double firing is therefore structurally impossible rather
    than merely unconfigured.
    """

    callback = build()
    assert not hasattr(callback, "on_step_end")
    assert not hasattr(callback, "on_train_end")
    # Nothing is configured either: subscriptions are class-owned under ADR-E002
    # and the constructors no longer accept a positional trigger list.
    assert not hasattr(callback, "triggers")


def test_the_trainer_passes_state_at_the_boundary_these_callbacks_subscribe_to(
    tmp_path: Path,
) -> None:
    """End-to-end: the real trainer's emit really carries the state.

    The unit tests above emit `TrainingIterationCompleted` themselves, so they
    would still pass if `tpen.training.trainer` dropped its ``state=``. This one
    drives the real loop, which is the site where that mistake would be made.
    """

    checker = _RecordingChecker()
    callbacks = [
        DataIntegrity(),
        GradientStats(),
        SamplerHealth(),
        RuntimeEquivariance(checkers=[checker], fail_fast=False),
    ]
    logger = RecordingLogger()
    context = make_run_context(tmp_path, callbacks=callbacks, loggers=[logger])
    model = build_tiny_spenn()
    trainer = VMCTrainer(max_steps=2, log_every_n_steps=1)

    trainer.fit(
        model=model,
        sampler=build_tiny_sampler(),
        hamiltonian_terms=[KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()],
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        context=context,
        emit=lambda name, *, state=None, payload=None, step=None: None,
    )

    for namespace in (
        "checks/data_integrity",
        "checks/gradient",
        "checks/sampler",
        "checks/equivariance/recording",
    ):
        assert logger.steps(namespace) == [0, 1], f"{namespace} was not delivered every step"
    # The checker received the loop's own mutable state object, not a copy.
    assert all(isinstance(state, TrainerState) for state in checker.states)
