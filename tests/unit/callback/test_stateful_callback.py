"""Mechanism tests for delivering a domain state to a typed callback.

These exercise the mechanism with two deliberately domain-neutral state
classes rather than `tpen.training.state.TrainerState`, because the mechanism
is not training-specific: the point of `tpen.events.DomainState` being an empty
marker is that a second domain with a completely different coordinate can use
the same delivery path. No production callback subscribes to a state yet, so
every assertion here is about the mechanism itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any, ClassVar

import pytest
from omegaconf import OmegaConf
from typeguard import suppress_type_checks

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.callback import Cadence, Callback, StatefulCallback, SubscriptionGroup
from tpen.events import DomainState, Ended, Event as TypedEvent
from tpen.events import Occurrence, Operation, Started, Subscription, ended, started


@dataclass
class _StepState(DomainState):
    """Domain state coordinated by an integer step, mutated in place."""

    step: int = -1
    marks: list[str] = field(default_factory=list)


@dataclass
class _NamespaceState(DomainState):
    """A second domain's state, coordinated by a namespace string.

    It shares no field with `_StepState`, which is the situation the empty
    `DomainState` marker exists to accommodate.
    """

    namespace: str = "suite"


@dataclass(frozen=True)
class _Pulse(TypedEvent):
    label: str


@dataclass(frozen=True)
class _Work(Operation):
    label: str


class _StatelessRecorder(Callback):
    """Plain callback: its impl accepts exactly two arguments.

    The arity is the assertion. If the dispatcher ever delivered a state to a
    state-free callback, this would fail with a ``TypeError`` rather than
    quietly ignoring the extra argument.
    """

    def __init__(self, *groups: SubscriptionGroup) -> None:
        super().__init__(typed_groups=groups)
        self.seen: list[TypedEvent] = []

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        del context
        self.seen.append(occurrence.event)


class _StepRecorder(StatefulCallback[_StepState]):
    """Stateful callback observing the integer-step domain."""

    state_type: ClassVar[type[DomainState]] = _StepState

    def __init__(self, *groups: SubscriptionGroup) -> None:
        super().__init__(typed_groups=groups)
        self.seen: list[tuple[TypedEvent, _StepState, int]] = []

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: _StepState,
    ) -> None:
        del context
        # The step is read at DELIVERY time. The state is mutable, so reading it
        # after the run would not show what this boundary actually observed.
        self.seen.append((occurrence.event, state, state.step))


class _NamespaceRecorder(StatefulCallback[_NamespaceState]):
    """Stateful callback observing the string-namespace domain."""

    state_type: ClassVar[type[DomainState]] = _NamespaceState

    def __init__(self, *groups: SubscriptionGroup) -> None:
        super().__init__(typed_groups=groups)
        self.seen: list[str] = []

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: _NamespaceState,
    ) -> None:
        del occurrence, context
        self.seen.append(state.namespace)


class _InheritedImplRecorder(StatefulCallback[_StepState]):
    """Declares a domain but overrides nothing.

    Delivery therefore reaches `StatefulCallback.handle_occurrence_impl`, the
    generic base's no-op, which is what puts its ``state: StateT`` annotation in
    front of typeguard.
    """

    state_type: ClassVar[type[DomainState]] = _StepState


def _pulse_group(cadence: Cadence | None = None) -> SubscriptionGroup:
    return SubscriptionGroup(selectors=(Subscription.of(_Pulse),), cadence=cadence)


def test_stateful_callback_receives_the_state_and_a_plain_callback_does_not(
    tmp_path: Path,
) -> None:
    stateful = _StepRecorder(_pulse_group())
    stateless = _StatelessRecorder(_pulse_group())
    context = _context(tmp_path, callbacks=[stateful, stateless])
    state = _StepState(step=4)

    context.emit(_Pulse("one"), state=state)

    assert stateful.seen == [(_Pulse("one"), state, 4)]
    # The delivered object is the emitter's own state, not a copy of it.
    assert stateful.seen[0][1] is state
    assert stateless.seen == [_Pulse("one")]


def test_a_callback_declaring_another_domain_is_skipped_not_failed(
    tmp_path: Path,
) -> None:
    step_recorder = _StepRecorder(_pulse_group(Cadence()))
    namespace_recorder = _NamespaceRecorder(_pulse_group())
    stateless = _StatelessRecorder(_pulse_group())
    context = _context(
        tmp_path, callbacks=[step_recorder, namespace_recorder, stateless]
    )

    # A mixed run emits several domains' states. The step-domain callback has
    # nothing to observe here, and that must not end the run.
    context.emit(_Pulse("evaluation"), state=_NamespaceState(namespace="hooke"))

    assert step_recorder.seen == []
    assert namespace_recorder.seen == ["hooke"]
    assert stateless.seen == [_Pulse("evaluation")]
    # Skipping happens before delivery, so the skipped callback's cadence gate
    # is untouched: a foreign domain's occurrence cannot consume its schedule.
    gate = step_recorder._typed_group_states[0].gate
    assert gate is not None and gate.num_calls == 0


def test_emit_and_scope_without_a_state_behave_as_before(tmp_path: Path) -> None:
    stateful = _StepRecorder(
        _pulse_group(), SubscriptionGroup(selectors=(started(_Work), ended(_Work)))
    )
    stateless = _StatelessRecorder(
        _pulse_group(), SubscriptionGroup(selectors=(started(_Work), ended(_Work)))
    )
    context = _context(tmp_path, callbacks=[stateful, stateless])

    occurrence = context.emit(_Pulse("stateless"))
    with context.scope(_Work("stateless")) as started_occurrence:
        pass

    assert occurrence.count == 1
    assert started_occurrence.count == 1
    # Every pre-existing emitter passes no state, so it reaches exactly the
    # callbacks it reached before: a stateful callback has nothing to observe.
    assert stateful.seen == []
    assert [type(event) for event in stateless.seen] == [_Pulse, Started, Ended]


def test_the_ended_boundary_sees_mutations_made_inside_the_scope_body(
    tmp_path: Path,
) -> None:
    callback = _StepRecorder(SubscriptionGroup(selectors=(started(_Work), ended(_Work))))
    context = _context(tmp_path, callbacks=[callback])
    state = _StepState(step=0)

    with context.scope(_Work("body"), state=state):
        state.step = 1
        state.marks.append("mutated")

    # Both boundaries carry the same reference, so the ended boundary observes
    # the body's work. The event still says only WHEN; the state says WHAT.
    assert [(type(event), step) for event, _, step in callback.seen] == [
        (Started, 0),
        (Ended, 1),
    ]
    assert all(seen is state for _, seen, _ in callback.seen)


def test_the_inherited_no_op_impl_accepts_a_concrete_domain_state(
    tmp_path: Path,
) -> None:
    # THE typeguard target for this change. `StatefulCallback` annotates its
    # generic base methods `state: StateT`, the suite runs typeguard over
    # ``tpen``, and this is the only test whose delivery actually reaches those
    # un-narrowed annotations. Python does not retain a subclass's type argument
    # at runtime, so typeguard can resolve `StateT` no further than its bound,
    # `DomainState` -- which a concrete domain state satisfies. If this fails,
    # the base annotations must become `state: DomainState` and delivery keeps
    # being routed by the `state_type` ClassVar, which is what enforces domain
    # separation in production anyway (production runs carry no instrumentation).
    # No custom constructor here, so the plan goes through the inherited
    # keyword: the first positional parameter of the shared core is `triggers`.
    callback = _InheritedImplRecorder(typed_groups=(_pulse_group(Cadence()),))
    context = _context(tmp_path, callbacks=[callback])

    context.emit(_Pulse("no-op"), state=_StepState(step=2))

    # Assert delivery really happened, so the test cannot pass by being skipped.
    gate = callback._typed_group_states[0].gate
    assert gate is not None and gate.num_calls == 1


def test_a_stateful_subclass_must_declare_a_state_type() -> None:
    with pytest.raises(TypeError, match="state_type"):

        class _Undeclared(StatefulCallback[_StepState]):
            """Declares a static type argument but no runtime ``state_type``."""


def test_the_abstract_base_itself_cannot_be_instantiated() -> None:
    # The base declares no `state_type`, so an instance of it could never be
    # delivered anything. It fails here rather than with an ``AttributeError``
    # raised from inside dispatch, part-way through a run.
    with pytest.raises(TypeError, match="abstract"):
        StatefulCallback()


@pytest.mark.parametrize(
    "declared",
    [int, "_StepState", _StepState(), None],
    ids=["unrelated-type", "name-string", "instance", "none"],
)
def test_a_declared_state_type_must_be_a_domain_state_class(declared: Any) -> None:
    # Rejected at class creation, so a typo cannot survive until dispatch, where
    # `isinstance(state, state_type)` would raise mid-run instead.
    with pytest.raises(TypeError, match="DomainState subclass"):
        type("_BadDeclaration", (StatefulCallback,), {"state_type": declared})


def test_emit_and_scope_reject_a_state_that_is_not_a_domain_state(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)

    # typeguard would raise its own error first, and this asserts the runtime
    # guard that production runs (which carry no instrumentation) rely on.
    with suppress_type_checks():
        with pytest.raises(TypeError, match="DomainState"):
            context.emit(_Pulse("bad"), state=object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="DomainState"):
            with context.scope(_Work("bad"), state=object()):  # type: ignore[arg-type]
                pass


def test_domain_state_never_reaches_the_durable_occurrence_record(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)

    context.emit(_Pulse("recorded"), state=_StepState(step=9))

    records = [
        json.loads(line)
        for line in context.path("occurrences.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    # The typed edge describes the moment only. Data travels beside the
    # occurrence, never inside it or inside its record.
    assert records[0]["fields"] == {"label": "recorded"}
    assert "state" not in records[0]


def test_shared_cadence_machinery_gates_a_stateful_callback(tmp_path: Path) -> None:
    callback = _StepRecorder(_pulse_group(Cadence(every_n=2)))
    context = _context(tmp_path, callbacks=[callback])
    state = _StepState()

    for index in range(4):
        state.step = index
        context.emit(_Pulse(f"pulse-{index}"), state=state)

    # The cadence gate, subscription plan, and context reset live in the shared
    # core, so extracting them did not leave the stateful sibling ungated.
    assert [step for _, _, step in callback.seen] == [0, 2]


def test_stateful_callback_is_a_sibling_of_callback_not_a_subclass() -> None:
    # The two delivery arities differ, so an inheritance edge either way would be
    # a substitutability violation and would make the dispatcher's isinstance
    # discrimination meaningless.
    assert not issubclass(StatefulCallback, Callback)
    assert not issubclass(Callback, StatefulCallback)


def test_stateful_callback_satisfies_the_configured_callback_interface() -> None:
    # `tpen.run` rejects a configured callback that lacks either dispatch entry
    # point, so the legacy `handle` must stay reachable from the shared core.
    from tpen.run import _validate_callbacks

    _validate_callbacks([_StepRecorder(_pulse_group()), _StatelessRecorder()])


def test_trainer_state_is_the_training_domain_state() -> None:
    # Imported inside the test: `tpen.training.state` pulls in torch, and this
    # module otherwise stays torch-free, like `tpen.callback` itself.
    from tpen.training.state import TrainerState

    assert issubclass(TrainerState, DomainState)


def _context(
    tmp_path: Path,
    *,
    callbacks: list[Any] | None = None,
    loggers: list[Any] | None = None,
) -> RunContext:
    """Return a real `RunContext` writing artifacts under ``tmp_path``."""

    artifact_manager = ArtifactManager(
        tmp_path,
        experiment="stateful-callback",
        sector="unit",
        run_id="stateful-callback-unit",
        layout="flat",
    )
    artifact_manager.make_dirs()
    metadata = RunMetadata(
        run_id="stateful-callback-unit",
        run_name="stateful-callback-unit",
        timestamp="2026-08-11T12:00:00+00:00",
        timezone="UTC",
        git_commit="test-sha",
        git_branch="test-branch",
        dirty_worktree=False,
        command="pytest",
        config_path="test.yaml",
        resolved_config_path=str(artifact_manager.path("resolved_config.yaml")),
        run_dir=str(artifact_manager.run_dir),
        device="cpu",
        dtype="float64",
    )
    return RunContext(
        cfg=OmegaConf.create({}),
        source_cfg=OmegaConf.create({}),
        artifact_manager=artifact_manager,
        metadata=metadata,
        clock=RunClock(timezone="UTC", tzinfo=UTC),
        callbacks=[] if callbacks is None else callbacks,
        loggers=[] if loggers is None else loggers,
    )
