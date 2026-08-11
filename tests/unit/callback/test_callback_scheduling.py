"""Tests for legacy scheduling and typed occurrence-count cadence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tpen.artifacts import RunContext
from tpen.callback import (
    Cadence,
    CadenceGate,
    Callback,
    Event,
    SubscriptionGroup,
)
from tpen.events import DomainState, Ended, Event as TypedEvent, Occurrence, Operation, Started
from tpen.events import Subscription, ended, started


class Recorder(Callback):
    """Record the step of every legacy event actually handled."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(["step_end"], **kwargs)
        self.handled: list[int] = []

    def on_step_end(self, event: Event) -> None:
        assert event.step is not None
        self.handled.append(event.step)


@dataclass(frozen=True)
class _Pulse(TypedEvent):
    label: str


@dataclass(frozen=True)
class _OtherPulse(TypedEvent):
    label: str


@dataclass(frozen=True)
class _BasePulse(TypedEvent):
    label: str


@dataclass(frozen=True)
class _ChildPulse(_BasePulse):
    pass


@dataclass(frozen=True)
class _Work(Operation):
    label: str


@dataclass(frozen=True)
class _ChildWork(_Work):
    pass


class _TypedRecorder(Callback):
    def __init__(self, *groups: SubscriptionGroup) -> None:
        super().__init__(typed_groups=groups)
        self.handled: list[Occurrence[TypedEvent]] = []

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
    ) -> None:
        del context
        self.handled.append(occurrence)


class _ResetRecorder(_TypedRecorder):
    def __init__(self, *groups: SubscriptionGroup) -> None:
        self.reset_count = 0
        super().__init__(*groups)

    def _reset_typed_state(self) -> None:
        self.reset_count += 1


class _AlwaysEqualContext(RunContext):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _AlwaysEqualContext)


class _DispatchContext(RunContext):
    def __init__(self, *callbacks: Callback) -> None:
        self.callbacks = list(callbacks)
        self._occurrence_counts: dict[type[TypedEvent] | type[Operation], int] = {}

    def _dispatch_occurrence(
        self, occurrence: Occurrence[Any], *, state: DomainState | None = None
    ) -> None:
        # These groups are all state-free `Callback` subclasses, so this double
        # accepts the widened dispatch signature and drops the state.
        del state
        for callback in self.callbacks:
            callback.handle_occurrence(occurrence, self)


def _context() -> RunContext:
    return object.__new__(RunContext)


def _dispatch(
    callback: Callback,
    event: TypedEvent,
    count: int,
    context: RunContext,
) -> None:
    callback.handle_occurrence(Occurrence(event=event, count=count), context)


def _drive(callback: Callback, steps) -> None:
    for step in steps:
        callback.handle(
            Event(
                name="step_end",
                context=None,  # type: ignore[arg-type]
                state=None,
                payload={"step": step},
                step=step,
            )
        )


def test_legacy_positional_constructor_remains_compatible() -> None:
    callback = Callback(("step_end",), 2, 3, 4, 0.5, 7)

    assert callback.triggers == ("step_end",)
    assert callback.every_n_steps == 2
    assert callback.start_step == 3
    assert callback.max_calls == 4
    assert callback.probability == 0.5
    assert callback.seed == 7


def test_every_n_steps_filters_by_step() -> None:
    callback = Recorder(every_n_steps=2)

    _drive(callback, range(0, 6))

    assert callback.handled == [0, 2, 4]


def test_start_step_delays_first_run() -> None:
    callback = Recorder(every_n_steps=1, start_step=3)

    _drive(callback, range(0, 6))

    assert callback.handled == [3, 4, 5]


def test_max_calls_counts_actual_executions() -> None:
    callback = Recorder(every_n_steps=1, max_calls=2)

    _drive(callback, range(1, 10))

    assert callback.handled == [1, 2]
    assert callback.num_calls == 2


def test_probability_zero_never_runs() -> None:
    callback = Recorder(probability=0.0)

    _drive(callback, range(1, 21))

    assert callback.handled == []
    assert callback.num_calls == 0


def test_probability_one_always_runs_when_scheduled() -> None:
    callback = Recorder(every_n_steps=2, probability=1.0)

    _drive(callback, range(0, 6))

    assert callback.handled == [0, 2, 4]


def test_probability_is_deterministic_with_seed() -> None:
    first = Recorder(probability=0.5, seed=1234)
    second = Recorder(probability=0.5, seed=1234)

    _drive(first, range(1, 51))
    _drive(second, range(1, 51))

    assert first.handled == second.handled
    assert 0 < len(first.handled) < 50


def test_probability_is_applied_after_step_filters() -> None:
    callback = Recorder(every_n_steps=2, probability=0.5, seed=7)

    _drive(callback, range(0, 40))

    assert all(step % 2 == 0 for step in callback.handled)
    assert len(callback.handled) < 20


def test_invalid_legacy_probability_raises() -> None:
    with pytest.raises(ValueError, match="probability"):
        Recorder(probability=1.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"every_n": 0}, "every_n"),
        ({"start": 0}, "start"),
        ({"max_calls": -1}, "max_calls"),
        ({"probability": -0.1}, "probability"),
        ({"probability": 1.1}, "probability"),
    ],
)
def test_invalid_typed_cadence_raises(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Cadence(**kwargs)  # type: ignore[arg-type]


def test_cadence_draws_only_after_the_count_window_passes() -> None:
    gate = CadenceGate(Cadence(every_n=2, start=2, probability=0.5, seed=1))

    assert not gate.should_run(1)
    assert gate.should_run(2)
    assert not gate.should_run(3)
    assert not gate.should_run(4)
    assert gate.num_calls == 1


def test_cadence_max_calls_precedes_probability_and_reset_reseeds() -> None:
    gate = CadenceGate(Cadence(max_calls=1, probability=0.5, seed=1))

    assert gate.should_run(1)
    assert not gate.should_run(2)
    gate.reset()
    assert gate.num_calls == 0
    assert gate.should_run(1)


def test_nonmatching_occurrence_consumes_neither_rng_nor_counter() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(
            selectors=(Subscription.of(_Pulse),),
            cadence=Cadence(probability=0.5, seed=1),
        )
    )
    context = _DispatchContext(callback)

    other = context.emit(_OtherPulse("nonmatching"))
    pulse = context.emit(_Pulse("first matching"))

    assert (other.count, pulse.count) == (1, 1)
    assert [item.event for item in callback.handled] == [_Pulse("first matching")]
    gate = callback._typed_group_states[0].gate
    assert gate is not None and gate.num_calls == 1


def test_end_only_group_observes_hidden_start_and_consumes_rejections() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(
            selectors=(ended(_Work),),
            cadence=Cadence(every_n=2),
        )
    )
    context = _context()

    first = _Work("first")
    _dispatch(callback, Started(first), 1, context)
    _dispatch(callback, Ended(first), 1, context)
    second = _Work("second")
    _dispatch(callback, Started(second), 2, context)
    state = callback._typed_group_states[0]
    assert state.open_pairs == {(type(second), 2): False}
    _dispatch(callback, Ended(second), 2, context)

    assert [item.event for item in callback.handled] == [Ended(first)]
    assert state.open_pairs == {}
    assert state.gate is not None and state.gate.num_calls == 1


def test_orphan_ended_is_not_delivered_and_consumes_no_gate_state() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(
            selectors=(ended(_Work),),
            cadence=Cadence(probability=0.5, seed=1),
        )
    )
    context = _context()

    _dispatch(callback, Ended(_Work("orphan")), 1, context)
    live = _Work("paired")
    _dispatch(callback, Started(live), 2, context)
    _dispatch(callback, Ended(live), 2, context)

    assert [item.event for item in callback.handled] == [Ended(live)]
    state = callback._typed_group_states[0]
    assert state.open_pairs == {}
    assert state.gate is not None and state.gate.num_calls == 1


def test_maxed_lifecycle_start_caches_false_until_end_cleanup() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(
            selectors=(ended(_Work),),
            cadence=Cadence(max_calls=0),
        )
    )
    context = _context()
    operation = _Work("maxed")

    _dispatch(callback, Started(operation), 1, context)
    state = callback._typed_group_states[0]
    assert state.open_pairs == {(type(operation), 1): False}
    _dispatch(callback, Ended(operation), 1, context)

    assert callback.handled == []
    assert state.open_pairs == {}
    assert state.gate is not None and state.gate.num_calls == 0


def test_start_only_group_observes_hidden_end_and_pops_pair_state() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(
            selectors=(started(_Work),),
            cadence=Cadence(probability=0.5, seed=1),
        )
    )
    context = _context()

    first = _Work("first")
    _dispatch(callback, Started(first), 1, context)
    _dispatch(callback, Ended(first), 1, context)
    second = _Work("second")
    _dispatch(callback, Started(second), 2, context)
    _dispatch(callback, Ended(second), 2, context)

    assert [item.event for item in callback.handled] == [Started(first)]
    assert callback._typed_group_states[0].open_pairs == {}


def test_started_and_ended_share_one_draw_and_one_num_calls_increment() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(
            selectors=(started(_Work), ended(_Work)),
            cadence=Cadence(probability=0.5, seed=1),
        )
    )
    context = _context()

    first = _Work("first")
    _dispatch(callback, Started(first), 1, context)
    _dispatch(callback, Ended(first), 1, context)
    second = _Work("second")
    _dispatch(callback, Started(second), 2, context)
    _dispatch(callback, Ended(second), 2, context)

    assert [item.event for item in callback.handled] == [Started(first), Ended(first)]
    gate = callback._typed_group_states[0].gate
    assert gate is not None and gate.num_calls == 1


def test_nested_same_type_scopes_pair_by_concrete_type_and_count() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(
            selectors=(started(_Work), ended(_Work)),
            cadence=Cadence(),
        )
    )
    context = _DispatchContext(callback)
    outer = _Work("outer")
    inner = _Work("inner")

    with context.scope(outer):
        with context.scope(inner):
            pass

    assert [(item.event, item.count) for item in callback.handled] == [
        (Started(outer), 1),
        (Started(inner), 2),
        (Ended(inner), 2),
        (Ended(outer), 1),
    ]
    state = callback._typed_group_states[0]
    assert state.open_pairs == {}
    assert state.gate is not None and state.gate.num_calls == 2


def test_mixed_overlapping_selectors_in_one_group_deliver_once() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(
            selectors=(started(_Work), started(_ChildWork), ended(_Work)),
        )
    )
    context = _context()
    operation = _ChildWork("child")

    _dispatch(callback, Started(operation), 1, context)
    _dispatch(callback, Ended(operation), 1, context)

    assert [item.event for item in callback.handled] == [
        Started(operation),
        Ended(operation),
    ]
    assert callback._typed_group_states[0].gate is None


def test_base_event_subscription_preserves_independent_concrete_counts() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(selectors=(Subscription.of(_BasePulse),))
    )
    context = _DispatchContext(callback)

    base_one = context.emit(_BasePulse("base one"))
    child = context.emit(_ChildPulse("child"))
    base_two = context.emit(_BasePulse("base two"))

    assert (base_one.count, child.count, base_two.count) == (1, 1, 2)
    assert [item.event for item in callback.handled] == [
        _BasePulse("base one"),
        _ChildPulse("child"),
        _BasePulse("base two"),
    ]


def test_max_calls_is_global_within_one_group() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(
            selectors=(Subscription.of(_Pulse), Subscription.of(_OtherPulse)),
            cadence=Cadence(max_calls=1),
        )
    )
    context = _context()

    _dispatch(callback, _Pulse("first"), 1, context)
    _dispatch(callback, _OtherPulse("second"), 1, context)

    assert [item.event for item in callback.handled] == [_Pulse("first")]


def test_groups_have_independent_rng_streams() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(
            selectors=(Subscription.of(_Pulse),),
            cadence=Cadence(probability=0.5, seed=1),
        ),
        SubscriptionGroup(
            selectors=(Subscription.of(_OtherPulse),),
            cadence=Cadence(probability=0.5, seed=1),
        ),
    )
    context = _context()

    _dispatch(callback, _Pulse("first"), 1, context)
    _dispatch(callback, _OtherPulse("second"), 1, context)

    assert [item.event for item in callback.handled] == [
        _Pulse("first"),
        _OtherPulse("second"),
    ]


@pytest.mark.parametrize(
    "first,second",
    [
        (started(_Work), started(_ChildWork)),
        (Subscription.of(_BasePulse), Subscription.of(_ChildPulse)),
    ],
)
def test_overlapping_deliveries_across_groups_are_rejected(
    first: Subscription,
    second: Subscription,
) -> None:
    with pytest.raises(ValueError, match="overlapping"):
        _TypedRecorder(
            SubscriptionGroup(selectors=(first,)),
            SubscriptionGroup(selectors=(second,)),
        )


def test_distinct_start_and_end_delivery_groups_are_allowed() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(selectors=(started(_Work),)),
        SubscriptionGroup(selectors=(ended(_Work),)),
    )
    context = _context()
    operation = _Work("split")

    _dispatch(callback, Started(operation), 1, context)
    _dispatch(callback, Ended(operation), 1, context)

    assert [item.event for item in callback.handled] == [
        Started(operation),
        Ended(operation),
    ]


def test_context_identity_change_resets_gate_rng_counter_and_hook() -> None:
    callback = _ResetRecorder(
        SubscriptionGroup(
            selectors=(Subscription.of(_Pulse),),
            cadence=Cadence(probability=0.5, seed=1),
        )
    )
    first_context = object.__new__(_AlwaysEqualContext)
    second_context = object.__new__(_AlwaysEqualContext)
    state = callback._typed_group_states[0]
    original_gate = state.gate
    assert original_gate is not None
    assert first_context == second_context
    assert first_context is not second_context

    _dispatch(callback, _Pulse("first"), 1, first_context)
    _dispatch(callback, _Pulse("second"), 2, first_context)
    _dispatch(callback, _Pulse("fresh"), 1, second_context)

    assert [item.event.label for item in callback.handled] == ["first", "fresh"]
    assert state.gate is original_gate
    assert original_gate.num_calls == 1
    assert callback.reset_count == 2


def test_context_identity_change_clears_open_lifecycle_pairs() -> None:
    callback = _TypedRecorder(
        SubscriptionGroup(
            selectors=(ended(_Work),),
            cadence=Cadence(probability=0.5, seed=1),
        )
    )
    first_context = _context()
    second_context = _context()
    operation = _Work("abandoned")

    _dispatch(callback, Started(operation), 1, first_context)
    assert callback._typed_group_states[0].open_pairs
    _dispatch(callback, _Pulse("context-boundary"), 1, second_context)

    state = callback._typed_group_states[0]
    assert state.open_pairs == {}
    assert state.gate is not None and state.gate.num_calls == 0
