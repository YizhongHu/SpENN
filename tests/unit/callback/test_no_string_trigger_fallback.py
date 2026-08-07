"""Typed dispatch never falls back to legacy string trigger names."""

from __future__ import annotations

from dataclasses import dataclass

from tpen.artifacts import RunContext
from tpen.callback import Callback, SubscriptionGroup
from tpen.events import Event, Occurrence, Subscription


@dataclass(frozen=True)
class _Pulse(Event):
    value: int


class _Recorder(Callback):
    def __init__(self, *, triggers=(), typed_groups=()) -> None:
        super().__init__(triggers=triggers, typed_groups=typed_groups)
        self.values: list[int] = []

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[Event],
        context: RunContext,
    ) -> None:
        del context
        assert isinstance(occurrence.event, _Pulse)
        self.values.append(occurrence.event.value)


def test_legacy_trigger_name_does_not_select_typed_event() -> None:
    callback = _Recorder(triggers=("_Pulse", "pulse", "Pulse"))
    context = object.__new__(RunContext)

    callback.handle_occurrence(Occurrence(event=_Pulse(1), count=1), context)

    assert callback.values == []


def test_typed_selector_is_required_for_typed_delivery() -> None:
    callback = _Recorder(
        triggers=("unrelated",),
        typed_groups=(
            SubscriptionGroup(selectors=(Subscription.of(_Pulse),)),
        ),
    )
    context = object.__new__(RunContext)

    callback.handle_occurrence(Occurrence(event=_Pulse(2), count=1), context)

    assert callback.values == [2]
