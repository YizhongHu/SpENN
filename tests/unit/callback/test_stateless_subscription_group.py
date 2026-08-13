"""A subscription group may declare it needs no domain state (item ``24f91145``).

This is the ADR-E008 amendment under test. Before it, `tpen.callback.Status` and
`tpen.callback.ArtifactIndex` could not subscribe the typed run lifecycle at all:
they are `StatefulCallback`s, the run lifecycle belongs to no domain and carries
no state, and `tpen.artifacts.RunContext._dispatch_occurrence` decided delivery
once per CALLBACK on ``isinstance(state, callback.state_type)``. A run-level
occurrence was therefore skipped for them silently, and subscribing it would have
stopped ``status.json`` and the empty-suite ``diagnostics/index.json`` being
written with no error anywhere.

The decision was per-GROUP delivery rather than splitting those classes in two,
because splitting changes a config-facing ``_target_`` and this repo has already
been bitten by exactly that (``pair_stability_v3/*`` still names ``spenn.callback.*``).

Everything here drives the REAL dispatcher. A `RunContext` stand-in would
override ``_dispatch_occurrence``, which is half of the routing under test.

The mechanism lands here with ZERO consumers, the same shape as PR #174, which
delivered ADR-E008 itself before any callback used it. Migrating `Status` and
`ArtifactIndex` is the next slice; the pins in
``tests/unit/callback/test_typed_run_lifecycle.py`` record that they are now
*able* to move rather than blocked from moving.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tpen.artifacts import RunContext
from tpen.callback import Cadence, Callback, StatefulCallback, SubscriptionGroup
from tpen.events import DomainState, Occurrence, Operation, Subscription
from tpen.events import Event as TypedEvent
from tpen.run_events import RunCompleted, RunStarted
from tests.helpers.run_context import make_run_context

_DIAGNOSTIC_LOGGER = "tpen.callback"


# --------------------------------------------------------------------------
# A two-domain vocabulary, so "skipped because it is not mine" has something
# to be distinct from
# --------------------------------------------------------------------------


class _OwnState(DomainState):
    """The domain the callbacks below declare."""


class _ForeignState(DomainState):
    """Some other domain's state, so a legitimate skip has a cause."""


class _IterationDone(TypedEvent):
    """A domain boundary, emitted with state."""


class _TaskRun(Operation):
    """A scoped domain operation."""


class _OtherRun(Operation):
    """A second scoped operation, so lifecycle shapes can be told apart."""


class _BothGroups(StatefulCallback[_OwnState]):
    """The capability itself: one domain group beside one state-free group."""

    state_type: ClassVar[type[DomainState]] = _OwnState

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            typed_groups=(
                SubscriptionGroup(selectors=(Subscription.of(_IterationDone),)),
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(RunStarted),
                        Subscription.of(RunCompleted),
                    ),
                    stateless=True,
                ),
            ),
            **kwargs,
        )
        self.with_state: list[tuple[Any, DomainState]] = []
        self.without_state: list[Any] = []

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext, state: _OwnState
    ) -> None:
        del context
        self.with_state.append((occurrence.event, state))

    def handle_stateless_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        del context
        self.without_state.append(occurrence.event)


class _DomainOnly(StatefulCallback[_OwnState]):
    """Today's shape, unchanged: every group wants the domain's state."""

    state_type: ClassVar[type[DomainState]] = _OwnState

    def __init__(self, cadence: Cadence | None = None, **kwargs: Any) -> None:
        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(Subscription.of(_IterationDone),), cadence=cadence
                ),
            ),
            **kwargs,
        )
        self.seen: list[Any] = []

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext, state: _OwnState
    ) -> None:
        del context, state
        self.seen.append(occurrence.event)


# --------------------------------------------------------------------------
# The new capability
# --------------------------------------------------------------------------


def test_a_stateless_group_receives_a_state_free_run_level_event(tmp_path: Path) -> None:
    """The exact delivery `Status` and `ArtifactIndex` were denied.

    The same instance declares a domain group, so this is not "a callback that
    happens to be state-free": it is one callback observing both.
    """

    callback = _BothGroups()
    context = make_run_context(tmp_path, callbacks=[callback])

    context.emit(RunStarted())
    context.emit(RunCompleted())

    assert callback.without_state == [RunStarted(), RunCompleted()]
    assert callback.with_state == []


def test_a_domain_group_on_the_same_callback_still_receives_its_state(
    tmp_path: Path,
) -> None:
    """The stateless group does not cost the stateful one its state."""

    callback = _BothGroups()
    context = make_run_context(tmp_path, callbacks=[callback])
    state = _OwnState()

    context.emit(_IterationDone(), state=state)

    assert len(callback.with_state) == 1
    assert callback.with_state[0][1] is state
    assert callback.without_state == []


def test_no_occurrence_reaches_both_hooks(tmp_path: Path) -> None:
    """Structurally impossible, not merely unobserved.

    `tpen.callback.cadence.validate_subscription_groups` rejects overlapping
    deliveries, so at most one group matches any occurrence and the two hooks
    cannot both fire. That check is what this property rests on, which is why
    the test below proves a stateless group is not exempt from it.
    """

    callback = _BothGroups()
    context = make_run_context(tmp_path, callbacks=[callback])

    context.emit(RunStarted())
    context.emit(_IterationDone(), state=_OwnState())
    context.emit(RunCompleted())
    context.emit(_IterationDone(), state=_ForeignState())

    assert len(callback.without_state) == 2
    assert len(callback.with_state) == 1


def test_a_stateless_group_receives_both_boundaries_of_a_scope(tmp_path: Path) -> None:
    """Lifecycle pairing is a group property and survives the new route."""

    class _ScopedStateless(StatefulCallback[_OwnState]):
        state_type: ClassVar[type[DomainState]] = _OwnState

        def __init__(self) -> None:
            super().__init__(
                typed_groups=(
                    SubscriptionGroup(selectors=(Subscription.of(_IterationDone),)),
                    SubscriptionGroup(
                        selectors=(
                            Subscription.started(_TaskRun),
                            Subscription.ended(_TaskRun),
                        ),
                        stateless=True,
                    ),
                )
            )
            self.seen: list[str] = []

        def handle_occurrence_impl(
            self,
            occurrence: Occurrence[TypedEvent],
            context: RunContext,
            state: _OwnState,
        ) -> None:
            del occurrence, context, state

        def handle_stateless_occurrence_impl(
            self, occurrence: Occurrence[TypedEvent], context: RunContext
        ) -> None:
            del context
            self.seen.append(type(occurrence.event).__name__)

    callback = _ScopedStateless()
    context = make_run_context(tmp_path, callbacks=[callback])

    with context.scope(_TaskRun()):
        pass

    assert callback.seen == ["Started", "Ended"]


def test_a_stateless_group_still_honours_its_own_cadence(tmp_path: Path) -> None:
    """``stateless`` changes the delivery shape, not the schedule."""

    class _Cadenced(StatefulCallback[_OwnState]):
        state_type: ClassVar[type[DomainState]] = _OwnState

        def __init__(self) -> None:
            super().__init__(
                typed_groups=(
                    SubscriptionGroup(selectors=(Subscription.of(_IterationDone),)),
                    SubscriptionGroup(
                        selectors=(Subscription.of(RunStarted),),
                        cadence=Cadence(every_n=2, start=1),
                        stateless=True,
                    ),
                )
            )
            self.calls = 0

        def handle_occurrence_impl(
            self,
            occurrence: Occurrence[TypedEvent],
            context: RunContext,
            state: _OwnState,
        ) -> None:
            del occurrence, context, state

        def handle_stateless_occurrence_impl(
            self, occurrence: Occurrence[TypedEvent], context: RunContext
        ) -> None:
            del occurrence, context
            self.calls += 1

    callback = _Cadenced()
    context = make_run_context(tmp_path, callbacks=[callback])

    for _ in range(4):
        context.emit(RunStarted())

    assert callback.calls == 2


# --------------------------------------------------------------------------
# Nothing that worked before moved
# --------------------------------------------------------------------------


def test_a_foreign_domains_state_is_still_skipped_silently(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The routine skip stays routine, and stays quiet.

    A mixed run emits several domains' boundaries, so this happens constantly;
    a diagnostic here would drown the one that matters.
    """

    callback = _DomainOnly()
    context = make_run_context(tmp_path, callbacks=[callback])

    with caplog.at_level(logging.WARNING, logger=_DIAGNOSTIC_LOGGER):
        context.emit(_IterationDone(), state=_ForeignState())

    assert callback.seen == []
    assert caplog.records == []


def test_a_foreign_occurrence_does_not_spend_the_groups_cadence_budget(
    tmp_path: Path,
) -> None:
    """The state check runs BEFORE the gate, and that ordering is observable.

    Asking `_typed_group_delivers` advances the group's cadence gate as a side
    effect. When the dispatcher filtered whole callbacks, a foreign domain's
    occurrence never reached that question; if the per-group check were placed
    after it instead of before, a single foreign occurrence would consume this
    group's entire one-call budget and starve its own boundary.
    """

    callback = _DomainOnly(cadence=Cadence(max_calls=1))
    context = make_run_context(tmp_path, callbacks=[callback])

    context.emit(_IterationDone(), state=_ForeignState())
    context.emit(_IterationDone())
    context.emit(_IterationDone(), state=_OwnState())

    assert len(callback.seen) == 1


def test_a_plain_callback_still_receives_every_selected_occurrence(
    tmp_path: Path,
) -> None:
    """`Callback` delivery is untouched: state present or absent, it fires."""

    class _Plain(Callback):
        def __init__(self) -> None:
            super().__init__(
                typed_groups=(
                    SubscriptionGroup(selectors=(Subscription.of(_IterationDone),)),
                )
            )
            self.calls = 0

        def handle_occurrence_impl(
            self, occurrence: Occurrence[TypedEvent], context: RunContext
        ) -> None:
            del occurrence, context
            self.calls += 1

    callback = _Plain()
    context = make_run_context(tmp_path, callbacks=[callback])

    context.emit(_IterationDone())
    context.emit(_IterationDone(), state=_OwnState())
    context.emit(_IterationDone(), state=_ForeignState())

    assert callback.calls == 3


# --------------------------------------------------------------------------
# The declaration cannot disagree with the class it is on
# --------------------------------------------------------------------------


def test_a_plain_callback_rejects_a_stateless_group() -> None:
    """Vacuous there, and actively misleading: it implies a stateful sibling."""

    class _Plain(Callback):
        def __init__(self) -> None:
            super().__init__(
                typed_groups=(
                    SubscriptionGroup(
                        selectors=(Subscription.of(RunStarted),), stateless=True
                    ),
                )
            )

    with pytest.raises(TypeError, match="meaningful only on a StatefulCallback"):
        _Plain()


def test_an_all_stateless_stateful_callback_is_rejected() -> None:
    """A ``state_type`` that can never route a delivery is a wiring error.

    "Can never" is a fact about the CLASS: this one overrides no
    ``handle_occurrence_impl``, so no configuration of it could consume
    ``_OwnState``, and it should have been a `Callback`.
    """

    class _AllStateless(StatefulCallback[_OwnState]):
        state_type: ClassVar[type[DomainState]] = _OwnState

        def __init__(self) -> None:
            super().__init__(
                typed_groups=(
                    SubscriptionGroup(
                        selectors=(Subscription.of(RunStarted),), stateless=True
                    ),
                )
            )

        def handle_stateless_occurrence_impl(
            self, occurrence: Occurrence[TypedEvent], context: RunContext
        ) -> None:
            del occurrence, context

    with pytest.raises(TypeError, match="make it a Callback"):
        _AllStateless()


def test_an_all_stateless_plan_is_allowed_when_the_class_can_route_state() -> None:
    """The group plan is per-INSTANCE; ``state_type`` is per-CLASS.

    `tpen.callback.Status` is why this distinction had to be drawn. With
    ``train_lines`` off it declares only the state-free run-lifecycle group, yet
    the same class with ``train_lines`` on routes
    `tpen.training.state.TrainerState` through ``handle_occurrence_impl``.
    Judging the plan alone would have made the SHIPPED DEFAULT of that callback
    unconstructible, and "make it a `Callback` instead" is false advice for a
    class that genuinely observes a domain.

    Overriding ``handle_occurrence_impl`` is the class-level evidence that the
    ``state_type`` can route something, so it is what the check asks about.
    """

    class _StatefulByConfiguration(StatefulCallback[_OwnState]):
        state_type: ClassVar[type[DomainState]] = _OwnState

        def __init__(self, *, domain_group: bool) -> None:
            super().__init__(
                typed_groups=(
                    (SubscriptionGroup(selectors=(Subscription.of(_IterationDone),)),)
                    if domain_group
                    else ()
                )
                + (
                    SubscriptionGroup(
                        selectors=(Subscription.of(RunStarted),), stateless=True
                    ),
                )
            )
            self.seen: list[TypedEvent] = []

        def handle_occurrence_impl(
            self,
            occurrence: Occurrence[TypedEvent],
            context: RunContext,
            state: _OwnState,
        ) -> None:
            del context, state
            self.seen.append(occurrence.event)

        def handle_stateless_occurrence_impl(
            self, occurrence: Occurrence[TypedEvent], context: RunContext
        ) -> None:
            del context
            self.seen.append(occurrence.event)

    assert _StatefulByConfiguration(domain_group=False) is not None
    assert _StatefulByConfiguration(domain_group=True) is not None


def test_an_all_stateless_plan_still_needs_the_stateless_hook() -> None:
    """Narrowing the all-stateless check must not open the silent-drop hole.

    A class that overrides ``handle_occurrence_impl`` now passes the
    all-stateless check, so the "forgot the hook" check is the only thing left
    catching an all-stateless plan whose deliveries would land in the inherited
    no-op.
    """

    class _RoutesStateButForgotTheHook(StatefulCallback[_OwnState]):
        state_type: ClassVar[type[DomainState]] = _OwnState

        def __init__(self) -> None:
            super().__init__(
                typed_groups=(
                    SubscriptionGroup(
                        selectors=(Subscription.of(RunStarted),), stateless=True
                    ),
                )
            )

        def handle_occurrence_impl(
            self,
            occurrence: Occurrence[TypedEvent],
            context: RunContext,
            state: _OwnState,
        ) -> None:
            del occurrence, context, state

    with pytest.raises(TypeError, match="silently discarded"):
        _RoutesStateButForgotTheHook()


def test_a_stateless_group_without_its_hook_is_rejected() -> None:
    """The new route must not become a new way to lose deliveries quietly."""

    class _ForgotHook(StatefulCallback[_OwnState]):
        state_type: ClassVar[type[DomainState]] = _OwnState

        def __init__(self) -> None:
            super().__init__(
                typed_groups=(
                    SubscriptionGroup(selectors=(Subscription.of(_IterationDone),)),
                    SubscriptionGroup(
                        selectors=(Subscription.of(RunStarted),), stateless=True
                    ),
                )
            )

        def handle_occurrence_impl(
            self,
            occurrence: Occurrence[TypedEvent],
            context: RunContext,
            state: _OwnState,
        ) -> None:
            del occurrence, context, state

    with pytest.raises(TypeError, match="silently discarded"):
        _ForgotHook()


def test_a_stateful_callback_with_no_groups_at_all_still_constructs() -> None:
    """"No groups" is a third claim again: it subscribes nothing at all.

    This described `Status(train_lines=False)` when the mechanism landed with no
    consumers. That instance now declares the state-free run-lifecycle group, so
    the case it guards is a hypothetical one -- kept because the check above it
    reads ``if not stateless: return`` and an empty plan must fall through it
    rather than be judged by it.
    """

    class _NoGroups(StatefulCallback[_OwnState]):
        state_type: ClassVar[type[DomainState]] = _OwnState

    assert _NoGroups(typed_groups=()) is not None


def test_a_non_bool_stateless_declaration_is_rejected() -> None:
    """Typeguard checks parameters and returns, never dataclass FIELDS.

    So the annotation on `SubscriptionGroup.stateless` guards nothing at
    runtime, and a truthy non-bool would otherwise select the stateless route
    silently -- the failure shape this whole change removes.
    """

    with pytest.raises(TypeError, match="stateless must be a bool"):
        SubscriptionGroup(selectors=(Subscription.of(RunStarted),), stateless=1)


@pytest.mark.parametrize("second", ["identical", "subclass"])
def test_a_stateless_group_is_not_exempt_from_the_overlap_check(second: str) -> None:
    """Weakening this would let one occurrence reach both hooks.

    ADR-E002 records the check as by ``issubclass`` and not by identity, so both
    spellings are covered.
    """

    class _Subclass(_IterationDone):
        pass

    subject = _IterationDone if second == "identical" else _Subclass

    class _Overlapping(StatefulCallback[_OwnState]):
        state_type: ClassVar[type[DomainState]] = _OwnState

        def __init__(self) -> None:
            super().__init__(
                typed_groups=(
                    SubscriptionGroup(selectors=(Subscription.of(_IterationDone),)),
                    SubscriptionGroup(
                        selectors=(Subscription.of(subject),), stateless=True
                    ),
                )
            )

        def handle_occurrence_impl(
            self,
            occurrence: Occurrence[TypedEvent],
            context: RunContext,
            state: _OwnState,
        ) -> None:
            del occurrence, context, state

        def handle_stateless_occurrence_impl(
            self, occurrence: Occurrence[TypedEvent], context: RunContext
        ) -> None:
            del occurrence, context

    with pytest.raises(ValueError, match="overlapping deliveries"):
        _Overlapping()


# --------------------------------------------------------------------------
# The two skips are distinguishable, which is the constraint the decision set
# --------------------------------------------------------------------------


def test_a_selected_boundary_arriving_with_no_state_at_all_is_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The wiring error, told apart from the routine skip by ONE fact.

    Some other domain's state arrived means this callback is not the audience,
    which is routine. NOTHING arriving on a boundary this group actually
    selected means nobody could have been -- either the emitter omitted its
    ``state=`` or the group should have declared ``stateless=True``. That is a
    mistake every time, and it is precisely the silence that trapped `Status`
    and `ArtifactIndex`.
    """

    callback = _DomainOnly()
    context = make_run_context(tmp_path, callbacks=[callback])

    with caplog.at_level(logging.WARNING, logger=_DIAGNOSTIC_LOGGER):
        context.emit(_IterationDone())

    assert callback.seen == []
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    for fragment in ("_DomainOnly", "group 0", "_OwnState", "_IterationDone"):
        assert fragment in message


def test_the_report_is_a_warning_rather_than_an_exception(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Deliberate, and the pinned counterpart is why.

    ``test_a_boundary_emitted_without_state_delivers_nothing`` fixes that a
    boundary emitted without state delivers nothing rather than killing the run,
    and that behaviour stays correct for groups that DO want state. A callback
    misconfiguration must not take down training, so the mismatch is made
    audible rather than fatal.
    """

    callback = _DomainOnly()
    context = make_run_context(tmp_path, callbacks=[callback])

    with caplog.at_level(logging.WARNING, logger=_DIAGNOSTIC_LOGGER):
        context.emit(_IterationDone())

    assert caplog.records[0].levelno == logging.WARNING


def test_the_report_fires_once_per_event_shape_rather_than_once_per_occurrence(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The selected boundary may be a per-step one, so this cannot repeat."""

    callback = _DomainOnly()
    context = make_run_context(tmp_path, callbacks=[callback])

    with caplog.at_level(logging.WARNING, logger=_DIAGNOSTIC_LOGGER):
        for _ in range(50):
            context.emit(_IterationDone())

    assert len(caplog.records) == 1


def test_lifecycle_shapes_are_reported_separately(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``Started`` alone is not the shape; the operation type is half of it."""

    class _ScopeWatcher(StatefulCallback[_OwnState]):
        state_type: ClassVar[type[DomainState]] = _OwnState

        def __init__(self) -> None:
            super().__init__(
                typed_groups=(
                    SubscriptionGroup(
                        selectors=(
                            Subscription.started(_TaskRun),
                            Subscription.ended(_TaskRun),
                            Subscription.started(_OtherRun),
                        )
                    ),
                )
            )

        def handle_occurrence_impl(
            self,
            occurrence: Occurrence[TypedEvent],
            context: RunContext,
            state: _OwnState,
        ) -> None:
            del occurrence, context, state

    context = make_run_context(tmp_path, callbacks=[_ScopeWatcher()])

    with caplog.at_level(logging.WARNING, logger=_DIAGNOSTIC_LOGGER):
        with context.scope(_TaskRun()):
            pass
        with context.scope(_OtherRun()):
            pass

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 3
    assert any("Started[_TaskRun]" in message for message in messages)
    assert any("Ended[_TaskRun]" in message for message in messages)
    assert any("Started[_OtherRun]" in message for message in messages)


def test_an_unselected_state_free_boundary_is_not_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The report is about a group's OWN boundary, not about every emit.

    Without this, every state-free run-level event would be reported by every
    stateful callback in the run, which is noise indistinguishable from signal.
    """

    callback = _DomainOnly()
    context = make_run_context(tmp_path, callbacks=[callback])

    with caplog.at_level(logging.WARNING, logger=_DIAGNOSTIC_LOGGER):
        context.emit(RunStarted())

    assert caplog.records == []


def test_a_stateless_groups_own_delivery_is_never_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A group that declared no state is not disappointed to receive none."""

    callback = _BothGroups()
    context = make_run_context(tmp_path, callbacks=[callback])

    with caplog.at_level(logging.WARNING, logger=_DIAGNOSTIC_LOGGER):
        context.emit(RunStarted())

    assert callback.without_state == [RunStarted()]
    assert caplog.records == []


def test_the_report_dedupe_resets_with_the_owning_context(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A second run through a reused instance is a second chance to be told."""

    callback = _DomainOnly()
    first = make_run_context(tmp_path / "a", callbacks=[callback])
    second = make_run_context(tmp_path / "b", callbacks=[callback])

    with caplog.at_level(logging.WARNING, logger=_DIAGNOSTIC_LOGGER):
        first.emit(_IterationDone())
        second.emit(_IterationDone())

    assert len(caplog.records) == 2
