"""Pure, pin-aware checkpoint retention policies.

Retention policies only plan lifecycle decisions.  They consume immutable
checkpoint references and an already-materialized snapshot of active pin
records; they never inspect the filesystem, mutate a ledger, or delete an
artifact.  A later executor can therefore apply a :class:`RetentionSnapshot`
under its own lock without repeating policy decisions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

from .pins import PinRecord


RETENTION_SNAPSHOT_SCHEMA = "tpen.checkpoint-retention/v1"


@runtime_checkable
class RetentionRef(Protocol):
    """Immutable reference fields needed by retention planning."""

    checkpoint_dir: Path
    completed_updates: int
    next_iteration: int
    content_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return the complete, JSON-safe reference mapping."""


RetentionPinState: TypeAlias = Iterable[PinRecord]


@dataclass(frozen=True, slots=True)
class RetentionDecision:
    """One immutable retain/delete decision for an exact reference.

    Parameters
    ----------
    ref : RetentionRef
        The exact immutable reference targeted by this decision.
    action : {"retain", "delete"}
        The action a later executor may apply.  The policy itself never
        performs that action.
    reason : str
        Stable machine-readable explanation for the decision.
    protected : bool
        Whether the reference is retained by a mandatory safety rule rather
        than by the configured retention window.
    """

    ref: RetentionRef
    action: str
    reason: str
    protected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe decision mapping."""

        return {
            "action": self.action,
            "protected": self.protected,
            "reason": self.reason,
            "ref": _ref_mapping(self.ref),
        }


@dataclass(frozen=True, slots=True)
class RetentionSnapshot:
    """Deterministic, serializable output of one retention policy decision.

    The ``decisions`` tuple is ordered by the total reference ordering used by
    the policy.  ``to_json`` uses sorted JSON keys and compact separators, so
    equivalent inputs in different iteration orders produce byte-identical
    snapshots.
    """

    policy: str
    status: str
    decisions: tuple[RetentionDecision, ...]
    config: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "config", tuple(self.config))

    @property
    def retained_refs(self) -> tuple[RetentionRef, ...]:
        """Return exact references whose planned action is retain."""

        return tuple(decision.ref for decision in self.decisions if decision.action == "retain")

    @property
    def deletion_targets(self) -> tuple[RetentionRef, ...]:
        """Return exact references a later executor may delete."""

        return tuple(decision.ref for decision in self.decisions if decision.action == "delete")

    # These short aliases keep call sites readable while the longer names
    # make the no-deletion boundary explicit in documentation.
    @property
    def retain(self) -> tuple[RetentionRef, ...]:
        """Alias for :attr:`retained_refs`."""

        return self.retained_refs

    @property
    def delete(self) -> tuple[RetentionRef, ...]:
        """Alias for :attr:`deletion_targets`."""

        return self.deletion_targets

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-safe decision snapshot."""

        return {
            "schema": RETENTION_SNAPSHOT_SCHEMA,
            "policy": self.policy,
            "status": self.status,
            "config": dict(self.config),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "retain": [_ref_mapping(ref) for ref in self.retained_refs],
            "delete": [_ref_mapping(ref) for ref in self.deletion_targets],
        }

    def to_json(self) -> str:
        """Return canonical compact JSON for durable receipts or comparison."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    serialize = to_dict


class RetentionPolicy(Protocol):
    """Protocol shared by pure checkpoint retention policies."""

    def decide(
        self,
        refs: Iterable[RetentionRef],
        pin_state: RetentionPinState | None = None,
        *,
        latest: RetentionRef | Iterable[RetentionRef] | None = None,
        selected: RetentionRef | Iterable[RetentionRef] | None = None,
    ) -> RetentionSnapshot:
        """Produce a retention snapshot without performing side effects."""


@dataclass(frozen=True, slots=True)
class RetainAll:
    """Retain every supplied complete checkpoint reference."""

    def decide(
        self,
        refs: Iterable[RetentionRef],
        pin_state: RetentionPinState | None = None,
        *,
        latest: RetentionRef | Iterable[RetentionRef] | None = None,
        selected: RetentionRef | Iterable[RetentionRef] | None = None,
    ) -> RetentionSnapshot:
        """Return retain decisions in deterministic reference order.

        Pin state and selection are accepted for a common policy interface but
        do not alter an all-retain policy.  If either state is malformed, the
        result remains all-retain and is marked fail-closed.
        """

        normalized_refs, refs_status = _normalize_refs(refs)
        status = _combine_status(
            refs_status, _state_status(normalized_refs, pin_state, latest, selected)
        )
        reason = "policy_retain_all" if status == "ready" else f"fail_closed_{status}"
        return _snapshot(
            "retain_all",
            status,
            tuple(
                RetentionDecision(ref, "retain", reason, status != "ready")
                for ref in normalized_refs
            ),
        )

    def plan(
        self,
        refs: Iterable[RetentionRef],
        pin_state: RetentionPinState | None = None,
        *,
        latest: RetentionRef | Iterable[RetentionRef] | None = None,
        selected: RetentionRef | Iterable[RetentionRef] | None = None,
    ) -> RetentionSnapshot:
        """Alias for :meth:`decide`."""

        return self.decide(refs, pin_state, latest=latest, selected=selected)


@dataclass(frozen=True, slots=True)
class KeepLast:
    """Retain the newest ``limit`` references independently per stream root.

    Parameters
    ----------
    limit : int or None
        Number of references retained in each ``checkpoint_dir.parent``
        stream root.  ``None`` is an explicit unbounded/retain-all setting.

    Notes
    -----
    This is the pure planning replacement for the existing
    ``save_checkpoint(..., keep_last=...)`` and ``prune_old_checkpoints``
    behavior.  The legacy parameter remains unchanged and continues to own
    its execution path until a later integration layer migrates it explicitly.
    """

    limit: int | None

    def __post_init__(self) -> None:
        # Do not use isinstance here: bool is an int subclass and must not
        # silently become KeepLast(1).
        if self.limit is not None and type(self.limit) is not int:
            raise TypeError("limit must be an int or None")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive when set")

    @classmethod
    def from_legacy(cls, keep_last: int | None) -> "KeepLast":
        """Create a planning policy from the legacy ``keep_last`` value.

        This adapter makes migration explicit.  It does not alter the
        existing ``save_checkpoint`` or ``prune_old_checkpoints`` execution
        path, and callers must still hand the resulting policy a complete
        immutable-ref and pin-state snapshot.
        """

        return cls(keep_last)

    from_legacy_keep_last = from_legacy

    def decide(
        self,
        refs: Iterable[RetentionRef],
        pin_state: RetentionPinState | None = None,
        *,
        latest: RetentionRef | Iterable[RetentionRef] | None = None,
        selected: RetentionRef | Iterable[RetentionRef] | None = None,
    ) -> RetentionSnapshot:
        """Plan per-root retention, fail-closed on incomplete safety state."""

        normalized_refs, refs_status = _normalize_refs(refs)
        pins, pin_status = _normalize_pin_state(pin_state)
        latest_refs, latest_status = _normalize_optional_refs(latest)
        state = _combine_status(refs_status, pin_status, latest_status)
        if state != "ready":
            return _retain_all_snapshot(
                "keep_last", normalized_refs, state, {"limit": self.limit}
            )

        if latest is None:
            latest_refs = _latest_per_root(normalized_refs)
        elif not _has_complete_latest_coverage(normalized_refs, latest_refs):
            return _retain_all_snapshot(
                "keep_last",
                normalized_refs,
                "incomplete_latest_coverage",
                {"limit": self.limit},
            )
        protected_keys = {_ref_key(ref) for ref in latest_refs}
        protected_keys.update(_ref_key(pin.ref) for pin in pins)
        if not protected_keys.issubset({_ref_key(ref) for ref in normalized_refs}):
            return _retain_all_snapshot(
                "keep_last", normalized_refs, "missing_pin_ref", {"limit": self.limit}
            )

        if self.limit is None:
            decisions = tuple(
                RetentionDecision(ref, "retain", "keep_last_unbounded", False)
                for ref in normalized_refs
            )
            return _snapshot("keep_last", "ready", decisions, {"limit": self.limit})

        groups = _group_by_root(normalized_refs)
        window_keys: set[str] = set()
        for group in groups.values():
            window_keys.update(_ref_key(ref) for ref in group[-self.limit :])

        decisions = tuple(
            _decision_for(ref, protected_keys, window_keys, "keep_last")
            for ref in normalized_refs
        )
        return _snapshot("keep_last", "ready", decisions, {"limit": self.limit})

    def plan(
        self,
        refs: Iterable[RetentionRef],
        pin_state: RetentionPinState | None = None,
        *,
        latest: RetentionRef | Iterable[RetentionRef] | None = None,
        selected: RetentionRef | Iterable[RetentionRef] | None = None,
    ) -> RetentionSnapshot:
        """Alias for :meth:`decide`."""

        return self.decide(refs, pin_state, latest=latest, selected=selected)


@dataclass(frozen=True, slots=True)
class HoldUntilSelection:
    """Hold all references until an explicit selection snapshot is supplied.

    Before selection, ``selected`` is absent and the policy retains every
    reference.  Once a complete selection is supplied, selected references,
    every latest recovery reference, and every ref carrying a live pin are
    retained; other references become exact deletion targets in the snapshot.
    """

    def decide(
        self,
        refs: Iterable[RetentionRef],
        pin_state: RetentionPinState | None = None,
        *,
        latest: RetentionRef | Iterable[RetentionRef] | None = None,
        selected: RetentionRef | Iterable[RetentionRef] | None = None,
    ) -> RetentionSnapshot:
        """Plan post-selection cleanup, or retain all while selection is absent."""

        normalized_refs, refs_status = _normalize_refs(refs)
        pins, pin_status = _normalize_pin_state(pin_state)
        latest_refs, latest_status = _normalize_optional_refs(latest)
        if selected is None:
            return _retain_all_snapshot(
                "hold_until_selection", normalized_refs, "selection_pending"
            )
        selected_refs, selected_status = _normalize_optional_refs(selected)
        state = _combine_status(refs_status, pin_status, latest_status, selected_status)
        if state == "ready" and not selected_refs:
            state = "incomplete_selection"
        if state != "ready":
            return _retain_all_snapshot("hold_until_selection", normalized_refs, state)

        if latest is None:
            latest_refs = _latest_per_root(normalized_refs)
        elif not _has_complete_latest_coverage(normalized_refs, latest_refs):
            return _retain_all_snapshot(
                "hold_until_selection", normalized_refs, "incomplete_latest_coverage"
            )
        selected_keys = {_ref_key(ref) for ref in selected_refs}
        protected_keys = {_ref_key(ref) for ref in latest_refs}
        protected_keys.update(_ref_key(pin.ref) for pin in pins)
        known_keys = {_ref_key(ref) for ref in normalized_refs}
        if not (selected_keys | protected_keys).issubset(known_keys):
            return _retain_all_snapshot(
                "hold_until_selection", normalized_refs, "missing_selection_ref"
            )

        decisions = tuple(
            _decision_for(
                ref,
                protected_keys,
                selected_keys,
                "hold_until_selection",
            )
            for ref in normalized_refs
        )
        return _snapshot("hold_until_selection", "ready", decisions)

    def plan(
        self,
        refs: Iterable[RetentionRef],
        pin_state: RetentionPinState | None = None,
        *,
        latest: RetentionRef | Iterable[RetentionRef] | None = None,
        selected: RetentionRef | Iterable[RetentionRef] | None = None,
    ) -> RetentionSnapshot:
        """Alias for :meth:`decide`."""

        return self.decide(refs, pin_state, latest=latest, selected=selected)


def _snapshot(
    policy: str,
    status: str,
    decisions: tuple[RetentionDecision, ...],
    config: Mapping[str, Any] | None = None,
) -> RetentionSnapshot:
    return RetentionSnapshot(
        policy=policy,
        status=status,
        decisions=decisions,
        config=() if config is None else tuple(sorted(config.items())),
    )


def _retain_all_snapshot(
    policy: str,
    refs: tuple[RetentionRef, ...],
    status: str,
    config: Mapping[str, Any] | None = None,
) -> RetentionSnapshot:
    reason = f"fail_closed_{status}"
    return _snapshot(
        policy,
        status,
        tuple(RetentionDecision(ref, "retain", reason, True) for ref in refs),
        config,
    )


def _decision_for(
    ref: RetentionRef,
    protected_keys: set[str],
    retain_keys: set[str],
    policy: str,
) -> RetentionDecision:
    key = _ref_key(ref)
    if key in protected_keys:
        return RetentionDecision(ref, "retain", "protected", True)
    if key in retain_keys:
        reason = "keep_last" if policy == "keep_last" else "selected"
        return RetentionDecision(ref, "retain", reason, False)
    reason = "outside_keep_last" if policy == "keep_last" else "not_selected"
    return RetentionDecision(ref, "delete", reason, False)


def _normalize_refs(
    refs: Iterable[RetentionRef],
) -> tuple[tuple[RetentionRef, ...], str]:
    """Deduplicate exact refs and reject conflicting identities for one path."""

    unique: dict[str, RetentionRef] = {}
    identity_by_path: dict[Path, str] = {}
    status = "ready"
    for ref in refs:
        ref_key = _ref_key(ref)
        checkpoint_dir = Path(ref.checkpoint_dir)
        known_identity = identity_by_path.setdefault(checkpoint_dir, ref_key)
        if known_identity != ref_key:
            status = "ambiguous_ref_state"
        unique.setdefault(ref_key, ref)
    normalized = tuple(
        sorted(
            unique.values(),
            key=_ref_order,
        )
    )
    return normalized, status


def _normalize_optional_refs(
    refs: RetentionRef | Iterable[RetentionRef] | None,
) -> tuple[tuple[RetentionRef, ...], str]:
    if refs is None:
        return (), "ready"
    if isinstance(refs, RetentionRef):
        return _normalize_refs((refs,))
    try:
        return _normalize_refs(refs)
    except (AttributeError, TypeError, ValueError):
        return (), "ambiguous_state"


def _normalize_pin_state(
    pin_state: RetentionPinState | None,
) -> tuple[tuple[PinRecord, ...], str]:
    if pin_state is None:
        return (), "missing_pin_state"
    try:
        entries = tuple(pin_state)
    except TypeError:
        return (), "ambiguous_pin_state"

    by_token: dict[str, str] = {}
    by_content_id: dict[str, str] = {}
    unique: dict[str, PinRecord] = {}
    try:
        for pin in entries:
            serialized = _canonical_json(pin.to_dict())
            token = pin.token
            content_id = pin.ref.content_id
            ref_key = _ref_key(pin.ref)
            previous = by_token.get(token)
            if previous is not None and previous != serialized:
                return (), "ambiguous_pin_state"
            known_ref = by_content_id.get(content_id)
            if known_ref is not None and known_ref != ref_key:
                return (), "ambiguous_pin_state"
            by_token[token] = serialized
            by_content_id[content_id] = ref_key
            unique.setdefault(token, pin)
    except (AttributeError, KeyError, TypeError, ValueError):
        return (), "ambiguous_pin_state"
    return tuple(unique.values()), "ready"


def _state_status(
    refs: tuple[RetentionRef, ...],
    pin_state: RetentionPinState | None,
    latest: RetentionRef | Iterable[RetentionRef] | None,
    selected: RetentionRef | Iterable[RetentionRef] | None,
) -> str:
    _, pin_status = _normalize_pin_state(pin_state)
    _, latest_status = _normalize_optional_refs(latest)
    selected_refs, selected_status = _normalize_optional_refs(selected)
    state = _combine_status(pin_status, latest_status, selected_status)
    if state != "ready" or selected is None:
        return state
    known_keys = {_ref_key(ref) for ref in refs}
    if not {_ref_key(ref) for ref in selected_refs}.issubset(known_keys):
        return "missing_selection_ref"
    return "ready"


def _combine_status(*statuses: str) -> str:
    for status in statuses:
        if status != "ready":
            return status
    return "ready"


def _latest_per_root(refs: tuple[RetentionRef, ...]) -> tuple[RetentionRef, ...]:
    return tuple(group[-1] for group in _group_by_root(refs).values() if group)


def _has_complete_latest_coverage(
    refs: tuple[RetentionRef, ...], latest_refs: tuple[RetentionRef, ...]
) -> bool:
    known_keys = {_ref_key(ref) for ref in refs}
    latest_keys = {_ref_key(ref) for ref in latest_refs}
    expected_roots = set(_group_by_root(refs))
    observed_roots = [str(Path(ref.checkpoint_dir).parent) for ref in latest_refs]
    return (
        latest_keys.issubset(known_keys)
        and len(observed_roots) == len(expected_roots)
        and set(observed_roots) == expected_roots
    )


def _group_by_root(refs: tuple[RetentionRef, ...]) -> dict[str, tuple[RetentionRef, ...]]:
    grouped: dict[str, list[RetentionRef]] = {}
    for ref in refs:
        grouped.setdefault(str(Path(ref.checkpoint_dir).parent), []).append(ref)
    return {root: tuple(group) for root, group in grouped.items()}


def _ref_order(ref: RetentionRef) -> tuple[int, int, str]:
    completed_updates = ref.completed_updates
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError("completed_updates must be a non-negative int")
    next_iteration = ref.next_iteration
    if type(next_iteration) is not int or next_iteration < 0:
        raise ValueError("next_iteration must be a non-negative int")
    return completed_updates, next_iteration, _ref_key(ref)


def _ref_key(ref: RetentionRef) -> str:
    return _canonical_json(_ref_mapping(ref))


def _ref_mapping(ref: RetentionRef) -> dict[str, Any]:
    data = ref.to_dict()
    if type(data) is dict:
        return data
    try:
        return dict(data)
    except (TypeError, ValueError) as exc:
        raise TypeError("checkpoint reference must serialize to a mapping") from exc


def _canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "RETENTION_SNAPSHOT_SCHEMA",
    "HoldUntilSelection",
    "KeepLast",
    "RetainAll",
    "RetentionDecision",
    "RetentionPolicy",
    "RetentionPinState",
    "RetentionRef",
    "RetentionSnapshot",
]
