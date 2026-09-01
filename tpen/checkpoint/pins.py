"""Durable checkpoint pin and release tokens.

Pins are lifecycle records kept separately from :class:`CheckpointRef`.
The ledger is intentionally independent of callbacks, training, evaluation,
retention planning, and deletion: a caller supplies an immutable checkpoint
reference and receives an explicit token that protects that reference until a
matching release is recorded.

The on-disk format is append-only JSONL.  Each mutation replays the ledger
under an exclusive advisory lock, compares parsed records for idempotence, and
appends one fsynced record.  Replay is therefore O(n) in the number of ledger
records, just as the publication catalog is O(n) in its existing publications.
The final unterminated record is treated as a torn append and can be repaired
only by a writer holding the exclusive lock.  Readers take a shared lock and
never repair durable state.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol, TypeAlias, runtime_checkable

try:  # pragma: no cover - the supported platforms are POSIX clusters and macOS.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - keeps read-only import portable.
    _fcntl = None


PIN_LEDGER_FILENAME = "pins.jsonl"
PIN_RECORD_SCHEMA = "tpen.checkpoint-pin/v1"
PinLedgerRecord: TypeAlias = "PinRecord | ReleaseRecord"

_LOGGER = logging.getLogger(__name__)


class PinLedgerError(ValueError):
    """Raised when a pin ledger cannot be used safely."""


@runtime_checkable
class CheckpointRefLike(Protocol):
    """The immutable reference surface required by the pin ledger."""

    content_id: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable serialized reference."""

    def validate(self) -> CheckpointRefLike:
        """Validate the reference against its current artifact bytes."""


@dataclass(frozen=True, slots=True)
class PinRecord:
    """One immutable durable hold on a checkpoint reference.

    Parameters
    ----------
    token : str
        Caller-chosen unique token.  Tokens are never reused after release.
    ref : CheckpointRefLike
        Immutable checkpoint identity.  Its lifecycle is not modified by this
        record; the serialized ref remains nested under ``ref`` unchanged.
    owner : str
        Stable owner identity for the hold.
    reason : str
        Stable human- or machine-readable purpose for the hold.  An active
        evaluation is represented by an appropriate reason, without this
        module depending on evaluation types.
    """

    token: str
    ref: CheckpointRefLike
    owner: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "token", _required_text(self.token, "token"))
        object.__setattr__(self, "owner", _required_text(self.owner, "owner"))
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-safe pin operation record."""

        return {
            "schema": PIN_RECORD_SCHEMA,
            "operation": "pin",
            "token": self.token,
            "ref": _serialize_ref(self.ref),
            "owner": self.owner,
            "reason": self.reason,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PinRecord:
        """Deserialize and validate one pin operation record."""

        _require_record_keys(
            data,
            {"schema", "operation", "token", "ref", "owner", "reason"},
            "pin",
        )
        if data["schema"] != PIN_RECORD_SCHEMA or data["operation"] != "pin":
            raise ValueError("unsupported checkpoint pin record schema or operation")
        return cls(
            token=data["token"],
            ref=_deserialize_ref(data["ref"]),
            owner=data["owner"],
            reason=data["reason"],
        )

    from_dict = from_mapping


@dataclass(frozen=True, slots=True)
class ReleaseRecord:
    """One immutable durable release operation for a pin token."""

    token: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "token", _required_text(self.token, "token"))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-safe release operation record."""

        return {
            "schema": PIN_RECORD_SCHEMA,
            "operation": "release",
            "token": self.token,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ReleaseRecord:
        """Deserialize and validate one release operation record."""

        _require_record_keys(data, {"schema", "operation", "token"}, "release")
        if data["schema"] != PIN_RECORD_SCHEMA or data["operation"] != "release":
            raise ValueError("unsupported checkpoint release record schema or operation")
        return cls(token=data["token"])

    from_dict = from_mapping


class PinStore:
    """Append-only durable store for checkpoint pin and release tokens.

    Parameters
    ----------
    path : pathlib.Path or str
        JSONL ledger path.  A conventional run-local path can be obtained with
        :func:`checkpoint_pins_path`.

    Notes
    -----
    Every mutating operation replays the complete ledger under an exclusive
    ``fcntl.flock`` lock, so its read cost is O(n) in the number of records.
    Lock acquisition failure is fatal: the store never mutates a ledger
    without the lock.  Readers use a shared lock for their snapshot and ignore
    (without repairing) a genuinely torn final append left by a crash.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def pin(
        self,
        ref: CheckpointRefLike,
        token: str,
        owner: str,
        reason: str,
    ) -> PinRecord:
        """Durably acquire ``token`` for ``ref``.

        Equal parsed records are idempotent and do not append.  A token cannot
        be rebound to another reference, owner, or reason.  A relocation with
        the same path-independent ``content_id`` is also rejected because the
        complete serialized refs disagree and would make the protected path
        ambiguous.
        """

        normalized_ref = _normalize_ref(ref)
        # Validate before opening the ledger so an unknown checkpoint cannot
        # create durable state while failing closed.
        _validate_live_ref(normalized_ref)
        requested = PinRecord(token=token, ref=normalized_ref, owner=owner, reason=reason)
        with self._write_ledger() as handle:
            state, tail_offset = _scan_ledger(handle.read(), self.path, report_torn=False)
            existing = state.pin_history.get(requested.token)
            if existing is not None:
                if requested.token in state.release_history:
                    raise PinLedgerError(
                        f"pin token {requested.token!r} was already released and cannot be reused"
                    )
                if existing.to_dict() == requested.to_dict():
                    return existing
                raise PinLedgerError(f"conflicting pin records for token {requested.token!r}")
            _check_ref_conflict(state, requested)
            _append_record(handle, requested.to_dict(), self.path, tail_offset)
            return requested

    acquire = pin
    hold = pin

    def release(self, token: str) -> ReleaseRecord:
        """Durably release a token.

        Releasing an already-released token is an idempotent no-op.  A token
        never seen in the ledger is ambiguous and fails closed without an
        append.
        """

        token = _required_text(token, "token")
        if not self.path.is_file():
            raise PinLedgerError(f"cannot release unknown pin token {token!r}")
        with self._write_ledger() as handle:
            state, tail_offset = _scan_ledger(handle.read(), self.path, report_torn=False)
            if token not in state.pin_history:
                if token in state.release_history:
                    return state.release_history[token]
                raise PinLedgerError(f"cannot release unknown pin token {token!r}")
            existing = state.release_history.get(token)
            if existing is not None:
                return existing
            requested = ReleaseRecord(token=token)
            _append_record(handle, requested.to_dict(), self.path, tail_offset)
            return requested

    def records(self) -> tuple[PinLedgerRecord, ...]:
        """Return all valid operation records in ledger order."""

        state = self._read_state()
        return tuple(state.events)

    read = records
    load = records

    def iter_records(self) -> Iterator[PinLedgerRecord]:
        """Yield valid operation records in ledger order."""

        yield from self.records()

    def active_pins(self) -> tuple[PinRecord, ...]:
        """Return currently active pin records in first-seen order."""

        state = self._read_state()
        return tuple(state.active.values())

    pins = active_pins
    active = active_pins

    def pins_for(self, ref: CheckpointRefLike) -> tuple[PinRecord, ...]:
        """Return active pins for one immutable ref, failing on ambiguity."""

        serialized = _serialize_lookup_ref(ref)
        state = self._read_state()
        content_id = serialized.get("content_id")
        known = state.refs_by_content_id.get(content_id)
        if known is not None and known != serialized:
            raise PinLedgerError(
                f"ambiguous checkpoint ref for content_id {content_id!r}"
            )
        return tuple(
            record
            for record in state.active.values()
            if record.ref.content_id == content_id
        )

    def is_pinned(self, ref: CheckpointRefLike) -> bool:
        """Return whether at least one active token protects ``ref``."""

        return bool(self.pins_for(ref))

    def pinned_refs(self) -> tuple[CheckpointRefLike, ...]:
        """Return distinct active refs in first-seen content-id order."""

        refs: dict[str, CheckpointRefLike] = {}
        for record in self.active_pins():
            refs.setdefault(record.ref.content_id, record.ref)
        return tuple(refs.values())

    def _read_state(self) -> _LedgerState:
        try:
            with self.path.open("rb") as handle:
                _lock(handle, "shared")
                try:
                    data = handle.read()
                finally:
                    _unlock(handle)
        except FileNotFoundError:
            return _LedgerState()
        except OSError as exc:
            raise PinLedgerError(f"cannot read pin ledger {self.path}: {exc}") from exc
        return _scan_ledger(data, self.path, report_torn=True)[0]

    @contextmanager
    def _write_ledger(self) -> Iterator[BinaryIO]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = self.path.open("a+b")
        except OSError as exc:
            raise PinLedgerError(f"cannot open pin ledger {self.path}: {exc}") from exc
        try:
            _lock(handle, "exclusive")
        except BaseException:
            handle.close()
            raise
        try:
            handle.seek(0)
            yield handle
        finally:
            try:
                _unlock(handle)
            finally:
                handle.close()


CheckpointPin = PinRecord
CheckpointRelease = ReleaseRecord
CheckpointPinStore = PinStore
PinLedger = PinStore


def checkpoint_pins_path(checkpoint_root: str | Path) -> Path:
    """Return the conventional durable pin ledger path for a checkpoint root."""

    return Path(checkpoint_root) / PIN_LEDGER_FILENAME


pin_store_path = checkpoint_pins_path


def _validate_live_ref(ref: CheckpointRefLike) -> CheckpointRefLike:
    normalized_ref = _normalize_ref(ref)
    normalized_ref.validate()
    return normalized_ref


def _normalize_ref(ref: CheckpointRefLike) -> CheckpointRefLike:
    checkpoint_ref_type, _, _ = _reference_helpers()
    if type(ref) is not checkpoint_ref_type:
        raise TypeError(f"expected CheckpointRef, got {type(ref).__name__}")
    _serialize_ref(ref)
    return ref


def _serialize_lookup_ref(ref: CheckpointRefLike) -> dict[str, Any]:
    checkpoint_ref_type, _, _ = _reference_helpers()
    if type(ref) is not checkpoint_ref_type:
        raise TypeError(f"expected CheckpointRef, got {type(ref).__name__}")
    return _serialize_ref(ref)


def _reference_helpers() -> tuple[type[Any], Any, Any]:
    """Load checkpoint identity helpers only when the ledger is used.

    Keeping this import lazy lets the module itself remain a stdlib-only
    policy object for the local import smoke; importing ``tpen.checkpoint`` as
    a package otherwise loads the optional training stack through its legacy
    package exports.
    """

    try:
        from .reference import CheckpointRef, deserialize_checkpoint_ref, serialize_checkpoint_ref
    except ImportError as exc:  # pragma: no cover - only exercised by an isolated copy.
        raise PinLedgerError("checkpoint reference helpers are unavailable") from exc
    return CheckpointRef, deserialize_checkpoint_ref, serialize_checkpoint_ref


def _serialize_ref(ref: CheckpointRefLike) -> dict[str, Any]:
    _, _, serialize_checkpoint_ref = _reference_helpers()
    return serialize_checkpoint_ref(ref)


def _deserialize_ref(data: Any) -> CheckpointRefLike:
    _, deserialize_checkpoint_ref, _ = _reference_helpers()
    return deserialize_checkpoint_ref(data)


def _required_text(value: Any, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must be non-empty")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{label} must not contain newlines")
    return value


def _require_record_keys(
    data: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if type(data) is not dict:
        raise TypeError(f"{label} record must be a JSON object")
    if set(data) != expected:
        raise ValueError(
            f"invalid {label} record fields: expected {sorted(expected)!r}, got {sorted(data)!r}"
        )


class _LedgerState:
    """Mutable replay accumulator, never exposed outside this module."""

    def __init__(self) -> None:
        self.events: list[PinLedgerRecord] = []
        self.pin_history: dict[str, PinRecord] = {}
        self.release_history: dict[str, ReleaseRecord] = {}
        self.active: dict[str, PinRecord] = {}
        self.refs_by_content_id: dict[str, dict[str, Any]] = {}

    def copy(self) -> _LedgerState:
        copied = _LedgerState()
        copied.events = list(self.events)
        copied.pin_history = dict(self.pin_history)
        copied.release_history = dict(self.release_history)
        copied.active = dict(self.active)
        copied.refs_by_content_id = dict(self.refs_by_content_id)
        return copied


def _scan_ledger(
    data: bytes, path: Path, *, report_torn: bool
) -> tuple[_LedgerState, int | None]:
    state = _LedgerState()
    offset = 0
    for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
        terminated = line.endswith(b"\n")
        payload = line[:-1] if terminated else line
        if not payload.strip():
            if not terminated:
                _report_torn(path, len(data) - offset, "ignored" if report_torn else None)
                return state, offset
            offset += len(line)
            continue
        try:
            decoded = payload.decode("utf-8")
            raw = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if not terminated:
                _report_torn(path, len(data) - offset, "ignored" if report_torn else None)
                return state, offset
            raise PinLedgerError(
                f"invalid checkpoint pin ledger record at {path}:{line_number}: {exc}"
            ) from exc
        try:
            record = _decode_record(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise PinLedgerError(
                f"invalid checkpoint pin ledger record at {path}:{line_number}: {exc}"
            ) from exc
        candidate = state.copy()
        _apply_record(candidate, record, path, line_number)
        if not terminated:
            # Semantic validation happened before this branch.  A valid but
            # unterminated append is the only EOF shape eligible for repair.
            _report_torn(path, len(data) - offset, "ignored" if report_torn else None)
            return state, offset
        state = candidate
        offset += len(line)
    return state, None


def _decode_record(raw: Any) -> PinLedgerRecord:
    if type(raw) is not dict:
        raise TypeError("record must be a JSON object")
    operation = raw.get("operation")
    if operation == "pin":
        return PinRecord.from_mapping(raw)
    if operation == "release":
        return ReleaseRecord.from_mapping(raw)
    raise ValueError(f"unknown pin ledger operation {operation!r}")


def _apply_record(
    state: _LedgerState, record: PinLedgerRecord, path: Path, line_number: int
) -> None:
    if type(record) is PinRecord:
        existing = state.pin_history.get(record.token)
        if existing is not None:
            if record.token in state.release_history:
                raise PinLedgerError(
                    f"pin ledger reuses released token {record.token!r} at {path}:{line_number}"
                )
            if existing.to_dict() != record.to_dict():
                raise PinLedgerError(
                    f"conflicting pin records for token {record.token!r} at {path}:{line_number}"
                )
            state.events.append(record)
            return
        _check_ref_conflict(state, record, path=path, line_number=line_number)
        state.pin_history[record.token] = record
        state.active[record.token] = record
        state.events.append(record)
        return

    existing_release = state.release_history.get(record.token)
    if existing_release is not None:
        if existing_release.to_dict() != record.to_dict():
            raise PinLedgerError(
                f"conflicting release records for token {record.token!r} at {path}:{line_number}"
            )
        state.events.append(record)
        return
    if record.token not in state.pin_history:
        raise PinLedgerError(
            f"release for unknown pin token {record.token!r} at {path}:{line_number}"
        )
    state.release_history[record.token] = record
    state.active.pop(record.token, None)
    state.events.append(record)


def _check_ref_conflict(
    state: _LedgerState,
    record: PinRecord,
    *,
    path: Path | None = None,
    line_number: int | None = None,
) -> None:
    serialized = record.to_dict()["ref"]
    content_id = record.ref.content_id
    known = state.refs_by_content_id.get(content_id)
    if known is not None and known != serialized:
        suffix = "" if path is None else f" at {path}:{line_number}"
        raise PinLedgerError(
            f"ambiguous checkpoint ref for content_id {content_id!r}{suffix}"
        )
    state.refs_by_content_id[content_id] = serialized


def _append_record(
    handle: BinaryIO, record: Mapping[str, Any], path: Path, tail_offset: int | None
) -> None:
    if tail_offset is not None:
        handle.seek(0, os.SEEK_END)
        tail_length = handle.tell() - tail_offset
        handle.seek(tail_offset)
        handle.truncate()
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError as exc:
            raise PinLedgerError(f"cannot fsync torn-tail repair for {path}: {exc}") from exc
        _report_torn(path, tail_length, "discarded")
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    ) + b"\n"
    handle.seek(0, os.SEEK_END)
    try:
        written = os.write(handle.fileno(), payload)
        if written != len(payload):
            raise PinLedgerError(
                f"short append to checkpoint pin ledger {path}: {written}/{len(payload)} bytes"
            )
        os.fsync(handle.fileno())
    except OSError as exc:
        raise PinLedgerError(f"cannot durably append checkpoint pin record to {path}: {exc}") from exc


def _report_torn(path: Path, byte_length: int, action: str | None) -> None:
    if action is None:
        return
    if action == "discarded":
        _LOGGER.warning(
            "discarded torn EOF tail from checkpoint pin ledger %s (byte_length=%d)",
            path,
            byte_length,
        )
    else:
        _LOGGER.warning(
            "ignored torn EOF tail in checkpoint pin ledger %s (byte_length=%d)",
            path,
            byte_length,
        )


def _lock(handle: BinaryIO, mode: str) -> None:
    if _fcntl is None:
        raise PinLedgerError("advisory file locking is unavailable; refusing to mutate pin ledger")
    operation = _fcntl.LOCK_EX if mode == "exclusive" else _fcntl.LOCK_SH
    try:
        _fcntl.flock(handle.fileno(), operation)
    except OSError as exc:
        raise PinLedgerError(
            f"cannot acquire {mode} pin-ledger lock; refusing to proceed: {exc}"
        ) from exc


def _unlock(handle: BinaryIO) -> None:
    if _fcntl is None:
        return
    try:
        _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
    except OSError as exc:
        raise PinLedgerError(f"cannot release pin-ledger lock safely: {exc}") from exc


__all__ = [
    "PIN_LEDGER_FILENAME",
    "PIN_RECORD_SCHEMA",
    "PinLedgerRecord",
    "CheckpointPin",
    "CheckpointPinStore",
    "CheckpointRefLike",
    "CheckpointRelease",
    "PinLedger",
    "PinLedgerError",
    "PinRecord",
    "PinStore",
    "ReleaseRecord",
    "checkpoint_pins_path",
    "pin_store_path",
]
