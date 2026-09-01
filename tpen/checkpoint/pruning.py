"""Fail-closed execution of frozen checkpoint retention snapshots.

Retention policies only produce immutable plans.  This module is the one
destructive boundary: it accepts the exact references in a
:class:`~tpen.checkpoint.retention.RetentionSnapshot`, records every state
transition durably, and removes only a quarantined copy of a target.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

try:  # pragma: no cover - supported execution is POSIX.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX import smoke.
    _fcntl = None

from .artifact import (
    CheckpointPruneError,
    LATEST_JSON,
    _latest_pointer_target,
    list_complete_checkpoints,
    require_complete_checkpoint_dir,
)
from .pins import PinStore, checkpoint_pins_path
from .reference import CheckpointRef
from .retention import KeepLast, RetentionDecision, RetentionSnapshot
from .hashing import file_sha256


PRUNE_RECEIPT_SCHEMA = "tpen.checkpoint-prune/v1"
PRUNE_RECEIPTS_FILENAME = "prune_receipts.jsonl"


@dataclass(frozen=True, slots=True)
class PruneReport:
    """Summary of one frozen-snapshot execution."""

    policy_digest: str
    deleted: tuple[Path, ...] = ()
    skipped: tuple[Path, ...] = ()
    retained: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class _FrozenTarget:
    ref: CheckpointRef
    path: Path
    path_text: str
    reason: str
    quarantine: Path


def prune_receipts_path(checkpoint_root: str | Path) -> Path:
    """Return the run-local append-only prune receipt path."""

    return Path(checkpoint_root) / PRUNE_RECEIPTS_FILENAME


def sweep_published_checkpoints(
    checkpoint_root: str | Path,
    *,
    keep_last: int | None,
) -> None:
    """Materialize one post-publication snapshot and execute it.

    This is the compatibility/integration adapter for the historical
    ``save_checkpoint(..., keep_last=...)`` parameter.  Planning occurs once,
    after publication and ``latest.json`` commit; the destructive executor
    below receives only the resulting frozen snapshot.
    """

    if keep_last is None:
        return
    if type(keep_last) is not int:
        raise TypeError("keep_last must be an int or None")
    if keep_last < 1:
        raise ValueError(f"keep_last must be positive when set, got {keep_last}")

    root = Path(checkpoint_root)
    checkpoints = list_complete_checkpoints(root)
    latest_path = root / LATEST_JSON
    if not checkpoints:
        # Absent latest on an empty root is a valid no-op.  A present pointer
        # is still validated so corruption never becomes an excuse to act.
        if latest_path.exists() or latest_path.is_symlink():
            _latest_pointer_target(root)
        return

    pointer_target = _latest_pointer_target(root)
    refs = tuple(CheckpointRef.from_directory(path) for path in checkpoints)
    latest_ref = None if pointer_target is None else CheckpointRef.from_directory(pointer_target)
    pin_path = checkpoint_pins_path(root)
    pin_store = PinStore(pin_path) if pin_path.is_file() else None
    pin_state = () if pin_store is None else pin_store.active_pins()
    snapshot = KeepLast(keep_last).decide(refs, pin_state=pin_state, latest=latest_ref)
    execute_retention_snapshot(snapshot, checkpoint_root=root, pin_store=pin_store)


def execute_retention_snapshot(
    snapshot: RetentionSnapshot,
    *,
    checkpoint_root: str | Path,
    pin_store: PinStore | None = None,
    target_paths: Iterable[str | Path] | None = None,
) -> PruneReport:
    """Execute exact delete decisions from one frozen retention snapshot.

    The executor never discovers candidates.  Its only delete candidates are
    the ``delete`` decisions in ``snapshot``.  ``target_paths`` is an
    optional caller-side narrowing for audits; every supplied path must match
    a delete decision's literal serialized path exactly.  It cannot add a
    target that is absent from the snapshot.

    Parameters
    ----------
    snapshot : RetentionSnapshot
        Already-materialized retention decisions.  A non-ready snapshot is a
        fail-closed refusal, not an instruction to infer a safer plan.
    checkpoint_root : str or pathlib.Path
        One configured checkpoint stream root.  Every target must be a direct
        non-symlink child of this root.
    pin_store : PinStore, optional
        When supplied, each target is checked against the current durable pin
        ledger immediately before quarantine.  This is an extra safety guard;
        it never changes the frozen target set.
    target_paths : iterable of str or pathlib.Path, optional
        Exact literal paths to execute from the snapshot's delete decisions.

    Returns
    -------
    PruneReport
        Policy digest plus exact original paths deleted or skipped as already
        completed.

    Raises
    ------
    CheckpointPruneError
        If any safety condition is not provable.  Preflight validates every
        selected target before the first target is moved.
    """

    if type(snapshot) is not RetentionSnapshot:
        raise TypeError("snapshot must be a RetentionSnapshot")
    if snapshot.status != "ready":
        raise CheckpointPruneError(
            "retention snapshot is not executable; refusing to prune: "
            f"status={snapshot.status!r}"
        )

    try:
        snapshot_bytes = snapshot.to_json().encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointPruneError(
            "retention snapshot is not canonically serializable; refusing to prune"
        ) from exc
    policy_digest = hashlib.sha256(snapshot_bytes).hexdigest()
    root = _require_checkpoint_root(checkpoint_root)
    deletion_decisions = _frozen_delete_decisions(snapshot)
    selected = _select_targets(deletion_decisions, target_paths)
    retained = tuple(
        Path(decision.ref.checkpoint_dir)
        for decision in snapshot.decisions
        if decision.action == "retain"
    )

    # A corrupt present pointer is a refusal even if a malformed snapshot
    # would otherwise happen to contain no deletion targets.  Genuine absence
    # remains the deliberate legacy policy and is represented by None.
    pointer_target = _latest_pointer_target(root)
    receipt_path = prune_receipts_path(root)
    skipped: list[Path] = []
    ready: list[_FrozenTarget] = []

    with _ReceiptLog(receipt_path) as receipts:
        for decision in selected:
            frozen = _freeze_target(decision, root, policy_digest)
            if receipts.is_completed(frozen, policy_digest):
                if frozen.path.exists() or frozen.path.is_symlink() or _exists(frozen.quarantine):
                    raise CheckpointPruneError(
                        "completed prune receipt conflicts with a live target: "
                        f"{frozen.path}"
                    )
                skipped.append(frozen.path)
                continue
            if _exists(frozen.quarantine):
                raise CheckpointPruneError(
                    "preserved prune quarantine already exists; refusing to remove or reuse it: "
                    f"{frozen.quarantine}"
                )
            _preflight_target(frozen, root, pointer_target, pin_store)
            ready.append(frozen)

        deleted: list[Path] = []
        for frozen in ready:
            # Re-read both mutable safety ledgers immediately before the
            # atomic move.  A state change is a refusal, never a new decision.
            current_pointer = _latest_pointer_target(root)
            if current_pointer is not None and _same_path(current_pointer, frozen.path):
                raise CheckpointPruneError(
                    f"latest pointer protects frozen prune target: {frozen.path}"
                )
            _check_pin(frozen, pin_store)
            receipts.append(_receipt(frozen, policy_digest, "planned"))
            try:
                frozen.path.rename(frozen.quarantine)
            except OSError as exc:
                receipts.append(_receipt(frozen, policy_digest, "failed", error=str(exc)))
                raise CheckpointPruneError(
                    f"could not quarantine frozen prune target: {frozen.path}"
                ) from exc
            receipts.append(_receipt(frozen, policy_digest, "quarantined"))
            try:
                # The original path has an atomic-ish boundary now.  If this
                # recursive removal fails, the quarantine remains visible and
                # can be recovered by an explicitly authorized operator.
                _validate_quarantine_bytes(frozen)
                shutil.rmtree(frozen.quarantine)
            except Exception as exc:
                receipts.append(_receipt(frozen, policy_digest, "failed", error=str(exc)))
                raise CheckpointPruneError(
                    "frozen prune target removal failed; quarantine preserved for recovery: "
                    f"{frozen.quarantine}"
                ) from exc
            receipts.append(_receipt(frozen, policy_digest, "deleted"))
            deleted.append(frozen.path)

    return PruneReport(
        policy_digest=policy_digest,
        deleted=tuple(deleted),
        skipped=tuple(skipped),
        retained=retained,
    )


# Short aliases keep callers that speak in terms of a sweep readable while
# retaining one implementation and one frozen-snapshot contract.
sweep_checkpoint_root = execute_retention_snapshot
prune_snapshot = execute_retention_snapshot


def _require_checkpoint_root(checkpoint_root: str | Path) -> Path:
    root = Path(checkpoint_root)
    if root.is_symlink() or not root.is_dir():
        raise CheckpointPruneError(
            f"checkpoint root must be an existing non-symlink directory: {root}"
        )
    return root


def _frozen_delete_decisions(snapshot: RetentionSnapshot) -> tuple[RetentionDecision, ...]:
    decisions: list[RetentionDecision] = []
    for decision in snapshot.decisions:
        if type(decision) is not RetentionDecision:
            raise CheckpointPruneError(
                "retention snapshot contains a non-canonical decision; refusing to prune"
            )
        if decision.action not in {"retain", "delete"}:
            raise CheckpointPruneError(
                f"retention snapshot contains unknown action {decision.action!r}"
            )
        if type(decision.ref) is not CheckpointRef:
            raise CheckpointPruneError(
                "retention snapshot contains a non-canonical checkpoint ref; refusing to prune"
            )
        serialized = decision.ref.to_dict()
        if serialized.get("checkpoint_dir") != str(decision.ref.checkpoint_dir):
            raise CheckpointPruneError(
                "checkpoint ref path is not its literal serialized path; refusing to prune"
            )
        if decision.action == "delete":
            decisions.append(decision)
    return tuple(decisions)


def _select_targets(
    decisions: tuple[RetentionDecision, ...],
    target_paths: Iterable[str | Path] | None,
) -> tuple[RetentionDecision, ...]:
    by_literal: dict[str, RetentionDecision] = {}
    for decision in decisions:
        literal = str(decision.ref.checkpoint_dir)
        if literal in by_literal:
            raise CheckpointPruneError(
                f"duplicate frozen delete path is ambiguous: {literal}"
            )
        by_literal[literal] = decision
    if target_paths is None:
        return decisions
    if type(target_paths) is str or type(target_paths) is Path:
        requested = (target_paths,)
    else:
        try:
            requested = tuple(target_paths)
        except TypeError as exc:
            raise TypeError("target_paths must be an iterable of paths") from exc
    selected_literals: set[str] = set()
    for requested_path in requested:
        literal = str(requested_path)
        if literal not in by_literal:
            raise CheckpointPruneError(
                f"target is not literally present in frozen snapshot: {literal}"
            )
        if literal in selected_literals:
            raise CheckpointPruneError(f"duplicate requested prune target: {literal}")
        selected_literals.add(literal)
    return tuple(decision for decision in decisions if str(decision.ref.checkpoint_dir) in selected_literals)


def _freeze_target(
    decision: RetentionDecision,
    root: Path,
    policy_digest: str,
) -> _FrozenTarget:
    ref = decision.ref
    path = Path(ref.checkpoint_dir)
    path_text = str(ref.checkpoint_dir)
    resolved = path.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    if path.is_symlink() or resolved.parent != root_resolved:
        raise CheckpointPruneError(
            "frozen prune target must be one literal non-symlink child of its root: "
            f"{path_text}"
        )
    quarantine = root / f".prune-quarantine-{path.name}-{policy_digest}"
    if quarantine == path or quarantine.parent != root:
        raise CheckpointPruneError(f"invalid deterministic prune quarantine: {quarantine}")
    return _FrozenTarget(
        ref=ref,
        path=path,
        path_text=path_text,
        reason=decision.reason,
        quarantine=quarantine,
    )


def _preflight_target(
    frozen: _FrozenTarget,
    root: Path,
    pointer_target: Path | None,
    pin_store: PinStore | None,
) -> None:
    if pointer_target is not None and _same_path(pointer_target, frozen.path):
        raise CheckpointPruneError(
            f"latest pointer protects frozen prune target: {frozen.path}"
        )
    if not frozen.path.exists() or not frozen.path.is_dir() or frozen.path.is_symlink():
        raise CheckpointPruneError(
            f"frozen prune target is missing or not a directory: {frozen.path}"
        )
    try:
        require_complete_checkpoint_dir(frozen.path)
        frozen.ref.validate()
    except (OSError, TypeError, ValueError) as exc:
        raise CheckpointPruneError(
            f"frozen prune target is not an unchanged complete checkpoint: {frozen.path}"
        ) from exc
    _check_pin(frozen, pin_store)
    if _exists(frozen.quarantine):
        raise CheckpointPruneError(
            f"deterministic prune quarantine already exists: {frozen.quarantine}"
        )
    if frozen.path.parent.resolve(strict=False) != root.resolve(strict=False):
        raise CheckpointPruneError(
            f"frozen prune target escaped checkpoint root: {frozen.path}"
        )


def _check_pin(frozen: _FrozenTarget, pin_store: PinStore | None) -> None:
    if pin_store is None:
        return
    if not pin_store.path.is_file():
        raise CheckpointPruneError(
            f"pin ledger is missing; cannot prove frozen target is unpinned: {pin_store.path}"
        )
    try:
        pins = pin_store.pins_for(frozen.ref)
    except Exception as exc:
        raise CheckpointPruneError(
            f"could not prove frozen target is unpinned: {frozen.path}"
        ) from exc
    if pins:
        raise CheckpointPruneError(
            f"frozen prune target carries a live pin: {frozen.path}"
        )


def _validate_quarantine_bytes(frozen: _FrozenTarget) -> None:
    """Check the moved tree without treating its hidden name as a checkpoint."""

    try:
        require_complete_checkpoint_dir(frozen.quarantine)
        manifest_digest = file_sha256(frozen.quarantine / "manifest.json")
        model_digest = file_sha256(frozen.quarantine / "model.pt")
    except (OSError, TypeError, ValueError) as exc:
        raise CheckpointPruneError(
            f"quarantined checkpoint no longer matches its frozen identity: "
            f"{frozen.quarantine}"
        ) from exc
    if (
        manifest_digest != frozen.ref.manifest_sha256
        or model_digest != frozen.ref.model_sha256
    ):
        raise CheckpointPruneError(
            f"quarantined checkpoint bytes changed after snapshot: {frozen.quarantine}"
        )


def _same_path(first: Path, second: Path) -> bool:
    return first.resolve(strict=False) == second.resolve(strict=False)


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _receipt(
    frozen: _FrozenTarget,
    policy_digest: str,
    event: str,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": PRUNE_RECEIPT_SCHEMA,
        "event": event,
        "ref": frozen.ref.to_dict(),
        "path": frozen.path_text,
        "quarantine_path": str(frozen.quarantine),
        "reason": frozen.reason,
        "policy_digest": policy_digest,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if error is not None:
        record["error"] = error
    return record


class _ReceiptLog:
    """One exclusively locked append-only receipt stream."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: BinaryIO | None = None
        self._records: list[dict[str, Any]] = []

    def __enter__(self) -> "_ReceiptLog":
        if _fcntl is None:
            raise CheckpointPruneError(
                "advisory file locking is unavailable; refusing to prune"
            )
        handle: BinaryIO | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
            handle.seek(0)
            self._records = _parse_receipts(handle.read(), self.path)
            self._handle = handle
            return self
        except CheckpointPruneError:
            if handle is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
                handle.close()
            raise
        except OSError as exc:
            if handle is not None:
                handle.close()
            raise CheckpointPruneError(
                f"cannot acquire prune receipt lock; refusing to prune: {self.path}"
            ) from exc

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
        except OSError as unlock_error:
            if exc is None:
                raise CheckpointPruneError(
                    f"cannot release prune receipt lock safely: {self.path}"
                ) from unlock_error
        finally:
            handle.close()

    def is_completed(self, frozen: _FrozenTarget, policy_digest: str) -> bool:
        key = _receipt_key(_receipt(frozen, policy_digest, "deleted"))
        return any(
            record.get("event") == "deleted" and _receipt_key(record) == key
            for record in self._records
        )

    def append(self, record: dict[str, Any]) -> None:
        handle = self._handle
        if handle is None:
            raise CheckpointPruneError("prune receipt log is not open")
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        ) + b"\n"
        handle.seek(0, os.SEEK_END)
        try:
            written = os.write(handle.fileno(), payload)
            if written != len(payload):
                raise OSError(f"short append {written}/{len(payload)} bytes")
            os.fsync(handle.fileno())
        except (OSError, TypeError, ValueError) as exc:
            raise CheckpointPruneError(
                f"cannot durably append prune receipt: {self.path}"
            ) from exc
        self._records.append(record)


def _receipt_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("policy_digest", "")),
        str(record.get("path", "")),
        json.dumps(record.get("ref"), sort_keys=True, separators=(",", ":"), allow_nan=False),
    )


def _parse_receipts(data: bytes, path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    required = {
        "schema",
        "event",
        "ref",
        "path",
        "quarantine_path",
        "reason",
        "policy_digest",
        "timestamp",
    }
    for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n"):
            raise CheckpointPruneError(
                f"unterminated prune receipt at {path}:{line_number}; refusing to prune"
            )
        try:
            record = json.loads(line[:-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointPruneError(
                f"invalid prune receipt at {path}:{line_number}; refusing to prune"
            ) from exc
        if type(record) is not dict or not required.issubset(record):
            raise CheckpointPruneError(
                f"invalid prune receipt fields at {path}:{line_number}; refusing to prune"
            )
        if record["schema"] != PRUNE_RECEIPT_SCHEMA or record["event"] not in {
            "planned",
            "quarantined",
            "deleted",
            "failed",
        }:
            raise CheckpointPruneError(
                f"invalid prune receipt state at {path}:{line_number}; refusing to prune"
            )
        if type(record["ref"]) is not dict:
            raise CheckpointPruneError(
                f"invalid prune receipt ref at {path}:{line_number}; refusing to prune"
            )
        try:
            CheckpointRef.from_mapping(record["ref"])
            json.dumps(record, sort_keys=True, allow_nan=False)
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointPruneError(
                f"invalid prune receipt content at {path}:{line_number}; refusing to prune"
            ) from exc
        records.append(record)
    return records


__all__ = [
    "PRUNE_RECEIPT_SCHEMA",
    "PRUNE_RECEIPTS_FILENAME",
    "PruneReport",
    "execute_retention_snapshot",
    "prune_receipts_path",
    "prune_snapshot",
    "sweep_published_checkpoints",
    "sweep_checkpoint_root",
]
