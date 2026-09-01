"""Failure-protocol tests for the durable checkpoint pin ledger."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import tpen.checkpoint.pins as pins_module
from tpen.checkpoint import PinLedgerError, PinStore
from tpen.checkpoint.reference import CheckpointRef


def _ref(root: Path, step: int = 7) -> CheckpointRef:
    checkpoint_dir = root / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(f"model-{step}".encode())
    manifest = {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        "completed_updates": step - 1,
        "created_at_unix": 123.0,
        "files": {"model": "model.pt"},
        "hashes": {},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": f"run-{step}", "git_sha": "deadbeef"},
    }
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return CheckpointRef.from_directory(checkpoint_dir)


@pytest.mark.parametrize("operation", ["pin", "release"])
@pytest.mark.xfail(
    strict=True,
    reason="F2-S1 9cfd4867-a96f-43de-87ff-6f3bcc06863f: retry must re-establish durability",
)
def test_write_then_barrier_failure_is_not_reported_as_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    ref = _ref(tmp_path / "checkpoint")
    path = tmp_path / "pins.jsonl"
    store = PinStore(path)
    if operation == "release":
        store.pin(ref, "token", "owner", "reason")
    calls = 0
    real_fsync = os.fsync

    def fail_once(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected durability failure")
        real_fsync(fd)

    monkeypatch.setattr(pins_module.os, "fsync", fail_once)
    if operation == "pin":
        with pytest.raises(PinLedgerError, match="durably append"):
            store.pin(ref, "token", "owner", "reason")
        with pytest.raises(PinLedgerError, match="durably append"):
            store.pin(ref, "token", "owner", "reason")
    else:
        calls = 0
        with pytest.raises(PinLedgerError, match="durably append"):
            store.release("token")
        with pytest.raises(PinLedgerError, match="durably append"):
            store.release("token")

    assert calls >= 2


def test_first_ledger_creation_uses_the_promised_file_durability_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = _ref(tmp_path / "checkpoint")
    calls: list[int] = []
    real_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(pins_module.os, "fsync", record_fsync)
    PinStore(tmp_path / "pins.jsonl").pin(ref, "token", "owner", "reason")

    # The production contract promises the ledger-file barrier. It does not
    # promise a parent-directory barrier, so no directory fsync is asserted.
    assert len(calls) == 1


@pytest.mark.xfail(
    strict=True,
    reason="F2-S1 9cfd4867-a96f-43de-87ff-6f3bcc06863f: ledger path follows symlinks",
)
def test_symlink_ledger_fails_closed_without_mutating_external_target(tmp_path: Path) -> None:
    ref = _ref(tmp_path / "checkpoint")
    target = tmp_path / "external.jsonl"
    original = PinStore(target)
    original.pin(ref, "token", "owner", "reason")
    before = target.read_bytes()
    link = tmp_path / "pins.jsonl"
    link.symlink_to(target)

    with pytest.raises(PinLedgerError):
        PinStore(link).release("token")

    assert target.read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [b"{\"schema\":", b"{\"schema\": \"wrong\"}\r", b"not-json\v", b"[]\f"],
)
def test_invalid_interior_records_are_distinct_from_a_torn_tail(
    tmp_path: Path, payload: bytes
) -> None:
    ref = _ref(tmp_path / "checkpoint")
    path = tmp_path / "pins.jsonl"
    store = PinStore(path)
    store.pin(ref, "token", "owner", "reason")
    path.write_bytes(path.read_bytes() + payload + b"\n")
    before = path.read_bytes()

    with pytest.raises(PinLedgerError, match="invalid checkpoint pin ledger record"):
        store.records()

    assert path.read_bytes() == before
