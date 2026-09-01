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
        store.pin(ref, "token", "owner", "reason")
    else:
        calls = 0
        with pytest.raises(PinLedgerError, match="durably append"):
            store.release("token")
        store.release("token")

    assert calls >= 2
    if operation == "pin":
        assert [record.token for record in store.active_pins()] == ["token"]
    else:
        assert store.active_pins() == ()


def test_first_ledger_creation_fsyncs_file_and_parent_directory(
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

    assert len(calls) == 2
    assert calls[0] != calls[1]


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

def test_crlf_boundary_keeps_following_records_visible_and_ledger_intact(
    tmp_path: Path,
) -> None:
    first = _ref(tmp_path / "first", step=7)
    second = _ref(tmp_path / "second", step=8)
    path = tmp_path / "pins.jsonl"
    store = PinStore(path)
    store.pin(first, "first", "owner", "reason")
    store.pin(second, "second", "owner", "reason")
    first_row, separator, following_rows = path.read_bytes().partition(b"\n")
    assert separator == b"\n"
    path.write_bytes(first_row + b"\r\n" + following_rows)
    before = path.read_bytes()

    assert [record.token for record in store.records()] == ["first", "second"]

    assert path.read_bytes() == before


def test_raw_cr_inside_record_fails_closed_without_hiding_or_truncating_later_rows(
    tmp_path: Path,
) -> None:
    first = _ref(tmp_path / "first", step=7)
    second = _ref(tmp_path / "second", step=8)
    third = _ref(tmp_path / "third", step=9)
    fourth = _ref(tmp_path / "fourth", step=10)
    path = tmp_path / "pins.jsonl"
    store = PinStore(path)
    store.pin(first, "first", "owner", "reason")
    store.pin(second, "second", "owner", "reason")
    store.pin(third, "third", "owner", "reason")
    rows = path.read_bytes().split(b"\n")
    rows[1] = rows[1].replace(b'"reason":"reason"', b'"reason":"raw\rcr"')
    path.write_bytes(b"\n".join(rows))
    before = path.read_bytes()

    with pytest.raises(PinLedgerError, match="invalid checkpoint pin ledger record"):
        store.records()
    with pytest.raises(PinLedgerError, match="invalid checkpoint pin ledger record"):
        store.pin(fourth, "fourth", "owner", "reason")

    assert path.read_bytes() == before
