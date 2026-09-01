"""Torch-free tests for durable checkpoint pin and release records."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tpen.checkpoint import (
    PIN_RECORD_SCHEMA,
    PinLedgerError,
    PinRecord,
    PinStore,
    ReleaseRecord,
    checkpoint_pins_path,
)
from tpen.checkpoint.reference import CheckpointRef


def _manifest(step: int, *, model_name: str = "model.pt") -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        "completed_updates": step - 1,
        "created_at_unix": 123.0,
        "files": {"model": model_name},
        "hashes": {},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": "run", "git_sha": "deadbeef"},
    }


def _write_checkpoint(root: Path, step: int = 7, *, model_name: str = "model.pt") -> Path:
    checkpoint_dir = root / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / model_name).write_bytes(b"immutable-model")
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(_manifest(step, model_name=model_name), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return checkpoint_dir


def _ref(root: Path, step: int = 7) -> CheckpointRef:
    return CheckpointRef.from_directory(_write_checkpoint(root, step=step))


def test_pin_record_is_typed_and_keeps_lifecycle_out_of_checkpoint_ref(tmp_path: Path) -> None:
    ref = _ref(tmp_path / "checkpoint")
    ledger = PinStore(tmp_path / "pins.jsonl")

    record = ledger.pin(ref, "eval-7", "evaluator/run-1", "active-evaluation")

    assert type(record) is PinRecord
    assert record.ref == ref
    assert record.owner == "evaluator/run-1"
    assert record.reason == "active-evaluation"
    assert record.ref.to_dict() == ref.to_dict()
    serialized = json.loads((tmp_path / "pins.jsonl").read_text(encoding="utf-8"))
    assert serialized == record.to_dict()
    assert serialized["schema"] == PIN_RECORD_SCHEMA
    assert set(serialized["ref"]) == set(ref.to_dict())
    assert not {"pinned", "released", "owner", "reason"}.intersection(serialized["ref"])


def test_pin_replay_is_parsed_equal_and_does_not_append(tmp_path: Path) -> None:
    ref = _ref(tmp_path / "checkpoint")
    ledger = PinStore(tmp_path / "pins.jsonl")

    first = ledger.pin(ref, "token", "owner", "reason")
    original = ledger.path.read_bytes()

    second = ledger.pin(ref, "token", "owner", "reason")

    assert second == first
    assert ledger.path.read_bytes() == original
    assert ledger.active_pins() == (first,)


def test_multiple_tokens_can_hold_one_ref_with_distinct_ownership(tmp_path: Path) -> None:
    ref = _ref(tmp_path / "checkpoint")
    ledger = PinStore(tmp_path / "pins.jsonl")

    first = ledger.pin(ref, "selection", "selector", "hold-until-selection")
    second = ledger.pin(ref, "evaluation", "evaluator", "active-evaluation")

    assert ledger.active_pins() == (first, second)
    assert ledger.pins_for(ref) == (first, second)
    assert ledger.is_pinned(ref)
    assert ledger.pinned_refs() == (ref,)


def test_pin_token_conflicts_are_loud_and_leave_ledger_unchanged(tmp_path: Path) -> None:
    first_ref = _ref(tmp_path / "first")
    second_ref = _ref(tmp_path / "second")
    assert first_ref.content_id == second_ref.content_id
    ledger = PinStore(tmp_path / "pins.jsonl")
    ledger.pin(first_ref, "token", "owner", "reason")
    before = ledger.path.read_bytes()

    with pytest.raises(PinLedgerError, match="conflicting pin records"):
        ledger.pin(first_ref, "token", "other-owner", "reason")
    with pytest.raises(PinLedgerError, match="ambiguous checkpoint ref"):
        ledger.pin(second_ref, "other-token", "owner", "reason")

    assert ledger.path.read_bytes() == before


def test_release_is_durable_and_releasing_twice_is_idempotent(tmp_path: Path) -> None:
    ref = _ref(tmp_path / "checkpoint")
    ledger = PinStore(tmp_path / "pins.jsonl")
    pin = ledger.pin(ref, "token", "owner", "reason")

    first_release = ledger.release("token")
    before_repeat = ledger.path.read_bytes()
    second_release = ledger.release("token")

    assert type(first_release) is ReleaseRecord
    assert second_release == first_release
    assert ledger.path.read_bytes() == before_repeat
    assert ledger.active_pins() == ()
    assert ledger.records() == (pin, first_release)


def test_unknown_release_fails_closed_without_creating_or_changing_ledger(tmp_path: Path) -> None:
    path = tmp_path / "pins.jsonl"
    ledger = PinStore(path)

    with pytest.raises(PinLedgerError, match="unknown pin token"):
        ledger.release("never-seen")

    assert not path.exists()


def test_unknown_checkpoint_ref_fails_closed_before_first_append(tmp_path: Path) -> None:
    missing_ref = CheckpointRef(
        checkpoint_dir=tmp_path / "missing" / "step_000007",
        schema_version=2,
        kind="tpen.checkpoint",
        next_iteration=7,
        completed_updates=6,
        manifest_sha256="0" * 64,
        model_sha256="1" * 64,
        provenance={"run_id": "run"},
    )
    path = tmp_path / "pins.jsonl"

    with pytest.raises((FileNotFoundError, ValueError)):
        PinStore(path).pin(missing_ref, "token", "owner", "reason")

    assert not path.exists()


def test_reusing_a_released_token_is_a_conflict(tmp_path: Path) -> None:
    ref = _ref(tmp_path / "checkpoint")
    ledger = PinStore(tmp_path / "pins.jsonl")
    ledger.pin(ref, "token", "owner", "reason")
    ledger.release("token")
    before = ledger.path.read_bytes()

    with pytest.raises(PinLedgerError, match="already released"):
        ledger.pin(ref, "token", "owner", "reason")

    assert ledger.path.read_bytes() == before


def test_reader_ignores_but_does_not_repair_a_torn_final_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ref = _ref(tmp_path / "checkpoint")
    ledger = PinStore(tmp_path / "pins.jsonl")
    pin = ledger.pin(ref, "token", "owner", "reason")
    tail = b'{"schema":"tpen.checkpoint-pin/v1","operation":"pin"'
    ledger.path.write_bytes(ledger.path.read_bytes() + tail)
    before = ledger.path.read_bytes()
    caplog.set_level(logging.WARNING, logger="tpen.checkpoint.pins")

    assert PinStore(ledger.path).active_pins() == (pin,)

    assert ledger.path.read_bytes() == before
    assert f"byte_length={len(tail)}" in caplog.text
    assert "ignored torn EOF tail" in caplog.text


def test_writer_repairs_only_torn_eof_tail_before_appending(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first_ref = _ref(tmp_path / "first", step=7)
    second_ref = _ref(tmp_path / "second", step=8)
    ledger = PinStore(tmp_path / "pins.jsonl")
    ledger.pin(first_ref, "first", "owner", "reason")
    tail = b'{"schema":"tpen.checkpoint-pin/v1","operation":"pin"'
    ledger.path.write_bytes(ledger.path.read_bytes() + tail)
    caplog.set_level(logging.WARNING, logger="tpen.checkpoint.pins")

    ledger.pin(second_ref, "second", "owner", "reason")

    rows = ledger.path.read_bytes().splitlines()
    assert len(rows) == 2
    assert all(tail not in row for row in rows)
    assert [record.token for record in ledger.active_pins()] == ["first", "second"]
    assert f"byte_length={len(tail)}" in caplog.text
    assert "discarded torn EOF tail" in caplog.text


def test_interior_corruption_is_an_error_and_is_never_repaired(tmp_path: Path) -> None:
    first_ref = _ref(tmp_path / "first", step=7)
    second_ref = _ref(tmp_path / "second", step=8)
    ledger = PinStore(tmp_path / "pins.jsonl")
    ledger.pin(first_ref, "first", "owner", "reason")
    first_row = ledger.path.read_bytes()
    ledger.pin(second_ref, "second", "owner", "reason")
    second_row = ledger.path.read_bytes().splitlines(keepends=True)[1]
    corrupted = first_row + b"not-json\n" + second_row
    ledger.path.write_bytes(corrupted)

    with pytest.raises(PinLedgerError, match="invalid checkpoint pin ledger record"):
        ledger.active_pins()
    assert ledger.path.read_bytes() == corrupted


def test_semantically_invalid_unterminated_record_is_not_repaired(tmp_path: Path) -> None:
    path = tmp_path / "pins.jsonl"
    invalid = b'{"schema":"wrong"}'
    path.write_bytes(invalid)

    with pytest.raises(PinLedgerError, match="invalid checkpoint pin ledger record"):
        PinStore(path).records()
    assert path.read_bytes() == invalid


def test_pin_and_release_rows_round_trip_after_reopening_store(tmp_path: Path) -> None:
    ref = _ref(tmp_path / "checkpoint")
    path = checkpoint_pins_path(tmp_path)
    ledger = PinStore(path)
    pin = ledger.pin(ref, "token", "owner", "reason")
    release = ledger.release("token")

    reopened = PinStore(path)

    assert reopened.records() == (pin, release)
    assert reopened.active_pins() == ()
