"""Failure-protocol tests for checkpoint retention state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen.checkpoint import HoldUntilSelection, KeepLast, RetentionSnapshot
from tpen.checkpoint.reference import CheckpointRef


def _ref(root: Path, step: int, marker: str) -> CheckpointRef:
    checkpoint_dir = root / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(marker.encode())
    manifest = {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        "completed_updates": step - 1,
        "created_at_unix": 123.0,
        "files": {"model": "model.pt"},
        "hashes": {"marker": marker},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": marker, "git_sha": "deadbeef"},
    }
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return CheckpointRef.from_directory(checkpoint_dir)


def _actions(snapshot: RetentionSnapshot) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(decision.ref.checkpoint_dir), decision.action)
        for decision in snapshot.decisions
    )


def test_empty_selection_is_explicit_and_retains_latest_recovery_refs(tmp_path: Path) -> None:
    refs = (
        _ref(tmp_path / "first", 1, "first-1"),
        _ref(tmp_path / "first", 2, "first-2"),
    )

    snapshot = HoldUntilSelection().decide(
        refs, pin_state=(), latest=refs[-1], selected=()
    )

    assert snapshot.status == "ready"
    assert snapshot.deletion_targets == (refs[0],)
    assert snapshot.retained_refs == (refs[1],)


@pytest.mark.xfail(
    strict=True,
    reason="F2-S2 8671d0c5-89ff-4b20-9aa0-0778f7207126: incomplete latest coverage must fail closed",
)
def test_incomplete_explicit_latest_coverage_retains_everything(tmp_path: Path) -> None:
    first = _ref(tmp_path / "first", 1, "first-1")
    first_latest = _ref(tmp_path / "first", 2, "first-2")
    second = _ref(tmp_path / "second", 1, "second-1")
    second_latest = _ref(tmp_path / "second", 2, "second-2")

    snapshot = KeepLast(1).decide(
        (first, first_latest, second, second_latest),
        pin_state=(),
        latest=(first_latest,),
    )

    assert snapshot.status == "incomplete_latest_coverage"
    assert snapshot.deletion_targets == ()
    assert all(decision.action == "retain" for decision in snapshot.decisions)


def test_duplicate_immutable_ref_alias_is_deduplicated_deterministically(tmp_path: Path) -> None:
    ref = _ref(tmp_path / "stream", 1, "one")

    first = KeepLast(1).decide((ref, ref), pin_state=())
    second = KeepLast(1).decide((ref,), pin_state=())

    assert first.to_json() == second.to_json()
    assert first.decisions == (second.decisions[0],)

