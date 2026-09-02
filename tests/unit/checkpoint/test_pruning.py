"""Torch-free tests for fail-closed checkpoint pruning."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

import tpen.checkpoint.pruning as pruning_module
from tpen.checkpoint import (
    CheckpointPruneError,
    CheckpointRef,
    KeepLast,
    PinStore,
    RetentionDecision,
    RetentionSnapshot,
    execute_retention_snapshot,
    prune_receipts_path,
)
from tpen.checkpoint.artifact import prune_old_checkpoints, write_latest


def _manifest(step: int, marker: str) -> dict[str, object]:
    return {
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


def _ref(root: Path, step: int, marker: str | None = None) -> CheckpointRef:
    marker = marker or f"model-{step}"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = root / f"step_{step:06d}"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.pt").write_bytes(marker.encode("utf-8"))
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(_manifest(step, marker), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return CheckpointRef.from_directory(checkpoint_dir)


def _snapshot(
    refs: tuple[CheckpointRef, ...],
    *,
    latest: CheckpointRef | None = None,
    pin_state=(),
) -> RetentionSnapshot:
    return KeepLast(1).decide(refs, pin_state=pin_state, latest=latest)


def _empty_pin_store(root: Path) -> PinStore:
    path = root / "pins.jsonl"
    path.touch()
    return PinStore(path)


def test_absent_latest_on_empty_root_is_a_noop(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()

    prune_old_checkpoints(root, keep_last=1)

    assert not list(root.iterdir())


def test_corrupt_latest_retains_all_complete_directories_and_raises(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    first = _ref(root, 1)
    second = _ref(root, 2)
    (root / "latest.json").write_text("{not-json\n", encoding="utf-8")
    before = {
        ref.checkpoint_dir.name: {
            path.name: path.read_bytes() for path in ref.checkpoint_dir.iterdir()
        }
        for ref in (first, second)
    }

    with pytest.raises(CheckpointPruneError, match="invalid latest"):
        prune_old_checkpoints(root, keep_last=1)

    assert first.checkpoint_dir.is_dir()
    assert second.checkpoint_dir.is_dir()
    assert {
        ref.checkpoint_dir.name: {
            path.name: path.read_bytes() for path in ref.checkpoint_dir.iterdir()
        }
        for ref in (first, second)
    } == before


def test_snapshot_status_that_retains_on_missing_pin_state_raises_without_delete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    refs = tuple(_ref(root, step) for step in (1, 2))
    snapshot = KeepLast(1).decide(refs)

    with pytest.raises(CheckpointPruneError, match="not executable"):
        execute_retention_snapshot(snapshot, checkpoint_root=root)

    assert all(ref.checkpoint_dir.is_dir() for ref in refs)


def test_target_not_literally_in_snapshot_is_refused_and_retained(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    first = _ref(root, 1)
    second = _ref(root, 2)
    snapshot = _snapshot((first,))

    with pytest.raises(CheckpointPruneError, match="not literally present"):
        execute_retention_snapshot(
            snapshot,
            checkpoint_root=root,
            target_paths=(str(second.checkpoint_dir),),
        )

    assert first.checkpoint_dir.is_dir()
    assert second.checkpoint_dir.is_dir()


def test_pinned_target_is_refused_and_retained(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    ref = _ref(root, 1)
    pin_store = PinStore(root / "pins.jsonl")
    pin_store.pin(ref, "evaluation-token", "evaluator", "active evaluation")
    snapshot = RetentionSnapshot(
        policy="test",
        status="ready",
        decisions=(RetentionDecision(ref, "delete", "forced-test-delete"),),
    )

    with pytest.raises(CheckpointPruneError, match="live pin"):
        execute_retention_snapshot(snapshot, checkpoint_root=root, pin_store=pin_store)

    assert ref.checkpoint_dir.is_dir()


def test_latest_target_is_refused_and_retained(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    ref = _ref(root, 1)
    _empty_pin_store(root)
    write_latest(root, ref.checkpoint_dir, step=1, created_at_unix=0.0)
    snapshot = RetentionSnapshot(
        policy="test",
        status="ready",
        decisions=(RetentionDecision(ref, "delete", "forced-test-delete"),),
    )

    with pytest.raises(CheckpointPruneError, match="latest pointer protects"):
        execute_retention_snapshot(snapshot, checkpoint_root=root)

    assert ref.checkpoint_dir.is_dir()


def test_release_then_sweep_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    first = _ref(root, 1)
    second = _ref(root, 2)
    write_latest(root, second.checkpoint_dir, step=2, created_at_unix=0.0)
    pin_store = PinStore(root / "pins.jsonl")
    pin_store.pin(first, "evaluation-token", "evaluator", "active evaluation")

    held = _snapshot((first, second), latest=second, pin_state=pin_store.active_pins())
    assert held.deletion_targets == ()
    held_report = execute_retention_snapshot(
        held,
        checkpoint_root=root,
        pin_store=pin_store,
    )
    assert held_report.deleted == ()
    assert first.checkpoint_dir.is_dir()

    pin_store.release("evaluation-token")
    released = _snapshot((first, second), latest=second, pin_state=pin_store.active_pins())
    first_report = execute_retention_snapshot(
        released,
        checkpoint_root=root,
        pin_store=pin_store,
    )
    assert first_report.deleted == (first.checkpoint_dir,)
    assert not first.checkpoint_dir.exists()
    receipt_before = prune_receipts_path(root).read_bytes()

    second_report = execute_retention_snapshot(
        released,
        checkpoint_root=root,
        pin_store=pin_store,
    )
    assert second_report.deleted == ()
    assert second_report.skipped == (first.checkpoint_dir,)
    assert prune_receipts_path(root).read_bytes() == receipt_before
    assert second.checkpoint_dir.is_dir()


def test_successful_prune_receipts_bind_exact_ref_path_reason_digest_and_time(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    first = _ref(root, 1)
    second = _ref(root, 2)
    _empty_pin_store(root)
    write_latest(root, second.checkpoint_dir, step=2, created_at_unix=0.0)
    snapshot = _snapshot((first, second), latest=second)

    report = execute_retention_snapshot(snapshot, checkpoint_root=root)
    records = [
        json.loads(line)
        for line in prune_receipts_path(root).read_text(encoding="utf-8").splitlines()
    ]
    deleted = [record for record in records if record["event"] == "deleted"]

    assert report.deleted == (first.checkpoint_dir,)
    assert len(deleted) == 1
    receipt = deleted[0]
    assert receipt["ref"] == first.to_dict()
    assert receipt["path"] == str(first.checkpoint_dir)
    assert receipt["reason"] == "outside_keep_last"
    assert receipt["policy_digest"] == report.policy_digest
    datetime.fromisoformat(receipt["timestamp"])
    assert [record["event"] for record in records] == [
        "planned",
        "quarantined",
        "deleted",
    ]


def test_mid_deletion_failure_preserves_quarantine_and_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checkpoints"
    first = _ref(root, 1)
    second = _ref(root, 2)
    _empty_pin_store(root)
    write_latest(root, second.checkpoint_dir, step=2, created_at_unix=0.0)
    snapshot = _snapshot((first, second), latest=second)

    def partial_remove(path: Path) -> None:
        (path / "model.pt").unlink()
        raise OSError("injected mid-deletion failure")

    monkeypatch.setattr(pruning_module.shutil, "rmtree", partial_remove)
    with pytest.raises(CheckpointPruneError, match="quarantine preserved"):
        execute_retention_snapshot(snapshot, checkpoint_root=root)

    quarantines = list(root.glob(".prune-quarantine-*"))
    assert not first.checkpoint_dir.exists()
    assert len(quarantines) == 1
    assert (quarantines[0] / "manifest.json").is_file()
    assert (quarantines[0] / "COMPLETE").is_file()
    records = [
        json.loads(line)
        for line in prune_receipts_path(root).read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["event"] == "failed"
    assert records[-1]["quarantine_path"] == str(quarantines[0])

    with pytest.raises(CheckpointPruneError, match="quarantine already exists"):
        execute_retention_snapshot(snapshot, checkpoint_root=root)
    assert quarantines[0].is_dir()


def test_quarantine_is_not_a_complete_checkpoint_candidate(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    ref = _ref(root, 1)
    quarantine = root / ".prune-quarantine-preserved"
    ref.checkpoint_dir.rename(quarantine)

    assert prune_old_checkpoints(root, keep_last=1) is None
    assert quarantine.is_dir()


def test_incomplete_frozen_target_is_refused_and_retained(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    ref = _ref(root, 1)
    _empty_pin_store(root)
    incomplete = root / "step_000002"
    incomplete.mkdir()
    forged = replace(ref, checkpoint_dir=incomplete)
    snapshot = RetentionSnapshot(
        policy="test",
        status="ready",
        decisions=(RetentionDecision(forged, "delete", "forced-test-delete"),),
    )

    with pytest.raises(CheckpointPruneError, match="unchanged complete"):
        execute_retention_snapshot(snapshot, checkpoint_root=root)

    assert incomplete.is_dir()
