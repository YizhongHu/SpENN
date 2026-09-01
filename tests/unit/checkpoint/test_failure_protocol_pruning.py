"""Failure-protocol tests for checkpoint pruning integration boundaries."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from dataclasses import replace

import pytest

import tpen.checkpoint.pruning as pruning_module
from tpen.checkpoint import (
    CheckpointPruneError,
    CheckpointCatalog,
    CheckpointRef,
    KeepLast,
    PinStore,
    RetentionDecision,
    RetentionSnapshot,
    execute_retention_snapshot,
    sweep_published_checkpoints,
)


def _manifest(step: int, marker: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        "completed_updates": step - 1,
        "created_at_unix": float(step),
        "files": {"model": "model.pt"},
        "hashes": {"marker": marker},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": "pruning-test", "git_sha": "deadbeef"},
    }


def _checkpoint(root: Path, step: int) -> CheckpointRef:
    checkpoint = root / f"step_{step:06d}"
    checkpoint.mkdir(parents=True)
    marker = f"model-{step}"
    (checkpoint / "model.pt").write_bytes(marker.encode())
    (checkpoint / "manifest.json").write_text(
        json.dumps(_manifest(step, marker), sort_keys=True) + "\n", encoding="utf-8"
    )
    (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return CheckpointRef.from_directory(checkpoint)


@pytest.mark.xfail(
    strict=True,
    reason="F2-S3 13432597: sweep must consume only the published catalog snapshot",
)
def test_sweep_consumes_only_published_catalog_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    unpublished = _checkpoint(root, 1)
    published = _checkpoint(root, 2)
    CheckpointCatalog(root / "publications.jsonl").publish(published)
    (root / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint_dir": published.checkpoint_dir.name,
                "step": published.next_iteration,
                "created_at_unix": 2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sweep_published_checkpoints(root, keep_last=1)

    assert unpublished.checkpoint_dir.is_dir()
    assert published.checkpoint_dir.is_dir()


def _delete_snapshot(ref: CheckpointRef) -> RetentionSnapshot:
    return RetentionSnapshot(
        policy="failure-protocol-test",
        status="ready",
        decisions=(RetentionDecision(ref, "delete", "forced-test-delete"),),
    )


@pytest.mark.xfail(
    strict=True,
    reason="F2-S3 13432597: canonical pins must not be bypassed by an omitted or unrelated store",
)
def test_canonical_pin_cannot_be_bypassed_by_missing_or_unrelated_store(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    ref = _checkpoint(root, 1)
    canonical = PinStore(root / "pins.jsonl")
    canonical.pin(ref, "evaluation-token", "evaluator", "active evaluation")
    unrelated = PinStore(tmp_path / "other-pins.jsonl")

    for supplied_store in (None, unrelated):
        with pytest.raises(CheckpointPruneError, match="live pin"):
            execute_retention_snapshot(
                _delete_snapshot(ref),
                checkpoint_root=root,
                pin_store=supplied_store,
            )
        assert ref.checkpoint_dir.is_dir()


@pytest.mark.xfail(
    strict=True,
    reason="F2-S3 13432597: contradictory snapshots must fail before receipt mutation",
)
def test_contradictory_typed_snapshot_fails_before_receipt_or_rename(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    ref = _checkpoint(root, 1)
    contradictory = replace(ref, model_sha256="0" * 64)

    with pytest.raises(CheckpointPruneError, match="unchanged complete"):
        execute_retention_snapshot(
            _delete_snapshot(contradictory), checkpoint_root=root
        )

    assert ref.checkpoint_dir.is_dir()
    assert not (root / "prune_receipts.jsonl").exists()


def test_aliased_typed_snapshot_fails_before_receipt_or_rename(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    ref = _checkpoint(root, 1)
    aliased = RetentionSnapshot(
        policy="failure-protocol-test",
        status="ready",
        decisions=(
            RetentionDecision(ref, "delete", "first"),
            RetentionDecision(ref, "delete", "alias"),
        ),
    )

    with pytest.raises(CheckpointPruneError, match="duplicate frozen delete path"):
        execute_retention_snapshot(aliased, checkpoint_root=root)

    assert ref.checkpoint_dir.is_dir()
    assert not (root / "prune_receipts.jsonl").exists()


@pytest.mark.parametrize("path_kind", ["symlink", "regular-file"])
@pytest.mark.xfail(
    strict=True,
    reason="F2-S3 13432597: nonregular targets must fail before receipt mutation",
)
def test_nonregular_or_symlink_target_fails_closed(
    tmp_path: Path, path_kind: str
) -> None:
    root = tmp_path / "checkpoints"
    ref = _checkpoint(root, 1)
    original = ref.checkpoint_dir
    if path_kind == "symlink":
        external = tmp_path / "external"
        _checkpoint(tmp_path, 2).checkpoint_dir.rename(external)
        original.rename(root / "original")
        (root / "step_000001").symlink_to(external, target_is_directory=True)
    else:
        original.rename(root / "original")
        (root / "step_000001").write_text("not a directory\n", encoding="utf-8")
    forged = replace(ref, checkpoint_dir=root / "step_000001")

    with pytest.raises(CheckpointPruneError, match="non-symlink child|missing or not a directory"):
        execute_retention_snapshot(_delete_snapshot(forged), checkpoint_root=root)

    assert (root / "step_000001").exists() or (root / "step_000001").is_symlink()
    assert not (root / "prune_receipts.jsonl").exists()


def _single_prune_setup(root: Path) -> tuple[CheckpointRef, RetentionSnapshot]:
    ref = _checkpoint(root, 1)
    return ref, _delete_snapshot(ref)


def test_planned_receipt_failure_recovers_once_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    ref, snapshot = _single_prune_setup(root)
    original_append = pruning_module._ReceiptLog.append
    failed = False

    def fail_planned(log: object, record: dict[str, object]) -> None:
        nonlocal failed
        if not failed and record["event"] == "planned":
            failed = True
            raise CheckpointPruneError("injected planned receipt failure")
        original_append(log, record)  # type: ignore[arg-type]

    monkeypatch.setattr(pruning_module._ReceiptLog, "append", fail_planned)
    with pytest.raises(CheckpointPruneError, match="planned receipt failure"):
        execute_retention_snapshot(snapshot, checkpoint_root=root)
    assert ref.checkpoint_dir.is_dir()

    report = execute_retention_snapshot(snapshot, checkpoint_root=root)
    assert report.deleted == (ref.checkpoint_dir,)
    assert not ref.checkpoint_dir.exists()


def test_quarantine_rename_failure_recovers_once_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    ref, snapshot = _single_prune_setup(root)
    original_rename = Path.rename
    failed = False

    def fail_once(path: Path, target: Path) -> Path:
        nonlocal failed
        if not failed and path == ref.checkpoint_dir:
            failed = True
            raise OSError("injected quarantine rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_once)
    with pytest.raises(CheckpointPruneError, match="could not quarantine"):
        execute_retention_snapshot(snapshot, checkpoint_root=root)
    assert ref.checkpoint_dir.is_dir()

    report = execute_retention_snapshot(snapshot, checkpoint_root=root)
    assert report.deleted == (ref.checkpoint_dir,)


@pytest.mark.xfail(
    strict=True,
    reason="F2-S3 13432597: a receipt failure after quarantine must recover on retry",
)
def test_quarantine_receipt_failure_recovers_once_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    ref, snapshot = _single_prune_setup(root)
    original_append = pruning_module._ReceiptLog.append
    failed = False

    def fail_quarantine(log: object, record: dict[str, object]) -> None:
        nonlocal failed
        if not failed and record["event"] == "quarantined":
            failed = True
            raise CheckpointPruneError("injected quarantine receipt failure")
        original_append(log, record)  # type: ignore[arg-type]

    monkeypatch.setattr(pruning_module._ReceiptLog, "append", fail_quarantine)
    with pytest.raises(CheckpointPruneError, match="quarantine receipt failure"):
        execute_retention_snapshot(snapshot, checkpoint_root=root)
    assert not ref.checkpoint_dir.exists()

    report = execute_retention_snapshot(snapshot, checkpoint_root=root)
    assert report.deleted == (ref.checkpoint_dir,)


@pytest.mark.xfail(
    strict=True,
    reason="F2-S3 13432597: removal failure must converge exactly once on retry",
)
def test_removal_failure_recovers_once_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    ref, snapshot = _single_prune_setup(root)
    original_rmtree = pruning_module.shutil.rmtree
    failed = False

    def fail_removal(path: Path) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected removal failure")
        original_rmtree(path)

    monkeypatch.setattr(pruning_module.shutil, "rmtree", fail_removal)
    with pytest.raises(CheckpointPruneError, match="removal failed"):
        execute_retention_snapshot(snapshot, checkpoint_root=root)
    assert not ref.checkpoint_dir.exists()

    report = execute_retention_snapshot(snapshot, checkpoint_root=root)
    assert report.deleted == (ref.checkpoint_dir,)


@pytest.mark.xfail(
    strict=True,
    reason="F2-S3 13432597: final receipt fsync failure must converge exactly once on retry",
)
def test_final_receipt_fsync_failure_recovers_once_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoints"
    ref, snapshot = _single_prune_setup(root)
    original_fsync = pruning_module.os.fsync
    calls = 0

    def fail_final_receipt(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected final receipt fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(pruning_module.os, "fsync", fail_final_receipt)
    with pytest.raises(CheckpointPruneError, match="durably append"):
        execute_retention_snapshot(snapshot, checkpoint_root=root)
    assert not ref.checkpoint_dir.exists()

    report = execute_retention_snapshot(snapshot, checkpoint_root=root)
    assert report.deleted == (ref.checkpoint_dir,)


def _pin_in_process(
    path: str, checkpoint_path: str, start: object, barrier: object, done: object
) -> None:
    start.wait()  # type: ignore[attr-defined]
    barrier.wait()  # type: ignore[attr-defined]
    ref = CheckpointRef.from_directory(checkpoint_path)
    PinStore(path).pin(ref, "evaluation-token", "evaluator", "active evaluation")
    done.set()  # type: ignore[attr-defined]


def test_posix_evaluator_pin_and_prune_are_linearizable(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("requires POSIX advisory locks")
    root = tmp_path / "checkpoints"
    ref = _checkpoint(root, 1)
    pin_path = root / "pins.jsonl"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    done = context.Event()
    barrier = context.Barrier(2)
    process = context.Process(
        target=_pin_in_process,
        args=(str(pin_path), str(ref.checkpoint_dir), start, barrier, done),
    )
    process.start()
    start.set()
    barrier.wait(timeout=30.0)
    assert done.wait(timeout=30.0)
    process.join(timeout=30.0)
    assert not process.is_alive()
    assert process.exitcode == 0

    with pytest.raises(CheckpointPruneError, match="live pin"):
        execute_retention_snapshot(
            _delete_snapshot(ref), checkpoint_root=root, pin_store=PinStore(pin_path)
        )
    assert ref.checkpoint_dir.is_dir()
