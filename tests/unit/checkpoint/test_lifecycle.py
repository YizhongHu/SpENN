"""Contracts for the checkpoint lifecycle foundation.

These tests do not assert deletion safety.  No destructive caller participates
in the protocol at this layer.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from tpen.checkpoint.lifecycle import (
    CHECKPOINT_LIFECYCLE_LOCK_FILENAME,
    CheckpointRoot,
    DeletionCapability,
    DistinctNodeFlockReceipt,
    LifecycleError,
    LifecycleLockMode,
    LifecycleProtocol,
    LockOrder,
    checkpoint_lifecycle_lock,
    lock_order,
    open_regular_file,
    require_deletion_capability,
)


JOIN_TIMEOUT_SECONDS = 30


def _root(path: Path) -> CheckpointRoot:
    path.mkdir()
    return CheckpointRoot(path)


def _capability(root: CheckpointRoot) -> DeletionCapability:
    return DeletionCapability(
        root=root,
        receipt=DistinctNodeFlockReceipt(
            receipt_id="cannon-distinct-node-receipt",
            mount_device=root.device,
            protocol=LifecycleProtocol.TWO_AUTHORITY_V1,
        ),
    )


def test_checkpoint_root_canonicalizes_one_identity(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()

    assert CheckpointRoot(root_path / ".") == CheckpointRoot(root_path)
    assert CheckpointRoot(root_path).lock_path == root_path / CHECKPOINT_LIFECYCLE_LOCK_FILENAME


def test_checkpoint_root_mismatch_fails_loudly(tmp_path: Path) -> None:
    first = _root(tmp_path / "first")
    second = _root(tmp_path / "second")

    with pytest.raises(LifecycleError) as exc_info:
        first.require_same(second)

    assert exc_info.value.code == "lifecycle_root_mismatch"
    assert exc_info.value.artifact == first.lock_path


def test_checkpoint_root_detects_path_replacement(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root = _root(root_path)
    root_path.rename(tmp_path / "detached")
    root_path.mkdir()

    with pytest.raises(LifecycleError) as exc_info:
        root.require_current_identity()

    assert exc_info.value.code == "lifecycle_root_replaced"


def test_shared_locks_overlap(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")
    first_acquired = threading.Event()
    second_acquired = threading.Event()
    release = threading.Event()

    def hold_first() -> None:
        with checkpoint_lifecycle_lock(root, LifecycleLockMode.SHARED):
            first_acquired.set()
            assert release.wait(JOIN_TIMEOUT_SECONDS)

    def hold_second() -> None:
        assert first_acquired.wait(JOIN_TIMEOUT_SECONDS)
        with checkpoint_lifecycle_lock(root, LifecycleLockMode.SHARED):
            second_acquired.set()
            assert release.wait(JOIN_TIMEOUT_SECONDS)

    threads = [threading.Thread(target=hold_first), threading.Thread(target=hold_second)]
    for thread in threads:
        thread.start()
    assert second_acquired.wait(JOIN_TIMEOUT_SECONDS)
    release.set()
    for thread in threads:
        thread.join(JOIN_TIMEOUT_SECONDS)
        assert not thread.is_alive()


def test_exclusive_waits_for_shared_lock(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")
    shared_acquired = threading.Event()
    exclusive_attempted = threading.Event()
    release_shared = threading.Event()
    order: list[str] = []

    def hold_shared() -> None:
        with checkpoint_lifecycle_lock(root, LifecycleLockMode.SHARED):
            order.append("shared")
            shared_acquired.set()
            assert release_shared.wait(JOIN_TIMEOUT_SECONDS)

    def hold_exclusive() -> None:
        assert shared_acquired.wait(JOIN_TIMEOUT_SECONDS)
        exclusive_attempted.set()
        with checkpoint_lifecycle_lock(root, LifecycleLockMode.EXCLUSIVE):
            order.append("exclusive")

    shared = threading.Thread(target=hold_shared)
    exclusive = threading.Thread(target=hold_exclusive)
    shared.start()
    exclusive.start()
    assert exclusive_attempted.wait(JOIN_TIMEOUT_SECONDS)
    assert order == ["shared"]
    release_shared.set()
    for thread in (shared, exclusive):
        thread.join(JOIN_TIMEOUT_SECONDS)
        assert not thread.is_alive()
    assert order == ["shared", "exclusive"]


def test_lock_artifact_is_permanent_and_reused(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")
    with checkpoint_lifecycle_lock(root, LifecycleLockMode.SHARED):
        identity = root.lock_path.stat()
    assert root.lock_path.exists()

    with checkpoint_lifecycle_lock(root, LifecycleLockMode.EXCLUSIVE):
        reused = root.lock_path.stat()

    assert (reused.st_dev, reused.st_ino) == (identity.st_dev, identity.st_ino)
    assert root.lock_path.exists()


def test_lifecycle_lock_refuses_symlink(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")
    target = tmp_path / "target"
    target.touch()
    root.lock_path.symlink_to(target)

    with pytest.raises(LifecycleError) as exc_info:
        with checkpoint_lifecycle_lock(root, LifecycleLockMode.SHARED):
            pass

    assert exc_info.value.code == "lifecycle_lock_open"


def test_lifecycle_lock_refuses_non_regular_artifact(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")
    os.mkfifo(root.lock_path)

    with pytest.raises(LifecycleError) as exc_info:
        with checkpoint_lifecycle_lock(root, LifecycleLockMode.EXCLUSIVE):
            pass

    assert exc_info.value.code == "lifecycle_lock_not_regular"


def test_open_regular_file_refuses_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("content")
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(LifecycleError) as exc_info:
        with open_regular_file(link):
            pass

    assert exc_info.value.code == "opened_regular_file"


def test_open_regular_file_refuses_non_regular_file(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()

    with pytest.raises(LifecycleError) as exc_info:
        with open_regular_file(directory):
            pass

    assert exc_info.value.code == "opened_not_regular"


def test_open_descriptor_detects_path_replacement(tmp_path: Path) -> None:
    artifact = tmp_path / "authority.jsonl"
    replacement = tmp_path / "replacement"
    artifact.write_text("old")
    replacement.write_text("new")

    with open_regular_file(artifact) as opened:
        os.replace(replacement, artifact)
        with pytest.raises(LifecycleError) as exc_info:
            opened.require_path_identity()

    assert exc_info.value.code == "opened_path_replaced"


def test_capability_absence_is_distinguishable(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")

    with pytest.raises(LifecycleError) as exc_info:
        require_deletion_capability(
            None, root=root, protocol=LifecycleProtocol.TWO_AUTHORITY_V1
        )

    assert exc_info.value.code == "deletion_capability_absent"
    assert "distinct-node" in str(exc_info.value)


def test_capability_malformation_is_distinguishable(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")

    with pytest.raises(LifecycleError) as exc_info:
        require_deletion_capability(
            object(), root=root, protocol=LifecycleProtocol.TWO_AUTHORITY_V1
        )

    assert exc_info.value.code == "deletion_capability_malformed"


def test_capability_root_mismatch_is_distinguishable(tmp_path: Path) -> None:
    expected = _root(tmp_path / "expected")
    other = _root(tmp_path / "other")

    with pytest.raises(LifecycleError) as exc_info:
        require_deletion_capability(
            _capability(other),
            root=expected,
            protocol=LifecycleProtocol.TWO_AUTHORITY_V1,
        )

    assert exc_info.value.code == "deletion_capability_root_mismatch"


def test_capability_mount_mismatch_is_distinguishable(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")
    capability = DeletionCapability(
        root=root,
        receipt=DistinctNodeFlockReceipt(
            receipt_id="wrong-mount",
            mount_device=root.device + 1,
            protocol=LifecycleProtocol.TWO_AUTHORITY_V1,
        ),
    )

    with pytest.raises(LifecycleError) as exc_info:
        require_deletion_capability(
            capability, root=root, protocol=LifecycleProtocol.TWO_AUTHORITY_V1
        )

    assert exc_info.value.code == "deletion_capability_mount_mismatch"


def test_capability_protocol_mismatch_is_distinguishable(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")
    capability = _capability(root)

    with pytest.raises(LifecycleError) as exc_info:
        require_deletion_capability(
            capability,
            root=root,
            protocol=LifecycleProtocol.LEGACY_SINGLE_AUTHORITY,
        )

    assert exc_info.value.code == "deletion_capability_protocol_mismatch"


def test_valid_capability_is_returned_unchanged(tmp_path: Path) -> None:
    root = _root(tmp_path / "root")
    capability = _capability(root)

    assert (
        require_deletion_capability(
            capability, root=root, protocol=LifecycleProtocol.TWO_AUTHORITY_V1
        )
        is capability
    )


def test_receipt_rejects_bare_boolean_escape_hatch() -> None:
    with pytest.raises(LifecycleError) as exc_info:
        DistinctNodeFlockReceipt(  # type: ignore[arg-type]
            receipt_id="receipt",
            mount_device=1,
            protocol=True,
        )

    assert exc_info.value.code == "deletion_capability_malformed"


def test_lock_order_vocabulary_accepts_declared_order() -> None:
    with lock_order(LockOrder.RECEIPT):
        with lock_order(LockOrder.LIFECYCLE):
            with lock_order(LockOrder.CATALOG):
                with lock_order(LockOrder.PROTECTION_LEDGER):
                    pass


def test_lock_order_vocabulary_rejects_inversion() -> None:
    with lock_order(LockOrder.CATALOG):
        with pytest.raises(LifecycleError) as exc_info:
            with lock_order(LockOrder.LIFECYCLE):
                pass

    assert exc_info.value.code == "lifecycle_lock_order"
