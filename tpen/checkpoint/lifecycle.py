"""Checkpoint lifecycle coordination primitives.

This module is foundation only.  No publication, protection, selection, or
pruning call site participates at this tip, so deletion safety is explicitly
not established here.

Correctness of the lock protocol assumes that ``flock`` is coherent across
every node that shares a checkpoint filesystem.  A process cannot establish
that operational property; a mount-bound, distinct-node receipt must do so.
"""

from __future__ import annotations

import contextvars
import enum
import fcntl
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


CHECKPOINT_LIFECYCLE_LOCK_FILENAME = ".checkpoint-lifecycle.lock"


class LifecycleError(ValueError):
    """Refusal raised when a lifecycle precondition is not satisfied."""

    def __init__(self, message: str, *, code: str, artifact: Path | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.artifact = artifact


class LifecycleLockMode(enum.Enum):
    """Typed lifecycle-lock acquisition modes."""

    SHARED = enum.auto()
    EXCLUSIVE = enum.auto()


class LifecycleProtocol(enum.Enum):
    """Protocol identity carried by operational capability receipts."""

    TWO_AUTHORITY_V1 = "two-authority-v1"
    LEGACY_SINGLE_AUTHORITY = "legacy-single-authority"


class LockOrder(enum.IntEnum):
    """Global acquisition order for destructive checkpoint operations."""

    RECEIPT = 1
    LIFECYCLE = 2
    CATALOG = 3
    PROTECTION_LEDGER = 4


@dataclass(frozen=True)
class CheckpointRoot:
    """Canonical identity of one existing checkpoint root."""

    path: Path
    device: int = field(init=False)
    inode: int = field(init=False)

    def __post_init__(self) -> None:
        canonical = Path(self.path).resolve(strict=True)
        metadata = canonical.stat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise LifecycleError(
                f"checkpoint root is not a directory: {canonical}",
                code="lifecycle_root_not_directory",
                artifact=canonical,
            )
        object.__setattr__(self, "path", canonical)
        object.__setattr__(self, "device", metadata.st_dev)
        object.__setattr__(self, "inode", metadata.st_ino)

    @property
    def lock_path(self) -> Path:
        """Return the sole lifecycle-lock pathname for this root."""
        return self.path / CHECKPOINT_LIFECYCLE_LOCK_FILENAME

    def require_same(self, other: "CheckpointRoot") -> None:
        """Refuse split-brain participation by a different root identity."""
        if self != other:
            raise LifecycleError(
                f"lifecycle root mismatch: expected {self.path}, got {other.path}",
                code="lifecycle_root_mismatch",
                artifact=self.lock_path,
            )

    def require_current_identity(self) -> None:
        """Refuse when the canonical pathname no longer names this root."""
        try:
            metadata = self.path.stat()
        except OSError as exc:
            raise LifecycleError(
                f"checkpoint root is no longer readable: {self.path}; restore it before retrying",
                code="lifecycle_root_unavailable",
                artifact=self.path,
            ) from exc
        if (metadata.st_dev, metadata.st_ino) != (self.device, self.inode):
            raise LifecycleError(
                f"checkpoint root identity changed: {self.path}; reconstruct the root identity before retrying",
                code="lifecycle_root_replaced",
                artifact=self.path,
            )


@dataclass(frozen=True)
class FileIdentity:
    """Identity captured from an opened regular-file descriptor."""

    device: int
    inode: int
    size: int


@dataclass(frozen=True)
class OpenedRegularFile:
    """Validated descriptor and the identity observed when it was opened."""

    path: Path
    fd: int
    identity: FileIdentity

    def require_path_identity(self) -> None:
        """Refuse when the pathname no longer names the opened descriptor.

        The post-read check closes ordinary unlink-and-replace races.  It does
        not claim protection against an adversarial inode-reuse (ABA) attack.
        """
        try:
            metadata = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise LifecycleError(
                f"opened artifact no longer has a readable pathname: {self.path}; retry safely",
                code="opened_path_unavailable",
                artifact=self.path,
            ) from exc
        if (metadata.st_dev, metadata.st_ino) != (
            self.identity.device,
            self.identity.inode,
        ):
            raise LifecycleError(
                f"opened artifact was replaced: {self.path}; retry safely",
                code="opened_path_replaced",
                artifact=self.path,
            )


@dataclass(frozen=True)
class DistinctNodeFlockReceipt:
    """Typed evidence that one mount passed the operational flock check."""

    receipt_id: str
    mount_device: int
    protocol: LifecycleProtocol

    def __post_init__(self) -> None:
        if (
            not isinstance(self.receipt_id, str)
            or not self.receipt_id.strip()
            or not isinstance(self.mount_device, int)
            or isinstance(self.mount_device, bool)
            or self.mount_device < 0
            or not isinstance(self.protocol, LifecycleProtocol)
        ):
            raise LifecycleError(
                "flock receipt is malformed; run the distinct-node lifecycle-lock check again",
                code="deletion_capability_malformed",
            )


@dataclass(frozen=True)
class DeletionCapability:
    """Mount-, root-, and protocol-bound authority prerequisite.

    Possessing this value does not make deletion safe at this tip: no
    destructive call site consumes it and neither authority store is wired to
    the lifecycle lock yet.
    """

    root: CheckpointRoot
    receipt: DistinctNodeFlockReceipt


_LOCK_STACK: contextvars.ContextVar[tuple[LockOrder, ...]] = contextvars.ContextVar(
    "checkpoint_lifecycle_lock_order", default=()
)


@contextmanager
def lock_order(level: LockOrder) -> Iterator[None]:
    """Declare a held lock level and fail loudly on order inversion."""
    if not isinstance(level, LockOrder):
        raise TypeError("lock order level must be a LockOrder")
    held = _LOCK_STACK.get()
    if held and level <= held[-1]:
        raise LifecycleError(
            f"checkpoint lock-order violation: cannot acquire {level.name} after {held[-1].name}",
            code="lifecycle_lock_order",
        )
    token = _LOCK_STACK.set((*held, level))
    try:
        yield
    finally:
        _LOCK_STACK.reset(token)


@contextmanager
def open_regular_file(path: str | Path, flags: int = os.O_RDONLY) -> Iterator[OpenedRegularFile]:
    """Open a nonblocking, non-symlink regular file and retain its descriptor."""
    artifact = Path(path)
    safe_flags = flags | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise LifecycleError(
            f"O_NOFOLLOW is unavailable; cannot safely open {artifact}",
            code="opened_nofollow_unavailable",
            artifact=artifact,
        )
    try:
        fd = os.open(artifact, safe_flags | nofollow)
    except OSError as exc:
        raise LifecycleError(
            f"cannot safely open regular file {artifact}: {exc}",
            code="opened_regular_file",
            artifact=artifact,
        ) from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise LifecycleError(
                f"opened artifact is not a regular file: {artifact}",
                code="opened_not_regular",
                artifact=artifact,
            )
        opened = OpenedRegularFile(
            path=artifact,
            fd=fd,
            identity=FileIdentity(metadata.st_dev, metadata.st_ino, metadata.st_size),
        )
        opened.require_path_identity()
        yield opened
    finally:
        os.close(fd)


@contextmanager
def checkpoint_lifecycle_lock(
    root: CheckpointRoot, mode: LifecycleLockMode
) -> Iterator[None]:
    """Hold the root's permanent shared or exclusive lifecycle lock.

    The artifact is created if absent and is never unlinked, truncated, or
    replaced here.  This primitive alone establishes no deletion safety.
    """
    if not isinstance(root, CheckpointRoot):
        raise TypeError("root must be a CheckpointRoot")
    if not isinstance(mode, LifecycleLockMode):
        raise TypeError("mode must be a LifecycleLockMode")
    root.require_current_identity()
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise LifecycleError(
            f"O_NOFOLLOW is unavailable; cannot safely open {root.lock_path}",
            code="lifecycle_lock_nofollow_unavailable",
            artifact=root.lock_path,
        )
    try:
        fd = os.open(root.lock_path, flags | nofollow, 0o666)
    except OSError as exc:
        raise LifecycleError(
            f"cannot open lifecycle lock {root.lock_path}: {exc}; inspect the artifact and mount",
            code="lifecycle_lock_open",
            artifact=root.lock_path,
        ) from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise LifecycleError(
                f"lifecycle lock is not a regular file: {root.lock_path}; replace it manually only while quiescent",
                code="lifecycle_lock_not_regular",
                artifact=root.lock_path,
            )
        opened_lock = OpenedRegularFile(
            path=root.lock_path,
            fd=fd,
            identity=FileIdentity(metadata.st_dev, metadata.st_ino, metadata.st_size),
        )
        opened_lock.require_path_identity()
        operation = fcntl.LOCK_SH if mode is LifecycleLockMode.SHARED else fcntl.LOCK_EX
        with lock_order(LockOrder.LIFECYCLE):
            try:
                fcntl.flock(fd, operation)
            except OSError as exc:
                raise LifecycleError(
                    f"cannot acquire lifecycle lock {root.lock_path}: {exc}; verify mount flock support",
                    code="lifecycle_lock_acquire",
                    artifact=root.lock_path,
                ) from exc
            opened_lock.require_path_identity()
            try:
                yield
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError as exc:
                    raise LifecycleError(
                        f"cannot release lifecycle lock {root.lock_path}: {exc}",
                        code="lifecycle_lock_release",
                        artifact=root.lock_path,
                    ) from exc
    finally:
        os.close(fd)


def require_deletion_capability(
    capability: object,
    *,
    root: CheckpointRoot,
    protocol: LifecycleProtocol,
) -> DeletionCapability:
    """Validate and return the exact operational capability required by a caller."""
    if capability is None:
        raise LifecycleError(
            f"deletion capability is absent for {root.path}; run the distinct-node flock check on this mount",
            code="deletion_capability_absent",
            artifact=root.lock_path,
        )
    root.require_current_identity()
    if not isinstance(capability, DeletionCapability):
        raise LifecycleError(
            f"deletion capability is malformed for {root.path}; supply a typed DeletionCapability",
            code="deletion_capability_malformed",
            artifact=root.lock_path,
        )
    if capability.root != root:
        raise LifecycleError(
            f"deletion capability names root {capability.root.path}, not {root.path}; rerun verification for this root",
            code="deletion_capability_root_mismatch",
            artifact=root.lock_path,
        )
    if capability.receipt.mount_device != root.device:
        raise LifecycleError(
            f"deletion capability names mount device {capability.receipt.mount_device}, not {root.device}; rerun verification on this mount",
            code="deletion_capability_mount_mismatch",
            artifact=root.lock_path,
        )
    if capability.receipt.protocol is not protocol:
        raise LifecycleError(
            f"deletion capability uses protocol {capability.receipt.protocol.value}, not {protocol.value}; rerun verification for the required protocol",
            code="deletion_capability_protocol_mismatch",
            artifact=root.lock_path,
        )
    return capability


__all__ = [
    "CHECKPOINT_LIFECYCLE_LOCK_FILENAME",
    "CheckpointRoot",
    "DeletionCapability",
    "DistinctNodeFlockReceipt",
    "FileIdentity",
    "LifecycleError",
    "LifecycleLockMode",
    "LifecycleProtocol",
    "LockOrder",
    "OpenedRegularFile",
    "checkpoint_lifecycle_lock",
    "lock_order",
    "open_regular_file",
    "require_deletion_capability",
]
