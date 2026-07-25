"""Read-only source and result-tree receipts for V4-0 ownership checks."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from routes import load_legacy_source_manifest


def inventory_source_tree(root: Path) -> tuple[dict[str, object], ...]:
    """Return deterministic content receipts for the pinned source closure."""

    root = Path(root).resolve(strict=True)
    rows: list[dict[str, object]] = []
    for entry in load_legacy_source_manifest()["files"]:
        relative = str(entry["path"])
        path = root / relative
        row = _metadata_row(path, root)
        if row["type"] == "file":
            row["sha256"] = sha256_file(path)
        rows.append(row)
    return tuple(rows)


def inventory_results_tree(root: Path) -> tuple[dict[str, object], ...]:
    """Return deterministic metadata receipts without hashing result payloads."""

    root = Path(root)
    if not root.exists() and not root.is_symlink():
        return ()
    canonical_parent = root.parent.resolve(strict=True)
    rows: list[dict[str, object]] = []
    for current, directory_names, filenames in os.walk(root, followlinks=False):
        directory_names.sort()
        filenames.sort()
        current_path = Path(current)
        for name in (*directory_names, *filenames):
            rows.append(_metadata_row(current_path / name, canonical_parent))
    return tuple(sorted(rows, key=lambda row: str(row["path"])))


def _metadata_row(path: Path, base: Path) -> dict[str, object]:
    stat = path.lstat()
    relative = path.relative_to(base)
    if path.is_symlink():
        kind = "symlink"
    elif path.is_dir():
        kind = "directory"
    elif path.is_file():
        kind = "file"
    else:
        kind = "other"
    row: dict[str, object] = {
        "path": relative.as_posix(),
        "type": kind,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "mode": int(stat.st_mode),
    }
    if kind == "symlink":
        row["link_target"] = os.readlink(path)
    return row


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
