"""Checkpoint directory artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tpen.artifacts import write_json

COMPLETE_MARKER = "COMPLETE"
LATEST_JSON = "latest.json"


class CheckpointPruneError(ValueError):
    """Raised when checkpoint pruning cannot prove its safety preconditions."""


def checkpoint_step_dir_name(step: int) -> str:
    """Return the canonical directory name for a checkpoint step."""

    if int(step) < 0:
        raise ValueError(f"checkpoint step must be nonnegative, got {step}")
    return f"step_{int(step):06d}"


def is_complete_checkpoint_dir(path: str | Path) -> bool:
    """Return whether `path` is a complete checkpoint directory."""

    checkpoint_dir = Path(path)
    return (
        checkpoint_dir.is_dir()
        and not checkpoint_dir.name.endswith(".tmp")
        and (checkpoint_dir / "manifest.json").is_file()
        and (checkpoint_dir / COMPLETE_MARKER).is_file()
    )


def require_complete_checkpoint_dir(path: str | Path) -> Path:
    """Return `path` as a checkpoint directory or fail loudly."""

    checkpoint_dir = Path(path)
    if checkpoint_dir.name.endswith(".tmp"):
        raise ValueError(f"checkpoint tmp directory is not valid: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
    if not (checkpoint_dir / COMPLETE_MARKER).is_file():
        raise ValueError(f"checkpoint directory lacks COMPLETE marker: {checkpoint_dir}")
    if not (checkpoint_dir / "manifest.json").is_file():
        raise ValueError(f"checkpoint directory lacks manifest.json: {checkpoint_dir}")
    return checkpoint_dir


def read_latest(checkpoint_root: str | Path) -> dict[str, Any]:
    """Read the latest checkpoint pointer from `checkpoint_root/latest.json`."""

    latest_path = Path(checkpoint_root) / LATEST_JSON
    if not latest_path.is_file():
        raise FileNotFoundError(f"latest checkpoint pointer not found: {latest_path}")
    with latest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data.get("checkpoint_dir"):
        raise ValueError(f"invalid latest checkpoint pointer: {latest_path}")
    return data


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Resolve a checkpoint root, latest pointer, or step directory to a valid step dir."""

    candidate = Path(path)
    if candidate.is_file() and candidate.name == LATEST_JSON:
        pointer = read_latest(candidate.parent)
        return require_complete_checkpoint_dir(candidate.parent / str(pointer["checkpoint_dir"]))
    if candidate.is_dir() and (candidate / LATEST_JSON).is_file() and not (candidate / "manifest.json").exists():
        pointer = read_latest(candidate)
        return require_complete_checkpoint_dir(candidate / str(pointer["checkpoint_dir"]))
    return require_complete_checkpoint_dir(candidate)


def write_latest(checkpoint_root: Path, checkpoint_dir: Path, *, step: int, created_at_unix: float) -> None:
    """Atomically update `latest.json` to point at `checkpoint_dir`."""

    latest_path = checkpoint_root / LATEST_JSON
    tmp_path = checkpoint_root / f"{LATEST_JSON}.tmp"
    write_json(
        tmp_path,
        {
            "checkpoint_dir": checkpoint_dir.name,
            "step": int(step),
            "created_at_unix": float(created_at_unix),
        },
    )
    tmp_path.replace(latest_path)


def list_complete_checkpoints(checkpoint_root: str | Path) -> list[Path]:
    """Return complete checkpoint step directories ordered by step."""

    root = Path(checkpoint_root)
    checkpoints = [path for path in root.glob("step_*") if is_complete_checkpoint_dir(path)]
    return sorted(checkpoints, key=_checkpoint_sort_key)


def prune_old_checkpoints(checkpoint_root: str | Path, *, keep_last: int | None) -> None:
    """Remove older complete checkpoint directories when `keep_last` is set.

    The directory `latest.json` points at is always spared, even when it falls
    outside the newest `keep_last`. `latest.json` is the pointer every resume
    path resolves through, so deleting its target would leave the run
    unresumable through its own pointer. The pointer target is spared *in
    addition* to the newest `keep_last`; the keep window itself is unchanged.

    Parameters
    ----------
    checkpoint_root : str or pathlib.Path
        Checkpoint root holding ``step_*`` directories and ``latest.json``.
    keep_last : int or None
        Number of newest complete checkpoints to keep. ``None`` disables
        pruning entirely.

    Raises
    ------
    ValueError
        If `keep_last` is set but not positive.
    """

    if keep_last is None:
        return
    if type(keep_last) is not int:
        raise TypeError("keep_last must be an int or None")
    keep = keep_last
    if keep < 1:
        raise ValueError(f"keep_last must be positive when set, got {keep_last}")
    from .pruning import sweep_published_checkpoints

    sweep_published_checkpoints(checkpoint_root, keep_last=keep)


def _latest_pointer_target(checkpoint_root: Path) -> Path | None:
    """Return the valid target of a present latest pointer, or ``None`` absent.

    Absence is a deliberate legacy/manual-prune policy.  Once a pointer file
    exists, however, every failure to read or validate it is a safety error:
    treating a damaged pointer as absent would remove the resume checkpoint's
    protection.
    """

    latest_path = checkpoint_root / LATEST_JSON
    if latest_path.is_symlink():
        raise CheckpointPruneError(
            f"latest checkpoint pointer must not be a symlink: {latest_path}"
        )
    if not latest_path.exists():
        return None
    if not latest_path.is_file():
        raise CheckpointPruneError(
            f"latest checkpoint pointer is not a regular file: {latest_path}"
        )
    try:
        pointer = read_latest(checkpoint_root)
        pointer_name = pointer["checkpoint_dir"]
        if type(pointer_name) is not str or not pointer_name.strip():
            raise ValueError("checkpoint_dir must be a non-empty string")
        pointer_path = Path(pointer_name)
        if (
            pointer_path.is_absolute()
            or pointer_path.name != pointer_name
            or pointer_name in {".", ".."}
        ):
            raise ValueError("checkpoint_dir must name one direct child")
        target = checkpoint_root / pointer_name
        if target.is_symlink():
            raise ValueError("latest target must not be a symlink")
        if target.resolve(strict=False).parent != checkpoint_root.resolve(strict=False):
            raise ValueError("latest target must be inside its checkpoint root")
        return require_complete_checkpoint_dir(target)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise CheckpointPruneError(
            f"invalid latest checkpoint pointer; refusing to prune: {latest_path}"
        ) from exc


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.removeprefix("step_")), path.name
    except ValueError:
        return -1, path.name
