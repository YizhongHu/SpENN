"""Row claim, deadline-guard, and completion-check primitives.

These helpers decide whether a chunked local/Submitit worker should skip,
claim, or reclaim one row of work, based on artifacts already written under a
run's attempt directory (``status.json``, ``checkpoints/latest.json``). They
are shared by ``pair_stability_v2``/``pair_stability_v3``'s ``launch.py`` and
by :mod:`experiments.toolkit.executors`.

The checkpoint pointer these functions read (``checkpoints/latest.json``,
written by ``spenn.checkpoint``) is unrelated to the attempt-lineage
``latest.json`` pointer owned by a study's ``utils.layout`` module; the two
share a filename but not a schema.

This module also collects ``final_train.py``'s, ``final_eval.py``'s, and
``validate.py``'s own, independently-implemented checkpoint-discovery and
readiness checks. They are kept as distinct functions rather than merged with
``_attempt_already_completed`` above: each answers a different question
(is this row ready for the next stage, has it fully completed, what is the
highest complete checkpoint to resume from) and they can legitimately
disagree with each other and with ``_attempt_already_completed`` on the same
attempt directory. Unifying them would be a behavior change, not a
relocation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

_DEFAULT_TIMEZONE = "America/New_York"


def parse_deadline_unix(value: str | None) -> float | None:
    """Return a UNIX deadline from seconds or an ISO timestamp."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    timestamp = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError(
            "local deadline must be UNIX seconds or an ISO timestamp"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(_DEFAULT_TIMEZONE))
    return parsed.timestamp()


def local_claim_deadline_unix(args: argparse.Namespace) -> float | None:
    """Return the local claim deadline from CLI or Slurm environment."""

    explicit = getattr(args, "local_deadline", None)
    if explicit:
        return parse_deadline_unix(explicit)
    return parse_deadline_unix(os.environ.get("SLURM_JOB_END_TIME"))


def _deadline_guard_reached(deadline_unix: float | None, guard_min: int | None) -> bool:
    """Return whether a local worker should stop claiming new rows."""

    if deadline_unix is None:
        return False
    guard_seconds = max(0, int(guard_min or 0)) * 60
    if guard_seconds <= 0:
        return False
    return time.time() >= float(deadline_unix) - guard_seconds


def _deadline_guard_payload(
    *,
    index: int,
    command: str,
    claim_label: str | None,
    deadline_unix: float | None,
    guard_min: int | None,
) -> dict[str, Any]:
    """Return a row status payload for a deadline-guarded skipped claim."""

    remaining_min = None
    if deadline_unix is not None:
        remaining_min = (float(deadline_unix) - time.time()) / 60
    return {
        "status": "skipped_deadline_guard",
        "chunk_index": index,
        "command": command,
        "claim_label": claim_label,
        "deadline_unix": deadline_unix,
        "guard_min": guard_min,
        "remaining_min": remaining_min,
    }


def _write_status(path: str | Path | None, payload: dict[str, Any]) -> None:
    """Best-effort JSON status writer for launcher/chunk bookkeeping."""

    if path is None:
        return
    status_path = Path(path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _claim_path_for_status(path: str | Path | None) -> Path | None:
    """Return the atomic launch claim path next to a row status file."""

    if path is None:
        return None
    return Path(path).with_name("launcher_claim.json")


def claim_paths_for_statuses(paths: Sequence[str | Path | None] | None) -> list[Path | None] | None:
    """Return per-row claim paths for mixed CPU/CUDA submissions."""

    if paths is None:
        return None
    return [_claim_path_for_status(path) for path in paths]


def _attempt_already_completed(status_path: str | Path | None) -> bool:
    """Return whether the row already has a completed run checkpoint."""

    if status_path is None:
        return False
    attempt_dir = Path(status_path).parent
    checkpoint = attempt_dir / "checkpoints" / "latest.json"
    status_file = attempt_dir / "status.json"
    if not checkpoint.is_file() or not status_file.is_file():
        return False
    status = _read_json_mapping(status_file)
    if status is None:
        return False
    return status.get("status") == "completed"


def _terminal_row_status(status_path: str | Path | None) -> str | None:
    """Return the terminal status that makes an old row claim reclaimable."""

    if status_path is None:
        return None
    status_path = Path(status_path)
    launcher_status = _read_json_mapping(status_path)
    if launcher_status and launcher_status.get("status") in {"failed", "stopped"}:
        return str(launcher_status["status"])
    run_status = _read_json_mapping(status_path.parent / "status.json")
    if run_status and run_status.get("status") in {"failed", "stopped"}:
        return str(run_status["status"])
    return None


def _write_claim(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _claim_row(path: str | Path | None, payload: dict[str, Any], status_path: str | Path | None = None) -> bool:
    """Atomically claim one row for a racing CPU/CUDA submission."""

    if path is None:
        return True
    claim_path = Path(path)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        terminal_status = _terminal_row_status(status_path)
        if terminal_status is None:
            return False
        lock_path = claim_path.with_name(f"{claim_path.name}.reclaim.lock")
        try:
            lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            return False
        try:
            with os.fdopen(lock_fd, "w") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "created_at_unix": time.time()}) + "\n")
            terminal_status = _terminal_row_status(status_path)
            if terminal_status is None:
                return False
            _write_claim(
                claim_path,
                {
                    **payload,
                    "reclaimed": True,
                    "reclaim_reason": terminal_status,
                    "previous_claim": _read_json_mapping(claim_path),
                },
            )
            return True
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return True


def _checkpoint_ready(train_attempt: Path) -> bool:
    """Return whether a train attempt exposes a latest checkpoint pointer."""

    return (train_attempt / "checkpoints" / "latest.json").is_file()


def _checkpoint_step(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.removeprefix("step_")), path.name
    except ValueError:
        return -1, path.name


def _complete_checkpoint_dirs(attempt_dir: Path) -> list[Path]:
    checkpoint_dir = attempt_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []
    checkpoints = [
        path
        for path in checkpoint_dir.glob("step_*")
        if path.is_dir() and not path.name.endswith(".tmp") and (path / "COMPLETE").is_file()
    ]
    return sorted(checkpoints, key=_checkpoint_step)


def _latest_complete_checkpoint(attempt_dir: Path) -> Path | None:
    checkpoints = _complete_checkpoint_dirs(attempt_dir)
    return checkpoints[-1] if checkpoints else None


def _latest_restorable_checkpoint(attempt_dir: Path) -> Path | None:
    """Return highest complete checkpoint containing a manifest."""

    checkpoints = [
        checkpoint
        for checkpoint in _complete_checkpoint_dirs(attempt_dir)
        if (checkpoint / "manifest.json").is_file()
    ]
    return checkpoints[-1] if checkpoints else None


def _final_train_completed(attempt_dir: Path) -> bool:
    status_path = attempt_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text()).get("status")
    except Exception:
        return False
    return status == "completed" and _latest_complete_checkpoint(attempt_dir) is not None


def _resume_overrides(attempt_dir: Path) -> list[str]:
    checkpoint = _latest_complete_checkpoint(attempt_dir)
    if checkpoint is None:
        return []
    if _final_train_completed(attempt_dir):
        return []
    return [
        f"load.path={checkpoint}",
        "load.mode=train_resume",
    ]


def _resolved_checkpoint(train_attempt: Path) -> dict[str, Any] | None:
    selection_path = train_attempt / "selected_checkpoint.json"
    if not selection_path.is_file():
        return None
    selection = json.loads(selection_path.read_text())
    pointer = Path(str(selection.get("checkpoint_pointer", "")))
    if not pointer.is_file():
        return None
    pointer_data = json.loads(pointer.read_text())
    checkpoint_name = pointer_data.get("checkpoint_dir")
    if not checkpoint_name:
        return None
    checkpoint_dir = pointer.parent / str(checkpoint_name)
    if not checkpoint_dir.is_dir():
        return None
    if not (checkpoint_dir / "COMPLETE").is_file() or not (checkpoint_dir / "manifest.json").is_file():
        return None
    return {
        "selection_path": str(selection_path),
        "selection_policy": selection.get("selection_policy", ""),
        "checkpoint_pointer": str(pointer),
        "checkpoint_pointer_data": pointer_data,
        "resolved_checkpoint_dir": str(checkpoint_dir),
    }


def _resolved_latest_checkpoint(train_attempt: Path) -> dict[str, Any] | None:
    """Resolve highest complete restorable checkpoint in an attempt."""

    checkpoint_dir = _latest_restorable_checkpoint(train_attempt)
    if checkpoint_dir is None:
        return None
    pointer = train_attempt / "checkpoints" / "latest.json"
    pointer_data = _read_json_mapping(pointer) if pointer.is_file() else None
    return {
        "selection_policy": "latest_complete_checkpoint",
        "checkpoint_pointer": str(pointer) if pointer.is_file() else "",
        "checkpoint_pointer_data": pointer_data or {},
        "resolved_checkpoint_dir": str(checkpoint_dir),
    }
