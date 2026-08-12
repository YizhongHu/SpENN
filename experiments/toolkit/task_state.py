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

Two row-claim policies coexist here, deliberately
--------------------------------------------------

``_claim_row``
    *Release-by-reclaim.* One claim file per row; a later worker reclaims the
    row once its status is ``failed`` or ``stopped``. Used by Slurm
    mixed-profile submission (``pair_stability_v3``'s ``launch.py`` and
    :mod:`experiments.toolkit.executors`), where at most one submitter is
    racing per command profile and an operator's re-submit is what retries a
    row.

``claim_row_for_pass``
    *Pass-scoped, never released* (ADR-C012). Used by the allocation-pool
    worker model, where N long-lived workers share one immutable plan. Retry
    comes from starting a **new pass id**, never from releasing a claim.

The two are not interchangeable, and the choice is dictated by the concurrency
model rather than by taste. Under N concurrent workers, release-by-reclaim lets
every *other* worker re-claim and re-run a deterministically failing row inside
one pass: that is exactly the 2026-08-07 Polaris incident, in which one broken
row executed four times in a single pass and broke both "no duplicate rows" and
"failures remain isolated". Under a single racing submitter per profile, that
same policy is what lets a re-submit pick a failed row back up, so it is kept.

Pick pass-scoped claims for worker pools, ``_claim_row`` for submission races.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote
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
    """Atomically claim one row for a racing CPU/CUDA submission.

    This is the *release-by-reclaim* policy: a claim on a row whose status has
    reached ``failed`` or ``stopped`` is taken over by the next caller, so an
    operator's re-submit retries the row. That is correct for mixed-profile
    Slurm submission, where at most one submitter races per command profile.

    It is **not** safe for a worker pool. Do not call this from an
    allocation-pool worker: with N concurrent workers, a deterministically
    failing row is re-claimed and re-run by every other worker within a single
    pass. Use :func:`claim_row_for_pass` there instead, and see the "Two
    row-claim policies" section of the module docstring for why both exist.
    """

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


# ---------------------------------------------------------------------------
# Pass-scoped row claims (ADR-C012) -- the allocation-pool policy
# ---------------------------------------------------------------------------

PASS_CLAIMS_DIRNAME = "_claims"
"""Name of the claim namespace directory created under a run root."""

ATTEMPT_DIR_PREFIX = "attempt"
"""Prefix of the per-execution directories created by :func:`next_attempt_dir`."""

_UNUSABLE_CLAIM_NAMES = {"", ".", ".."}


def _claim_component(value: str) -> str:
    """Return one path-safe directory name for a pass id or row id.

    Percent-encoding is used rather than character stripping because it is
    injective: ``quote`` escapes ``%`` itself, so two distinct row ids can never
    collapse onto one claim directory. A collision here would silently drop a
    row from a whole pass, which is worse than an ugly directory name.
    """

    name = quote(str(value), safe="")
    if name in _UNUSABLE_CLAIM_NAMES:
        raise ValueError(f"claim key {value!r} has no usable directory name")
    return name


def pass_claims_dir(run_root: str | Path, pass_id: str) -> Path:
    """Return the claim namespace for one pass under ``run_root``.

    Parameters
    ----------
    run_root : str or Path
        Durable run root shared by every worker in the allocation.
    pass_id : str
        Identifier of the current pass. A retry uses a *new* pass id, which
        yields a fresh, empty claim namespace.

    Returns
    -------
    Path
        ``<run_root>/_claims/<pass_id>``. Not created by this call.
    """

    return Path(run_root) / PASS_CLAIMS_DIRNAME / _claim_component(pass_id)


def claim_row_for_pass(
    run_root: str | Path,
    pass_id: str,
    row: str,
    payload: dict[str, Any] | None = None,
) -> bool:
    """Claim one row for one pass, atomically and permanently (ADR-C012).

    The claim is the directory ``<run_root>/_claims/<pass_id>/<row>``, created
    with a single ``mkdir``. ``mkdir`` is atomic on Lustre, so exactly one of
    any number of concurrent callers observes success.

    A claim is **never released**. A caller that loses the race must move on to
    the next row: it must not inspect the row's status and must not reclaim.
    Retry is expressed by running a new pass, whose claim namespace is empty;
    rows that already succeeded are skipped by the ordinary completion checks
    (:func:`_attempt_already_completed`), so a fresh pass re-runs only the rows
    that did not complete.

    Parameters
    ----------
    run_root : str or Path
        Durable run root shared by every worker in the allocation.
    pass_id : str
        Identifier of the current pass.
    row : str
        Identifier of the row being claimed, unique within the plan. Any string
        is accepted; it is percent-encoded into a single directory name.
    payload : dict, optional
        Extra provenance merged into the claim receipt written inside the claim
        directory. The receipt is provenance only -- the directory is the claim.

    Returns
    -------
    bool
        ``True`` if this caller now owns the row for this pass, ``False`` if
        another caller already owns it. A ``False`` result is an ordinary,
        expected outcome under contention and is worth logging as a lost claim.

    See Also
    --------
    _claim_row : the release-by-reclaim policy used for Slurm submission races.
    """

    claim_dir = pass_claims_dir(run_root, pass_id) / _claim_component(row)
    try:
        # ``parents=True`` creates intermediates permissively, but the leaf is
        # still one bare ``os.mkdir`` -- that is the atomic step that decides
        # the race.
        claim_dir.mkdir(parents=True)
    except FileExistsError:
        return False
    receipt = {
        "pass_id": str(pass_id),
        "row": str(row),
        "pid": os.getpid(),
        "claimed_at_unix": time.time(),
        **(payload or {}),
    }
    try:
        (claim_dir / "claim.json").write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n")
    except OSError:
        # The directory is the claim; a failed receipt write must never hand the
        # row to a second worker.
        pass
    return True


def next_attempt_dir(row_dir: str | Path) -> Path:
    """Create and return the first free ``<row_dir>/attempt<N>`` directory.

    Numbering starts at 1 and the directory is created by this call, so the
    returned path is always fresh: a re-run of a row never overwrites the
    evidence left by an earlier attempt.

    Creation uses ``mkdir`` on each candidate rather than counting existing
    directories, so two callers cannot select the same ``N``. Within one pass
    that race cannot happen anyway -- :func:`claim_row_for_pass` admits a single
    worker per row -- but attempts also accumulate across passes, and the loop
    keeps the numbering correct without depending on that.

    Parameters
    ----------
    row_dir : str or Path
        Directory holding one row's attempts. Created if missing.

    Returns
    -------
    Path
        The newly created attempt directory.
    """

    row_path = Path(row_dir)
    row_path.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        attempt_dir = row_path / f"{ATTEMPT_DIR_PREFIX}{number}"
        try:
            attempt_dir.mkdir()
        except FileExistsError:
            number += 1
            continue
        return attempt_dir


def allocation_deadline_unix(
    explicit: str | float | None = None,
    *,
    env_var: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> float | None:
    """Return the UNIX deadline an allocation-pool worker must stop claiming by.

    Resolution order: an explicit value, then ``env_var`` if the caller named
    one, then ``SLURM_JOB_END_TIME``. Values are parsed by
    :func:`parse_deadline_unix`, so UNIX seconds and ISO timestamps are both
    accepted.

    There is deliberately no PBS branch. Per ADR-C008 facility selection is by
    environment variable only and is never committed configuration, so a PBS
    allocation supplies its end time through ``env_var`` (or an explicit value
    derived from ``qstat``) rather than through facility-specific code here.
    The resulting value feeds the already-tracked
    :func:`_deadline_guard_reached`; only the value was missing.

    Parameters
    ----------
    explicit : str or float, optional
        Deadline supplied directly, e.g. from a CLI flag. Wins over both
        environment lookups. Blank strings are treated as absent.
    env_var : str, optional
        Name of a facility-supplied environment variable to consult before the
        Slurm fallback.
    environ : Mapping, optional
        Environment mapping to read. Defaults to :data:`os.environ`.

    Returns
    -------
    float or None
        UNIX deadline in seconds, or ``None`` when no source supplied one.
    """

    env = os.environ if environ is None else environ
    if explicit is not None and str(explicit).strip():
        return parse_deadline_unix(str(explicit))
    names = ([env_var] if env_var else []) + ["SLURM_JOB_END_TIME"]
    for name in names:
        value = env.get(name)
        if value is not None and str(value).strip():
            return parse_deadline_unix(str(value))
    return None


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
