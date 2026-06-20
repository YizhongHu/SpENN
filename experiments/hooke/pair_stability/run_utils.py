"""Shared utilities for the pair-stability study scripts (PR8.8).

The stage-layout vocabulary, timezone/attempt-id helpers, run-id grammar, JSON
IO, and staged-directory path helpers used by ``plan.py``, ``orchestrator.py``,
``collect.py``, and ``select_champions.py``. Kept stdlib-only so every study
script can import it without pulling in torch.

Result layout (under ``results_root``)::

    00_grid/{attempt_id}/...
    01_train/{run_id}/{attempt_id}/...
    02_validation/{run_id}/{attempt_id}/...
    03_collect/{attempt_id}/...
    04_select/{attempt_id}/...

Every directory under a stage (or under a stage's run id) is an attempt, so
there is no intermediate ``attempts/`` segment; attempt ids name the leaves
directly.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Stage directory names; the numbers document artifact inheritance order.
# ---------------------------------------------------------------------------
STAGE_GRID = "00_grid"
STAGE_TRAIN = "01_train"
STAGE_VALIDATION = "02_validation"
STAGE_COLLECT = "03_collect"
STAGE_SELECT = "04_select"

# Grid axis order (also the deterministic Cartesian-product nesting order).
GRID_AXES = ("architecture", "normalization", "lr", "channels", "seed")


# ---------------------------------------------------------------------------
# Timezone and attempt ids
# ---------------------------------------------------------------------------
# The planner is the source of truth for the study timezone; study
# timestamps (attempt ids, manifest ``created_at``) and the ``run.timezone``
# override it injects all derive from it. ``plan.py --timezone`` overrides
# this default.
DEFAULT_STUDY_TIMEZONE = "America/New_York"


def resolve_timezone(name: str | None = None) -> ZoneInfo:
    """Return the study timezone, defaulting to America/New_York.

    Parameters
    ----------
    name : str or None
        IANA timezone name. ``None`` selects :data:`DEFAULT_STUDY_TIMEZONE`.

    Returns
    -------
    zoneinfo.ZoneInfo
        The resolved timezone.
    """

    return ZoneInfo(name or DEFAULT_STUDY_TIMEZONE)


# Resolved default timezone used to stamp attempt ids.
STUDY_TIMEZONE = resolve_timezone()


def new_attempt_id(moment: datetime | None = None, *, tz: ZoneInfo = STUDY_TIMEZONE) -> str:
    """Return an attempt id of the form ``YYYYMMDDTHHMMSS-0400`` in ``tz``.

    The trailing UTC offset keeps the id unambiguous and directory-safe. Ids
    sort chronologically by name within a fixed offset; the only exception is
    the one-hour DST fall-back fold, where lexical order can briefly disagree
    with real time (the authoritative latest pointer is ``latest.json``).

    Parameters
    ----------
    moment : datetime or None
        Instant to format; defaults to the current time in ``tz``.
    tz : zoneinfo.ZoneInfo
        Wall clock for the id; defaults to :data:`STUDY_TIMEZONE`.

    Returns
    -------
    str
        The attempt id.
    """

    moment = moment or datetime.now(tz)
    return moment.astimezone(tz).strftime("%Y%m%dT%H%M%S%z")


# ---------------------------------------------------------------------------
# Run-id grammar
# ---------------------------------------------------------------------------
def format_lr(lr: float) -> str:
    """Return a compact, deterministic learning-rate label, e.g. ``1e-3``."""

    mantissa, _, exponent = f"{float(lr):.1e}".partition("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def run_id_for(point: dict[str, Any]) -> str:
    """Return the deterministic run id for one grid point."""

    return (
        f"arch-{point['architecture']}"
        f"_norm-{point['normalization']}"
        f"_lr-{format_lr(point['lr'])}"
        f"_ch-{int(point['channels'])}"
        f"_seed-{int(point['seed'])}"
    )


_RUN_ID_PATTERN = re.compile(
    r"^arch-(?P<architecture>.+)_norm-(?P<normalization>[^_]+)"
    r"_lr-(?P<lr>[^_]+)_ch-(?P<channels>\d+)_seed-(?P<seed>\d+)$"
)


def parse_run_id(run_id: str) -> dict[str, Any]:
    """Recover the grid choices encoded in a run id."""

    match = _RUN_ID_PATTERN.match(run_id)
    if match is None:
        raise ValueError(f"run id {run_id!r} does not match the pair_stability convention")
    fields = match.groupdict()
    return {
        "architecture": fields["architecture"],
        "normalization": fields["normalization"],
        "lr": fields["lr"],
        "channels": int(fields["channels"]),
        "seed": int(fields["seed"]),
    }


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------
def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as pretty JSON, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def read_json(path: Path) -> Any:
    """Read JSON from ``path``."""

    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Stage path layout
# ---------------------------------------------------------------------------
def stage_dir(results_root: str | Path, stage: str) -> Path:
    """Return the directory for a numbered stage."""

    return Path(results_root) / stage


def grid_attempt_dir(results_root: str | Path, attempt_id: str) -> Path:
    """Return the ``00_grid`` attempt directory."""

    return stage_dir(results_root, STAGE_GRID) / attempt_id


def train_run_dir(results_root: str | Path, run_id: str) -> Path:
    """Return the per-run-id directory under ``01_train``."""

    return stage_dir(results_root, STAGE_TRAIN) / run_id


def validation_run_dir(results_root: str | Path, run_id: str) -> Path:
    """Return the per-run-id directory under ``02_validation``."""

    return stage_dir(results_root, STAGE_VALIDATION) / run_id


def train_attempt_dir(results_root: str | Path, run_id: str, attempt_id: str) -> Path:
    """Return the train attempt directory for a run id."""

    return train_run_dir(results_root, run_id) / attempt_id


def validation_attempt_dir(results_root: str | Path, run_id: str, attempt_id: str) -> Path:
    """Return the validation attempt directory for a run id."""

    return validation_run_dir(results_root, run_id) / attempt_id


def attempt_ids(parent: str | Path) -> list[str]:
    """Return sorted attempt-id directory names directly under ``parent``.

    Excludes the ``latest`` convenience symlink. Returns ``[]`` when ``parent``
    is not a directory. Because attempt ids sort chronologically by name, the
    last element is the most recent (modulo the DST fold noted in
    :func:`new_attempt_id`).
    """

    parent = Path(parent)
    if not parent.is_dir():
        return []
    return sorted(
        child.name for child in parent.iterdir() if child.is_dir() and child.name != "latest"
    )


def write_latest(stage_path: Path, attempt_id: str) -> None:
    """Record the most recent attempt id under ``stage_path``.

    Writes a portable ``latest.json`` pointer and additionally attempts a
    ``latest`` symlink (best effort; durable provenance uses explicit attempt
    ids, never ``latest``).
    """

    write_json(stage_path / "latest.json", {"attempt_id": attempt_id, "path": attempt_id})
    link = stage_path / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(attempt_id, target_is_directory=True)
    except OSError:
        # Symlinks may be unsupported on the target filesystem; latest.json suffices.
        pass
