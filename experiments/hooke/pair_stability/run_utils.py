"""Shared utilities for the pair-stability study scripts (PR8.8).

The stage-layout vocabulary, timezone/attempt-id helpers, run-id grammar, JSON
IO, and staged-directory path helpers used by ``plan.py``, ``train.py``,
``validate.py``, ``collect.py``, and ``select_champions.py``. Kept stdlib-only
so every study script can import it without pulling in torch.

Result layout (under ``results_root``)::

    00_grid/{attempt_id}/...
    01_train/{run_id}/{attempt_id}/...
    02_validation/{run_id}/{attempt_id}/...
    03_collect/{attempt_id}/...
    04_select/{attempt_id}/...
    05_final_grid/{attempt_id}/...
    06_final_train/{final_run_id}/{attempt_id}/...
    07_final_eval/{final_run_id}/{attempt_id}/...
    08_final_collect/{attempt_id}/...
    09_final_report/{attempt_id}/...

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
STAGE_FINAL_GRID = "05_final_grid"
STAGE_FINAL_TRAIN = "06_final_train"
STAGE_FINAL_EVAL = "07_final_eval"
STAGE_FINAL_COLLECT = "08_final_collect"
STAGE_FINAL_REPORT = "09_final_report"

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


def final_grid_attempt_dir(results_root: str | Path, attempt_id: str) -> Path:
    """Return the ``05_final_grid`` attempt directory."""

    return stage_dir(results_root, STAGE_FINAL_GRID) / attempt_id


def final_train_run_dir(results_root: str | Path, final_run_id: str) -> Path:
    """Return the per-final-run-id directory under ``06_final_train``."""

    return stage_dir(results_root, STAGE_FINAL_TRAIN) / final_run_id


def final_eval_run_dir(results_root: str | Path, final_run_id: str) -> Path:
    """Return the per-final-run-id directory under ``07_final_eval``."""

    return stage_dir(results_root, STAGE_FINAL_EVAL) / final_run_id


def train_attempt_dir(results_root: str | Path, run_id: str, attempt_id: str) -> Path:
    """Return the train attempt directory for a run id."""

    return train_run_dir(results_root, run_id) / attempt_id


def validation_attempt_dir(results_root: str | Path, run_id: str, attempt_id: str) -> Path:
    """Return the validation attempt directory for a run id."""

    return validation_run_dir(results_root, run_id) / attempt_id


def final_train_attempt_dir(results_root: str | Path, final_run_id: str, attempt_id: str) -> Path:
    """Return the final-train attempt directory for a final run id."""

    return final_train_run_dir(results_root, final_run_id) / attempt_id


def final_eval_attempt_dir(results_root: str | Path, final_run_id: str, attempt_id: str) -> Path:
    """Return the final-eval attempt directory for a final run id."""

    return final_eval_run_dir(results_root, final_run_id) / attempt_id


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
        child.name
        for child in parent.iterdir()
        if child.is_dir() and child.name not in {"latest", "latest-smoke"}
    )


ATTEMPT_METADATA = "attempt_metadata.json"


def _latest_payload(parent: str | Path, *, filename: str = "latest.json") -> dict[str, Any] | None:
    """Return one latest-pointer payload when present."""

    latest = Path(parent) / filename
    if not latest.is_file():
        return None
    payload = read_json(latest)
    return payload if isinstance(payload, dict) else None


def read_latest_attempt_id(parent: str | Path, *, filename: str = "latest.json") -> str | None:
    """Return the latest-pointer attempt id under ``parent`` when present."""

    payload = _latest_payload(parent, filename=filename)
    attempt_id = None if payload is None else payload.get("attempt_id")
    return str(attempt_id) if attempt_id else None


def attempt_metadata(parent: str | Path, attempt_id: str) -> dict[str, Any]:
    """Return metadata recorded for one stage attempt."""

    metadata_path = Path(parent) / str(attempt_id) / ATTEMPT_METADATA
    if not metadata_path.is_file():
        return {}
    metadata = read_json(metadata_path)
    return metadata if isinstance(metadata, dict) else {}


def attempt_smoke(parent: str | Path, attempt_id: str) -> bool | None:
    """Return an attempt's smoke flag when known."""

    metadata = attempt_metadata(parent, attempt_id)
    if "smoke" in metadata:
        return bool(metadata["smoke"])
    return None


def _attempt_matches_smoke(parent: Path, attempt_id: str, smoke: bool | None) -> bool:
    """Return whether an attempt matches a requested smoke lineage."""

    if smoke is None:
        return True
    known_smoke = attempt_smoke(parent, attempt_id)
    if known_smoke is None:
        # Backward compatibility for full-run artifacts that predate explicit
        # attempt metadata.
        return smoke is False
    return known_smoke is smoke


def _pointer_matches_smoke(parent: Path, payload: dict[str, Any], smoke: bool | None) -> bool:
    """Return whether a latest-pointer payload matches a smoke lineage."""

    attempt_id = str(payload.get("attempt_id") or "")
    if not attempt_id or not (parent / attempt_id).is_dir():
        return False
    if smoke is None:
        return True
    if "smoke" in payload:
        return bool(payload["smoke"]) is smoke
    return _attempt_matches_smoke(parent, attempt_id, smoke)


def latest_attempt_id(parent: str | Path, *, smoke: bool | None = None) -> str | None:
    """Return the preferred latest attempt id under ``parent``.

    Smoke/full identity is read from pointer payloads or
    ``attempt_metadata.json``. Attempt ids are names only and are not parsed for
    lineage.
    """

    parent = Path(parent)
    pointer_names = ["latest-smoke.json"] if smoke is True else ["latest.json"]
    for pointer_name in pointer_names:
        payload = _latest_payload(parent, filename=pointer_name)
        if payload is not None and _pointer_matches_smoke(parent, payload, smoke):
            return str(payload["attempt_id"])
    if smoke is False:
        payload = _latest_payload(parent, filename="latest-full.json")
        if payload is not None and _pointer_matches_smoke(parent, payload, smoke):
            return str(payload["attempt_id"])
    candidates = [
        attempt_id
        for attempt_id in attempt_ids(parent)
        if _attempt_matches_smoke(parent, attempt_id, smoke)
    ]
    return candidates[-1] if candidates else None


def _write_attempt_metadata(stage_path: Path, attempt_id: str, *, smoke: bool) -> None:
    """Record attempt lineage metadata independent of its name."""

    write_json(
        stage_path / str(attempt_id) / ATTEMPT_METADATA,
        {"attempt_id": str(attempt_id), "smoke": bool(smoke)},
    )


def _write_latest_pointer(stage_path: Path, filename: str, attempt_id: str, *, smoke: bool) -> None:
    """Write one portable latest pointer."""

    write_json(
        stage_path / filename,
        {"attempt_id": str(attempt_id), "path": str(attempt_id), "smoke": bool(smoke)},
    )


def _write_latest_symlink(stage_path: Path, link_name: str, attempt_id: str) -> None:
    """Best-effort latest symlink update."""

    link = stage_path / link_name
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(attempt_id, target_is_directory=True)
    except OSError:
        # Symlinks may be unsupported on the target filesystem; JSON pointers suffice.
        pass


def _write_primary_latest(stage_path: Path, attempt_id: str, *, smoke: bool) -> None:
    """Write the primary latest pointer and symlink."""

    _write_latest_pointer(stage_path, "latest.json", attempt_id, smoke=smoke)
    _write_latest_symlink(stage_path, "latest", attempt_id)


def smoke_attempt_id(base_attempt_id: str) -> str:
    """Return a human-readable smoke attempt name.

    This is a naming convention only; smoke/full lineage is recorded in
    ``attempt_metadata.json`` and latest-pointer payloads.
    """

    return base_attempt_id if base_attempt_id.endswith("-smoke") else f"{base_attempt_id}-smoke"


def write_latest(stage_path: Path, attempt_id: str, *, smoke: bool = False) -> None:
    """Record latest attempt ids under ``stage_path``.

    ``latest.json`` remains the normal full-run pointer whenever a full attempt
    exists. Smoke attempts update ``latest-smoke.json`` and update
    ``latest.json`` only when no full attempt is known, so a smoke diagnostic
    cannot silently become the default upstream input for a later real run.
    """

    stage_path = Path(stage_path)
    _write_attempt_metadata(stage_path, attempt_id, smoke=smoke)
    if smoke:
        _write_latest_pointer(stage_path, "latest-smoke.json", attempt_id, smoke=True)
        _write_latest_symlink(stage_path, "latest-smoke", attempt_id)
        if latest_attempt_id(stage_path, smoke=False) is None:
            _write_primary_latest(stage_path, attempt_id, smoke=True)
        return
    _write_latest_pointer(stage_path, "latest-full.json", attempt_id, smoke=False)
    _write_primary_latest(stage_path, attempt_id, smoke=False)
