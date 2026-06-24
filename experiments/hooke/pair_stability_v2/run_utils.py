"""Shared utilities for staged study scripts.

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
from typing import Any, Sequence
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

SCAN_SEED_AXIS = "seed"
DEFAULT_SEED_OVERRIDES = {
    "scan_train": {
        "run_parameters.seed": "scan_seed",
        "runtime.seed": "scan_seed",
        "sampler.seed": "scan_seed",
    },
    "validation": {
        "run_parameters.seed": "scan_seed",
        "runtime.seed": "scan_seed",
        "evaluation.seed": "scan_seed",
    },
    "final_train": {
        "run_parameters.seed": "final_train_model_seed",
        "runtime.seed": "final_train_model_seed",
        "sampler.seed": "final_train_sampler_seed",
    },
    "final_eval": {
        "run_parameters.seed": "final_eval_seed",
        "runtime.seed": "final_eval_seed",
        "evaluation.seed": "final_eval_seed",
    },
}
DEFAULT_FINAL_SEED_SEQUENCES = {
    "final_train_sampler_seed": {"start": 101, "step": 1},
    "final_train_model_seed": {"start": 1001, "step": 1},
    "final_eval_seed": {"start": 10001, "step": 1},
}
DEFAULT_STUDY_NAME = "study"
DEFAULT_CONFIG_SNAPSHOTS = {
    "train": "train_config.yaml",
    "validation": "validation_config.yaml",
}


def config_snapshot_names(configured: Any | None = None) -> dict[str, str]:
    """Return stage -> grid-attempt config snapshot filename."""

    source = DEFAULT_CONFIG_SNAPSHOTS if configured is None else configured
    if not isinstance(source, dict):
        raise ValueError("config_snapshots must be a mapping")
    snapshots = {str(stage): str(filename) for stage, filename in source.items()}
    for stage, filename in snapshots.items():
        if not filename or Path(filename).name != filename:
            raise ValueError(f"config_snapshots.{stage} must be a plain filename")
    return snapshots


def study_name(value: Any | None = None) -> str:
    """Return a normalized non-empty study name."""

    text = str(DEFAULT_STUDY_NAME if value is None else value).strip()
    return text or DEFAULT_STUDY_NAME


def study_name_from_manifest(manifest: dict[str, Any] | None) -> str:
    """Return the study name recorded in a stage manifest."""

    if isinstance(manifest, dict):
        return study_name(manifest.get("study"))
    return study_name()


def _safe_name(value: Any, *, separator: str) -> str:
    """Return ``value`` as a scheduler/config-safe single component."""

    text = study_name(value)
    text = text.replace("_", separator).replace("-", separator)
    text = re.sub(r"[^A-Za-z0-9]+", separator, text)
    text = re.sub(f"{re.escape(separator)}+", separator, text).strip(separator)
    return text or "study"


def log_prefix(study: Any | None = None) -> str:
    """Return the console log prefix for a study."""

    return f"[{study_name(study)}]"


def stage_job_name(study: Any, stage: str, *, smoke: bool = False) -> str:
    """Return a Slurm job name derived from study and stage."""

    suffix = f"{stage}-smoke" if smoke else stage
    return f"{_safe_name(study, separator='-')}-{suffix}"


def experiment_run_name(study: Any, stage: str) -> str:
    """Return a Hydra experiment.run_name derived from study and stage."""

    return f"{_safe_name(study, separator='_')}_{stage}"


def axis_value_label(value: Any) -> str:
    """Return a compact, deterministic axis-value label for ids."""

    if isinstance(value, float):
        return format_lr(value)
    text = str(value)
    try:
        numeric = float(text)
    except ValueError:
        return text
    if any(marker in text.lower() for marker in (".", "e")):
        return format_lr(numeric)
    return text


def axis_id_labels_from_manifest(
    manifest: dict[str, Any] | None,
    axes: Sequence[str],
) -> dict[str, str]:
    """Return axis -> id-label mapping from a manifest."""

    configured = manifest.get("axis_id_labels") if isinstance(manifest, dict) else None
    if not isinstance(configured, dict):
        configured = {}
    return {axis: str(configured.get(axis, axis)) for axis in axes}


def id_for_axes(
    point: dict[str, Any],
    axes: Sequence[str],
    labels: dict[str, str] | None = None,
) -> str:
    """Return a deterministic id from configured axes."""

    labels = labels or {}
    return "_".join(
        f"{labels.get(axis, axis)}-{axis_value_label(point.get(axis, ''))}"
        for axis in axes
    )


def grid_axes_from_manifest(manifest: dict[str, Any] | None) -> dict[str, tuple[str, ...] | str]:
    """Return major/minor/seed axis metadata from a grid or final-grid manifest."""

    manifest = manifest or {}
    seed_axis = str(manifest.get("scan_seed_axis", SCAN_SEED_AXIS))
    if "major_axes" in manifest or "minor_axes" in manifest:
        major_axes = tuple(str(axis) for axis in manifest.get("major_axes", ()))
        minor_axes = tuple(str(axis) for axis in manifest.get("minor_axes", ()))
    else:
        axes = tuple(str(axis) for axis in manifest.get("grid_axes", ()))
        major_axes = tuple(axis for axis in axes if axis != seed_axis)
        minor_axes = ()
    return {
        "major_axes": major_axes,
        "minor_axes": minor_axes,
        "scan_seed_axis": seed_axis,
        "config_axes": (*major_axes, *minor_axes),
        "run_axes": (*major_axes, *minor_axes, seed_axis),
    }


def seed_override_policy(configured: Any | None = None) -> dict[str, dict[str, str]]:
    """Return normalized stage -> override path -> named seed mapping."""

    source = DEFAULT_SEED_OVERRIDES if configured is None else configured
    if not isinstance(source, dict):
        raise ValueError("seed_overrides must be a mapping")
    policy: dict[str, dict[str, str]] = {}
    for stage, overrides in source.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"seed_overrides.{stage} must be a mapping")
        policy[str(stage)] = {str(path): str(seed_name) for path, seed_name in overrides.items()}
    return policy


def seed_override_values(
    policy: dict[str, dict[str, str]] | None,
    stage: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Resolve configured seed overrides for ``stage`` from named seed values."""

    resolved_policy = seed_override_policy(policy)
    overrides = resolved_policy.get(stage, {})
    resolved = {}
    for path, seed_name in overrides.items():
        if seed_name not in values:
            raise KeyError(f"seed policy for {stage!r} references missing seed {seed_name!r}")
        resolved[path] = values[seed_name]
    return resolved


def final_seed_sequences(configured: Any | None = None) -> dict[str, dict[str, int]]:
    """Return normalized final seed sequence specs."""

    source = DEFAULT_FINAL_SEED_SEQUENCES if configured is None else configured
    if not isinstance(source, dict):
        raise ValueError("final_seed_sequences must be a mapping")
    sequences: dict[str, dict[str, int]] = {}
    for name, spec in source.items():
        if not isinstance(spec, dict):
            raise ValueError(f"final_seed_sequences.{name} must be a mapping")
        sequences[str(name)] = {
            "start": int(spec.get("start", 0)),
            "step": int(spec.get("step", 1)),
        }
    return sequences


def final_seed_values(
    sequences: dict[str, dict[str, int]] | None,
    replicate_index: int,
) -> dict[str, int]:
    """Return named final seeds for one replicate index."""

    resolved_sequences = final_seed_sequences(sequences)
    return {
        name: int(spec["start"]) + int(replicate_index) * int(spec["step"])
        for name, spec in resolved_sequences.items()
    }


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
