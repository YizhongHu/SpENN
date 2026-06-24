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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SourceGrid:
    """Resolved source ``00_grid`` attempt for a downstream artifact."""

    attempt_id: str
    attempt_dir: Path
    manifest_path: Path

    def to_record(self) -> dict[str, str]:
        """Return a JSON-safe provenance record."""

        return {
            "grid_attempt_id": self.attempt_id,
            "grid_attempt_dir": str(self.attempt_dir),
            "manifest_path": str(self.manifest_path),
        }

    def read_manifest(self) -> dict[str, Any]:
        """Read this grid attempt's routine manifest.

        The manifest contains blinded alias values when the source attempt was
        planned blinded. The semantic mapping remains isolated in
        ``unblind.json`` and is not read here.
        """

        manifest = read_json(self.manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError(f"grid manifest must be a JSON object: {self.manifest_path}")
        return manifest


@dataclass(frozen=True)
class SourceAncestry:
    """Result roots traced through stage provenance records."""

    roots: frozenset[Path]
    warnings: tuple[str, ...] = ()


def read_json_object(path: str | Path, warnings: list[str] | None = None) -> dict[str, Any]:
    """Read a JSON object, optionally recording missing/invalid input as warnings."""

    path = Path(path)
    if not path.is_file():
        if warnings is not None:
            warnings.append(f"missing JSON file: {path}")
            return {}
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if warnings is not None:
            warnings.append(f"{path}: invalid JSON: {exc}")
            return {}
        raise
    if not isinstance(payload, dict):
        message = f"{path}: expected JSON object"
        if warnings is not None:
            warnings.append(message)
            return {}
        raise ValueError(message)
    return payload


def read_json_object_list(
    path: str | Path,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Read a JSON list of objects, optionally recording problems as warnings."""

    path = Path(path)
    if not path.is_file():
        if warnings is not None:
            warnings.append(f"missing JSON file: {path}")
            return []
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if warnings is not None:
            warnings.append(f"{path}: invalid JSON: {exc}")
            return []
        raise
    if not isinstance(payload, list):
        message = f"{path}: expected JSON list"
        if warnings is not None:
            warnings.append(message)
            return []
        raise ValueError(message)
    return [item for item in payload if isinstance(item, dict)]


def path_from_record(record: dict[str, Any], key: str) -> Path | None:
    """Return an absolute path from a provenance record field."""

    raw = record.get(key)
    if raw in (None, ""):
        return None
    return Path(str(raw)).resolve()


def source_grid_from_id(results_root: str | Path, grid_attempt_id: str) -> SourceGrid:
    """Return the ``00_grid`` source descriptor for ``grid_attempt_id``."""

    attempt_id = str(grid_attempt_id)
    attempt_dir = grid_attempt_dir(results_root, attempt_id).resolve()
    return SourceGrid(
        attempt_id=attempt_id,
        attempt_dir=attempt_dir,
        manifest_path=(attempt_dir / "manifest.json").resolve(),
    )


def source_grid_from_record(
    results_root: str | Path,
    record: dict[str, Any],
    *,
    warnings: list[str] | None = None,
) -> SourceGrid | None:
    """Resolve a source-grid record into a ``SourceGrid`` descriptor."""

    attempt_id = str(record.get("grid_attempt_id") or "").strip()
    attempt_dir = path_from_record(record, "grid_attempt_dir")
    if not attempt_id and attempt_dir is not None:
        attempt_id = attempt_dir.name
    if not attempt_id:
        return None
    source = source_grid_from_id(results_root, attempt_id)
    if attempt_dir is not None:
        source = SourceGrid(
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
            manifest_path=path_from_record(record, "manifest_path") or (attempt_dir / "manifest.json").resolve(),
        )
    if warnings is not None and not source.attempt_dir.is_dir():
        warnings.append(f"missing grid directory: {source.attempt_dir}")
    if warnings is not None and not source.manifest_path.is_file():
        warnings.append(f"missing grid manifest: {source.manifest_path}")
    return source


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
        # Backward compatibility for existing full-run artifacts that predate
        # explicit attempt metadata.
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


# ---------------------------------------------------------------------------
# Source ancestry
# ---------------------------------------------------------------------------
def source_grid_from_attempt(
    results_root: str | Path,
    attempt_dir: str | Path,
    *,
    warnings: list[str] | None = None,
) -> SourceGrid | None:
    """Trace an attempt's provenance back to its source ``00_grid`` attempt.

    The traversal follows explicit stage provenance records and stops at the
    routine grid manifest. It does not read ``unblind.json``.
    """

    results_root = Path(results_root).resolve()
    return _source_grid_from_attempt(results_root, Path(attempt_dir).resolve(), warnings=warnings, seen=set())


def trace_source_ancestry(
    results_root: str | Path,
    attempt_dir: str | Path,
    *,
    include_scan_run_roots: bool = True,
) -> SourceAncestry:
    """Trace result roots reachable from a stage attempt's provenance records."""

    roots: set[Path] = set()
    warnings: list[str] = []
    _trace_attempt_roots(
        Path(results_root).resolve(),
        Path(attempt_dir).resolve(),
        roots,
        warnings,
        include_scan_run_roots=include_scan_run_roots,
        seen=set(),
    )
    return SourceAncestry(frozenset(roots), tuple(warnings))


def _source_grid_from_attempt(
    results_root: Path,
    attempt_dir: Path,
    *,
    warnings: list[str] | None,
    seen: set[Path],
) -> SourceGrid | None:
    attempt_dir = attempt_dir.resolve()
    if attempt_dir in seen:
        return None
    seen.add(attempt_dir)

    if _stage_name(attempt_dir, results_root) == STAGE_GRID:
        return source_grid_from_id(results_root, attempt_dir.name)

    direct = _source_grid_from_direct_file(results_root, attempt_dir / "source_grid_attempt.json", warnings=warnings)
    if direct is not None:
        return direct

    train_source = _source_grid_from_direct_file(
        results_root,
        attempt_dir / "source_train_attempt.json",
        warnings=None,
    )
    if train_source is not None:
        return train_source

    source_train = _read_optional_object(attempt_dir / "source_train_attempt.json", warnings=warnings)
    train_dir = path_from_record(source_train, "train_attempt_dir")
    if train_dir is not None:
        source = _source_grid_from_attempt(results_root, train_dir, warnings=warnings, seen=seen)
        if source is not None:
            return source

    for filename, path_key, id_key, stage in (
        ("source_collection_attempt.json", "collection_attempt_dir", "collection_attempt_id", STAGE_COLLECT),
        ("source_selection_attempt.json", "selection_attempt_dir", "selection_attempt_id", STAGE_SELECT),
        ("source_final_grid_attempt.json", "final_grid_attempt_dir", "final_grid_attempt_id", STAGE_FINAL_GRID),
        ("source_final_train_attempt.json", "final_train_attempt_dir", "final_train_attempt_id", STAGE_FINAL_TRAIN),
    ):
        record = _read_optional_object(attempt_dir / filename, warnings=warnings)
        upstream = _upstream_attempt_dir(results_root, record, path_key=path_key, id_key=id_key, stage=stage)
        if upstream is None:
            continue
        source = _source_grid_from_attempt(results_root, upstream, warnings=warnings, seen=seen)
        if source is not None:
            return source

    for source in _read_optional_object_list(attempt_dir / "source_validation_attempts.json", warnings=warnings):
        validation_dir = path_from_record(source, "validation_attempt_dir")
        if validation_dir is None:
            continue
        source_grid = _source_grid_from_attempt(results_root, validation_dir, warnings=warnings, seen=seen)
        if source_grid is not None:
            return source_grid
    return None


def _source_grid_from_direct_file(
    results_root: Path,
    path: Path,
    *,
    warnings: list[str] | None,
) -> SourceGrid | None:
    if not path.is_file():
        return None
    return source_grid_from_record(results_root, read_json_object(path, warnings=warnings), warnings=warnings)


def _trace_attempt_roots(
    results_root: Path,
    attempt_dir: Path,
    roots: set[Path],
    warnings: list[str],
    *,
    include_scan_run_roots: bool,
    seen: set[Path],
) -> None:
    attempt_dir = attempt_dir.resolve()
    if attempt_dir in seen:
        return
    seen.add(attempt_dir)

    stage = _stage_name(attempt_dir, results_root)
    if stage == STAGE_GRID:
        _add_existing_root(roots, warnings, attempt_dir, "grid")
        return
    if stage not in {STAGE_TRAIN, STAGE_VALIDATION} or include_scan_run_roots:
        _add_existing_root(roots, warnings, attempt_dir, stage or "attempt")
    elif not attempt_dir.is_dir():
        warnings.append(f"missing {stage or 'attempt'} directory: {attempt_dir}")
        return

    direct_grid = _source_grid_from_direct_file(results_root, attempt_dir / "source_grid_attempt.json", warnings=warnings)
    if direct_grid is not None:
        _add_existing_root(roots, warnings, direct_grid.attempt_dir, "grid")

    source_train = _read_optional_object(attempt_dir / "source_train_attempt.json", warnings=warnings)
    direct_from_train = source_grid_from_record(results_root, source_train, warnings=warnings)
    if direct_from_train is not None:
        _add_existing_root(roots, warnings, direct_from_train.attempt_dir, "grid")
    train_dir = path_from_record(source_train, "train_attempt_dir")
    if train_dir is not None:
        _trace_attempt_roots(
            results_root,
            train_dir,
            roots,
            warnings,
            include_scan_run_roots=include_scan_run_roots,
            seen=seen,
        )

    for filename, path_key, id_key, upstream_stage in (
        ("source_collection_attempt.json", "collection_attempt_dir", "collection_attempt_id", STAGE_COLLECT),
        ("source_selection_attempt.json", "selection_attempt_dir", "selection_attempt_id", STAGE_SELECT),
        ("source_final_grid_attempt.json", "final_grid_attempt_dir", "final_grid_attempt_id", STAGE_FINAL_GRID),
        ("source_final_train_attempt.json", "final_train_attempt_dir", "final_train_attempt_id", STAGE_FINAL_TRAIN),
    ):
        record = _read_optional_object(attempt_dir / filename, warnings=warnings)
        upstream = _upstream_attempt_dir(results_root, record, path_key=path_key, id_key=id_key, stage=upstream_stage)
        if upstream is None:
            continue
        _trace_attempt_roots(
            results_root,
            upstream,
            roots,
            warnings,
            include_scan_run_roots=include_scan_run_roots,
            seen=seen,
        )

    for source in _read_optional_object_list(attempt_dir / "source_validation_attempts.json", warnings=warnings):
        validation_dir = path_from_record(source, "validation_attempt_dir")
        if validation_dir is None:
            continue
        _trace_attempt_roots(
            results_root,
            validation_dir,
            roots,
            warnings,
            include_scan_run_roots=include_scan_run_roots,
            seen=seen,
        )


def _read_optional_object(path: Path, *, warnings: list[str] | None) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return read_json_object(path, warnings=warnings)


def _read_optional_object_list(path: Path, *, warnings: list[str] | None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return read_json_object_list(path, warnings=warnings)


def _upstream_attempt_dir(
    results_root: Path,
    record: dict[str, Any],
    *,
    path_key: str,
    id_key: str,
    stage: str,
) -> Path | None:
    path = path_from_record(record, path_key)
    if path is not None:
        return path
    attempt_id = str(record.get(id_key) or "").strip()
    if not attempt_id:
        return None
    return stage_dir(results_root, stage) / attempt_id


def _stage_name(path: Path, results_root: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(results_root.resolve())
    except ValueError:
        return None
    return relative.parts[0] if relative.parts else None


def _add_existing_root(roots: set[Path], warnings: list[str], path: Path, label: str) -> bool:
    path = path.resolve()
    if path.is_dir():
        roots.add(path)
        return True
    warnings.append(f"missing {label} directory: {path}")
    return False
