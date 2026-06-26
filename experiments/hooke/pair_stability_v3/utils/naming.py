"""Study naming and axis-id helpers."""

from __future__ import annotations

import re
from typing import Any, Sequence

SCAN_SEED_AXIS = "seed"
DEFAULT_STUDY_NAME = "study"


def format_lr(lr: float) -> str:
    """Return a compact, deterministic learning-rate label, e.g. ``1e-3``."""

    mantissa, _, exponent = f"{float(lr):.1e}".partition("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


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
