"""Guard pair-stability-v4 writable roots independently of legacy stages."""

from __future__ import annotations

import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from strict_data import StrictDataError, load_json

STUDY_NAME = "pair_stability_v4"
ROOT_SCHEMA_VERSION = "pair-stability-v4/root/v1"
ROOT_SENTINEL = ".pair_stability_v4-root.json"
PURPOSE_EXPERIMENT = "experiment"
PURPOSE_OWNERSHIP_AUDIT = "ownership_audit"
ROOT_PURPOSES = frozenset({PURPOSE_EXPERIMENT, PURPOSE_OWNERSHIP_AUDIT})
ATTEMPT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
V3_STUDY_DIR = STUDY_DIR.parent / "pair_stability_v3"
V4_RESULTS_DIR = STUDY_DIR / "results"


def validate_lineage_id(lineage_id: str) -> str:
    """Return one filesystem-safe lineage identifier."""

    value = str(lineage_id)
    if not ATTEMPT_PATTERN.fullmatch(value):
        raise ValueError(
            "lineage/attempt ids must start with an alphanumeric character "
            "and contain only letters, digits, '.', '_', '+', or '-'"
        )
    return value


def initialize_root(
    path: Path,
    *,
    lineage_id: str,
    purpose: str = PURPOSE_EXPERIMENT,
) -> Path:
    """Create and identify one new pair-stability-v4 results root."""

    requested = _validate_requested_root(path)
    lineage_id = validate_lineage_id(lineage_id)
    purpose = _validate_purpose(purpose)

    if requested.exists():
        if requested.is_symlink():
            raise ValueError(f"v4 results root may not be a symlink: {requested}")
        if not requested.is_dir():
            raise ValueError(f"v4 results root must be a directory: {requested}")
        sentinel = requested / ROOT_SENTINEL
        if sentinel.exists():
            raise ValueError(
                f"refusing to reuse initialized v4 results root: {requested}"
            )
        if any(requested.iterdir()):
            raise ValueError(
                f"refusing nonempty directory without {ROOT_SENTINEL}: {requested}"
            )
    else:
        requested.mkdir(parents=True)

    canonical = requested.resolve(strict=True)
    _validate_canonical_root(canonical)
    payload = {
        "schema_version": ROOT_SCHEMA_VERSION,
        "study": STUDY_NAME,
        "canonical_root": str(canonical),
        "lineage_id": lineage_id,
        "purpose": purpose,
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(
            timespec="seconds"
        ),
    }
    _write_new_json(canonical / ROOT_SENTINEL, payload)
    return canonical


def require_v4_root(
    path: Path,
    *,
    lineage_id: str | None = None,
    purpose: str | None = None,
) -> Path:
    """Resolve and validate one existing pair-stability-v4 results root."""

    requested = _validate_requested_root(path)
    if requested.is_symlink():
        raise ValueError(f"v4 results root may not be a symlink: {requested}")
    canonical = requested.resolve(strict=True)
    _validate_canonical_root(canonical)
    if not canonical.is_dir():
        raise ValueError(f"v4 results root must be a directory: {canonical}")

    sentinel_path = canonical / ROOT_SENTINEL
    payload = _read_json_object(sentinel_path)
    expected = {
        "schema_version": ROOT_SCHEMA_VERSION,
        "study": STUDY_NAME,
        "canonical_root": str(canonical),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"invalid v4 root sentinel {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )

    sentinel_lineage = validate_lineage_id(str(payload.get("lineage_id") or ""))
    sentinel_purpose = _validate_purpose(str(payload.get("purpose") or ""))
    if lineage_id is not None and sentinel_lineage != validate_lineage_id(lineage_id):
        raise ValueError(
            f"v4 root belongs to lineage {sentinel_lineage!r}, "
            f"not {lineage_id!r}"
        )
    if purpose is not None and sentinel_purpose != _validate_purpose(purpose):
        raise ValueError(
            f"v4 root purpose is {sentinel_purpose!r}, not {purpose!r}"
        )
    return canonical


def root_metadata(path: Path) -> dict[str, Any]:
    """Return validated sentinel metadata for one v4 root."""

    root = require_v4_root(path)
    return _read_json_object(root / ROOT_SENTINEL)


def require_beneath_root(path: Path, root: Path) -> Path:
    """Resolve an artifact path and reject root, traversal, or symlink escape."""

    root = require_v4_root(root)
    candidate = Path(path)
    if ".." in candidate.parts:
        raise ValueError(f"artifact path contains traversal: {candidate}")
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"artifact path escapes v4 root: {candidate}")
    return resolved


def validate_root_links(root: Path) -> tuple[str, ...]:
    """Return internal links that resolve outside the guarded v4 root."""

    root = require_v4_root(root)
    unsafe: list[str] = []
    for current, directory_names, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in (*directory_names, *filenames):
            candidate = current_path / name
            if not candidate.is_symlink():
                continue
            resolved = candidate.resolve(strict=False)
            if resolved == root or root not in resolved.parents:
                unsafe.append(str(candidate.relative_to(root)))
    return tuple(sorted(unsafe))


def _validate_requested_root(path: Path) -> Path:
    requested = Path(path)
    if not requested.is_absolute():
        raise ValueError(f"v4 results root must be absolute: {requested}")
    if ".." in requested.parts:
        raise ValueError(f"v4 results root contains traversal: {requested}")
    repo = REPO_ROOT.resolve()
    if requested == repo or repo in requested.parents:
        v4_results = V4_RESULTS_DIR.absolute()
        if requested != v4_results and v4_results not in requested.parents:
            raise ValueError(
                "in-repository v4 roots must be the pair_stability_v4/results "
                f"directory or a descendant: {requested}"
            )
        resolved = requested.resolve(strict=False)
        if resolved != repo and repo not in resolved.parents:
            raise ValueError(
                f"in-repository root resolves outside repository: {requested}"
            )
    return requested


def _validate_canonical_root(root: Path) -> None:
    forbidden_exact = {
        Path("/"),
        Path.home().resolve(),
        REPO_ROOT.resolve(),
        STUDY_DIR.resolve(),
        V3_STUDY_DIR.resolve(),
    }
    if root in forbidden_exact:
        raise ValueError(f"refusing broad or source-owned results root: {root}")
    v3 = V3_STUDY_DIR.resolve()
    if v3 in root.parents:
        raise ValueError(f"refusing results root below legacy v3 source: {root}")
    repo = REPO_ROOT.resolve()
    if root == repo or repo in root.parents:
        v4_results = V4_RESULTS_DIR.resolve(strict=False)
        if root != v4_results and v4_results not in root.parents:
            raise ValueError(
                "in-repository v4 roots must be the pair_stability_v4/results "
                f"directory or a descendant: {root}"
            )


def _validate_purpose(purpose: str) -> str:
    value = str(purpose)
    if value not in ROOT_PURPOSES:
        raise ValueError(
            f"unknown v4 root purpose {value!r}; expected one of "
            f"{', '.join(sorted(ROOT_PURPOSES))}"
        )
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"missing v4 root sentinel: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"missing v4 root sentinel: {path}")
    try:
        value = load_json(path)
    except (OSError, StrictDataError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
