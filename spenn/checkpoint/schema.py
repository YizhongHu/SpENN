"""Checkpoint schema dispatch."""

from __future__ import annotations

import json
from pathlib import Path

from .manifest import CHECKPOINT_KIND, CHECKPOINT_SCHEMA_VERSION, CheckpointManifest

SUPPORTED_SCHEMA_VERSIONS = frozenset({CHECKPOINT_SCHEMA_VERSION})


def read_manifest(path: str | Path) -> CheckpointManifest:
    """Read and validate a checkpoint manifest."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    manifest = CheckpointManifest.from_mapping(data)
    validate_manifest_schema(manifest, path=manifest_path)
    return manifest


def validate_manifest_schema(manifest: CheckpointManifest, *, path: Path | None = None) -> None:
    """Fail if `manifest` is not a supported SpENN checkpoint schema."""

    label = "" if path is None else f"{path}: "
    if manifest.kind != CHECKPOINT_KIND:
        raise ValueError(f"{label}unsupported checkpoint kind {manifest.kind!r}")
    if manifest.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{label}unsupported checkpoint schema_version {manifest.schema_version}; "
            f"supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
