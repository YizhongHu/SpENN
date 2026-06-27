"""Config metadata helpers for staged studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
