"""Checkpoint manifest schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from spenn.artifacts import write_json

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "spenn.checkpoint"


@dataclass(frozen=True)
class CheckpointManifest:
    """Readable metadata for one checkpoint directory."""

    schema_version: int
    kind: str
    step: int
    created_at_unix: float
    files: dict[str, str]
    hashes: dict[str, str | None]
    runtime: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest mapping."""

        return {
            "schema_version": int(self.schema_version),
            "kind": self.kind,
            "step": int(self.step),
            "created_at_unix": float(self.created_at_unix),
            "files": dict(self.files),
            "hashes": dict(self.hashes),
            "runtime": dict(self.runtime),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CheckpointManifest":
        """Build a manifest from a loaded JSON mapping."""

        return cls(
            schema_version=int(data["schema_version"]),
            kind=str(data["kind"]),
            step=int(data["step"]),
            created_at_unix=float(data["created_at_unix"]),
            files={str(key): str(value) for key, value in dict(data["files"]).items()},
            hashes={str(key): value for key, value in dict(data.get("hashes", {})).items()},
            runtime=dict(data.get("runtime", {})),
            provenance=dict(data.get("provenance", {})),
        )

    def write(self, path: str | Path) -> None:
        """Write the manifest JSON."""

        write_json(Path(path), self.to_dict())
