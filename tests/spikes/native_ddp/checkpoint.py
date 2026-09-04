"""DCP payload storage and single-coordinator publication for the spike.

Model and optimizer state cross the distributed checkpoint API.  Sampler and
process RNG state remain rank-local sidecars because they are owned by one
worker and must not be reconstructed from a global application state dict.
The publication protocol is intentionally local to this package and does not
modify TPEN's production checkpoint path.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from tests.spikes.native_ddp.model_access import SemanticWavefunction
from tests.spikes.native_ddp.runtime import DistributedRuntime


class CheckpointTopologyMismatch(RuntimeError):
    """Raised before DCP mutation when a checkpoint's world size differs."""


class CheckpointCorrupt(RuntimeError):
    """Raised when a rank-local sidecar fails its recorded digest."""


@dataclass(frozen=True)
class FileDigest:
    """Closed-file size and SHA-256 digest used by publication validation."""

    relative_path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CheckpointPayloadStore:
    """Proposed typed native checkpoint adapter surface."""

    root: Path
    runtime: DistributedRuntime

    def save(
        self,
        model: SemanticWavefunction,
        optimizer: torch.optim.Optimizer,
        *,
        generation: int,
        sampler_state: dict[str, Any],
        rng_state: dict[str, Any],
        completed_updates: int,
        failure_rank: int | None = None,
        delay_rank: int | None = None,
        delay_seconds: float = 0.0,
    ) -> Path:
        """DCP-save payloads, validate every rank sidecar, then publish once."""

        if generation < 1:
            raise ValueError("checkpoint generation must be positive")
        stage = self.root / "staging" / f"gen-{generation:06d}"
        final = self.root / "generations" / f"gen-{generation:06d}"
        if self.runtime.rank == 0:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "staging").mkdir(parents=True, exist_ok=True)
            (self.root / "generations").mkdir(parents=True, exist_ok=True)
            if stage.exists() or final.exists():
                raise FileExistsError(f"checkpoint generation already exists: {generation}")
            stage.mkdir()
        self.runtime.barrier()

        model_state, optimizer_state = get_state_dict(model, optimizer)
        dcp.save(
            {"model": model_state, "optimizer": optimizer_state},
            storage_writer=dcp.FileSystemWriter(stage / "dcp"),
        )

        if self.runtime.rank == failure_rank:
            os._exit(7)
        if self.runtime.rank == delay_rank:
            import time

            time.sleep(delay_seconds)

        sidecar = stage / "sidecars" / f"rank-{self.runtime.rank:05d}.json"
        _atomic_write_json(
            sidecar,
            {
                "rank": self.runtime.rank,
                "world_size": self.runtime.world_size,
                "completed_updates": completed_updates,
                "sampler_state": _jsonable(sampler_state),
                "rng_state": _jsonable(rng_state),
            },
        )
        local_digest = _digest(sidecar, root=stage)
        self.runtime.barrier()
        gathered_digests = self.runtime.all_gather_objects(local_digest.as_dict())

        if self.runtime.rank == 0:
            for rank, payload in enumerate(gathered_digests):
                expected_path = stage / "sidecars" / f"rank-{rank:05d}.json"
                actual = _digest(expected_path, root=stage)
                if payload != actual.as_dict():
                    raise CheckpointCorrupt(
                        f"rank {rank} sidecar digest changed before publication"
                    )
            manifest = {
                "generation": generation,
                "world_size": self.runtime.world_size,
                "completed_updates": completed_updates,
                "publisher_rank": 0,
                "canonical_model_keys": list(model.state_dict().keys()),
                "files": [digest.as_dict() for digest in _file_digests(stage)],
            }
            _atomic_write_json(stage / "manifest.json", manifest)
            (stage / "COMPLETE").write_text("COMPLETE\n")
            stage.rename(final)
            _atomic_write_json(
                self.root / "latest.json",
                {"generation": generation, "path": str(final.relative_to(self.root))},
            )
        self.runtime.barrier()
        return final

    def load(
        self,
        model: SemanticWavefunction,
        optimizer: torch.optim.Optimizer,
        *,
        generation: int | None = None,
    ) -> dict[str, Any]:
        """Validate topology and sidecars, then DCP-load model/optimizer state."""

        metadata = self.runtime.broadcast_object(
            self._read_manifest(generation) if self.runtime.rank == 0 else None
        )
        if int(metadata["world_size"]) != self.runtime.world_size:
            raise CheckpointTopologyMismatch(
                f"checkpoint world_size={metadata['world_size']} does not match "
                f"runtime world_size={self.runtime.world_size}"
            )

        checkpoint_dir = self.root / str(metadata["path"])
        sidecar_path = checkpoint_dir / "sidecars" / f"rank-{self.runtime.rank:05d}.json"
        local_error: str | None = None
        try:
            expected = next(
                item
                for item in metadata["files"]
                if item["relative_path"] == sidecar_path.relative_to(checkpoint_dir).as_posix()
            )
            actual = _digest(sidecar_path, root=checkpoint_dir)
            if actual.as_dict() != expected:
                local_error = f"rank {self.runtime.rank} sidecar digest mismatch"
        except (OSError, StopIteration, ValueError, TypeError) as exc:
            local_error = f"rank {self.runtime.rank} sidecar validation failed: {exc}"

        errors = self.runtime.all_gather_objects(local_error)
        if any(error is not None for error in errors):
            raise CheckpointCorrupt("; ".join(error for error in errors if error is not None))

        model_state, optimizer_state = get_state_dict(model, optimizer)
        dcp.load(
            {"model": model_state, "optimizer": optimizer_state},
            storage_reader=dcp.FileSystemReader(checkpoint_dir / "dcp"),
        )
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state,
            optim_state_dict=optimizer_state,
        )
        sidecar_payload = json.loads(sidecar_path.read_text())
        return {
            "metadata": metadata,
            "sidecar": _from_jsonable(sidecar_payload),
        }

    def _read_manifest(self, generation: int | None) -> dict[str, Any]:
        if generation is None:
            latest = json.loads((self.root / "latest.json").read_text())
            relative_path = str(latest["path"])
        else:
            relative_path = f"generations/gen-{generation:06d}"
        checkpoint_dir = self.root / relative_path
        manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
        manifest["path"] = relative_path
        if not (checkpoint_dir / "COMPLETE").exists():
            raise CheckpointCorrupt(f"checkpoint generation is not complete: {relative_path}")
        return manifest


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, path)


def _digest(path: Path, *, root: Path) -> FileDigest:
    data = path.read_bytes()
    return FileDigest(
        relative_path=path.relative_to(root).as_posix(),
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _file_digests(root: Path) -> tuple[FileDigest, ...]:
    """List only closed regular files under a staging generation."""

    return tuple(
        _digest(path, root=root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "COMPLETE"}
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": value.detach().cpu().tolist(),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_jsonable(item) for item in value]}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__tensor__" in value:
        dtype = {
            "torch.float64": torch.float64,
            "torch.float32": torch.float32,
            "torch.uint8": torch.uint8,
            "torch.int64": torch.int64,
        }[str(value["dtype"])]
        return torch.tensor(value["__tensor__"], dtype=dtype)
    if "__tuple__" in value:
        return tuple(_from_jsonable(item) for item in value["__tuple__"])
    return {key: _from_jsonable(item) for key, item in value.items()}


__all__ = [
    "CheckpointCorrupt",
    "CheckpointPayloadStore",
    "CheckpointTopologyMismatch",
    "FileDigest",
]
