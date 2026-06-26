"""Resource profiles for replaceable experiment executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ResourceSpec:
    """Executor-facing resource request for one task."""

    profile: str
    device: str
    partition: str | None = None
    threads: int | None = None
    mem_gb: int | None = None
    gpus: int | None = None
    timeout_min: int | None = None
    uv_environment: str | None = None
    uv_extras: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "profile": self.profile,
            "device": self.device,
            "partition": self.partition,
            "threads": self.threads,
            "mem_gb": self.mem_gb,
            "gpus": self.gpus,
            "timeout_min": self.timeout_min,
            "uv_environment": self.uv_environment,
            "uv_extras": list(self.uv_extras),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ResourceSpec":
        """Build a resource spec from serialized data."""

        data = data or {}
        return cls(
            profile=str(data.get("profile") or "cpu"),
            device=str(data.get("device") or data.get("profile") or "cpu"),
            partition=_optional_str(data.get("partition")),
            threads=_optional_int(data.get("threads")),
            mem_gb=_optional_int(data.get("mem_gb")),
            gpus=_optional_int(data.get("gpus")),
            timeout_min=_optional_int(data.get("timeout_min")),
            uv_environment=_optional_str(data.get("uv_environment")),
            uv_extras=tuple(str(item) for item in data.get("uv_extras", ()) or ()),
            metadata=_mapping(data.get("metadata")),
        )


def resource_from_profile(
    *,
    profile: str,
    partition: str | None,
    timeout_min: int | None,
    mem_gb: int | None,
    cpus: int | None,
    gpus: int | None,
    uv_environment: str | None,
    uv_extras: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> ResourceSpec:
    """Create a ``ResourceSpec`` from resolved launcher profile values."""

    device = "cuda" if profile == "cuda" else "cpu"
    return ResourceSpec(
        profile=str(profile),
        device=device,
        partition=partition,
        threads=cpus,
        mem_gb=mem_gb,
        gpus=gpus,
        timeout_min=timeout_min,
        uv_environment=uv_environment,
        uv_extras=tuple(str(extra) for extra in uv_extras),
        metadata=dict(metadata or {}),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
