"""Explicit execution topology and rank-local resource profile artifacts.

Topology is supplied by the launcher.  This module deliberately does not read
launcher environment variables or infer a device from a rank.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

from tpen.accelerator import AcceleratorIdentity, AllocatorUnavailable, AllocatorUsage
from tpen.process_resources import ProcessResourceResult, ResourceUnavailable


class ProfileScope(Enum):
    """Ownership scope of one profile record."""

    PROCESS = "process"
    DEVICE = "device"
    NODE = "node"
    JOB = "job"


@dataclass(frozen=True)
class ExecutionTopology:
    """Immutable identity supplied by a launcher for one Python process.

    Rank fields may be ``None`` when the launcher has no value for that rank.
    ``None`` is preserved; it is never replaced by a guessed rank or a local
    rank used as a device index.
    """

    global_rank: int | None
    global_size: int
    local_rank: int | None
    local_size: int
    node_rank: int | None
    node_size: int
    host: str
    pid: int
    device: str
    job_id: str | None = None
    device_identity: AcceleratorIdentity | None = None

    def __post_init__(self) -> None:
        for name, size in (
            ("global_size", self.global_size),
            ("local_size", self.local_size),
            ("node_size", self.node_size),
        ):
            if size < 1:
                raise ValueError(f"{name} must be positive, got {size}")
        for rank_name, rank, size in (
            ("global_rank", self.global_rank, self.global_size),
            ("local_rank", self.local_rank, self.local_size),
            ("node_rank", self.node_rank, self.node_size),
        ):
            if rank is not None and not 0 <= rank < size:
                raise ValueError(f"{rank_name} must be in [0, {size})")
        if not self.host:
            raise ValueError("host must be nonempty")
        if self.pid < 1:
            raise ValueError("pid must be positive")
        if not self.device:
            raise ValueError("device must be nonempty")

    @classmethod
    def single_process(cls, *, device: str, job_id: str | None = None) -> "ExecutionTopology":
        """Create an explicit one-process topology for the local launcher."""

        return cls(
            global_rank=0,
            global_size=1,
            local_rank=0,
            local_size=1,
            node_rank=0,
            node_size=1,
            host=socket.gethostname(),
            pid=os.getpid(),
            device=device,
            job_id=job_id,
        )

    @property
    def rank_path_component(self) -> str:
        """Return the collision-free global rank component."""

        if self.global_rank is None:
            raise ValueError("global rank is unavailable; cannot create a rank-local path")
        return f"rank-{self.global_rank}"


Scalar = Union[bool, float, int]


@dataclass(frozen=True)
class ScalarMetric:
    """One JSON-safe scalar metric at the serialization boundary."""

    key: str
    value: Scalar


@dataclass(frozen=True)
class ProfileRecord:
    """Typed resource readings owned by one process and one topology."""

    scope: ProfileScope
    monotonic_time: float
    topology: ExecutionTopology
    process: ProcessResourceResult | None = None
    device: AllocatorUsage | None = None

    def __post_init__(self) -> None:
        if self.monotonic_time < 0:
            raise ValueError("monotonic_time must be nonnegative")
        if self.scope is ProfileScope.PROCESS and self.process is None:
            raise ValueError("PROCESS profile requires process readings")
        if self.scope is ProfileScope.DEVICE and self.device is None:
            raise ValueError("DEVICE profile requires device readings")
        if self.scope in (ProfileScope.NODE, ProfileScope.JOB):
            raise ValueError("NODE and JOB profiles require an explicit aggregate record")


def project_scalars(record: ProfileRecord) -> tuple[ScalarMetric, ...]:
    """Project one typed record deterministically without aggregating readings."""

    if record.scope is ProfileScope.PROCESS:
        assert record.process is not None
        readings = (
            ("user_cpu_seconds", record.process.user_cpu_seconds),
            ("system_cpu_seconds", record.process.system_cpu_seconds),
            ("read_block_operations", record.process.read_block_operations),
            ("write_block_operations", record.process.write_block_operations),
            ("voluntary_context_switches", record.process.voluntary_context_switches),
            ("involuntary_context_switches", record.process.involuntary_context_switches),
            ("peak_rss_mb", record.process.peak_rss_mb),
        )
        return _project_readings(readings)
    assert record.device is not None
    readings = (
        ("allocated_mb", record.device.allocated_mb),
        ("reserved_mb", record.device.reserved_mb),
        ("device_count", record.device.device_count),
    )
    return _project_readings(readings)


def _project_readings(readings: tuple[tuple[str, object], ...]) -> tuple[ScalarMetric, ...]:
    metrics: list[ScalarMetric] = []
    for key, value in readings:
        if isinstance(value, (ResourceUnavailable, AllocatorUnavailable)):
            metrics.append(ScalarMetric(f"{key}_unavailable", True))
        elif value is not None:
            if not isinstance(value, (bool, float, int)):
                raise TypeError(f"{key} is not a scalar: {type(value).__name__}")
            metrics.append(ScalarMetric(key, value))
    return tuple(metrics)


def reject_aggregation(records: tuple[ProfileRecord, ...]) -> None:
    """Reject numeric reduction of resource readings with a clear error."""

    if len(records) != 1:
        raise ValueError("resource profiles are rank-local; numeric aggregation is unsupported")


class RankLocalJSONLWriter:
    """Append profile records to ``profiles/rank-N/records.jsonl``."""

    def __init__(self, run_dir: Path | str, topology: ExecutionTopology) -> None:
        self.topology = topology
        self.path = Path(run_dir) / "profiles" / topology.rank_path_component / "records.jsonl"

    def write(self, record: ProfileRecord) -> None:
        """Append one scalar-projected, topology-stamped JSON record."""

        if record.topology != self.topology:
            raise ValueError("record topology does not belong to this rank-local writer")
        payload = {
            "scope": record.scope.value,
            "time_monotonic": record.monotonic_time,
            "global_rank": record.topology.global_rank,
            "global_size": record.topology.global_size,
            "local_rank": record.topology.local_rank,
            "local_size": record.topology.local_size,
            "node_rank": record.topology.node_rank,
            "node_size": record.topology.node_size,
            "host": record.topology.host,
            "pid": record.topology.pid,
            "device": record.topology.device,
            "job_id": record.topology.job_id,
            "device_identity": None
            if record.topology.device_identity is None
            else {
                "kind": record.topology.device_identity.kind.value,
                "index": record.topology.device_identity.index,
                "uuid": record.topology.device_identity.uuid,
            },
            "metrics": {metric.key: metric.value for metric in project_scalars(record)},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False))
            handle.write("\n")


__all__ = [
    "ExecutionTopology",
    "ProfileRecord",
    "ProfileScope",
    "RankLocalJSONLWriter",
    "ScalarMetric",
    "project_scalars",
    "reject_aggregation",
]
