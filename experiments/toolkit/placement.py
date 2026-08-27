"""Placement evidence and topology admission checks for allocation runs.

This module is deliberately standard-library-only.  Placement claims are made
from immutable records collected by workers, never from scheduler intent or
local GPU indices.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_POLARIS_HOST = re.compile(r"^x(?P<row>30|31|32)(?P<position>\d{2})c0s(?P<slot>\d{1,2})b(?P<board>[01])n0$")
_SLOTS = {1, 7, 13, 19, 25, 31, 37}


@dataclass(frozen=True)
class PolarisTopology:
    """Decoded, scheduler-cross-checkable topology for one Polaris host."""

    hostname: str
    rack: str
    chassis_slot: int
    board: int
    node: int
    tier0: str
    tier1: str


def decode_polaris_hostname(hostname: str) -> PolarisTopology:
    """Decode a Polaris hostname, failing closed for unknown formats."""

    short = str(hostname).strip().lower().split(".", 1)[0]
    match = _POLARIS_HOST.fullmatch(short)
    if match is None:
        raise ValueError(f"unknown Polaris hostname format: {hostname!r}")
    row = int(match["row"])
    position = int(match["position"])
    slot = int(match["slot"])
    board = int(match["board"])
    if slot not in _SLOTS or position < 1 or position > (16 if row == 30 else 12):
        raise ValueError(f"contradictory Polaris hostname fields: {hostname!r}")
    rack = f"x{row}{position:02d}"
    group = {30: (position - 1) // 4, 31: 4 + (position - 1) // 4, 32: 7 + (position - 1) // 4}[row]
    return PolarisTopology(short, rack, slot, board, 0, f"{rack}-g{group}", f"g{group}")


@dataclass(frozen=True)
class AttemptPlacement:
    """One worker's placement census, captured before its science subprocess."""

    attempt_id: str
    hostname: str
    fqdn: str
    pbs_job_id: str | None
    identity: str | None
    pid: int
    cpu_affinity: tuple[int, ...] | None
    cuda_visible_devices: str | None
    gpus: tuple[Mapping[str, Any], ...]
    cwd: str
    result_dir: str
    started_at_unix: float
    ended_at_unix: float | None = None
    returncode: int | None = None
    artefact_paths: tuple[str, ...] = ()

    @property
    def physical_gpu_ids(self) -> tuple[tuple[str, str], ...]:
        """Return physical identities; local GPU indices are intentionally ignored."""

        return tuple((self.hostname, str(gpu["uuid"])) for gpu in self.gpus if gpu.get("uuid"))


@dataclass(frozen=True)
class AllocationManifest:
    """Scheduler allocation facts and the raw nodefile evidence."""

    raw_nodefile: str
    raw_nodefile_sha256: str
    hosts: tuple[str, ...]
    topology: tuple[PolarisTopology, ...]
    scheduler_tiers: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    requested_nodes: int = 0
    nodes_per_block: int = 0
    launcher_class: str = ""
    launcher_args: tuple[str, ...] = ()
    expected_managers: int = 0
    expected_workers: int = 0
    manager_registration_timestamps: Mapping[str, float] = field(default_factory=dict)
    worker_registration_timestamps: Mapping[str, float] = field(default_factory=dict)
    topology_map_source: str = "https://docs.alcf.anl.gov/polaris/running-jobs/"
    topology_map_retrieved_date: str = "2026-08-27"

    @classmethod
    def from_nodefile(cls, raw_nodefile: str, *, requested_nodes: int, **kwargs: Any) -> "AllocationManifest":
        """Build a manifest while retaining the nodefile bytes unchanged."""

        hosts = tuple(dict.fromkeys(line.strip().lower() for line in raw_nodefile.splitlines() if line.strip()))
        topology = tuple(decode_polaris_hostname(host) for host in hosts)
        return cls(raw_nodefile, hashlib.sha256(raw_nodefile.encode()).hexdigest(), hosts, topology, requested_nodes=requested_nodes, **kwargs)


def validate_placement_evidence(
    *,
    manifest: AllocationManifest,
    attempts: Sequence[AttemptPlacement],
    dispatch_ids: Sequence[str],
    completed_result_dirs: Sequence[str | Path],
    admission_time_unix: float,
    completion_ok: Mapping[str, bool],
    manager_hosts: Sequence[str],
    worker_count: int | None = None,
    planned_result_dirs: Mapping[str, str | Path] | None = None,
) -> None:
    """Raise ``ValueError`` unless every placement claim is independently green."""

    allocated = set(manifest.hosts)
    if manifest.requested_nodes != len(allocated):
        raise ValueError("requested node count does not equal unique nodefile hosts")
    if set(manager_hosts) != allocated:
        raise ValueError("observed manager host set does not equal allocated host set")
    physical = [gpu_id for attempt in attempts for gpu_id in attempt.physical_gpu_ids]
    if len(physical) != 4 * manifest.requested_nodes or len(set(physical)) != 4 * manifest.requested_nodes:
        raise ValueError("placement census is not exactly four distinct physical GPUs per allocated node")
    if len(physical) != len(set(physical)):
        raise ValueError("duplicate simultaneous physical GPU use")
    if {attempt.hostname for attempt in attempts} != allocated:
        raise ValueError("placement records do not cover every allocated host")
    if worker_count is not None and worker_count != len(attempts):
        raise ValueError("observed worker count does not equal expected worker count")
    if any(not completion_ok.get(attempt.attempt_id, False) for attempt in attempts):
        raise ValueError("at least one row completion predicate failed")
    if not (len(dispatch_ids) == len(attempts) == len(completed_result_dirs)):
        raise ValueError("expected rows, dispatch records, placement records, and result directories differ")
    if set(dispatch_ids) != {attempt.attempt_id for attempt in attempts}:
        raise ValueError("dispatch records and placement records identify different rows")
    for attempt in attempts:
        if attempt.started_at_unix < admission_time_unix:
            raise ValueError(f"placement artefact predates admission: {attempt.attempt_id}")
        planned = Path(attempt.result_dir).resolve()
        if planned_result_dirs is not None:
            expected = Path(planned_result_dirs[attempt.attempt_id]).resolve()
            if planned != expected:
                raise ValueError("completed result directory is not the exact planned directory")
        for path in (Path(attempt.cwd), planned):
            if not path.is_absolute():
                raise ValueError("placement paths must be absolute")
        if any(not Path(artefact).resolve().is_relative_to(planned) for artefact in attempt.artefact_paths):
            raise ValueError("status or checkpoint artefact is outside the exact planned directory")
    if planned_result_dirs is not None and {Path(path).resolve() for path in completed_result_dirs} != {
        Path(path).resolve() for path in planned_result_dirs.values()
    }:
        raise ValueError("completed result directories do not match exact planned directories")
    for host, tiers in manifest.scheduler_tiers.items():
        decoded = next((item for item in manifest.topology if item.hostname == host), None)
        if decoded is None or any(tiers.get(key) != getattr(decoded, key) for key in ("tier0", "tier1")):
            raise ValueError(f"scheduler topology tiers contradict hostname decoding for {host!r}")


def capture_attempt_placement(*, attempt_id: str, cwd: str, result_dir: str, started_at_unix: float) -> dict[str, Any]:
    """Capture worker identity and physical GPU metadata before launching science."""

    hostname = socket.gethostname().lower()
    payload: dict[str, Any] = {
        "attempt_id": attempt_id,
        "hostname": hostname,
        "fqdn": socket.getfqdn(),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "identity": os.environ.get("PARSL_WORKER_ID") or os.environ.get("PARSL_MANAGER_ID"),
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cwd": str(Path(cwd).resolve()),
        "result_dir": str(Path(result_dir).resolve()),
        "started_at_unix": started_at_unix,
        "gpus": [],
    }
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True,
        )
    except OSError:
        query = None
    if query is not None and query.returncode == 0:
        for line in query.stdout.splitlines():
            name, uuid, bus_id = (part.strip() for part in line.split(",", 2))
            payload["gpus"].append({"name": name, "uuid": uuid, "pci_bus_id": bus_id})
    return payload


__all__ = ["AllocationManifest", "AttemptPlacement", "PolarisTopology", "capture_attempt_placement", "decode_polaris_hostname", "validate_placement_evidence"]
