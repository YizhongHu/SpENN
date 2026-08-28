"""Synthetic, off-cluster tests for placement admission evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.toolkit.placement import (
    AllocationManifest,
    AttemptPlacement,
    decode_polaris_hostname,
    validate_placement_evidence,
)


HOSTS = ("x3001c0s1b0n0", "x3001c0s7b0n0")
NOW = 1_000.0


def _attempt(host: str, attempt_id: str, *, uuids: tuple[str, ...] | None = None, started: float = NOW + 1, result: str | None = None) -> AttemptPlacement:
    uuids = uuids or tuple(f"GPU-{host}-{index}" for index in range(4))
    result_dir = result or f"/runs/results/{attempt_id}"
    return AttemptPlacement(
        attempt_id=attempt_id, hostname=host, fqdn=f"{host}.hsn.cm.polaris.alcf.anl.gov",
        pbs_job_id="123.polaris", identity="worker", pid=12,
        cpu_affinity=(0, 1), cuda_visible_devices="0,1,2,3",
        gpus=tuple({"name": "A100", "uuid": uuid, "pci_bus_id": f"0000:{index:02x}:00.0"} for index, uuid in enumerate(uuids)),
        cwd="/runs", result_dir=result_dir, started_at_unix=started,
        ended_at_unix=started + 1, returncode=0,
        artefact_paths=(f"{result_dir}/status.json", f"{result_dir}/checkpoints/latest.json"),
    )


def _green(*, attempts=None, managers=HOSTS, admission=NOW, planned=None, census=None, row_hosts=None, completed=None):
    attempts = attempts or tuple(_attempt(host, f"row-{index}") for index, host in enumerate(HOSTS))
    return validate_placement_evidence(
        manifest=AllocationManifest.from_nodefile("\n".join(HOSTS) + "\n", requested_nodes=2),
        attempts=attempts, dispatch_ids=tuple(item.attempt_id for item in attempts),
        completed_result_dirs=completed or tuple(item.result_dir for item in attempts), admission_time_unix=admission,
        completion_ok={item.attempt_id: True for item in attempts}, manager_hosts=managers,
        worker_count=len(attempts), planned_result_dirs=planned,
        gpu_census=census, row_hosts=row_hosts,
    )


def test_valid_two_node_census_is_green() -> None:
    _green()


def test_all_tasks_on_first_host_rejected() -> None:
    attempts = tuple(_attempt(HOSTS[0], f"row-{index}") for index in range(2))
    with pytest.raises(ValueError, match="placement records do not cover every allocated host"):
        _green(attempts=attempts, managers=HOSTS, census=(_attempt(HOSTS[0], "census-0"), _attempt(HOSTS[1], "census-1")), row_hosts=(HOSTS[0], HOSTS[0]))


def test_absent_manager_rejected() -> None:
    with pytest.raises(ValueError, match="manager host set"):
        _green(managers=(HOSTS[0],))


def test_repeated_physical_gpu_uuid_rejected() -> None:
    attempts = (_attempt(HOSTS[0], "row-0"), _attempt(HOSTS[1], "row-1"))
    census = (_attempt(HOSTS[0], "census-0", uuids=("same", "same", "same", "same")), _attempt(HOSTS[1], "census-1"))
    with pytest.raises(ValueError, match="four distinct physical GPUs"):
        _green(attempts=attempts, census=census)


def test_fewer_observed_nodes_than_allocated_rejected() -> None:
    attempts = (_attempt(HOSTS[0], "row-0"),)
    census = (_attempt(HOSTS[0], "census-0"), _attempt(HOSTS[1], "census-1"))
    with pytest.raises(ValueError, match="placement records do not cover every allocated host"):
        _green(attempts=attempts, census=census)


def test_stale_completion_evidence_rejected() -> None:
    attempts = tuple(_attempt(host, f"row-{index}", started=NOW - 1) for index, host in enumerate(HOSTS))
    census = (_attempt(HOSTS[0], "census-0"), _attempt(HOSTS[1], "census-1"))
    with pytest.raises(ValueError, match="predates admission"):
        _green(attempts=attempts, census=census)


def test_doubly_nested_result_directory_rejected() -> None:
    attempts = tuple(_attempt(host, f"row-{index}", result=f"/runs/results/row-{index}/row-{index}") for index, host in enumerate(HOSTS))
    census = (_attempt(HOSTS[0], "census-0"), _attempt(HOSTS[1], "census-1"))
    planned = {f"row-{index}": f"/runs/results/row-{index}" for index in range(2)}
    with pytest.raises(ValueError, match="completed result directory is not the exact planned directory"):
        _green(attempts=attempts, planned=planned, census=census, completed=tuple(planned.values()))


def test_unknown_and_contradictory_hostnames_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown Polaris hostname"):
        decode_polaris_hostname("node01")
    with pytest.raises(ValueError, match="contradictory"):
        decode_polaris_hostname("x3001c0s2b0n0")


def test_scheduler_tiers_are_cross_checked() -> None:
    manifest = AllocationManifest.from_nodefile("\n".join(HOSTS), requested_nodes=2, scheduler_tiers={HOSTS[0]: {"tier0": "wrong", "tier1": "g0"}})
    with pytest.raises(ValueError, match="scheduler topology tiers contradict hostname decoding"):
        validate_placement_evidence(
            manifest=manifest, attempts=(_attempt(HOSTS[0], "row-0"), _attempt(HOSTS[1], "row-1")),
            dispatch_ids=("row-0", "row-1"), completed_result_dirs=("/runs/results/row-0", "/runs/results/row-1"),
            admission_time_unix=NOW, completion_ok={"row-0": True, "row-1": True}, manager_hosts=HOSTS,
        )
