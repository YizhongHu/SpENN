"""Tests that the launcher wires topology and rank-local writing through context."""

from __future__ import annotations

import json

from omegaconf import OmegaConf

from tpen.distributed import ExecutionTopology, ProfileRecord, ProfileScope
from tpen.process_resources import ProcessResourceResult
from tpen.run import prepare_run_context


def test_prepare_context_owns_passed_topology_and_writer(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "distributed", "sector": "unit", "run_name": "topology"},
            "run": {"root": str(tmp_path), "run_id": "fixed", "dir": None},
            "runtime": {"device": "cuda:3", "dtype": "float64"},
            "callbacks": [],
            "loggers": [],
        }
    )
    topology = ExecutionTopology(
        global_rank=1,
        global_size=2,
        local_rank=0,
        local_size=1,
        node_rank=0,
        node_size=1,
        host="node-a",
        pid=42,
        device="cuda:3",
    )

    context = prepare_run_context(cfg, topology=topology)
    assert context.topology is topology
    context.write_profile(
        ProfileRecord(
            scope=ProfileScope.PROCESS,
            monotonic_time=1.25,
            topology=topology,
            process=ProcessResourceResult(
                user_cpu_seconds=1,
                system_cpu_seconds=2,
                read_block_operations=3,
                write_block_operations=4,
                voluntary_context_switches=5,
                involuntary_context_switches=6,
                peak_rss_mb=7,
            ),
        )
    )
    path = tmp_path / "distributed" / "unit" / "fixed" / "profiles" / "rank-1" / "records.jsonl"
    payload = json.loads(path.read_text())
    assert payload["global_rank"] == 1
    assert payload["device"] == "cuda:3"
    assert payload["metrics"]["peak_rss_mb"] == 7
