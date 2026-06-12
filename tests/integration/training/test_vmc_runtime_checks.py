"""Integration test: runtime-check callbacks during a VMC smoke training run.

Drives ``run_from_config`` through the Train runner with DataIntegrity,
GradientStats, and SamplerHealth callbacks active, and asserts the standard
artifacts plus the check-metric namespaces. No convergence assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from spenn.run import run_from_config

FIXTURE = Path(__file__).resolve().parents[1] / "artifacts" / "training" / "vmc_runtime_checks.yaml"


def _run(tmp_path: Path) -> Path:
    cfg = OmegaConf.load(FIXTURE)
    cfg.run.root = str(tmp_path)
    exit_code = run_from_config(cfg, config_path=str(FIXTURE), command="pytest")
    assert exit_code == 0
    run_dirs = list(tmp_path.glob("vmc_runtime_checks/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    return run_dirs[0]


def test_runtime_checks_run_writes_standard_artifacts(tmp_path) -> None:
    run_dir = _run(tmp_path)

    for artifact in (
        "config.yaml",
        "resolved_config.yaml",
        "metadata.json",
        "status.json",
        "metrics.jsonl",
        "checkpoints/latest.pt",
    ):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"


def test_runtime_checks_log_check_namespaces(tmp_path) -> None:
    run_dir = _run(tmp_path)

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    namespaces = {record.get("namespace") for record in records}

    for expected in ("train", "train/sampler", "checks/data_integrity", "checks/gradient", "checks/sampler"):
        assert expected in namespaces, f"missing namespace: {expected}"

    data_integrity = [r["metrics"] for r in records if r.get("namespace") == "checks/data_integrity"]
    assert data_integrity, "no data-integrity records"
    assert all(record["passed"] is True for record in data_integrity)
