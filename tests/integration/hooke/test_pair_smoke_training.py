"""Integration smoke test: the real Hooke pair config runs end-to-end.

Drives ``run_from_config`` through the real Train runner -> SpENNWaveFunction ->
MetropolisSampler -> Hooke Hamiltonian -> VMCTrainer with DataValidity,
GradientStats, SamplerHealth, RuntimeEquivariance (full_model + trace),
Checkpoint, and CSV/JSONL logging. No convergence or reference-energy assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from spenn.run import run_from_config

CONFIG = Path(__file__).resolve().parents[1] / "artifacts" / "hooke" / "pair_train.yaml"


def _run(tmp_path: Path) -> Path:
    cfg = OmegaConf.load(CONFIG)
    cfg.run.root = str(tmp_path)
    exit_code = run_from_config(cfg, config_path=str(CONFIG), command="pytest")
    assert exit_code == 0
    run_dirs = list(tmp_path.glob("hooke_pair_smoke/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    return run_dirs[0]


def test_pair_smoke_training_writes_standard_artifacts(tmp_path) -> None:
    run_dir = _run(tmp_path)

    for artifact in (
        "config.yaml",
        "resolved_config.yaml",
        "metadata.json",
        "status.json",
        "metrics.jsonl",
        "metrics.csv",
        "checkpoints/latest.pt",
    ):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"


def test_pair_smoke_training_logs_expected_namespaces(tmp_path) -> None:
    run_dir = _run(tmp_path)

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    namespaces = {record.get("namespace") for record in records}

    for expected in (
        "train",
        "train/sampler",
        "checks/data_validity",
        "checks/gradient",
        "checks/sampler",
        "checks/equivariance/full_model",
        "checks/equivariance/trace",
    ):
        assert expected in namespaces, f"missing namespace: {expected}"
