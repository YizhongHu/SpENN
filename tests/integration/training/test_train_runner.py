"""Integration test: the Train runner executes a VMC smoke loop end-to-end.

Drives the full configured path -- ``run_from_config`` -> ``Train`` runner ->
``make_optimizer`` -> ``VMCTrainer.fit`` -> sampler -> Hamiltonian terms ->
surrogate loss -> optimizer step -> loggers/callbacks -- and asserts the
standard run artifacts and finite ``train`` metrics. No convergence assertions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from spenn.run import run_from_config

FIXTURE = Path(__file__).resolve().parents[1] / "artifacts" / "training" / "vmc_smoke.yaml"

ALLOWED_NONFINITE_KEYS = {"energy_stderr"}


def _run(tmp_path: Path):
    cfg = OmegaConf.load(FIXTURE)
    cfg.run.root = str(tmp_path)
    exit_code = run_from_config(cfg, config_path=str(FIXTURE), command="pytest")
    assert exit_code == 0
    run_dirs = list(tmp_path.glob("vmc_smoke/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    return run_dirs[0]


def test_train_runner_writes_standard_artifacts(tmp_path) -> None:
    run_dir = _run(tmp_path)

    for artifact in (
        "config.yaml",
        "resolved_config.yaml",
        "metadata.json",
        "status.json",
        "metrics.csv",
        "metrics.jsonl",
        "checkpoints/latest.pt",
        "checkpoints/step_3.pt",
    ):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"


def test_train_runner_logs_finite_train_metrics(tmp_path) -> None:
    run_dir = _run(tmp_path)

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    train_records = [record["metrics"] for record in records if record.get("namespace") == "train"]
    sampler_records = [record["metrics"] for record in records if record.get("namespace") == "train/sampler"]
    perf_records = [record["metrics"] for record in records if record.get("namespace") == "train/perf"]
    runtime_records = [record["metrics"] for record in records if record.get("namespace") == "runtime"]
    assert len(train_records) == 3, "expected one train record per step"
    assert len(sampler_records) == 3, "expected one train/sampler record per step"
    assert len(perf_records) == 3, "expected one train/perf record per step"
    assert any("wall_time_sec" in record for record in runtime_records)

    last = train_records[-1]
    for key in (
        "loss",
        "energy",
        "energy_variance",
        "local_energy_n_finite",
        "local_energy_finite_fraction",
        "logabs_mean",
    ):
        assert key in last, f"missing metric: {key}"
    # The physical training estimator is logged as `energy`, never `energy_mean`.
    assert "energy_mean" not in last
    assert not any(key.startswith("sampler.") for key in last)
    assert "acceptance_rate" in sampler_records[-1]
    assert "n_walkers" in sampler_records[-1]
    assert "step_time_sec" in perf_records[-1]
    assert "step_time_sec_rolling_mean" in perf_records[-1]

    # JSONL serialization with allow_nan=False would already have failed the run
    # on any non-finite value; assert finiteness directly for good measure.
    for record in train_records:
        for key, value in record.items():
            if key in ALLOWED_NONFINITE_KEYS or not isinstance(value, (int, float)):
                continue
            assert math.isfinite(value), f"non-finite metric {key}={value}"
