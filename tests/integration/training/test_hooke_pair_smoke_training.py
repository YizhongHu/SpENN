"""Integration smoke test: the real Hooke pair config runs end-to-end.

Drives ``run_from_config`` through the real Train runner -> TPENWaveFunction ->
MetropolisSampler -> Hooke Hamiltonian -> VMCTrainer with DataIntegrity,
GradientStats, SamplerHealth, RuntimeEquivariance (full_model + trace),
Checkpoint, and CSV/JSONL logging. Train-end validation has been removed;
evaluation runs separately via the Evaluate runner. No convergence or
reference-energy assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from tpen.run import run_from_config

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
        "run_start.json",
        "checkpoints/latest.json",
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
        "checks/data_integrity",
        "checks/gradient",
        "checks/sampler",
        "checks/equivariance/full_model",
        "checks/equivariance/trace",
    ):
        assert expected in namespaces, f"missing namespace: {expected}"
    assert "checks/data_validity" not in namespaces
    # Train-end validation was removed; no validation/* namespaces are emitted.
    assert not any(str(ns).startswith("validation") for ns in namespaces)


def test_pair_smoke_training_equivariance_checks_actually_compared_something(tmp_path) -> None:
    # A present namespace is not the same as a performed check: every verdict
    # under `checks/equivariance/*` is well defined over an empty comparison
    # set, so `passed` alone cannot distinguish "equivariant" from "nothing was
    # compared". `n_comparisons` is the key that can, and this asserts the
    # configured run reaches it with a real number rather than a vacuous zero.
    run_dir = _run(tmp_path)

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]

    full_model = [r["metrics"] for r in records if r.get("namespace") == "checks/equivariance/full_model"]
    assert full_model, "no full-model equivariance records"
    for metrics in full_model:
        # n_particles = 2 admits exactly one non-identity permutation, and the
        # checker compares once per permutation.
        assert metrics["n_comparisons"] == 1
        assert metrics["passed"] is True

    trace = [r["metrics"] for r in records if r.get("namespace") == "checks/equivariance/trace"]
    assert trace, "no trace equivariance records"
    for metrics in trace:
        # The fixture sets compare_output: false, so the count is exactly one
        # comparison per shared trace key per permutation.
        assert metrics["n_comparisons"] == metrics["n_permutations_tested"] * metrics["n_trace_entries"]
        assert metrics["passed"] is True


def test_pair_smoke_training_geometry_metrics(tmp_path) -> None:
    run_dir = _run(tmp_path)

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]

    # Geometry diagnostics ride along with training sampler stats.
    sampler_records = [r["metrics"] for r in records if r.get("namespace") == "train/sampler"]
    assert sampler_records, "no train/sampler records"
    for key in ("radius_mean", "radius_q99", "electron_distance_q01", "position_rms"):
        assert key in sampler_records[-1], f"missing train/sampler/{key}"
