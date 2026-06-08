"""Integration tests: the Evaluate runner samples the exact Hooke energy.

These drive the full configured path -- ``run_from_config`` -> ``Evaluate``
runner -> sampler -> Hamiltonian terms -> local energy -> loggers/callbacks --
and assert the logged sampled energy matches the known exact energy.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import spenn.runner as runner_module
from spenn.physics.hamiltonian import summarize_local_energy
from spenn.run import run_from_config

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hooke"


def test_evaluate_uses_summary_helper_from_physics_hamiltonian() -> None:
    assert runner_module.summarize_local_energy is summarize_local_energy
    assert "summarize_local_energy" not in runner_module.__all__


def test_evaluate_does_not_accept_callbacks_or_loggers() -> None:
    params = inspect.signature(runner_module.Evaluate.__init__).parameters
    assert "callbacks" not in params
    assert "loggers" not in params


@pytest.mark.parametrize("fixture", ["exact_singlet.yaml", "exact_triplet.yaml"])
def test_evaluate_config_does_not_pass_callbacks_or_loggers(fixture: str) -> None:
    cfg = OmegaConf.load(FIXTURES / fixture)
    runner_keys = set(cfg.runner.keys())
    assert "callbacks" not in runner_keys
    assert "loggers" not in runner_keys


def _eval_metrics(run_root: Path) -> dict:
    jsonl_files = list(run_root.glob("**/metrics.jsonl"))
    assert len(jsonl_files) == 1, f"expected exactly one metrics.jsonl, found {jsonl_files}"
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines() if line.strip()]
    eval_records = [record["metrics"] for record in records if record.get("namespace") == "eval"]
    assert eval_records, "no eval metric records were logged"
    return eval_records[-1]


@pytest.mark.parametrize(
    ("fixture", "exact_energy"),
    [("exact_singlet.yaml", 2.0), ("exact_triplet.yaml", 1.25)],
)
def test_hooke_eval_runner_matches_exact_energy(tmp_path, fixture: str, exact_energy: float) -> None:
    config_path = FIXTURES / fixture
    cfg = OmegaConf.load(config_path)
    cfg.run.root = str(tmp_path)

    exit_code = run_from_config(cfg, config_path=str(config_path), command="pytest")
    assert exit_code == 0

    metrics = _eval_metrics(tmp_path)
    energy_atol = float(cfg.validation.energy_atol)
    variance_max = float(cfg.validation.variance_max)

    assert metrics["reference_energy"] == pytest.approx(exact_energy)
    assert metrics["n_finite_samples"] == metrics["n_samples"] == 512
    assert metrics["nonfinite_energy_fraction"] == 0.0
    assert metrics["abs_energy_error"] < energy_atol
    assert metrics["energy_variance"] < variance_max
    # return_terms: true -> per-term decomposition is logged as terms.<name>_mean
    # and terms.<name>_nonfinite_fraction.
    for term in ("kinetic", "harmonic_trap", "electron_electron"):
        assert f"terms.{term}_mean" in metrics
        assert metrics[f"terms.{term}_nonfinite_fraction"] == 0.0
    # sampler diagnostics are logged with sampler.* keys.
    assert metrics["sampler.n_walkers"] == 512
    assert "sampler.acceptance_rate" in metrics


@pytest.mark.parametrize("fixture", ["exact_singlet.yaml", "exact_triplet.yaml"])
def test_hooke_eval_runner_writes_standard_artifacts(tmp_path, fixture: str) -> None:
    config_path = FIXTURES / fixture
    cfg = OmegaConf.load(config_path)
    cfg.run.root = str(tmp_path)

    assert run_from_config(cfg, config_path=str(config_path), command="pytest") == 0

    run_dirs = list(tmp_path.glob("hooke_exact/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    run_dir = run_dirs[0]
    for artifact in (
        "config.yaml",
        "resolved_config.yaml",
        "metadata.json",
        "status.json",
        "report.md",
        "metrics.jsonl",
        "metrics.csv",
    ):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
