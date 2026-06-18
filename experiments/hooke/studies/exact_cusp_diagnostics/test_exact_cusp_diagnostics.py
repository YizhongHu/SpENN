"""Tests for the exact Hooke cusp evaluation experiment."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from omegaconf import OmegaConf

from spenn.run import run_from_config


ROOT = Path(__file__).resolve().parents[4]
STUDY_DIR = Path(__file__).resolve().parent
CONFIG = STUDY_DIR / "configs" / "exact_singlet_eval.yaml"


def _load_script(name: str) -> ModuleType:
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"exact_cusp_diagnostics_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot_pair_distance = _load_script("plot_pair_distance")


def test_config_wires_exact_singlet_eval_and_cusp_task() -> None:
    cfg = OmegaConf.load(CONFIG)

    assert cfg.experiment.name == "exact_cusp_diagnostics"
    assert cfg.experiment.sector == "singlet"
    assert cfg.run.timezone == "America/New_York"
    assert cfg.runtime.device == "cpu"
    assert cfg.runtime.dtype == "float64"
    assert cfg.model._target_ == "spenn.physics.hooke.HookeSingletExact"
    assert cfg.runner._target_ == "spenn.runner.Evaluate"
    assert cfg.runner.evaluator == "${evaluator}"

    raw = OmegaConf.to_container(cfg, resolve=False)
    assert raw["evaluator"]["tasks"] == ["${evaluation_tasks.energy}", "${evaluation_tasks.cusp}"]
    assert raw["evaluation_tasks"]["energy"]["summaries"][-1]["_target_"] == "spenn.evaluation.summaries.ReferenceEnergySummary"
    assert raw["evaluation_tasks"]["cusp"]["generator"]["_target_"] == "spenn.evaluation.generators.CuspGridGenerator"

    # Experiment scripts may read files, but should not import SpENN internals.
    script = (STUDY_DIR / "plot_pair_distance.py").read_text(encoding="utf-8")
    assert "import spenn" not in script
    assert "from spenn" not in script


@pytest.mark.integration
def test_exact_eval_writes_energy_and_cusp_metrics(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    cfg = OmegaConf.load(CONFIG)
    cfg.run.root = str(tmp_path)
    cfg.run.run_id = "test_exact_cusp"
    cfg.sampler_params.n_walkers = 16
    cfg.sampler_params.burn_in = 4
    cfg.sampler_params.n_steps = 2
    cfg.diagnostic_params.pair_distance_probe.n_points = 5
    cfg.diagnostic_params.pair_distance_probe.n_directions = 1
    cfg.diagnostic_params.pair_distance_probe.center_of_mass_radii = [0.0]
    cfg.diagnostic_params.pair_distance_probe.local_energy_chunk_size = 2

    assert run_from_config(cfg, config_path=str(CONFIG), command="pytest exact cusp diagnostics") == 0

    run_dir = tmp_path / "exact_cusp_diagnostics" / "singlet" / "test_exact_cusp"
    energy = _metrics(run_dir, "eval/energy")
    cusp = _metrics(run_dir, "eval/cusp")
    assert energy["reference_energy"] == pytest.approx(2.0)
    assert energy["local_energy_mean"] == pytest.approx(2.0, abs=1.0e-4)
    assert energy["local_energy_variance"] < 1.0e-8
    assert energy["local_energy_finite_fraction"] == 1.0
    assert cusp["c_minus_1_abs_max"] < 1.0e-4
    assert cusp["cusp_even_slope_abs_error"] < 1.0e-4


def _metrics(run_dir: Path, namespace: str) -> dict:
    metrics_path = run_dir / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [record["metrics"] for record in records if record.get("namespace") == namespace]
    assert matches
    return matches[-1]
