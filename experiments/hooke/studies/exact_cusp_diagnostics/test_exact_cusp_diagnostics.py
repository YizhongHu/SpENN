"""Tests for the exact Hooke cusp diagnostics experiment."""

from __future__ import annotations

import csv
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


def test_config_wires_exact_singlet_eval_and_pair_probe() -> None:
    cfg = OmegaConf.load(CONFIG)

    assert cfg.experiment.name == "exact_cusp_diagnostics"
    assert cfg.experiment.sector == "singlet"
    assert cfg.run.timezone == "America/New_York"
    assert cfg.runtime.device == "cpu"
    assert cfg.runtime.dtype == "float64"
    assert cfg.model._target_ == "spenn.physics.hooke.HookeSingletExact"
    assert cfg.runner._target_ == "spenn.runner.Evaluate"
    assert cfg.runner.return_terms is True

    runner = OmegaConf.to_container(cfg.runner, resolve=False)
    diagnostics = runner["diagnostics"]
    assert diagnostics[0]["_target_"] == "spenn.diagnostics.EnergyEvaluation"
    assert diagnostics[0]["reference_energy"] == "${references.exact_energy}"
    assert diagnostics[1]["_target_"] == "spenn.diagnostics.HookePairDistanceProbe"
    assert diagnostics[1]["artifact_path"] == "${run.dir}/diagnostics/pair_distance_probe/probe.csv"

    # Experiment scripts may read files, but should not import SpENN internals.
    script = (STUDY_DIR / "plot_pair_distance.py").read_text(encoding="utf-8")
    assert "import spenn" not in script
    assert "from spenn" not in script


@pytest.mark.integration
def test_exact_eval_writes_probe_and_plot(tmp_path: Path) -> None:
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
    metrics = _eval_metrics(run_dir)
    assert metrics["reference_energy"] == pytest.approx(2.0)
    assert metrics["energy"] == pytest.approx(2.0, abs=1.0e-4)
    assert metrics["energy_variance"] < 1.0e-8
    assert metrics["local_energy_finite_fraction"] == 1.0

    probe_csv = run_dir / "diagnostics" / "pair_distance_probe" / "probe.csv"
    with probe_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert {"pair_distance", "model_local_energy", "exact_local_energy"}.issubset(rows[0])
    energies = [float(row["model_local_energy"]) for row in rows]
    assert all(abs(value - 2.0) < 1.0e-4 for value in energies)

    output = plot_pair_distance.plot_pair_distance(run_dir=run_dir)
    assert output == run_dir / "diagnostics" / "pair_distance_probe" / "model_local_energy_vs_pair_distance.png"
    assert output.exists()


def _eval_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_records = [record["metrics"] for record in records if record.get("namespace") == "eval"]
    assert eval_records
    return eval_records[-1]
