"""Integration tests for Hooke multibody smoke runs."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import pytest

from experiments.hooke_multibody.plot_outputs import _to_float, plot_run
from experiments.hooke_multibody.process_outputs import process_run
from experiments.hooke_multibody.run_reference import DEFAULT_CONFIG as HOOKE_MULTIBODY_REFERENCE_CONFIG
from experiments.hooke_multibody.run_reference import run as run_reference
from experiments.hooke_multibody.run_spenn import load_config, run, run_spin_scan
from tests.helpers import (
    HOOKE_MULTIBODY_INTEGRATION_ARTIFACTS,
    assert_hooke_multibody_run_artifacts,
)


@pytest.mark.integration
def test_hooke_multibody_smoke_writes_artifacts_with_timestamp() -> None:
    cfg = load_config(HOOKE_MULTIBODY_INTEGRATION_ARTIFACTS / "smoke.yaml")
    cfg.run_id = "integration_hooke_multibody_smoke"

    summary = run(cfg, forwarded_overrides=["run_id=integration_hooke_multibody_smoke"])
    summary_artifact = assert_hooke_multibody_run_artifacts(
        summary["output_dir"],
        expect_plot_data=True,
        expect_checkpoint=False,
    )

    assert summary["can_reach_goal"] is True
    assert summary["run_id"] == "integration_hooke_multibody_smoke"
    assert re.fullmatch(r"\d{2}-\d{2}-\d{2}", summary["run_time"])
    assert summary_artifact["run_time"] == summary["run_time"]
    assert summary_artifact["config"]["run"]["time"] == summary["run_time"]
    assert summary_artifact["config"]["run_mode"] == "integration"
    assert summary_artifact["config"]["system"]["n_electrons"] == 3
    assert summary_artifact["config"]["system"]["n_up"] == 2
    assert summary_artifact["config"]["system"]["n_down"] == 1
    assert summary_artifact["config"]["diagnostics"]["cusp"]["average_opposite_directions"] is True
    assert summary_artifact["config"]["model"]["spenn"]["readout"]["eps"] == 1.0e-30
    assert "integration_test" in summary_artifact["config"]["tracking"]["tags"]

    metrics = summary_artifact["metrics"]
    assert math.isfinite(float(metrics["spenn/energy/mean"]))
    assert math.isfinite(float(metrics["spenn/local_energy/variance"]))
    assert math.isfinite(float(metrics["sampler/mean_pair_distance"]))
    assert float(metrics["sampler/min_pair_distance"]) > 0.0
    assert float(metrics["sampler/local_energy_sample_count"]) == 4.0
    assert "sampler/local_energy_autocorrelation_time" in metrics
    assert "sampler/local_energy_effective_sample_size" in metrics
    assert math.isfinite(float(metrics["radial_density/mean_radius"]))
    assert "cusp/same_count" in metrics
    assert "cusp/opposite_count" in metrics
    assert abs(float(metrics["cusp/cusp_only_same_mean_error"])) < 5.0e-2
    assert abs(float(metrics["cusp/cusp_only_opposite_mean_error"])) < 5.0e-2
    assert abs(float(metrics["cusp/same_mean_error"])) < 5.0e-2
    assert abs(float(metrics["cusp/opposite_mean_error"])) < 2.5e-1
    assert math.isfinite(float(metrics["cusp/smooth_residual_same_mean_slope"]))
    assert math.isfinite(float(metrics["cusp/smooth_residual_opposite_mean_slope"]))
    assert "antisymmetry/antisymmetry_error_max" in metrics
    assert "exact/energy" not in metrics
    assert "comparison/energy_abs_error" not in metrics

    run_dir = Path(summary["output_dir"])
    assert _csv_row_count(run_dir / "plots" / "pair_distance_histogram.csv") == 8
    assert _csv_row_count(run_dir / "plots" / "radial_density.csv") == 8
    assert _csv_row_count(run_dir / "plots" / "local_energy_samples.csv") == 4
    assert _csv_row_count(run_dir / "plots" / "pair_distance_samples.csv") == 12
    assert _csv_row_count(run_dir / "plots" / "cusp_slope_by_spin.csv") == 3
    assert _csv_row_count(run_dir / "plots" / "particle_antisymmetry.csv") == 2

    processed = process_run(run_dir)
    for name in [
        "spenn_observables.csv",
        "energy_trace.csv",
        "sampler_metrics.csv",
        "energy_plausibility.csv",
        "local_energy_samples.csv",
        "pair_distance_samples.csv",
        "pair_distance_histogram.csv",
        "radial_density.csv",
        "cusp_slope_by_spin.csv",
        "particle_antisymmetry.csv",
    ]:
        assert (run_dir / "data" / name).exists(), f"missing processed data file: {name}"
    assert "pair_distance_histogram.csv" in processed["data_files"]
    plausibility_rows = _csv_rows(run_dir / "data" / "energy_plausibility.csv")
    assert len(plausibility_rows) == 1
    assert plausibility_rows[0]["n_electrons"] == "3"
    assert plausibility_rows[0]["specht_M"] == "2"
    assert plausibility_rows[0]["reference_available"] == "False"
    assert plausibility_rows[0]["reference_method"] == "none"

    figures = plot_run(run_dir, figure_root=run_dir / "figures")
    assert figures
    assert all(path.exists() for path in figures)


@pytest.mark.integration
def test_hooke_multibody_spin_scan_uses_one_timestamp_and_writes_scan_artifacts() -> None:
    cfg = load_config(HOOKE_MULTIBODY_INTEGRATION_ARTIFACTS / "smoke.yaml")
    cfg.scan = {"spin_partitions": [[2, 1], [1, 2]]}

    summary = run_spin_scan(
        cfg,
        forwarded_overrides=[
            "run_id=integration_hooke_multibody_spin_scan",
            "sampler.n_walkers=5",
            "training.vmc_steps=1",
            "trainer.max_steps=1",
        ],
    )

    run_dir = Path(summary["output_dir"])
    assert summary["mode"] == "spin_scan"
    assert summary["run_id"] == "integration_hooke_multibody_spin_scan"
    assert re.fullmatch(r"\d{2}-\d{2}-\d{2}", summary["run_time"])
    assert (run_dir / ".hydra" / "config.yaml").exists()
    assert (run_dir / ".hydra" / "overrides.yaml").exists()
    assert (run_dir / "metrics" / "spin_scan_summary.csv").exists()
    assert _csv_row_count(run_dir / "metrics" / "spin_scan_summary.csv") == 2
    assert summary["best_run"]["run_id"] in {run["run_id"] for run in summary["runs"]}
    assert summary["best_run"]["n_electrons"] == 3
    assert {run["run_time"] for run in summary["runs"]} == {summary["run_time"]}
    assert summary["config"]["sampler"]["n_walkers"] == 5
    assert summary["config"]["training"]["vmc_steps"] == 1
    assert all(run["run_id"].endswith(f"up{run['n_up']}_down{run['n_down']}") for run in summary["runs"])
    recorded_overrides = (run_dir / ".hydra" / "overrides.yaml").read_text(encoding="utf-8")
    assert "run_id=integration_hooke_multibody_spin_scan" in recorded_overrides
    assert f"run.time={summary['run_time']}" in recorded_overrides
    scan_rows = _csv_rows(run_dir / "metrics" / "spin_scan_summary.csv")
    assert {row["n_electrons"] for row in scan_rows} == {"3"}

    processed = process_run(run_dir)
    assert processed["mode"] == "spin_scan"
    assert (run_dir / "data" / "spin_scan_summary.csv").exists()
    assert (run_dir / "data" / "energy_plausibility.csv").exists()
    assert (run_dir / "artifacts" / "processed_summary.json").exists()
    plausibility_rows = _csv_rows(run_dir / "data" / "energy_plausibility.csv")
    assert len(plausibility_rows) == 2
    assert {row["n_electrons"] for row in plausibility_rows} == {"3"}
    assert {row["specht_M"] for row in plausibility_rows} == {"2"}
    assert {row["reference_method"] for row in plausibility_rows} == {"none"}

    figures = plot_run(run_dir, figure_root=run_dir / "figures")
    assert len(figures) == 1
    assert figures[0].name.endswith("_spin_scan_energy.png")
    assert figures[0].exists()


@pytest.mark.integration
def test_hooke_multibody_reference_summary_records_config_and_git() -> None:
    cfg = load_config(HOOKE_MULTIBODY_REFERENCE_CONFIG)
    cfg.run = {"id": "integration_hooke_multibody_reference"}

    summary = run_reference(cfg)
    artifact = _summary_json(Path(summary["output_dir"]))

    assert summary["run_id"] == "integration_hooke_multibody_reference"
    assert artifact["run_id"] == summary["run_id"]
    assert artifact["config"]["run"]["id"] == "integration_hooke_multibody_reference"
    assert artifact["config"]["run_id"] == "integration_hooke_multibody_reference"
    assert "git_commit" in artifact["git"]
    assert "dirty_git_state" in artifact["git"]


def test_hooke_multibody_plot_numeric_parser_rejects_nonfinite_values() -> None:
    assert _to_float("nan") is None
    assert _to_float("inf") is None
    assert _to_float("-inf") is None
    assert _to_float("1.25") == 1.25


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_json(run_dir: Path) -> dict[str, object]:
    with (run_dir / "artifacts" / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)
