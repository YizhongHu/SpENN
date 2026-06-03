"""Integration tests for Hooke multibody smoke runs."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import pytest

from experiments.hooke_multibody.plot_outputs import plot_run
from experiments.hooke_multibody.process_outputs import process_run
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
    cfg.run_id = "integration_hooke_multibody_spin_scan"

    summary = run_spin_scan(
        cfg,
        forwarded_overrides=[
            "run_id=integration_hooke_multibody_spin_scan",
            "sampler.n_walkers=5",
            "training.vmc_steps=1",
            "trainer.max_steps=1",
        ],
        run_id="integration_hooke_multibody_spin_scan",
    )

    run_dir = Path(summary["output_dir"])
    assert summary["mode"] == "spin_scan"
    assert re.fullmatch(r"\d{2}-\d{2}-\d{2}", summary["run_time"])
    assert (run_dir / ".hydra" / "config.yaml").exists()
    assert (run_dir / ".hydra" / "overrides.yaml").exists()
    assert (run_dir / "metrics" / "spin_scan_summary.csv").exists()
    assert _csv_row_count(run_dir / "metrics" / "spin_scan_summary.csv") == 2
    assert summary["best_run"]["run_id"] in {run["run_id"] for run in summary["runs"]}
    assert {run["run_time"] for run in summary["runs"]} == {summary["run_time"]}
    assert summary["config"]["sampler"]["n_walkers"] == 5
    assert summary["config"]["training"]["vmc_steps"] == 1
    assert all(run["run_id"].endswith(f"up{run['n_up']}_down{run['n_down']}") for run in summary["runs"])

    processed = process_run(run_dir)
    assert processed["mode"] == "spin_scan"
    assert (run_dir / "data" / "spin_scan_summary.csv").exists()
    assert (run_dir / "data" / "energy_plausibility.csv").exists()
    assert (run_dir / "artifacts" / "processed_summary.json").exists()
    plausibility_rows = _csv_rows(run_dir / "data" / "energy_plausibility.csv")
    assert len(plausibility_rows) == 2
    assert {row["specht_M"] for row in plausibility_rows} == {"2"}
    assert {row["reference_method"] for row in plausibility_rows} == {"none"}

    figures = plot_run(run_dir, figure_root=run_dir / "figures")
    assert len(figures) == 1
    assert figures[0].name.endswith("_spin_scan_energy.png")
    assert figures[0].exists()


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
