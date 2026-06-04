"""Integration tests for Hooke multibody smoke runs."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import pytest

from experiments.hooke_multibody.plot_outputs import _plot_spin_scan, _to_float, plot_run
from experiments.hooke_multibody.process_outputs import _baseline_comparison, process_run
from experiments.hooke_multibody.run_reference import DEFAULT_CONFIG as HOOKE_MULTIBODY_REFERENCE_CONFIG
from experiments.hooke_multibody.run_reference import run as run_reference
from experiments.hooke_multibody.run_spenn import load_config, run, run_spin_scan
from tests.helpers import (
    HOOKE_MULTIBODY_INTEGRATION_ARTIFACTS,
    assert_hooke_multibody_run_artifacts,
)


@pytest.mark.integration
def test_hooke_multibody_smoke_writes_artifacts_with_timestamp(tmp_path: Path) -> None:
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
    assert summary_artifact["config"]["validation"]["acceptance_min"] == 0.3
    assert summary_artifact["config"]["validation"]["acceptance_max"] == 0.7
    assert "integration_test" in summary_artifact["config"]["tracking"]["tags"]

    metrics = summary_artifact["metrics"]
    assert math.isfinite(float(metrics["spenn/energy/mean"]))
    assert math.isfinite(float(metrics["spenn/local_energy/variance"]))
    assert math.isfinite(float(metrics["sampler/mean_pair_distance"]))
    assert float(metrics["sampler/min_pair_distance"]) > 0.0
    assert float(metrics["sampler/local_energy_sample_count"]) == 32.0
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
    assert _csv_row_count(run_dir / "plots" / "local_energy_samples.csv") == 32
    assert _csv_row_count(run_dir / "plots" / "pair_distance_samples.csv") == 96
    assert _csv_row_count(run_dir / "plots" / "cusp_slope_by_spin.csv") == 3
    assert _csv_row_count(run_dir / "plots" / "particle_antisymmetry.csv") == 2

    processed = process_run(run_dir)
    for name in [
        "spenn_observables.csv",
        "energy_trace.csv",
        "eval_metrics.csv",
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
    eval_rows = _csv_rows(run_dir / "data" / "eval_metrics.csv")
    assert len(eval_rows) == 1
    assert math.isfinite(float(eval_rows[0]["spenn/energy/mean"]))
    assert math.isfinite(float(eval_rows[0]["spenn/local_energy/variance"]))
    assert "exact/energy" not in eval_rows[0]
    assert "comparison/energy_abs_error" not in eval_rows[0]
    (run_dir / "metrics" / "eval_metrics.csv").unlink()
    legacy_processed_dir = tmp_path / "legacy_eval_metrics"
    legacy_processed = process_run(run_dir, output_dir=legacy_processed_dir)
    assert "eval_metrics.csv" in legacy_processed["data_files"]
    legacy_eval_rows = _csv_rows(legacy_processed_dir / "data" / "eval_metrics.csv")
    assert math.isfinite(float(legacy_eval_rows[0]["spenn/energy/mean"]))
    assert math.isfinite(float(legacy_eval_rows[0]["spenn/local_energy/variance"]))
    plausibility_rows = _csv_rows(run_dir / "data" / "energy_plausibility.csv")
    assert len(plausibility_rows) == 1
    assert plausibility_rows[0]["n_electrons"] == "3"
    assert plausibility_rows[0]["harmonic_omega"] == "0.5"
    assert plausibility_rows[0]["spatial_dim"] == "3"
    assert plausibility_rows[0]["specht_M"] == "2"
    assert plausibility_rows[0]["reference_available"] == "False"
    assert plausibility_rows[0]["reference_method"] == "none"
    assert plausibility_rows[0]["baseline_available"] == "False"

    reference_cfg = load_config(HOOKE_MULTIBODY_REFERENCE_CONFIG)
    reference_cfg.run = {"id": "integration_hooke_multibody_baseline_for_processing"}
    reference_summary = run_reference(reference_cfg)
    processed_with_reference = process_run(run_dir, reference_run=Path(reference_summary["output_dir"]))
    assert processed_with_reference["baseline_available"] is True
    assert (run_dir / "data" / "reference_observables.csv").exists()
    assert (run_dir / "data" / "reference_radial_density.csv").exists()
    assert (run_dir / "data" / "reference_pair_distance_density.csv").exists()
    baseline_rows = _csv_rows(run_dir / "data" / "energy_plausibility.csv")
    assert baseline_rows[0]["baseline_available"] == "True"
    assert baseline_rows[0]["baseline_method"] == "gaussian_hartree_variational"
    assert math.isfinite(float(baseline_rows[0]["baseline_energy"]))
    assert math.isfinite(float(baseline_rows[0]["energy_minus_baseline"]))
    reference_table_names = [
        "reference_observables.csv",
        "reference_radial_density.csv",
        "reference_pair_distance_density.csv",
    ]
    assert all((run_dir / "data" / name).exists() for name in reference_table_names)

    mismatched_reference_cfg = load_config(HOOKE_MULTIBODY_REFERENCE_CONFIG)
    mismatched_reference_cfg.run = {"id": "integration_hooke_multibody_mismatched_baseline"}
    mismatched_reference_cfg.system.harmonic_omega = 0.7
    mismatched_reference_summary = run_reference(mismatched_reference_cfg)
    in_place_mismatched_processed = process_run(
        run_dir,
        reference_run=Path(mismatched_reference_summary["output_dir"]),
    )
    assert in_place_mismatched_processed["baseline_available"] is False
    assert "reference_data_files" not in in_place_mismatched_processed
    assert all(not (run_dir / "data" / name).exists() for name in reference_table_names)

    mismatch_dir = tmp_path / "mismatched_reference_processing"
    mismatched_processed = process_run(
        run_dir,
        reference_run=Path(mismatched_reference_summary["output_dir"]),
        output_dir=mismatch_dir,
    )
    assert mismatched_processed["baseline_available"] is False
    assert "reference_data_files" not in mismatched_processed
    assert all(not (mismatch_dir / "data" / name).exists() for name in reference_table_names)
    mismatched_rows = _csv_rows(mismatch_dir / "data" / "energy_plausibility.csv")
    assert mismatched_rows[0]["baseline_available"] == "False"
    assert mismatched_rows[0]["baseline_energy"] == ""

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
    assert {row["harmonic_omega"] for row in plausibility_rows} == {"0.5"}
    assert {row["spatial_dim"] for row in plausibility_rows} == {"3"}
    assert {row["specht_M"] for row in plausibility_rows} == {"2"}
    assert {row["reference_method"] for row in plausibility_rows} == {"none"}
    assert {row["baseline_available"] for row in plausibility_rows} == {"False"}

    figures = plot_run(run_dir, figure_root=run_dir / "figures")
    assert len(figures) == 1
    assert figures[0].name.endswith("_spin_scan_energy.png")
    assert figures[0].exists()

    reference_cfg = load_config(HOOKE_MULTIBODY_REFERENCE_CONFIG)
    reference_cfg.run = {"id": "integration_hooke_multibody_spin_scan_baseline"}
    reference_summary = run_reference(reference_cfg)
    processed_with_reference = process_run(run_dir, reference_run=Path(reference_summary["output_dir"]))
    assert processed_with_reference["baseline_available"] is True
    assert (run_dir / "data" / "reference_observables.csv").exists()
    assert (run_dir / "data" / "reference_radial_density.csv").exists()
    assert (run_dir / "data" / "reference_pair_distance_density.csv").exists()
    baseline_rows = _csv_rows(run_dir / "data" / "energy_plausibility.csv")
    assert len(baseline_rows) == 2
    assert {row["baseline_available"] for row in baseline_rows} == {"True"}
    assert {row["baseline_method"] for row in baseline_rows} == {"gaussian_hartree_variational"}
    for row in baseline_rows:
        baseline_energy = float(row["baseline_energy"])
        energy_mean = float(row["energy_mean"])
        energy_minus_baseline = float(row["energy_minus_baseline"])
        assert math.isfinite(baseline_energy)
        assert math.isfinite(float(row["energy_abs_minus_baseline"]))
        assert energy_minus_baseline == pytest.approx(energy_mean - baseline_energy)
    figures_with_baseline = plot_run(run_dir, figure_root=run_dir / "figures_with_baseline")
    assert len(figures_with_baseline) == 1
    assert figures_with_baseline[0].name.endswith("_spin_scan_energy.png")
    assert figures_with_baseline[0].exists()


@pytest.mark.integration
def test_hooke_multibody_reference_summary_records_config_and_git() -> None:
    cfg = load_config(HOOKE_MULTIBODY_REFERENCE_CONFIG)
    cfg.run = {"id": "integration_hooke_multibody_reference"}

    summary = run_reference(cfg)
    artifact = _summary_json(Path(summary["output_dir"]))

    assert summary["run_id"] == "integration_hooke_multibody_reference"
    assert summary["reference_available"] is False
    assert summary["baseline_available"] is True
    assert summary["baseline_method"] == "gaussian_hartree_variational"
    assert math.isfinite(float(summary["baseline_energy"]))
    assert artifact["run_id"] == summary["run_id"]
    assert artifact["config"]["run"]["id"] == "integration_hooke_multibody_reference"
    assert artifact["config"]["run_id"] == "integration_hooke_multibody_reference"
    assert artifact["config"]["system"]["harmonic_omega"] == 0.5
    assert artifact["config"]["system"]["spatial_dim"] == 3
    assert artifact["reference_available"] is False
    assert artifact["baseline_available"] is True
    reference_rows = _csv_rows(Path(summary["output_dir"]) / "data" / "reference_observables.csv")
    assert reference_rows[0]["harmonic_omega"] == "0.5"
    assert reference_rows[0]["spatial_dim"] == "3"
    assert (Path(summary["output_dir"]) / "data" / "reference_radial_density.csv").exists()
    assert (Path(summary["output_dir"]) / "data" / "reference_pair_distance_density.csv").exists()
    assert "git_commit" in artifact["git"]
    assert "dirty_git_state" in artifact["git"]


def test_hooke_multibody_plot_numeric_parser_rejects_nonfinite_values() -> None:
    assert _to_float("nan") is None
    assert _to_float("inf") is None
    assert _to_float("-inf") is None
    assert _to_float("1.25") == 1.25


@pytest.mark.parametrize(
    "system",
    [
        {"n_electrons": 4, "harmonic_omega": 0.5, "spatial_dim": 3},
        {"n_electrons": 3, "harmonic_omega": 0.7, "spatial_dim": 3},
        {"n_electrons": 3, "harmonic_omega": 0.5, "spatial_dim": 2},
    ],
)
def test_hooke_multibody_baseline_comparison_requires_matching_system(system: dict[str, object]) -> None:
    reference_summary = {
        "baseline_available": True,
        "baseline_method": "gaussian_hartree_variational",
        "baseline_energy": 3.8,
        "config": {"system": {"n_electrons": 3, "harmonic_omega": 0.5, "spatial_dim": 3}},
    }

    result = _baseline_comparison(4.1, reference_summary, system=system)

    assert result["baseline_available"] is False
    assert result["baseline_energy"] == ""


def test_hooke_multibody_baseline_comparison_accepts_matching_system_and_legacy_n() -> None:
    reference_summary = {
        "baseline_available": True,
        "baseline_method": "gaussian_hartree_variational",
        "baseline_energy": 3.8,
        "config": {"system": {"n_electrons": 3, "harmonic_omega": 0.5, "spatial_dim": 3}},
    }

    result = _baseline_comparison(
        4.1,
        reference_summary,
        system={"n_electrons": 3, "harmonic_omega": 0.5, "spatial_dim": 3},
    )
    legacy_result = _baseline_comparison(4.1, reference_summary, n_electrons=3)

    assert result["baseline_available"] is True
    assert result["energy_minus_baseline"] == pytest.approx(0.3)
    assert legacy_result["baseline_available"] is True


def test_hooke_multibody_spin_scan_plot_draws_baseline_line(tmp_path: Path) -> None:
    fake_plt = _FakePyplot()

    written = _plot_spin_scan(
        fake_plt,
        [
            {"n_up": "2", "n_down": "1", "energy_mean": "4.0", "local_energy_variance": "0.1", "acceptance_rate": "0.5"},
            {"n_up": "1", "n_down": "2", "energy_mean": "4.2", "local_energy_variance": "0.2", "acceptance_rate": "0.6"},
        ],
        tmp_path / "scan.png",
        baseline=3.8,
    )

    assert written is True
    assert (tmp_path / "scan.png").exists()
    assert fake_plt.axes[0].horizontal_lines == [{"y": 3.8, "label": "Gaussian Hartree"}]


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_json(run_dir: Path) -> dict[str, object]:
    with (run_dir / "artifacts" / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


class _FakeAxis:
    def __init__(self) -> None:
        self.horizontal_lines: list[dict[str, object]] = []

    def bar(self, *args, **kwargs) -> None:
        return None

    def axhline(self, y: float, **kwargs) -> None:
        self.horizontal_lines.append({"y": y, "label": kwargs.get("label")})

    def legend(self, **kwargs) -> None:
        return None

    def set_ylabel(self, value: str) -> None:
        return None

    def set_title(self, value: str) -> None:
        return None

    def set_xticks(self, value: list[int]) -> None:
        return None

    def set_xticklabels(self, value: list[str]) -> None:
        return None

    def set_xlabel(self, value: str) -> None:
        return None

    def grid(self, *args, **kwargs) -> None:
        return None


class _FakeFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, path: Path, **kwargs) -> None:
        path.write_bytes(b"fake png")


class _FakePyplot:
    def __init__(self) -> None:
        self.axes = [_FakeAxis(), _FakeAxis(), _FakeAxis()]

    def subplots(self, *args, **kwargs) -> tuple[_FakeFigure, list[_FakeAxis]]:
        return _FakeFigure(), self.axes

    def close(self, figure: _FakeFigure) -> None:
        return None
