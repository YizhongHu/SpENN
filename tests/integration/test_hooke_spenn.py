"""Integration tests for Hooke exact-vs-SpENN comparison runs."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from experiments.hooke.run_spenn import run
from tests.helpers import HOOKE_INTEGRATION_ARTIFACTS, assert_hooke_run_artifacts, load_test_config


@pytest.mark.parametrize("sector", ["singlet", "triplet"])
@pytest.mark.integration
def test_hooke_spenn_writes_comparison_artifacts_without_plot_data(sector: str) -> None:
    cfg = load_test_config(HOOKE_INTEGRATION_ARTIFACTS / f"spenn_{sector}.yaml")
    cfg.run_id = f"pytest_hooke_spenn_{sector}"

    summary = run(cfg, forwarded_overrides=[f"run_id=pytest_hooke_spenn_{sector}"])
    summary_artifact = assert_hooke_run_artifacts(summary["output_dir"], expect_plot_data=False)

    assert summary["can_reach_goal"] is True
    assert summary["run_id"] == f"pytest_hooke_spenn_{sector}"
    assert summary_artifact["config"]["run_mode"] == "integration"
    assert summary_artifact["config"]["artifacts"]["write_plot_data"] is False
    expected_tags = {"integration_test", "hooke", "spenn", sector}
    assert expected_tags <= set(summary_artifact["config"]["tracking"]["tags"])

    metrics = summary_artifact["metrics"]
    assert _keys_with_prefix(metrics, "exact/"), "missing exact-reference metrics"
    assert _keys_with_prefix(metrics, "spenn/"), "missing SpENN metrics"
    assert _keys_with_prefix(metrics, "comparison/"), "missing exact-vs-SpENN comparison metrics"

    error_key = _first_key_containing(metrics, prefix="comparison/", fragments=("energy", "abs_error"))
    assert error_key is not None, "missing comparison energy absolute-error metric"
    error_value = float(metrics[error_key])
    assert math.isfinite(error_value)
    assert error_value <= cfg.validation.comparison_energy_abs_error_tolerance
    assert math.isfinite(float(metrics["comparison/radial_logabs_rmse"]))
    assert float(metrics["comparison/radial_logabs_rmse"]) <= cfg.validation.radial_logabs_rmse_tolerance
    assert math.isfinite(float(metrics["comparison/cusp_slope_error"]))
    assert abs(float(metrics["comparison/cusp_slope_error"])) <= cfg.validation.cusp_slope_tolerance
    assert float(metrics["comparison/sign_alignment_accuracy"]) >= cfg.validation.sign_alignment_min
    assert summary_artifact["config"]["diagnostics"]["exchange"]["exchange_mode"] == "particle_antisymmetric"
    assert float(metrics["comparison/antisymmetry_error_max"]) <= cfg.validation.exchange_error_tolerance
    assert float(metrics["comparison/sign_flip_accuracy"]) >= cfg.validation.sign_flip_min

    comparison_metrics = Path(summary["output_dir"]) / "metrics" / "comparison_metrics.csv"
    assert comparison_metrics.exists()
    train_metrics = _read_metric_rows(Path(summary["output_dir"]) / "metrics" / "train_metrics.csv")
    assert train_metrics
    assert "training/vmc_step" in train_metrics[0]
    assert "training/supervised_loss" not in train_metrics[0]
    assert {row.get("phase") for row in train_metrics} <= {None, ""}


def _keys_with_prefix(metrics: Mapping[str, Any], prefix: str) -> set[str]:
    return {key for key in metrics if key.startswith(prefix)}


def _first_key_containing(metrics: Mapping[str, Any], *, prefix: str, fragments: tuple[str, ...]) -> str | None:
    for key in metrics:
        if key.startswith(prefix) and all(fragment in key for fragment in fragments):
            return key
    return None


def _read_metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
