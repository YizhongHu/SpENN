"""Integration tests for Hooke multibody smoke runs."""

from __future__ import annotations

import math
import re

import pytest

from experiments.hooke_multibody.run_spenn import load_config, run
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
    assert "exact/energy" not in metrics
    assert "comparison/energy_abs_error" not in metrics
