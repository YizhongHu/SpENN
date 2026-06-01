"""Integration tests for exact Hooke benchmark runs."""

from __future__ import annotations

import pytest

from experiments.hooke.run_exact import run
from tests.helpers import HOOKE_INTEGRATION_ARTIFACTS, assert_hooke_run_artifacts, load_test_config


@pytest.mark.parametrize(
    ("sector", "exact_energy"),
    [
        ("singlet", 2.0),
        ("triplet", 1.25),
    ],
)
@pytest.mark.integration
def test_exact_hooke_run_writes_integration_artifacts_without_plot_data(sector: str, exact_energy: float) -> None:
    cfg = load_test_config(HOOKE_INTEGRATION_ARTIFACTS / f"{sector}.yaml")
    cfg.run_id = f"pytest_hooke_{sector}"

    summary = run(cfg, forwarded_overrides=[f"run_id=pytest_hooke_{sector}"])
    summary_artifact = assert_hooke_run_artifacts(summary["output_dir"], expect_plot_data=False)

    assert summary["can_reach_goal"] is True
    assert summary["energy_exact"] == exact_energy
    assert summary["energy_abs_error"] <= cfg.validation.sample_energy_tolerance
    assert summary_artifact["config"]["run_mode"] == "integration"
    assert summary_artifact["config"]["artifacts"]["write_plot_data"] is False
    assert "integration_test" in summary_artifact["config"]["tracking"]["tags"]
    metrics = summary_artifact["metrics"]
    exchange_mode = summary_artifact["config"]["diagnostics"]["exchange"]["exchange_mode"]
    if sector == "singlet":
        assert exchange_mode == "spatial_singlet"
        assert float(metrics["symmetry/symmetry_error_max"]) <= cfg.validation.exchange_error_tolerance
        assert float(metrics["symmetry/sign_match_accuracy"]) >= cfg.validation.sign_match_min
    else:
        assert exchange_mode == "particle_antisymmetric"
        assert float(metrics["symmetry/antisym_error_max"]) <= cfg.validation.exchange_error_tolerance
        assert float(metrics["symmetry/sign_flip_accuracy"]) >= cfg.validation.sign_flip_min
