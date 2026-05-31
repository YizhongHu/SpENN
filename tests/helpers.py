"""Shared helpers for unit and integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
TEST_ARTIFACTS = ROOT / "tests" / "artifacts"
INTEGRATION_ARTIFACTS = ROOT / "tests" / "integration" / "artifacts"
HOOKE_INTEGRATION_ARTIFACTS = INTEGRATION_ARTIFACTS / "hooke"


def load_test_config(path: Path) -> DictConfig:
    """Load a test-owned OmegaConf config.

    Parameters
    ----------
    path : pathlib.Path
        Absolute config path, or a path relative to the repository root.

    Returns
    -------
    omegaconf.DictConfig
        Loaded configuration.
    """

    config_path = path if path.is_absolute() else ROOT / path
    return OmegaConf.load(config_path)


def summary_path(output_dir: str | Path) -> Path:
    """Return the summary artifact path for a run directory.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Run output directory.

    Returns
    -------
    pathlib.Path
        Path to ``artifacts/summary.json``.
    """

    return Path(output_dir) / "artifacts" / "summary.json"


def load_summary(output_dir: str | Path) -> dict[str, Any]:
    """Load a run summary artifact.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Run output directory.

    Returns
    -------
    dict
        Parsed summary JSON.
    """

    with summary_path(output_dir).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def assert_hooke_run_artifacts(output_dir: str | Path, *, expect_plot_data: bool) -> dict[str, Any]:
    """Assert the standard Hooke run artifact layout.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Run output directory returned by ``experiments.hooke.run_exact.run``.
    expect_plot_data : bool
        Whether CSV plot-data artifacts should exist.

    Returns
    -------
    dict
        Parsed summary JSON for additional assertions.
    """

    run_dir = Path(output_dir)
    required = [
        run_dir / ".hydra" / "config.yaml",
        run_dir / ".hydra" / "overrides.yaml",
        run_dir / "artifacts" / "summary.json",
        run_dir / "checkpoints" / "final_model.pt",
        run_dir / "metrics" / "energy_trace.csv",
        run_dir / "metrics" / "sampler_metrics.csv",
        run_dir / "metrics" / "train_metrics.csv",
    ]
    for path in required:
        assert path.exists(), f"missing expected artifact: {path}"
    plot_paths = [
        run_dir / "plots" / "local_energy_histogram.csv",
        run_dir / "plots" / "r12_histogram.csv",
        run_dir / "plots" / "cusp_diagnostic_plot.csv",
        run_dir / "plots" / "wavefunction_radial_cut.csv",
    ]
    for path in plot_paths:
        assert path.exists() is expect_plot_data, f"unexpected plot-data state for {path}"
    plots_dir = run_dir / "plots"
    if expect_plot_data:
        assert plots_dir.exists(), f"missing plots directory: {plots_dir}"
    else:
        assert not plots_dir.exists() or not any(plots_dir.iterdir()), f"unexpected plot artifacts under {plots_dir}"
    return load_summary(run_dir)
