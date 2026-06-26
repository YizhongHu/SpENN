"""PR7 workflow-readiness tests for legacy Hooke pair test configs.

Runs the real ``experiments/hooke/configs/smoke`` train and eval configs
end-to-end through ``run_from_config`` and asserts the run-directory artifacts,
metric namespaces, and timing/perf metrics that benchmark runs rely on. Also
keeps the SLURM submission scripts line-oriented and batch-safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from spenn.run import run_from_config

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "experiments" / "hooke" / "configs" / "smoke"
SLURM_DIR = ROOT / "experiments" / "hooke" / "slurm"

TRAIN_CONFIG = CONFIG_DIR / "pair_train.yaml"
EVAL_CONFIG = CONFIG_DIR / "pair_eval.yaml"

EXPECTED_ARTIFACTS = (
    "config.yaml",
    "resolved_config.yaml",
    "metadata.json",
    "status.json",
    "metrics.csv",
    "metrics.jsonl",
)


def _run_config(config: Path, tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    """Run one legacy test config into ``tmp_path`` and return the run directory."""

    cfg = OmegaConf.load(config)
    cfg.run.root = str(tmp_path)
    cfg.run.timezone = "UTC"  # tests log in UTC; experiments use America/New_York
    # Keep the process-global "spenn" terminal logger unconfigured: it would
    # set propagate=False and break caplog-based unit tests that run later.
    cfg.terminal.enabled = False
    # Tests must stay hermetic: if the on-disk config enables the W&B logger,
    # force offline mode and keep its files inside tmp_path.
    for logger_cfg in cfg.loggers:
        if str(logger_cfg.get("_target_", "")).endswith("WandB"):
            logger_cfg["mode"] = "offline"
            logger_cfg["dir"] = str(tmp_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    exit_code = run_from_config(cfg, config_path=str(config), command="pytest")
    assert exit_code == 0
    run_dirs = list(tmp_path.glob("hooke_pair_smoke/pair/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    return run_dirs[0]


def _metrics_by_namespace(run_dir: Path) -> dict[str, set[str]]:
    """Map each logged namespace to the set of metric keys it received."""

    namespaces: dict[str, set[str]] = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        namespaces.setdefault(record["namespace"], set()).update(record["metrics"])
    return namespaces


def test_legacy_test_configs_load_and_interpolate() -> None:
    """Both legacy test configs load and fully resolve without a live run."""

    for config in (TRAIN_CONFIG, EVAL_CONFIG):
        cfg = OmegaConf.load(config)
        # run.dir is populated by run setup; give it a value so ${run.dir}
        # interpolations resolve during this static check.
        cfg.run.dir = "/tmp/spenn-config-check"
        OmegaConf.resolve(cfg)


def test_legacy_train_config_runs_and_logs_perf_metrics(tmp_path: Path) -> None:
    """The legacy train config runs and emits artifacts plus perf metrics."""

    run_dir = _run_config(TRAIN_CONFIG, tmp_path)

    for artifact in (*EXPECTED_ARTIFACTS, "run_start.json", "checkpoints/latest.json"):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"

    namespaces = _metrics_by_namespace(run_dir)
    assert "energy" in namespaces["train"]
    assert "acceptance_rate" in namespaces["train/sampler"]
    assert "passed" in namespaces["checks/data_integrity"]
    assert "passed" in namespaces["checks/equivariance/full_model"]
    assert "wall_time_sec" in namespaces["runtime"]
    assert {"step_time_sec", "step_time_sec_rolling_mean"} <= namespaces["train/perf"]


def test_legacy_eval_config_runs_and_emits_energy_metrics(tmp_path: Path) -> None:
    """The legacy eval config runs EnergyEvaluation with reference errors."""

    run_dir = _run_config(EVAL_CONFIG, tmp_path)

    for artifact in EXPECTED_ARTIFACTS:
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"

    namespaces = _metrics_by_namespace(run_dir)
    assert {"energy", "energy_stderr", "energy_error", "energy_abs_error"} <= namespaces["eval"]
    # return_terms=true + include_terms=true produce per-term decompositions.
    assert "energy_term_kinetic" in namespaces["eval"]
    # Sampler stats use the namespaced path, not dotted keys inside "eval".
    assert "acceptance_rate" in namespaces["eval/sampler"]
    assert not any("." in key for key in namespaces["eval"])
    assert "wall_time_sec" in namespaces["eval/perf"]
    assert "time_sec" in namespaces["diagnostics/energy"]
    assert "wall_time_sec" in namespaces["runtime"]


def test_eval_config_keeps_reference_energy_out_of_system() -> None:
    """Reference energies live under ``references``, never ``system``."""

    cfg = OmegaConf.load(EVAL_CONFIG)
    assert OmegaConf.select(cfg, "references.exact_energy") is not None
    assert OmegaConf.select(cfg, "system.exact_energy") is None


def test_metadata_records_hardware_and_runtime_provenance(tmp_path: Path) -> None:
    """metadata.json carries hardware/runtime (and SLURM, when present) blocks."""

    run_dir = _run_config(EVAL_CONFIG, tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text())

    hardware = metadata["hardware"]
    assert hardware["hostname"]
    assert isinstance(hardware["cuda_available"], bool)
    assert isinstance(hardware["cuda_device_count"], int)

    runtime = metadata["runtime"]
    assert runtime["device"] == "cpu"
    assert runtime["dtype"] == "float64"
    assert runtime["python_version"]
    assert "torch_version" in runtime

    # SLURM block exists; contents depend on whether the test runs under SLURM.
    assert isinstance(metadata["slurm"], dict)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_legacy_train_config_runs_on_cuda(tmp_path: Path) -> None:
    """GPU smoke path: the train config runs with runtime.device=cuda."""

    run_dir = _run_config(
        TRAIN_CONFIG,
        tmp_path,
        overrides={"runtime": {"device": "cuda"}, "timing": {"cuda_synchronize": True}},
    )

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["runtime"]["device"] == "cuda"
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"


@pytest.mark.parametrize(
    "config,job_type",
    [(TRAIN_CONFIG, "train"), (EVAL_CONFIG, "eval")],
    ids=["train", "eval"],
)
def test_wandb_offline_logger_creates_event_files(
    config: Path,
    job_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WandB offline logger runs end-to-end and writes offline event files."""

    pytest.importorskip("wandb")

    monkeypatch.setenv("WANDB_SILENT", "true")

    cfg = OmegaConf.load(config)
    cfg.run.root = str(tmp_path)
    cfg.run.timezone = "UTC"
    cfg.terminal.enabled = False
    # Pass dir explicitly so each test's wandb files land in its own tmp_path
    # regardless of any WANDB_DIR global state left by a previous test.
    cfg.loggers.append(
        OmegaConf.create(
            {
                "_target_": "spenn.logging.WandB",
                "project": "spenn-qmc",
                "mode": "offline",
                "dir": str(tmp_path),
                "group": "hooke_pair_smoke",
                "job_type": job_type,
                "mirror_scalars": True,
                "dashboard_aliases": True,
                "health_flags": True,
                "log_artifacts": False,
            }
        )
    )

    exit_code = run_from_config(cfg, config_path=str(config), command="pytest")
    assert exit_code == 0

    # Verify that at least one offline W&B run directory with an event file exists.
    offline_runs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert offline_runs, f"no offline W&B run dirs found under {tmp_path / 'wandb'}"
    event_files = [f for run_dir in offline_runs for f in run_dir.iterdir() if f.suffix == ".wandb"]
    assert event_files, f"no .wandb event files in {offline_runs}"


@pytest.mark.parametrize(
    "script", ["train_pair_smoke.sbatch", "eval_pair_smoke.sbatch"]
)
def test_slurm_scripts_are_batch_safe(script: str) -> None:
    """SLURM scripts stay unbuffered, line-oriented, and exit-code preserving."""

    text = (SLURM_DIR / script).read_text()
    assert text.startswith("#!/bin/bash")
    assert "export PYTHONUNBUFFERED=1" in text
    assert "python -u" in text
    assert "set -euo pipefail" in text
    assert 'echo "[slurm] job_id=' in text
    assert "OMP_NUM_THREADS" in text
    # Submission forwards dotlist overrides to run.py.
    assert '"$@"' in text
