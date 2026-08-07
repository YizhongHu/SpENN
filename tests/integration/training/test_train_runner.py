"""Integration test: the Train runner executes a VMC smoke loop end-to-end.

Drives the full configured path -- ``run_from_config`` -> ``Train`` runner ->
``make_optimizer`` -> ``VMCTrainer.fit`` -> sampler -> Hamiltonian terms ->
surrogate loss -> optimizer step -> loggers/callbacks -- and asserts the
standard run artifacts and finite ``train`` metrics. No convergence assertions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from tpen.checkpoint import resolve_checkpoint_dir
from tpen.run import run_from_config

FIXTURE = Path(__file__).resolve().parents[1] / "artifacts" / "training" / "vmc_smoke.yaml"

ALLOWED_NONFINITE_KEYS = {"energy_stderr"}

# Every durable phase key the wired `TrainPhaseTiming` reports for one completed
# training iteration. Each name is owned by a concrete `TrainingPhase` type, so
# this tuple pins the public spelling of the whole `train/perf` phase surface.
PHASE_TIMING_KEYS = (
    "sampling_time_sec",
    "batch_build_time_sec",
    "local_energy_time_sec",
    "forward_time_sec",
    "objective_time_sec",
    "backward_time_sec",
    "optimizer_step_time_sec",
    "post_step_metrics_time_sec",
)


def _run(tmp_path: Path):
    cfg = OmegaConf.load(FIXTURE)
    cfg.run.root = str(tmp_path)
    exit_code = run_from_config(cfg, config_path=str(FIXTURE), command="pytest")
    assert exit_code == 0
    run_dirs = list(tmp_path.glob("vmc_smoke/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    return run_dirs[0]


def test_train_runner_writes_standard_artifacts(tmp_path) -> None:
    run_dir = _run(tmp_path)

    for artifact in (
        "config.yaml",
        "resolved_config.yaml",
        "metadata.json",
        "status.json",
        "metrics.csv",
        "metrics.jsonl",
        "run_start.json",
        "checkpoints/latest.json",
        # Cadence 2 writes step 2; train_end still writes terminal step 3.
        "checkpoints/step_000002/COMPLETE",
        # Checkpoint steps use the resume cursor, so a 3-step run ends at step 3.
        "checkpoints/step_000003/manifest.json",
        "checkpoints/step_000003/model.pt",
        "checkpoints/step_000003/COMPLETE",
    ):
        assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["hardware"]["hostname"]
    assert "cpu_count_logical" in metadata["hardware"]
    assert "cuda_available" in metadata["hardware"]
    assert metadata["runtime"]["device"] == "cpu"
    assert metadata["runtime"]["dtype"] == "float64"
    assert "python_version" in metadata["runtime"]
    assert "slurm" in metadata

    # Three attempted iterations, each of which applied its optimizer update.
    trainer_state = json.loads((run_dir / "checkpoints/step_000003/trainer.json").read_text())
    assert trainer_state == {"next_iteration": 3, "completed_updates": 3}

    latest = json.loads((run_dir / "checkpoints/latest.json").read_text())
    assert latest["checkpoint_dir"] == "step_000003"
    assert latest["step"] == 3
    assert resolve_checkpoint_dir(run_dir / "checkpoints") == run_dir / "checkpoints/step_000003"


def test_train_runner_logs_finite_train_metrics(tmp_path) -> None:
    run_dir = _run(tmp_path)

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    train_records = [record["metrics"] for record in records if record.get("namespace") == "train"]
    sampler_records = [record["metrics"] for record in records if record.get("namespace") == "train/sampler"]
    perf_records = [record["metrics"] for record in records if record.get("namespace") == "train/perf"]
    runtime_records = [record["metrics"] for record in records if record.get("namespace") == "runtime"]
    assert len(train_records) == 3, "expected one train record per step"
    assert len(sampler_records) == 3, "expected one train/sampler record per step"
    # Two callbacks write `train/perf`: TrainStepTiming reports whole-step wall
    # time at `step_end`, TrainPhaseTiming the typed phase breakdown at
    # `TrainingIterationCompleted`. Split them by key rather than by position.
    step_timing_records = [record for record in perf_records if "step_time_sec" in record]
    phase_timing_records = [record for record in perf_records if "step_time_sec" not in record]
    assert len(step_timing_records) == 3, "expected one step-timing record per step"
    assert len(phase_timing_records) == 3, "expected one phase-timing record per step"
    assert any("wall_time_sec" in record for record in runtime_records)

    last = train_records[-1]
    for key in (
        "loss",
        "energy",
        "energy_variance",
        "local_energy_n_finite",
        "local_energy_finite_fraction",
        "logabs_mean",
    ):
        assert key in last, f"missing metric: {key}"
    # The physical training estimator is logged as `energy`, never `energy_mean`.
    assert "energy_mean" not in last
    assert not any(key.startswith("sampler.") for key in last)
    assert "acceptance_rate" in sampler_records[-1]
    assert "n_walkers" in sampler_records[-1]
    assert "step_time_sec" in step_timing_records[-1]
    assert "step_time_sec_rolling_mean" in step_timing_records[-1]

    # This is the only test that drives several phase types through the real
    # RunContext -> _dispatch_occurrence -> Callback.handle_occurrence path, so
    # pin the exact key set and require finite durations, not mere presence.
    for record in phase_timing_records:
        assert set(record) == set(PHASE_TIMING_KEYS), f"unexpected phase keys: {sorted(record)}"
        for key in PHASE_TIMING_KEYS:
            value = record[key]
            assert isinstance(value, (int, float)), f"non-numeric phase metric {key}={value!r}"
            assert math.isfinite(value), f"non-finite phase metric {key}={value}"

    # JSONL serialization with allow_nan=False would already have failed the run
    # on any non-finite value; assert finiteness directly for good measure.
    for record in train_records:
        for key, value in record.items():
            if key in ALLOWED_NONFINITE_KEYS or not isinstance(value, (int, float)):
                continue
            assert math.isfinite(value), f"non-finite metric {key}={value}"
