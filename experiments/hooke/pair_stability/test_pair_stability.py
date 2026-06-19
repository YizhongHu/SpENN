"""Study-level tests for the pair-stability experiment package (PR8.8).

These cover grid/choice consistency, orchestrator dry-run artifacts, staged
results layout, attempt provenance, and a one-grid-point smoke run through the
normal run path. Reusable model-component math is tested under ``tests/`` and is
intentionally not retested here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
CONFIGS = STUDY_DIR / "configs"
PAIR_STABILITY = CONFIGS / "pair_stability.yaml"
PAIR_VALIDATION = CONFIGS / "pair_validation.yaml"
GRID = CONFIGS / "grid.yaml"

if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import collect  # noqa: E402
import orchestrator  # noqa: E402
import run_utils  # noqa: E402
import select_champions  # noqa: E402


# ---------------------------------------------------------------------------
# Attempt-id timezone / format
# ---------------------------------------------------------------------------
def test_new_attempt_id_uses_study_timezone() -> None:
    import re
    from datetime import datetime

    # Study timestamps share the run-log wall clock (America/New_York).
    assert str(run_utils.STUDY_TIMEZONE) == "America/New_York"
    # Summer is EDT (-0400), winter is EST (-0500); ids stay dir-safe.
    summer = datetime(2026, 6, 19, 0, 0, 0, tzinfo=run_utils.STUDY_TIMEZONE)
    winter = datetime(2026, 1, 15, 0, 0, 0, tzinfo=run_utils.STUDY_TIMEZONE)
    assert run_utils.new_attempt_id(summer) == "20260619T000000-0400"
    assert run_utils.new_attempt_id(winter) == "20260115T000000-0500"
    # The no-arg form is study-local and matches the id grammar.
    assert re.fullmatch(r"\d{8}T\d{6}[+-]\d{4}", run_utils.new_attempt_id())


def test_attempt_ids_sorted_and_skips_latest(tmp_path: Path) -> None:
    base = tmp_path / "stage"
    (base / "20260101T000000-0500").mkdir(parents=True)
    (base / "20260619T000000-0400").mkdir()
    (base / "latest.json").write_text("{}")
    try:
        (base / "latest").symlink_to("20260619T000000-0400")
    except OSError:
        pass
    # Chronological by name; the latest symlink and latest.json are excluded.
    assert run_utils.attempt_ids(base) == ["20260101T000000-0500", "20260619T000000-0400"]
    assert run_utils.attempt_ids(base / "missing") == []


# ---------------------------------------------------------------------------
# Grid / choice-library consistency
# ---------------------------------------------------------------------------
def test_grid_values_exist_in_choices() -> None:
    grid = OmegaConf.load(GRID)
    config = OmegaConf.load(PAIR_STABILITY)
    points = orchestrator.expand_grid(OmegaConf.to_container(grid.grid, resolve=True))
    # validate_grid raises if any architecture/normalization is unknown.
    orchestrator.validate_grid(points, config)

    architectures = set(orchestrator.architecture_tags(config))
    normalizations = orchestrator.normalization_names(config)
    for axis_value in grid.grid.architecture:
        assert str(axis_value) in architectures
    for axis_value in grid.grid.normalization:
        assert str(axis_value) in normalizations


def test_grid_does_not_include_raw_no_envelope() -> None:
    grid = OmegaConf.load(GRID)
    architectures = {str(value) for value in grid.grid.architecture}
    assert "raw_no_envelope" not in architectures
    assert not any(name.endswith("_no_envelope") for name in architectures)


def test_main_architectures_all_include_gaussian_envelope() -> None:
    config = OmegaConf.load(PAIR_STABILITY)
    grid = OmegaConf.load(GRID)
    for name in grid.grid.architecture:
        envelope = config.choices.architecture[name].envelope
        assert str(envelope._target_).endswith("HookeGaussianEnvelope")


# ---------------------------------------------------------------------------
# Orchestrator dry run / manifest / layout
# ---------------------------------------------------------------------------
def _small_grid(tmp_path: Path) -> Path:
    """Write a tiny grid (including a known target run) pointing at real configs."""

    grid = {
        "study": "pair_stability",
        "config": str(PAIR_STABILITY),
        "validation_config": str(PAIR_VALIDATION),
        "results_root": str(tmp_path / "results"),
        "grid": {
            "architecture": ["raw_envelope", "hermite_o3_envelope"],
            "normalization": ["N0", "N2"],
            "lr": [1.0e-3],
            "channels": [16],
            "seed": [0],
        },
    }
    path = tmp_path / "grid.yaml"
    OmegaConf.save(OmegaConf.create(grid), path)
    return path


TARGET_RUN_ID = "arch-hermite_o3_envelope_norm-N2_lr-1e-3_ch-16_seed-0"
ATTEMPT = "20260619T000000-0400"


def _plan(tmp_path: Path) -> Path:
    grid = _small_grid(tmp_path)
    results_root = tmp_path / "results"
    code = orchestrator.main(
        ["--grid", str(grid), "--results-root", str(results_root), "--attempt-id", ATTEMPT, "--backend", "plan"]
    )
    assert code == 0
    return results_root


def test_orchestrator_dry_run_writes_grid_attempt(tmp_path: Path) -> None:
    results_root = _plan(tmp_path)
    attempt = orchestrator.grid_attempt_dir(results_root, ATTEMPT)
    assert (attempt / "manifest.json").is_file()
    assert (attempt / "commands.sh").is_file()
    assert (attempt / "grid.yaml").is_file()
    assert (attempt / "pair_stability.yaml").is_file()
    assert (attempt / "jobs" / f"{TARGET_RUN_ID}.json").is_file()
    assert (results_root / "00_grid" / "latest.json").is_file()


def test_manifest_contains_expected_run_ids_and_overrides(tmp_path: Path) -> None:
    results_root = _plan(tmp_path)
    manifest = json.loads((orchestrator.grid_attempt_dir(results_root, ATTEMPT) / "manifest.json").read_text())
    run_ids = {job["run_id"] for job in manifest["jobs"]}
    assert TARGET_RUN_ID in run_ids
    assert manifest["n_jobs"] == 4  # 2 architectures x 2 normalizations

    job = next(job for job in manifest["jobs"] if job["run_id"] == TARGET_RUN_ID)
    assert "run_parameters.architecture=hermite_o3_envelope" in job["overrides"]
    assert "run_parameters.normalization=N2" in job["overrides"]
    assert "run_parameters.channels=16" in job["overrides"]
    assert "run_parameters.seed=0" in job["overrides"]
    assert job["choices"]["architecture"] == "hermite_o3_envelope"
    assert job["choices"]["channels"] == 16


def test_manifest_train_and_validation_dirs_follow_expected_layout(tmp_path: Path) -> None:
    results_root = _plan(tmp_path)
    manifest = json.loads((orchestrator.grid_attempt_dir(results_root, ATTEMPT) / "manifest.json").read_text())
    job = next(job for job in manifest["jobs"] if job["run_id"] == TARGET_RUN_ID)
    assert job["train_dir"] == str(results_root / "01_train" / TARGET_RUN_ID)
    assert job["validation_dir"] == str(results_root / "02_validation" / TARGET_RUN_ID)
    assert job["train_attempt_dir"] == str(results_root / "01_train" / TARGET_RUN_ID / ATTEMPT)


def test_commands_sh_contains_run_commands(tmp_path: Path) -> None:
    # The repo has no @hydra.main app, so submission uses the canonical run.py
    # command path (handed to the Submitit launcher by the submitit backend).
    results_root = _plan(tmp_path)
    commands = (orchestrator.grid_attempt_dir(results_root, ATTEMPT) / "commands.sh").read_text()
    assert "run.py" in commands
    assert "--config" in commands
    assert "run_parameters.architecture=hermite_o3_envelope" in commands
    assert f"run.run_id={TARGET_RUN_ID}/{ATTEMPT}" in commands


def test_train_run_dir_uses_stage_attempt_layout(tmp_path: Path) -> None:
    results_root = _plan(tmp_path)
    manifest = json.loads((orchestrator.grid_attempt_dir(results_root, ATTEMPT) / "manifest.json").read_text())
    job = next(job for job in manifest["jobs"] if job["run_id"] == TARGET_RUN_ID)
    overrides = job["overrides"]
    assert f"run.root={results_root / '01_train'}" in overrides
    assert "run.layout=flat" in overrides
    assert f"run.run_id={TARGET_RUN_ID}/{ATTEMPT}" in overrides


def test_orchestrator_always_injects_run_timezone_override(tmp_path: Path) -> None:
    grid = _small_grid(tmp_path)
    # The orchestrator owns the timezone and always injects it (the config is null).
    orchestrator.main(
        ["--grid", str(grid), "--results-root", str(tmp_path / "a"), "--attempt-id", ATTEMPT, "--backend", "plan"]
    )
    commands = (orchestrator.grid_attempt_dir(tmp_path / "a", ATTEMPT) / "commands.sh").read_text()
    assert "run.timezone=America/New_York" in commands
    # --timezone selects the injected zone.
    orchestrator.main(
        ["--grid", str(grid), "--results-root", str(tmp_path / "b"), "--attempt-id", ATTEMPT,
         "--backend", "plan", "--timezone", "UTC"]
    )
    commands_utc = (orchestrator.grid_attempt_dir(tmp_path / "b", ATTEMPT) / "commands.sh").read_text()
    assert "run.timezone=UTC" in commands_utc


def test_validation_records_source_train_attempt(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    point = {"architecture": "hermite_o3_envelope", "normalization": "N2", "lr": 1.0e-3, "channels": 16, "seed": 0}
    jobs = orchestrator.build_jobs(
        [point],
        attempt_id="T1",
        results_root=results_root,
        config=PAIR_STABILITY,
        tags_by_architecture={"hermite_o3_envelope": ["main"]},
        launcher="plan",
    )
    plan = orchestrator.plan_validation_attempt(
        jobs[0],
        results_root=results_root,
        train_attempt_id="T1",
        validation_attempt_id="V1",
        validation_config=PAIR_VALIDATION,
    )
    source_path = Path(plan["validation_attempt_dir"]) / "source_train_attempt.json"
    assert source_path.is_file()
    source = json.loads(source_path.read_text())
    assert source["run_id"] == TARGET_RUN_ID
    assert source["train_attempt_id"] == "T1"
    assert source["checkpoint_path"].endswith("checkpoints")
    assert "load.path=" in plan["command"]


def test_pair_validation_config_model_and_tasks_instantiate() -> None:
    import spenn.config  # noqa: F401 - registers the basis_feature_dim resolver
    from hydra.utils import instantiate

    cfg = OmegaConf.load(PAIR_VALIDATION)
    cfg.run_parameters.architecture = "hermite_o3_envelope"
    cfg.run_parameters.normalization = "N2"
    cfg.run_parameters.channels = 4
    OmegaConf.update(cfg, "run.dir", str(tmp_run_dir()), force_add=True)
    model = instantiate(cfg.model)
    evaluator = instantiate(cfg.evaluator)
    assert type(model).__name__ == "SpENNWaveFunction"
    assert [task.name for task in evaluator.tasks] == [
        "cusp",
        "tail",
        "stratified_geometry",
        "hooke_orbital",
        "energy",
    ]
    # Every evaluation task routes its artifacts under the validation run dir.
    for task in evaluator.tasks:
        assert str(task.output_dir).startswith(str(tmp_run_dir()))


def tmp_run_dir() -> Path:
    return Path("/tmp/pair_stability_eval_check")


# ---------------------------------------------------------------------------
# Collection / selection
# ---------------------------------------------------------------------------
def _fake_validation_attempt(
    results_root: Path, run_id: str, attempt_id: str, *, status: str, energy_error: float
) -> None:
    attempt = orchestrator.validation_attempt_dir(results_root, run_id, attempt_id)
    (attempt / "cusp").mkdir(parents=True, exist_ok=True)
    (attempt / "tail").mkdir(parents=True, exist_ok=True)
    (attempt / "status.json").write_text(json.dumps({"status": status}))
    (attempt / "source_train_attempt.json").write_text(
        json.dumps({"run_id": run_id, "train_attempt_id": "T1", "checkpoint_path": "ckpt"})
    )
    records = [
        {"namespace": "eval/energy", "metrics": {"reference_abs_error": energy_error, "energy_mean": 2.0}},
        {"namespace": "eval/cusp", "metrics": {"opposite_spin_cusp_slope": -0.5}},
    ]
    (attempt / "metrics.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_collect_reads_eval_diagnostics_from_validation_attempts(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    good = "arch-hermite_o3_envelope_norm-N0_lr-1e-3_ch-16_seed-0"
    bad = "arch-raw_envelope_norm-N0_lr-1e-3_ch-8_seed-1"
    _fake_validation_attempt(results_root, good, "V1", status="completed", energy_error=0.01)
    _fake_validation_attempt(results_root, bad, "V1", status="failed", energy_error=5.0)

    result = collect.collect(results_root=results_root, collect_attempt_id="C1")
    attempt = Path(result["attempt_dir"])
    assert attempt == results_root / "03_collect" / "C1"

    summary_rows = list(_read_csv(attempt / "summary.csv"))
    by_run = {row["run_id"]: row for row in summary_rows}
    assert set(by_run) == {good, bad}
    assert by_run[good]["eval/energy/reference_abs_error"] == "0.01"
    assert by_run[good]["train_attempt_id"] == "T1"
    assert by_run[good]["architecture"] == "hermite_o3_envelope"
    assert int(by_run[good]["n_diagnostics"]) >= 2

    failures = list(_read_csv(attempt / "failures.csv"))
    assert {row["run_id"] for row in failures} == {bad}

    consumed = json.loads((attempt / "source_validation_attempts.json").read_text())
    assert {entry["run_id"] for entry in consumed} == {good, bad}
    assert json.loads((attempt / "collection_report.json").read_text())["n_collected"] == 2


def test_select_champions_reads_collection_attempt(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    collection = results_root / "03_collect" / "C1"
    collection.mkdir(parents=True, exist_ok=True)
    rows = [
        {"run_id": "arch-hermite_o3_envelope_norm-N0_lr-1e-3_ch-16_seed-0", "architecture": "hermite_o3_envelope",
         "normalization": "N0", "lr": "1e-3", "channels": "16", "seed": "0", "status": "completed",
         "eval/energy/reference_abs_error": "0.01"},
        {"run_id": "arch-hermite_o3_envelope_norm-N2_lr-1e-3_ch-16_seed-0", "architecture": "hermite_o3_envelope",
         "normalization": "N2", "lr": "1e-3", "channels": "16", "seed": "0", "status": "completed",
         "eval/energy/reference_abs_error": "0.20"},
        {"run_id": "arch-raw_envelope_norm-N0_lr-1e-3_ch-8_seed-0", "architecture": "raw_envelope",
         "normalization": "N0", "lr": "1e-3", "channels": "8", "seed": "0", "status": "completed",
         "eval/energy/reference_abs_error": "0.05"},
    ]
    _write_csv(collection / "summary.csv", rows)

    result = select_champions.select(results_root=results_root, collection_attempt_id="C1", select_attempt_id="S1")
    attempt = Path(result["attempt_dir"])
    assert attempt == results_root / "04_select" / "S1"

    source = json.loads((attempt / "source_collection_attempt.json").read_text())
    assert source["collection_attempt_id"] == "C1"

    champions = {row["architecture"]: row for row in _read_csv(attempt / "champions.csv")}
    # Per architecture, the lowest reference error wins.
    assert champions["hermite_o3_envelope"]["run_id"].endswith("norm-N0_lr-1e-3_ch-16_seed-0")
    report = json.loads((attempt / "selection_report.json").read_text())
    assert report["overall_champion"].endswith("norm-N0_lr-1e-3_ch-16_seed-0")


# ---------------------------------------------------------------------------
# Smoke run through the normal run path
# ---------------------------------------------------------------------------
def test_pair_stability_smoke_run_instantiates_one_grid_point(tmp_path: Path) -> None:
    from spenn.run import run_from_config

    cfg = OmegaConf.load(PAIR_STABILITY)
    cfg.runtime.device = "cpu"
    cfg.run.root = str(tmp_path)
    cfg.run.run_id = f"smoke/{ATTEMPT}"
    cfg.run_parameters.architecture = "hermite_o2_envelope"
    cfg.run_parameters.normalization = "N1"
    cfg.run_parameters.channels = 4
    cfg.run_parameters.seed = 0
    cfg.sampler_params.n_walkers = 8
    cfg.sampler_params.burn_in = 2
    cfg.sampler_params.n_steps = 2
    cfg.validation_sampler_params.n_walkers = 8
    cfg.validation_sampler_params.burn_in = 2
    cfg.validation_sampler_params.n_steps = 2
    cfg.training.max_steps = 1
    cfg.checkpoint.every_n_steps = 1

    exit_code = run_from_config(cfg, config_path=str(PAIR_STABILITY), command="pytest")
    assert exit_code == 0

    run_dir = tmp_path / "smoke" / ATTEMPT
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    assert (run_dir / "checkpoints" / "latest.json").is_file()
    resolved = OmegaConf.load(run_dir / "resolved_config.yaml")
    assert resolved.run_parameters.architecture == "hermite_o2_envelope"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read_csv(path: Path):
    import csv

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    columns = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
