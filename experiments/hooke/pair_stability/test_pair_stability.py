"""Study-level tests for the pair-stability experiment package (PR8.8).

These cover grid/choice consistency, planner artifacts, staged results layout,
attempt provenance, and a one-grid-point smoke run through the normal run path.
Reusable model-component math is tested under ``tests/`` and is intentionally
not retested here.
"""

from __future__ import annotations

import json
import sys
import types
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
import final_collect  # noqa: E402
import final_eval  # noqa: E402
import final_plan  # noqa: E402
import final_report  # noqa: E402
import final_train  # noqa: E402
import launch  # noqa: E402
import plan  # noqa: E402
import run_utils  # noqa: E402
import select_champions  # noqa: E402
import train  # noqa: E402
import validate  # noqa: E402


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
    points = plan.expand_grid(OmegaConf.to_container(grid.grid, resolve=True))
    # validate_grid raises if any architecture/normalization is unknown.
    plan.validate_grid(points, config)

    architectures = set(plan.architecture_tags(config))
    normalizations = plan.normalization_names(config)
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
# Planner manifest / layout
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
    code = plan.main(["--grid", str(grid), "--results-root", str(results_root), "--attempt-id", ATTEMPT])
    assert code == 0
    return results_root


def test_plan_writes_grid_attempt(tmp_path: Path) -> None:
    results_root = _plan(tmp_path)
    attempt = run_utils.grid_attempt_dir(results_root, ATTEMPT)
    assert (attempt / "manifest.json").is_file()
    assert (attempt / "commands.sh").is_file()
    assert (attempt / "grid.yaml").is_file()
    assert (attempt / "pair_stability.yaml").is_file()
    assert (attempt / "pair_validation.yaml").is_file()
    assert (attempt / "jobs" / f"{TARGET_RUN_ID}.json").is_file()
    assert (results_root / "00_grid" / "latest.json").is_file()


def test_manifest_contains_expected_run_ids_and_overrides(tmp_path: Path) -> None:
    results_root = _plan(tmp_path)
    manifest = json.loads((run_utils.grid_attempt_dir(results_root, ATTEMPT) / "manifest.json").read_text())
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
    manifest = json.loads((run_utils.grid_attempt_dir(results_root, ATTEMPT) / "manifest.json").read_text())
    job = next(job for job in manifest["jobs"] if job["run_id"] == TARGET_RUN_ID)
    assert job["train_dir"] == str(results_root / "01_train" / TARGET_RUN_ID)
    assert job["validation_dir"] == str(results_root / "02_validation" / TARGET_RUN_ID)
    assert job["train_attempt_dir"] == str(results_root / "01_train" / TARGET_RUN_ID / ATTEMPT)


def test_commands_sh_contains_run_commands(tmp_path: Path) -> None:
    # The repo has no @hydra.main app, so submission uses the canonical run.py
    # command path (handed to the Submitit launcher by the submitit backend).
    results_root = _plan(tmp_path)
    commands = (run_utils.grid_attempt_dir(results_root, ATTEMPT) / "commands.sh").read_text()
    assert "run.py" in commands
    assert "--config" in commands
    assert "python -u run.py" in commands
    assert "run_parameters.architecture=hermite_o3_envelope" in commands
    assert f"run.run_id={TARGET_RUN_ID}/{ATTEMPT}" in commands


def test_train_run_dir_uses_stage_attempt_layout(tmp_path: Path) -> None:
    results_root = _plan(tmp_path)
    manifest = json.loads((run_utils.grid_attempt_dir(results_root, ATTEMPT) / "manifest.json").read_text())
    job = next(job for job in manifest["jobs"] if job["run_id"] == TARGET_RUN_ID)
    overrides = job["overrides"]
    assert f"run.root={results_root / '01_train'}" in overrides
    assert "run.layout=flat" in overrides
    assert f"run.run_id={TARGET_RUN_ID}/{ATTEMPT}" in overrides


def test_plan_always_injects_run_timezone_override(tmp_path: Path) -> None:
    grid = _small_grid(tmp_path)
    # The launcher owns the timezone and always injects it (the config is null).
    assert OmegaConf.load(PAIR_STABILITY).run.timezone is None
    assert OmegaConf.load(PAIR_VALIDATION).run.timezone is None
    plan.main(["--grid", str(grid), "--results-root", str(tmp_path / "a"), "--attempt-id", ATTEMPT])
    commands = (run_utils.grid_attempt_dir(tmp_path / "a", ATTEMPT) / "commands.sh").read_text()
    assert "run.timezone=America/New_York" in commands
    # --timezone selects the injected zone.
    plan.main(
        ["--grid", str(grid), "--results-root", str(tmp_path / "b"), "--attempt-id", ATTEMPT, "--timezone", "UTC"]
    )
    commands_utc = (run_utils.grid_attempt_dir(tmp_path / "b", ATTEMPT) / "commands.sh").read_text()
    assert "run.timezone=UTC" in commands_utc


def test_train_consumes_grid_attempt_and_writes_submission_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = _plan(tmp_path)
    submitted_commands = []
    status_paths = []

    def fake_submit_local(
        commands,
        *,
        repo_root: Path,
        chunk_size: int = launch.DEFAULT_CHUNK_SIZE,
        row_status_paths=None,
        chunk_status_dir=None,
    ):
        assert chunk_size == launch.DEFAULT_CHUNK_SIZE
        status_paths.extend(row_status_paths or [])
        assert chunk_status_dir == results_root / "01_train" / "chunk_status" / ATTEMPT
        submitted_commands.extend(commands)
        return [f"local-{index}" for index, _ in enumerate(commands)]

    monkeypatch.setattr(launch, "submit_local", fake_submit_local)
    code = train.main(
        ["--results-root", str(results_root), "--grid-attempt-id", ATTEMPT, "--backend", "local"]
    )
    assert code == 0
    assert len(submitted_commands) == 4
    default_script = submitted_commands[0][2]
    assert "export UV_PROJECT_ENVIRONMENT=.venv" in default_script
    assert "uv sync --extra cpu" in default_script
    assert "runtime.device=cpu" in default_script

    train_attempt = results_root / "01_train" / TARGET_RUN_ID / ATTEMPT
    source = json.loads((train_attempt / "source_grid_attempt.json").read_text())
    assert source["grid_attempt_id"] == ATTEMPT
    assert source["run_id"] == TARGET_RUN_ID
    assert (train_attempt / "command.txt").is_file()
    assert train_attempt / "launcher_status.json" in status_paths
    submission = json.loads((train_attempt / "submission.json").read_text())
    assert submission["launcher"] == "local"
    assert submission["launcher_job_id"] == "local-3"

    # The train launcher reads 00_grid but does not mutate the planned manifest.
    manifest = json.loads((run_utils.grid_attempt_dir(results_root, ATTEMPT) / "manifest.json").read_text())
    job = next(job for job in manifest["jobs"] if job["run_id"] == TARGET_RUN_ID)
    assert job["submitted"] is False
    assert job["launcher"] is None


def test_train_smoke_submits_two_short_runs_with_smoke_attempt_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = _plan(tmp_path)
    manifest = json.loads((run_utils.grid_attempt_dir(results_root, ATTEMPT) / "manifest.json").read_text())
    first_run_id = manifest["jobs"][0]["run_id"]
    submitted_commands = []
    status_paths = []

    def fake_submit_local(
        commands,
        *,
        repo_root: Path,
        chunk_size: int = launch.DEFAULT_CHUNK_SIZE,
        row_status_paths=None,
        chunk_status_dir=None,
    ):
        assert chunk_size == launch.DEFAULT_CHUNK_SIZE
        status_paths.extend(row_status_paths or [])
        assert chunk_status_dir == results_root / "01_train" / "chunk_status" / f"{ATTEMPT}-smoke"
        submitted_commands.extend(commands)
        return [f"local-smoke-{index}" for index, _ in enumerate(commands)]

    monkeypatch.setattr(launch, "submit_local", fake_submit_local)
    code = train.main(
        [
            "--results-root",
            str(results_root),
            "--grid-attempt-id",
            ATTEMPT,
            "--backend",
            "local",
            "--cuda",
            "--smoke",
        ]
    )

    assert code == 0
    assert len(submitted_commands) == 2
    script = submitted_commands[0][2]
    smoke_attempt = f"{ATTEMPT}-smoke"
    assert "export UV_PROJECT_ENVIRONMENT=.venv-gpu" in script
    assert "uv sync --extra cu126" in script
    assert f"run.run_id={first_run_id}/{smoke_attempt}" in script
    assert f"study.attempt_id={smoke_attempt}" in script
    assert "runtime.device=cuda" in script
    assert "training.max_steps=2" in script
    assert "sampler_params.n_walkers=128" in script
    assert "checkpoint.every_n_steps=1" in script

    smoke_attempt_dir = results_root / "01_train" / first_run_id / smoke_attempt
    source = json.loads((smoke_attempt_dir / "source_grid_attempt.json").read_text())
    assert source["grid_attempt_id"] == ATTEMPT
    assert (smoke_attempt_dir / "command.txt").is_file()
    assert smoke_attempt_dir / "launcher_status.json" in status_paths
    submission = json.loads((smoke_attempt_dir / "submission.json").read_text())
    assert submission["launcher_job_id"] == "local-smoke-0"
    assert f"run.run_id={first_run_id}/{smoke_attempt}" in submission["submitted_command"]


def test_smoke_attempt_id_is_idempotent() -> None:
    smoke_attempt = f"{ATTEMPT}-smoke"

    assert launch.smoke_attempt_id(ATTEMPT) == smoke_attempt
    assert launch.smoke_attempt_id(smoke_attempt) == smoke_attempt


def test_pair_stability_train_config_has_no_train_validation_remnants() -> None:
    cfg = OmegaConf.load(PAIR_STABILITY)

    assert "validation" not in cfg
    assert "validation_sampler_params" not in cfg
    assert "validation_sampler" not in cfg
    targets = [str(callback.get("_target_", "")) for callback in cfg.callbacks]
    assert "spenn.callback.Validation" not in targets


def test_environment_wrapper_aligns_uv_environment_and_runtime_device() -> None:
    planned_python = str(ROOT / ".venv" / "bin" / "python")
    submitted = launch.environment_shell_command(
        [planned_python, "-u", "run.py", "--config", "cfg.yaml", "runtime.device=cpu", "x=y"],
        repo_root=ROOT,
        uv_environment=".venv-gpu",
        uv_extras=["cu126"],
        device="cuda",
    )

    assert submitted[:2] == ["bash", "-lc"]
    script = submitted[2]
    assert f"cd {ROOT}" in script
    assert "export UV_PROJECT_ENVIRONMENT=.venv-gpu" in script
    assert "uv sync --extra cu126" in script
    assert "source .venv-gpu/bin/activate" in script
    assert "exec python -u run.py --config cfg.yaml x=y runtime.device=cuda" in script
    assert planned_python not in script


def test_submitit_uses_matching_cpu_or_cuda_slurm_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_chunks = []
    captured_parameters = []
    captured_map_calls = []

    class FakeExecutor:
        def __init__(self, folder: str):
            self.folder = folder

        def update_parameters(self, **kwargs):
            self.parameters = kwargs
            captured_parameters.append(kwargs)

        def map_array(self, fn, commands):
            captured_map_calls.append(fn)
            captured_chunks.extend(commands)
            return [types.SimpleNamespace(job_id=f"array-job_{index}") for index, _ in enumerate(commands)]

    fake_submitit = types.SimpleNamespace(
        AutoExecutor=FakeExecutor,
    )
    monkeypatch.setitem(sys.modules, "submitit", fake_submitit)

    cpu_args = train.parse_args(["--backend", "submitit"])
    assert launch.slurm_parameters(cpu_args, profile="cpu") == {
        "slurm_partition": "seas_compute,kozinsky_lab,sapphire",
        "timeout_min": 480,
        "mem_gb": 32,
        "cpus_per_task": 8,
        "tasks_per_node": 1,
        "slurm_array_parallelism": launch.DEFAULT_ARRAY_PARALLELISM,
    }
    cuda_args = train.parse_args(["--backend", "submitit", "--cuda"])
    cuda_slurm = launch.slurm_parameters(cuda_args, profile="cuda")
    assert cuda_slurm["slurm_partition"] == "seas_gpu,kozinsky_gpu"
    assert cuda_slurm["gpus_per_node"] == 1
    assert cuda_slurm["slurm_array_parallelism"] == launch.DEFAULT_ARRAY_PARALLELISM
    smoke_cpu = launch.slurm_parameters(cpu_args, profile="cpu", smoke=True)
    assert smoke_cpu["slurm_partition"] == "test"
    assert smoke_cpu["timeout_min"] == 15
    assert smoke_cpu["mem_gb"] == 16
    assert smoke_cpu["cpus_per_task"] == 4
    assert smoke_cpu["slurm_array_parallelism"] == 2
    smoke_cuda = launch.slurm_parameters(cuda_args, profile="cuda", smoke=True)
    assert smoke_cuda["slurm_partition"] == "gpu_test"
    assert smoke_cuda["timeout_min"] == 15
    assert smoke_cuda["gpus_per_node"] == 1
    assert smoke_cuda["slurm_array_parallelism"] == 2

    submitted = launch.environment_shell_command(
        ["python", "-u", "run.py", "--config", "cfg.yaml", "x=y"],
        repo_root=ROOT,
        uv_environment=".venv-gpu",
        uv_extras=["cu126"],
        device="cuda",
    )
    job_ids = launch.submit_submitit(
        [submitted, submitted],
        log_dir=tmp_path / "logs",
        job_name="pair-stability",
        slurm=cuda_slurm,
    )

    assert job_ids == ["array-job_0", "array-job_1"]
    assert captured_map_calls == [launch.run_command_chunk]
    assert captured_chunks == [[submitted], [submitted]]
    assert captured_parameters[0]["slurm_partition"] == "seas_gpu,kozinsky_gpu"
    assert captured_parameters[0]["gpus_per_node"] == 1
    assert captured_parameters[0]["slurm_array_parallelism"] == launch.DEFAULT_ARRAY_PARALLELISM
    assert captured_parameters[0]["slurm_setup"][0].startswith("export PYTHONPATH=")
    assert str(STUDY_DIR) in captured_parameters[0]["slurm_setup"][0]
    assert str(ROOT) in captured_parameters[0]["slurm_setup"][0]


def test_submitit_chunks_commands_evenly_and_expands_job_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_chunks = []

    class FakeExecutor:
        def __init__(self, folder: str):
            self.folder = folder

        def update_parameters(self, **kwargs):
            self.parameters = kwargs

        def map_array(self, fn, commands):
            assert fn is launch.run_command_chunk
            captured_chunks.extend(commands)
            return [types.SimpleNamespace(job_id=f"array-job_{index}") for index, _ in enumerate(commands)]

    monkeypatch.setitem(sys.modules, "submitit", types.SimpleNamespace(AutoExecutor=FakeExecutor))

    commands = [["bash", "-lc", f"echo {index}"] for index in range(540)]
    job_ids = launch.submit_submitit(
        commands,
        log_dir=tmp_path / "logs",
        job_name="pair-stability",
        slurm={},
        chunk_size=128,
    )

    assert [len(chunk) for chunk in captured_chunks] == [108, 108, 108, 108, 108]
    assert [command for chunk in captured_chunks for command in chunk] == commands
    assert job_ids == [f"array-job_{index}" for index in range(5) for _ in range(108)]


def test_eval_chunks_record_partial_failures_without_aborting(tmp_path: Path) -> None:
    commands = [
        ["bash", "-lc", "exit 0"],
        ["bash", "-lc", "exit 3"],
        ["bash", "-lc", "exit 0"],
    ]
    row_status_paths = [tmp_path / f"row-{index}.json" for index in range(3)]
    job_ids = launch.submit_local(
        commands,
        repo_root=ROOT,
        chunk_size=3,
        allow_partial_failures=True,
        row_status_paths=row_status_paths,
        chunk_status_dir=tmp_path / "chunks",
    )

    assert job_ids == ["local-chunk-0-rc0"] * 3
    statuses = [json.loads(path.read_text())["status"] for path in row_status_paths]
    assert statuses == ["success", "failed", "success"]
    chunk_status = json.loads((tmp_path / "chunks" / "chunk-0000.json").read_text())
    assert chunk_status["status"] == "partial_failed"
    assert chunk_status["n_failed"] == 1


def _write_selection_attempt(results_root: Path, attempt_id: str = "S1") -> Path:
    selection = results_root / "04_select" / attempt_id
    selection.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "architecture": "hermite_o3_envelope",
            "normalization": "N0",
            "winner_kind": "energy",
            "config_id": "arch-hermite_o3_envelope_norm-N0_lr-1e-3_ch-16",
            "lr": "1e-3",
            "channels": "16",
            "metric": "eval/stratified_geometry/local_energy_mean_seed_median",
            "metric_value": "2.0",
            "run_ids": "arch-hermite_o3_envelope_norm-N0_lr-1e-3_ch-16_seed-0",
        },
        {
            "architecture": "raw_envelope",
            "normalization": "N1",
            "winner_kind": "feature_trace",
            "config_id": "arch-raw_envelope_norm-N1_lr-3e-3_ch-8",
            "lr": "3e-3",
            "channels": "8",
            "metric": "eval/feature_trace_stability/feature_rms_q95_seed_median",
            "metric_value": "0.2",
            "run_ids": "arch-raw_envelope_norm-N1_lr-3e-3_ch-8_seed-0",
        },
    ]
    _write_csv(selection / "champions.csv", rows)
    return selection


def test_final_plan_writes_replicate_grid_with_seed_policy(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_selection_attempt(results_root, "S1")

    code = final_plan.main(
        [
            "--results-root",
            str(results_root),
            "--selection-attempt-id",
            "S1",
            "--attempt-id",
            "F1",
            "--replicates",
            "2",
        ]
    )

    assert code == 0
    attempt = results_root / "05_final_grid" / "F1"
    assert (attempt / "manifest.json").is_file()
    assert (attempt / "manifest.yaml").is_file()
    assert (attempt / "source_selection_attempt.json").is_file()
    assert (attempt / "source_champions.csv").is_file()
    jobs = _read_csv(attempt / "final_jobs.csv")
    assert len(jobs) == 4
    first = jobs[0]
    assert first["source_selection_attempt_id"] == "S1"
    assert first["source_champion_id"] == "champion-0000"
    assert first["replicate_index"] == "0"
    assert first["final_train_sampler_seed"] == "101"
    assert first["final_train_model_seed"] == "1001"
    assert first["final_eval_seed"] == "10001"
    second_rep = jobs[1]
    assert second_rep["replicate_index"] == "1"
    assert second_rep["final_train_sampler_seed"] == "102"
    assert second_rep["final_train_model_seed"] == "1002"
    assert second_rep["final_eval_seed"] == "10002"


def _write_final_grid(tmp_path: Path) -> tuple[Path, dict]:
    results_root = tmp_path / "results"
    _write_selection_attempt(results_root, "S1")
    final_plan.main(
        [
            "--results-root",
            str(results_root),
            "--selection-attempt-id",
            "S1",
            "--attempt-id",
            "F1",
            "--replicates",
            "1",
            "--limit-champions",
            "1",
        ]
    )
    job = json.loads(
        next((results_root / "05_final_grid" / "F1" / "jobs").glob("*.json")).read_text()
    )
    return results_root, job


def test_final_train_consumes_final_grid_and_records_checkpoint_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root, job = _write_final_grid(tmp_path)
    submitted_commands = []
    status_paths = []

    def fake_submit_local(
        commands,
        *,
        repo_root: Path,
        chunk_size: int = launch.DEFAULT_CHUNK_SIZE,
        row_status_paths=None,
        chunk_status_dir=None,
    ):
        status_paths.extend(row_status_paths or [])
        assert chunk_status_dir == results_root / "06_final_train" / "chunk_status" / "TF1"
        submitted_commands.extend(commands)
        return [f"local-final-{index}" for index, _ in enumerate(commands)]

    monkeypatch.setattr(launch, "submit_local", fake_submit_local)
    code = final_train.main(
        [
            "--results-root",
            str(results_root),
            "--final-grid-attempt-id",
            "F1",
            "--attempt-id",
            "TF1",
            "--backend",
            "local",
        ]
    )

    assert code == 0
    assert len(submitted_commands) == 1
    script = submitted_commands[0][2]
    assert "run_parameters.seed=1001" in script
    assert "sampler.seed=101" in script
    assert "run.timezone=America/New_York" in script
    assert "runtime.device=cpu" in script
    attempt = results_root / "06_final_train" / job["final_run_id"] / "TF1"
    assert json.loads((attempt / "source_final_grid_attempt.json").read_text())["final_grid_attempt_id"] == "F1"
    assert json.loads((attempt / "source_final_job.json").read_text())["final_train_model_seed"] == 1001
    assert attempt / "launcher_status.json" in status_paths
    selected = json.loads((attempt / "selected_checkpoint.json").read_text())
    assert selected["selection_policy"] == "latest_checkpoint_pointer"
    assert selected["checkpoint_pointer"].endswith("checkpoints/latest.json")
    submission = json.loads((attempt / "submission.json").read_text())
    assert submission["launcher_job_id"] == "local-final-0"


def _write_final_train_checkpoint(results_root: Path, final_run_id: str, attempt_id: str = "TF1") -> Path:
    train_attempt = results_root / "06_final_train" / final_run_id / attempt_id
    checkpoint_root = train_attempt / "checkpoints"
    checkpoint_dir = checkpoint_root / "step_000002"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "manifest.json").write_text("{}")
    (checkpoint_dir / "COMPLETE").write_text("complete\n")
    (checkpoint_root / "latest.json").write_text(
        json.dumps({"checkpoint_dir": "step_000002", "step": 2, "created_at_unix": 0.0})
    )
    (train_attempt / "selected_checkpoint.json").write_text(
        json.dumps(
            {
                "selection_policy": "latest_checkpoint_pointer",
                "checkpoint_dir": str(checkpoint_root),
                "checkpoint_pointer": str(checkpoint_root / "latest.json"),
                "resolved_checkpoint_dir": None,
            }
        )
    )
    return train_attempt


def test_final_eval_records_exact_checkpoint_and_uses_final_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root, job = _write_final_grid(tmp_path)
    _write_final_train_checkpoint(results_root, job["final_run_id"], "TF1")
    submitted_commands = []

    def fake_submit_local(
        commands,
        *,
        repo_root: Path,
        chunk_size: int = launch.DEFAULT_CHUNK_SIZE,
        allow_partial_failures: bool = False,
        row_status_paths=None,
        chunk_status_dir=None,
    ):
        assert chunk_size == launch.DEFAULT_CHUNK_SIZE
        submitted_commands.extend(commands)
        return [f"local-eval-{index}" for index, _ in enumerate(commands)]

    monkeypatch.setattr(launch, "submit_local", fake_submit_local)
    code = final_eval.main(
        [
            "--results-root",
            str(results_root),
            "--final-grid-attempt-id",
            "F1",
            "--attempt-id",
            "FE1",
            "--backend",
            "local",
        ]
    )

    assert code == 0
    assert len(submitted_commands) == 1
    script = submitted_commands[0][2]
    assert "evaluation.suite=final_eval" in script
    assert "evaluation.seed=10001" in script
    assert "run.timezone=America/New_York" in script
    assert "load.path=" in script
    assert "step_000002" in script
    attempt = results_root / "07_final_eval" / job["final_run_id"] / "FE1"
    checkpoint = json.loads((attempt / "evaluated_checkpoint.json").read_text())
    assert checkpoint["resolved_checkpoint_dir"].endswith("checkpoints/step_000002")
    submission = json.loads((attempt / "submission.json").read_text())
    assert submission["launcher_job_id"] == "local-eval-0"


def test_final_eval_auto_selects_latest_ready_smoke_final_train_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root, job = _write_final_grid(tmp_path)
    good_attempt_id = "20260621T171930-0400-smoke"
    bad_attempt_id = "20260621T171930-0400-smoke-smoke"
    _write_final_train_checkpoint(results_root, job["final_run_id"], good_attempt_id)
    bad_attempt = results_root / "06_final_train" / job["final_run_id"] / bad_attempt_id
    bad_attempt.mkdir(parents=True)
    submitted_commands = []

    def fake_submit_local(
        commands,
        *,
        repo_root: Path,
        chunk_size: int = launch.DEFAULT_CHUNK_SIZE,
        allow_partial_failures: bool = False,
        row_status_paths=None,
        chunk_status_dir=None,
    ):
        assert chunk_size == launch.DEFAULT_CHUNK_SIZE
        submitted_commands.extend(commands)
        return [f"local-eval-{index}" for index, _ in enumerate(commands)]

    monkeypatch.setattr(launch, "submit_local", fake_submit_local)
    code = final_eval.main(
        [
            "--results-root",
            str(results_root),
            "--final-grid-attempt-id",
            "F1",
            "--backend",
            "local",
            "--smoke",
        ]
    )

    assert code == 0
    assert len(submitted_commands) == 1
    script = submitted_commands[0][2]
    assert f"{good_attempt_id}/checkpoints/step_000002" in script
    assert bad_attempt_id not in script
    attempt = results_root / "07_final_eval" / job["final_run_id"] / "F1-smoke"
    source = json.loads((attempt / "source_final_train_attempt.json").read_text())
    assert source["final_train_attempt_id"] == good_attempt_id


def test_final_collect_reduces_raw_artifacts_and_final_report_reads_collect_only(tmp_path: Path) -> None:
    results_root, job = _write_final_grid(tmp_path)
    attempt = results_root / "07_final_eval" / job["final_run_id"] / "FE1"
    train_attempt = results_root / "06_final_train" / job["final_run_id"] / "FT1"
    (attempt / "cusp").mkdir(parents=True)
    (attempt / "tail").mkdir()
    (attempt / "stratified_geometry").mkdir()
    (attempt / "energy").mkdir()
    (attempt / "full_model_antisymmetry").mkdir()
    (attempt / "trace_equivariance").mkdir()
    train_attempt.mkdir(parents=True)
    (attempt / "status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "start_time": "2026-06-22T12:00:00-04:00",
                "end_time": "2026-06-22T12:00:03-04:00",
            }
        )
    )
    (train_attempt / "status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "start_time": "2026-06-22T11:00:00-04:00",
                "end_time": "2026-06-22T11:00:05-04:00",
            }
        )
    )
    (train_attempt / "metadata.json").write_text(json.dumps({"runtime": {"device": "cpu"}}))
    (train_attempt / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"namespace": "train", "step": 0, "metrics": {"energy": 2.6, "energy_stderr": 0.2, "energy_variance": 0.3, "grad_norm": 4.0}}),
                json.dumps({"namespace": "train/sampler", "step": 0, "metrics": {"acceptance_rate": 0.7}}),
            ]
        )
        + "\n"
    )
    (attempt / "source_final_job.json").write_text(json.dumps(job))
    (attempt / "source_final_train_attempt.json").write_text(
        json.dumps({"final_train_attempt_id": "FT1", "final_train_attempt_dir": str(train_attempt)})
    )
    (attempt / "evaluated_checkpoint.json").write_text(
        json.dumps({"resolved_checkpoint_dir": "checkpoints/step_000002"})
    )
    (attempt / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "namespace": "eval/energy",
                        "metrics": {
                            "local_energy_mean": 2.5,
                            "local_energy_stderr": 0.1,
                            "local_energy_variance": 0.2,
                        },
                    }
                ),
                json.dumps({"namespace": "eval/cusp", "metrics": {"cusp_even_slope_abs_error": 0.01}}),
                json.dumps({"namespace": "eval/tail", "metrics": {"local_energy_pathology_count": 1}}),
                json.dumps(
                    {
                        "namespace": "eval/full_model_antisymmetry",
                        "metrics": {"logabs_max_abs_error": 0.02, "failure_count": 0},
                    }
                ),
                json.dumps(
                    {
                        "namespace": "eval/trace_equivariance",
                        "metrics": {"failure_count": 0, "comparison_error_count": 0},
                    }
                ),
            ]
        )
        + "\n"
    )
    _write_csv(attempt / "cusp" / "cusp_profiles.csv", [{"r12": "0.1", "local_energy": "2.0"}])
    _write_csv(attempt / "tail" / "tail_profiles.csv", [{"radius": "1.0", "local_energy": "3.0"}])
    _write_csv(attempt / "stratified_geometry" / "stratified_metrics.csv", [{"stratum": "bulk", "local_energy": "2.1"}])
    _write_csv(attempt / "energy" / "mcmc_energy_samples.csv", [{"sample_index": "0", "local_energy": "2.4"}])
    _write_csv(
        attempt / "full_model_antisymmetry" / "transform_records.csv",
        [{"record_index": "0", "logabs_abs_error": "0.0"}],
    )
    _write_csv(attempt / "trace_equivariance" / "trace_records.csv", [{"key": "basis/output", "max_abs_error": "0"}])

    collect_result = final_collect.collect_final_outputs(
        results_root=results_root,
        collect_attempt_id="C1",
        final_eval_attempt_id="FE1",
    )
    result = final_report.build_report(
        results_root=results_root,
        report_attempt_id="R1",
        final_collect_attempt_id="C1",
    )

    collect_dir = Path(collect_result["attempt_dir"])
    assert collect_dir == results_root / "08_final_collect" / "C1"
    run_index = _read_csv(collect_dir / "run_index.csv")
    assert run_index[0]["final_run_id"] == job["final_run_id"]
    assert run_index[0]["winner_kind"] == "energy"
    assert run_index[0]["train_wall_time_sec"] == "5"
    energy_by_run = _read_csv(collect_dir / "energy_by_run.csv")
    assert energy_by_run[0]["energy_error"] == "0.5"
    histograms = _read_csv(collect_dir / "local_energy_histograms.csv")
    assert histograms[0]["basis_class"] == job["basis_envelope"]
    cusp = _read_csv(collect_dir / "cusp_profile_summary.csv")
    assert cusp[0]["local_energy_median"] == "2"
    training = _read_csv(collect_dir / "training_curve_summary.csv")
    assert training[0]["acceptance_rate"] == "0.7"

    report_dir = Path(result["attempt_dir"])
    assert report_dir == results_root / "09_final_report" / "R1"
    copied_energy = _read_csv(report_dir / "tables" / "energy_by_run.csv")
    assert copied_energy[0]["energy_error"] == "0.5"
    assert (report_dir / "figures" / "1A_real_scale_energy_error_heatmap.png").is_file()
    assert (report_dir / "figures" / "1A_log_scale_energy_error_heatmap.png").is_file()
    assert (report_dir / "figures" / "1C_energy_winner_local_energy_distribution_grid.png").is_file()
    assert (report_dir / "figures" / "1C_stability_winner_local_energy_distribution_grid.png").is_file()
    assert (report_dir / "figures" / "2A_energy_winner_cusp_local_energy_grid.png").is_file()
    assert (report_dir / "figures" / "2A_stability_winner_cusp_local_energy_grid.png").is_file()
    assert (report_dir / "figures" / "2B_energy_winner_cusp_logabs_grid.png").is_file()
    assert (report_dir / "figures" / "2B_stability_winner_cusp_logabs_grid.png").is_file()
    assert (report_dir / "figures" / "2C_energy_winner_cusp_finite_fraction_grid.png").is_file()
    assert (report_dir / "figures" / "2C_stability_winner_cusp_finite_fraction_grid.png").is_file()
    assert (report_dir / "figures" / "3A_tail_energy_winner_grid.png").is_file()
    assert (report_dir / "figures" / "3B_tail_stability_winner_grid.png").is_file()
    assert (report_dir / "figures" / "3C_tail_outlier_heatmap.png").is_file()
    assert (report_dir / "figures" / "4_stratified_geometry_aggregate_heatmap.png").is_file()
    assert (report_dir / "figures" / "4_stratified_geometry_aggregate_log_heatmap.png").is_file()
    assert (report_dir / "figures" / "4_stratified_geometry_bulk_heatmap.png").is_file()
    assert (report_dir / "figures" / "4_stratified_geometry_bulk_log_heatmap.png").is_file()
    assert (report_dir / "figures" / "5A_energy_winner_hooke_orbital_local_energy_distribution.png").is_file()
    assert (report_dir / "figures" / "5A_stability_winner_hooke_orbital_local_energy_distribution.png").is_file()
    assert (report_dir / "figures" / "6_symmetry_logabs_error_max_heatmap_grid.png").is_file()
    assert (report_dir / "figures" / "6_symmetry_logabs_error_median_heatmap_grid.png").is_file()
    assert (report_dir / "figures" / "6_symmetry_sign_mismatch_count_heatmap_grid.png").is_file()
    assert (report_dir / "figures" / "6_symmetry_parity_mismatch_count_heatmap_grid.png").is_file()
    assert (report_dir / "figures" / "6_symmetry_finite_fraction_heatmap_grid.png").is_file()
    assert (report_dir / "figures" / "7_trace_scalar_heatmap.png").is_file()
    assert (report_dir / "report.md").read_text().startswith("# Hooke Pair-Stability Final Report")


def test_final_report_heatmap_matrix_keeps_real_scale_signed_errors() -> None:
    y_labels, x_labels, matrix = final_report._heatmap_matrix(
        [
            {"basis": "raw", "normalization": "N0", "energy_error": "-0.25"},
            {"basis": "raw", "normalization": "N0", "energy_error": "-0.75"},
            {"basis": "raw", "normalization": "N1", "energy_error": "0.5"},
        ],
        row_key="basis",
        col_key="normalization",
        value_key="energy_error",
    )

    assert y_labels == ["raw"]
    assert x_labels == ["N0", "N1"]
    assert matrix == [[-0.5, 0.5]]


def test_final_report_winner_helpers_split_energy_and_stability_rows() -> None:
    rows = [{"winner_kind": "energy", "id": 1}, {"winner_kind": "stability", "id": 2}, {"winner_kind": "feature_trace", "id": 3}]

    assert [row["id"] for row in final_report._winner_rows(rows, "energy")] == [1]
    assert [row["id"] for row in final_report._winner_rows(rows, "stability")] == [2, 3]
    assert final_report._winner_filename("4", "energy", "plot.png") == "4_energy_winner_plot.png"


def test_final_report_scalar_metric_matrix_aggregates_multiple_metrics() -> None:
    y_labels, x_labels, matrix = final_report._scalar_metric_matrix(
        [
            {
                "basis_class": "raw_envelope",
                "normalization": "N0",
                "symmetry_task": "exchange",
                "logabs_error_max": "2.0",
                "sign_mismatch_count": "1",
            },
            {
                "basis_class": "raw_envelope",
                "normalization": "N0",
                "symmetry_task": "exchange",
                "logabs_error_max": "4.0",
                "sign_mismatch_count": "3",
            },
            {
                "basis_class": "raw_envelope",
                "normalization": "N0",
                "symmetry_task": "exchange",
                "finite_fraction": "1.0",
            },
        ],
        row_keys=("basis_class", "normalization", "symmetry_task"),
        metric_keys=("logabs_error_max", "sign_mismatch_count", "finite_fraction"),
    )

    assert y_labels == ["raw_envelope/N0/exchange"]
    assert x_labels == ["logabs_error_max", "sign_mismatch_count", "finite_fraction"]
    assert matrix == [[3.0, 2.0, 1.0]]


def test_final_report_symmetry_metric_grid_splits_winners_and_symmetries(tmp_path: Path) -> None:
    path = tmp_path / "symmetry_grid.png"
    rows = [
        {
            "basis_class": "raw_envelope",
            "normalization": "N0",
            "winner_kind": "energy",
            "symmetry_task": "full_model_antisymmetry",
            "logabs_error_max": "0.1",
        },
        {
            "basis_class": "raw_envelope",
            "normalization": "N0",
            "winner_kind": "stability",
            "symmetry_task": "full_model_antisymmetry",
            "logabs_error_max": "0.2",
        },
        {
            "basis_class": "hermite_o2_envelope",
            "normalization": "N1",
            "winner_kind": "energy",
            "symmetry_task": "rotation_consistency",
            "logabs_error_max": "0.3",
        },
    ]

    final_report._save_symmetry_metric_grid(path, rows, metric_key="logabs_error_max", title="symmetry grid")

    assert path.is_file()


def test_final_report_architecture_line_grid_splits_architectures(tmp_path: Path) -> None:
    path = tmp_path / "architecture_grid.png"
    rows = [
        {"basis_class": "raw_envelope", "normalization": "N0", "step": "0", "energy_mean": "1.0"},
        {"basis_class": "raw_envelope", "normalization": "N1", "step": "0", "energy_mean": "1.2"},
        {"basis_class": "hermite_o2_envelope", "normalization": "N0", "step": "0", "energy_mean": "2.0"},
        {"basis_class": "hermite_o2_envelope", "normalization": "N1", "step": "0", "energy_mean": "2.2"},
    ]

    final_report._save_architecture_line_grid(
        path,
        rows,
        x_key="step",
        y_key="energy_mean",
        group_keys=("normalization",),
        title="architecture grid",
        legend_title="normalization",
    )

    assert path.is_file()


def test_final_report_line_plot_can_force_large_external_legend(tmp_path: Path) -> None:
    path = tmp_path / "line_with_legend.png"
    rows = [
        {"step": "0", "energy_mean": str(index), "label": f"group-{index}"}
        for index in range(16)
    ]

    final_report._save_line_plot(
        path,
        rows,
        x_key="step",
        y_key="energy_mean",
        group_keys=("label",),
        title="large legend",
        legend="outside",
        legend_title="groups",
    )

    assert path.is_file()


def test_final_report_energy_variance_scatter_uses_abs_positive_log_points() -> None:
    points = final_report._energy_variance_points(
        [
            {
                "architecture": "raw_envelope",
                "basis": "ignored",
                "normalization": "N0",
                "energy_error": "-0.25",
                "local_energy_var": "10",
            },
            {
                "architecture": "hermite_o2_envelope",
                "normalization": "N1",
                "energy_error": "0",
                "local_energy_var": "1",
            },
            {
                "architecture": "hermite_o2_envelope",
                "normalization": "N2",
                "energy_error": "0.5",
                "local_energy_var": "0",
            },
        ]
    )

    assert points == [
        {
            "abs_energy_error": 0.25,
            "local_energy_var": 10.0,
            "architecture": "raw_envelope",
            "normalization": "N0",
            "winner_kind": "",
        }
    ]


def test_final_report_local_energy_grid_groups_by_norm_and_architecture() -> None:
    normalizations, architectures, groups = final_report._local_energy_distribution_groups(
        [
            {"basis_class": "raw_envelope", "normalization": "N1", "winner_kind": "energy", "bin_center": "1.0", "count": "2"},
            {"basis_class": "raw_envelope", "normalization": "N1", "winner_kind": "energy", "bin_center": "2.0", "count": "1"},
            {"basis_class": "hermite_o2_envelope", "normalization": "N0", "winner_kind": "stability", "bin_center": "3.0", "count": "4"},
        ]
    )

    assert normalizations == ["N0", "N1"]
    assert architectures == ["hermite_o2_envelope", "raw_envelope"]
    assert len(groups[("N1", "raw_envelope")]) == 2
    assert groups[("N0", "hermite_o2_envelope")][0]["count"] == "4"


def test_final_collect_local_energy_histograms_use_group_scoped_bins(tmp_path: Path) -> None:
    def context(run_id: str, architecture: str, values: list[float]) -> dict:
        attempt = tmp_path / run_id
        (attempt / "energy").mkdir(parents=True)
        _write_csv(attempt / "energy" / "mcmc_energy_samples.csv", [{"local_energy": str(value)} for value in values])
        return {
            "final_run_id": run_id,
            "attempt_dir": attempt,
            "job": {
                "basis_envelope": architecture,
                "normalization": "N0",
                "winner_kind": "energy",
                "replicate_index": "0",
            },
        }

    rows = final_collect._local_energy_histograms(
        [
            context("compact", "raw_envelope", [1.0, 2.0, 3.0]),
            context("outlier", "hermite_o3_envelope", [1000.0, 1100.0, 1200.0]),
        ]
    )
    compact = [row for row in rows if row["final_run_id"] == "compact"]
    outlier = [row for row in rows if row["final_run_id"] == "outlier"]

    assert max(float(row["bin_right"]) for row in compact) == pytest.approx(3.0)
    assert min(float(row["bin_left"]) for row in outlier) == pytest.approx(1000.0)


def test_final_report_local_energy_bar_series_sums_seed_bins() -> None:
    centers, counts, widths = final_report._local_energy_bar_series(
        [
            {"bin_left": "0", "bin_right": "1", "bin_center": "0.5", "count": "2"},
            {"bin_left": "0", "bin_right": "1", "bin_center": "0.5", "count": "3"},
            {"bin_left": "1", "bin_right": "2", "bin_center": "1.5", "count": "0"},
            {"bin_left": "2", "bin_right": "3", "bin_center": "2.5", "count": "4"},
        ]
    )

    assert centers == [0.5, 2.5]
    assert counts == [5.0, 4.0]
    assert widths == [1.0, 1.0]


def test_final_report_tail_grid_aggregates_paths_before_seed_variance() -> None:
    points = final_report._tail_seed_profile_points(
        [
            {
                "basis_class": "raw_envelope",
                "normalization": "N0",
                "winner_kind": "energy",
                "seed_index": "0",
                "com_id": "near",
                "tail_path": "a",
                "radius": "1.0",
                "local_energy_median": "2.0",
            },
            {
                "basis_class": "raw_envelope",
                "normalization": "N0",
                "winner_kind": "energy",
                "seed_index": "0",
                "com_id": "near",
                "tail_path": "b",
                "radius": "1.0",
                "local_energy_median": "4.0",
            },
            {
                "basis_class": "raw_envelope",
                "normalization": "N0",
                "winner_kind": "energy",
                "seed_index": "1",
                "com_id": "near",
                "tail_path": "a",
                "radius": "1.0",
                "local_energy_median": "5.0",
            },
            {
                "basis_class": "raw_envelope",
                "normalization": "N0",
                "winner_kind": "stability",
                "seed_index": "0",
                "com_id": "near",
                "tail_path": "a",
                "radius": "1.0",
                "local_energy_median": "100.0",
            },
        ],
        winner_kind="energy",
        value_key="local_energy_median",
    )

    row = points[("raw_envelope", "N0", "CoM near")][0]
    assert row["radius"] == 1.0
    assert row["mean"] == pytest.approx(4.0)
    assert row["variance"] == pytest.approx(2.0)
    assert row["n_seeds"] == 2


def _write_checkpoint_pointer(results_root: Path, run_id: str, attempt_id: str) -> Path:
    checkpoint_dir = run_utils.train_attempt_dir(results_root, run_id, attempt_id) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "latest.json").write_text(json.dumps({"path": "step_000000"}))
    return checkpoint_dir


def test_validate_records_source_train_attempt(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    point = {"architecture": "hermite_o3_envelope", "normalization": "N2", "lr": 1.0e-3, "channels": 16, "seed": 0}
    jobs = plan.build_jobs(
        [point],
        attempt_id="T1",
        results_root=results_root,
        config=PAIR_STABILITY,
        tags_by_architecture={"hermite_o3_envelope": ["main"]},
    )
    _write_checkpoint_pointer(results_root, TARGET_RUN_ID, "T1")
    args = validate.parse_args(["--backend", "local", "--train-attempt-id", "T1", "--attempt-id", "V1"])
    validation_jobs, skipped = validate.plan_validation_jobs(
        jobs,
        args=args,
        results_root=results_root,
        grid_attempt_id="G1",
        validation_config=PAIR_VALIDATION,
    )
    assert skipped == []
    validation_plan = validation_jobs[0]
    source_path = Path(validation_plan["validation_attempt_dir"]) / "source_train_attempt.json"
    assert source_path.is_file()
    source = json.loads(source_path.read_text())
    assert source["run_id"] == TARGET_RUN_ID
    assert source["grid_attempt_id"] == "G1"
    assert source["train_attempt_id"] == "T1"
    assert source["checkpoint_path"].endswith("checkpoints")
    assert "load.path=" in validation_plan["command"]
    assert "run.timezone=America/New_York" in validation_plan["command"]


def test_validate_auto_selection_ignores_smoke_train_attempts(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    point = {"architecture": "hermite_o3_envelope", "normalization": "N2", "lr": 1.0e-3, "channels": 16, "seed": 0}
    jobs = plan.build_jobs(
        [point],
        attempt_id="T1",
        results_root=results_root,
        config=PAIR_STABILITY,
        tags_by_architecture={"hermite_o3_envelope": ["main"]},
    )
    _write_checkpoint_pointer(results_root, TARGET_RUN_ID, "T1")
    _write_checkpoint_pointer(results_root, TARGET_RUN_ID, "T2-smoke")

    assert validate.latest_train_attempt_id(results_root, TARGET_RUN_ID, smoke=False) == "T1"
    assert validate.latest_train_attempt_id(results_root, TARGET_RUN_ID, smoke=True) == "T2-smoke"

    args = validate.parse_args(["--backend", "local", "--attempt-id", "V1"])
    validation_jobs, skipped = validate.plan_validation_jobs(
        jobs,
        args=args,
        results_root=results_root,
        grid_attempt_id="G1",
        validation_config=PAIR_VALIDATION,
    )
    assert skipped == []
    assert validation_jobs[0]["train_attempt_id"] == "T1"


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
        "full_model_antisymmetry",
        "trace_equivariance",
        "feature_trace_stability",
        "readout_trace_stability",
    ]
    # Every evaluation task routes its artifacts under the validation run dir.
    for task in evaluator.tasks:
        assert str(task.output_dir).startswith(str(tmp_run_dir()))


def test_pair_validation_final_eval_suite_is_report_grade() -> None:
    import spenn.config  # noqa: F401 - registers the basis_feature_dim resolver
    from hydra.utils import instantiate

    cfg = OmegaConf.load(PAIR_VALIDATION)
    cfg.evaluation.suite = "final_eval"
    cfg.run_parameters.architecture = "hermite_o3_envelope"
    cfg.run_parameters.normalization = "N2"
    cfg.run_parameters.channels = 4
    OmegaConf.update(cfg, "run.dir", str(tmp_run_dir()), force_add=True)
    evaluator = instantiate(cfg.evaluator)
    task_names = [task.name for task in evaluator.tasks]

    assert cfg.evaluation.artifact_level == "records"
    assert task_names[:4] == ["cusp", "tail", "stratified_geometry", "hooke_orbital"]
    assert "energy" in task_names
    assert "spatial_exchange_symmetry" in task_names
    assert "rotation_consistency" in task_names
    assert len(task_names) > 8
    assert cfg.evaluation_tasks.final_cusp.generator.n_points > cfg.evaluation_tasks.cusp.generator.n_points
    assert (
        cfg.evaluation_tasks.final_stratified_geometry.generator.n_samples
        > cfg.evaluation_tasks.stratified_geometry.generator.n_samples
    )


def tmp_run_dir() -> Path:
    return Path("/tmp/pair_stability_eval_check")


# ---------------------------------------------------------------------------
# Collection / selection
# ---------------------------------------------------------------------------
def _fake_validation_attempt(
    results_root: Path, run_id: str, attempt_id: str, *, status: str, stratified_energy: float
) -> None:
    attempt = run_utils.validation_attempt_dir(results_root, run_id, attempt_id)
    for task_name in collect.TASK_NAMES:
        (attempt / task_name).mkdir(parents=True, exist_ok=True)
    (attempt / "status.json").write_text(json.dumps({"status": status}))
    train_attempt = run_utils.train_attempt_dir(results_root, run_id, "T1")
    train_attempt.mkdir(parents=True, exist_ok=True)
    train_records = [
        {"namespace": "train", "metrics": {"energy": 2.5}},
        {"namespace": "runtime", "metrics": {"wall_time_sec": 123.0}},
    ]
    (train_attempt / "metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in train_records) + "\n"
    )
    (attempt / "source_train_attempt.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "train_attempt_id": "T1",
                "train_attempt_dir": str(train_attempt),
                "checkpoint_path": str(train_attempt / "checkpoints"),
            }
        )
    )
    records = [
        {"namespace": "eval/stratified_geometry", "metrics": {"local_energy_mean": stratified_energy}},
        {"namespace": "eval/cusp", "metrics": {"opposite_spin_cusp_slope": -0.5}},
    ]
    (attempt / "metrics.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_collect_reads_eval_diagnostics_from_validation_attempts(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    good = "arch-hermite_o3_envelope_norm-N0_lr-1e-3_ch-16_seed-0"
    bad = "arch-raw_envelope_norm-N0_lr-1e-3_ch-8_seed-1"
    _fake_validation_attempt(results_root, good, "V1", status="completed", stratified_energy=1.95)
    _fake_validation_attempt(results_root, bad, "V1", status="failed", stratified_energy=5.0)

    result = collect.collect(results_root=results_root, collect_attempt_id="C1")
    attempt = Path(result["attempt_dir"])
    assert attempt == results_root / "03_collect" / "C1"

    summary_rows = list(_read_csv(attempt / "summary.csv"))
    by_run = {row["run_id"]: row for row in summary_rows}
    assert set(by_run) == {good, bad}
    assert by_run[good]["eval/stratified_geometry/local_energy_mean"] == "1.95"
    assert by_run[good]["train/train/energy"] == "2.5"
    assert by_run[good]["train/runtime/wall_time_sec"] == "123.0"
    assert by_run[good]["train_attempt_id"] == "T1"
    assert by_run[good]["architecture"] == "hermite_o3_envelope"
    assert int(by_run[good]["n_diagnostics"]) == len(collect.TASK_NAMES)

    failures = list(_read_csv(attempt / "failures.csv"))
    assert {row["run_id"] for row in failures} == {bad}

    consumed = json.loads((attempt / "source_validation_attempts.json").read_text())
    assert {entry["run_id"] for entry in consumed} == {good, bad}
    assert json.loads((attempt / "collection_report.json").read_text())["n_collected"] == 2


def test_select_champions_reads_collection_attempt(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    collection = results_root / "03_collect" / "C1"
    collection.mkdir(parents=True, exist_ok=True)

    def row(
        architecture: str,
        normalization: str,
        lr: str,
        channels: str,
        seed: int,
        energy: float,
        feature: float,
        readout: float = 100.0,
        train_wall_time: float = 10.0,
    ) -> dict[str, str]:
        data = {
            "run_id": f"arch-{architecture}_norm-{normalization}_lr-{lr}_ch-{channels}_seed-{seed}",
            "architecture": architecture,
            "normalization": normalization,
            "lr": lr,
            "channels": channels,
            "seed": str(seed),
            "status": "completed",
            "eval/feature_trace_stability/feature_rms_q95": str(feature),
            "eval/readout_trace_stability/condition_number_q95": str(readout),
            "train/runtime/wall_time_sec": str(train_wall_time),
        }
        for task in select_champions.ENERGY_TASK_ORDER:
            data[f"eval/{task}/local_energy_mean"] = str(energy)
            data[f"eval/{task}/local_energy_stderr"] = "0.01"
        return data

    rows = [
        row("hermite_o3_envelope", "N0", "1e-3", "16", 0, 5.0, 0.020),
        row("hermite_o3_envelope", "N0", "1e-3", "16", 1, 5.2, 0.021),
        row("hermite_o3_envelope", "N0", "1e-3", "16", 2, 5.4, 0.022),
        row("hermite_o3_envelope", "N0", "3e-3", "16", 0, 2.0, 0.010, readout=40.0),
        row("hermite_o3_envelope", "N0", "3e-3", "16", 1, 2.2, 0.011, readout=42.0),
        row("hermite_o3_envelope", "N0", "3e-3", "16", 2, 2.4, 0.012, readout=44.0),
        row("raw_envelope", "N1", "1e-3", "8", 0, 3.0, 0.030),
        row("raw_envelope", "N1", "1e-3", "8", 1, 3.2, 0.031),
        row("raw_envelope", "N1", "3e-3", "8", 0, 4.0, 0.010),
        row("raw_envelope", "N1", "3e-3", "8", 1, 4.2, 0.011),
    ]
    _write_csv(collection / "summary.csv", rows)

    result = select_champions.select(results_root=results_root, collection_attempt_id="C1", select_attempt_id="S1")
    attempt = Path(result["attempt_dir"])
    assert attempt == results_root / "04_select" / "S1"

    source = json.loads((attempt / "source_collection_attempt.json").read_text())
    assert source["collection_attempt_id"] == "C1"

    champions = _read_csv(attempt / "champions.csv")
    assert len(champions) == 4
    by_group_kind = {
        (row["architecture"], row["normalization"], row["winner_kind"]): row for row in champions
    }
    energy = by_group_kind[("hermite_o3_envelope", "N0", "energy")]
    assert energy["config_id"] == "arch-hermite_o3_envelope_norm-N0_lr-3e-3_ch-16"
    assert energy["metric"] == "eval/stratified_geometry/local_energy_mean_seed_median"
    assert energy["metric_value"] == "2.2"
    assert energy["metric_seed_mean"] == "2.2"
    assert float(energy["metric_seed_stderr"]) == pytest.approx(0.1154700538)
    assert energy["stratified_geometry_energy_seed_median"] == "2.2"
    assert energy["stratified_geometry_energy_seed_mean"] == "2.2"
    assert float(energy["stratified_geometry_energy_seed_stderr"]) == pytest.approx(0.1154700538)
    assert energy["tail_energy_seed_median"] == "2.2"
    assert energy["cusp_energy_seed_median"] == "2.2"
    assert energy["hooke_orbital_energy_seed_median"] == "2.2"
    assert energy["feature_stability_seed_median"] == "0.011"
    assert energy["feature_stability_seed_mean"] == "0.011"
    assert energy["readout_stability_seed_median"] == "42"
    assert energy["readout_stability_seed_mean"] == "42"
    assert float(energy["readout_stability_seed_stderr"]) == pytest.approx(1.154700538)

    feature = by_group_kind[("hermite_o3_envelope", "N0", "feature_trace")]
    assert feature["config_id"] == "arch-hermite_o3_envelope_norm-N0_lr-1e-3_ch-16"
    assert feature["metric"] == "eval/feature_trace_stability/feature_rms_q95_seed_median"
    assert feature["feature_stability_seed_median"] == "0.021"
    assert feature["readout_stability_seed_median"] == "100"

    raw_energy = by_group_kind[("raw_envelope", "N1", "energy")]
    raw_feature = by_group_kind[("raw_envelope", "N1", "feature_trace")]
    assert raw_energy["config_id"] == "arch-raw_envelope_norm-N1_lr-1e-3_ch-8"
    assert raw_feature["config_id"] == "arch-raw_envelope_norm-N1_lr-3e-3_ch-8"

    report = json.loads((attempt / "selection_report.json").read_text())
    assert report["metric"] is None
    assert report["group_by"] == ["architecture", "normalization"]
    assert report["config_keys"] == ["architecture", "normalization", "lr", "channels"]
    assert report["energy_task_order"] == ["stratified_geometry", "tail", "cusp", "hooke_orbital"]
    assert report["reference_metrics"]["readout_stability"] == (
        "eval/readout_trace_stability/condition_number_q95"
    )
    assert report["overall_champion"] == "arch-hermite_o3_envelope_norm-N0_lr-3e-3_ch-16"
    assert report["feature_trace_metric"] == "eval/feature_trace_stability/feature_rms_q95_seed_median"
    assert report["n_champions"] == 4


def test_select_champions_falls_back_to_wall_time_after_energy_ties() -> None:
    rows = [
        {
            "run_id": "slow-seed-0",
            "architecture": "raw_envelope",
            "normalization": "N0",
            "lr": "1e-3",
            "channels": "8",
            "seed": "0",
            "status": "completed",
            "train/runtime/wall_time_sec": "20.0",
        },
        {
            "run_id": "fast-seed-0",
            "architecture": "raw_envelope",
            "normalization": "N0",
            "lr": "3e-3",
            "channels": "8",
            "seed": "0",
            "status": "completed",
            "train/runtime/wall_time_sec": "10.0",
        },
    ]
    for row in rows:
        for task in select_champions.ENERGY_TASK_ORDER:
            row[f"eval/{task}/local_energy_mean"] = "2.0"
            row[f"eval/{task}/local_energy_stderr"] = "0.1"

    selection = select_champions.select_champions(rows)

    assert selection["overall_champion"] == "arch-raw_envelope_norm-N0_lr-3e-3_ch-8"
    assert selection["overall_metric"] == "train/runtime/wall_time_sec_seed_median"
    champion = selection["champions"][0]
    assert champion["winner_kind"] == "energy"
    assert champion["config_id"] == "arch-raw_envelope_norm-N0_lr-3e-3_ch-8"
    assert champion["metric"] == "train/runtime/wall_time_sec_seed_median"


def test_select_champions_feature_trace_winner_skips_overall_energy_champion() -> None:
    rows = [
        {
            "run_id": "energy",
            "architecture": "raw_envelope",
            "normalization": "N0",
            "lr": "1e-3",
            "channels": "8",
            "seed": "0",
            "status": "completed",
            "eval/stratified_geometry/local_energy_mean": "1.0",
            "eval/stratified_geometry/local_energy_stderr": "0.01",
            "eval/feature_trace_stability/feature_rms_q95": "0.01",
        },
        {
            "run_id": "feature",
            "architecture": "raw_envelope",
            "normalization": "N0",
            "lr": "3e-3",
            "channels": "8",
            "seed": "0",
            "status": "completed",
            "eval/stratified_geometry/local_energy_mean": "2.0",
            "eval/stratified_geometry/local_energy_stderr": "0.01",
            "eval/feature_trace_stability/feature_rms_q95": "0.02",
        },
        {
            "run_id": "other",
            "architecture": "raw_envelope",
            "normalization": "N0",
            "lr": "1e-4",
            "channels": "8",
            "seed": "0",
            "status": "completed",
            "eval/stratified_geometry/local_energy_mean": "3.0",
            "eval/stratified_geometry/local_energy_stderr": "0.01",
            "eval/feature_trace_stability/feature_rms_q95": "0.03",
        },
    ]

    selection = select_champions.select_champions(rows)

    assert selection["overall_champion"] == "arch-raw_envelope_norm-N0_lr-1e-3_ch-8"
    assert selection["feature_trace_champion"] == "arch-raw_envelope_norm-N0_lr-3e-3_ch-8"
    assert selection["feature_trace_metric"] == "eval/feature_trace_stability/feature_rms_q95_seed_median"
    assert selection["feature_trace_metric_value"] == "0.02"


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
