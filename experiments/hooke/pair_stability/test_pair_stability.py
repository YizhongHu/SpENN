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
    # The planner owns the timezone and always injects it (the config is null).
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

    def fake_submit_local(commands, *, repo_root: Path, chunk_size: int = launch.DEFAULT_CHUNK_SIZE):
        assert chunk_size == launch.DEFAULT_CHUNK_SIZE
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

    def fake_submit_local(commands, *, repo_root: Path, chunk_size: int = launch.DEFAULT_CHUNK_SIZE):
        assert chunk_size == launch.DEFAULT_CHUNK_SIZE
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
    assert "validation_sampler_params.n_steps=5" in script
    assert "checkpoint.every_n_steps=1" in script

    smoke_attempt_dir = results_root / "01_train" / first_run_id / smoke_attempt
    source = json.loads((smoke_attempt_dir / "source_grid_attempt.json").read_text())
    assert source["grid_attempt_id"] == ATTEMPT
    submission = json.loads((smoke_attempt_dir / "submission.json").read_text())
    assert submission["launcher_job_id"] == "local-smoke-0"
    assert f"run.run_id={first_run_id}/{smoke_attempt}" in submission["submitted_command"]


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
    (attempt / "source_train_attempt.json").write_text(
        json.dumps({"run_id": run_id, "train_attempt_id": "T1", "checkpoint_path": "ckpt"})
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
    rows = [
        {"run_id": "arch-hermite_o3_envelope_norm-N0_lr-1e-3_ch-16_seed-0", "architecture": "hermite_o3_envelope",
         "normalization": "N0", "lr": "1e-3", "channels": "16", "seed": "0", "status": "completed",
         "eval/stratified_geometry/local_energy_mean": "1.90",
         "eval/stratified_geometry/local_energy_stderr": "0.20",
         "eval/tail/local_energy_mean": "2.40",
         "eval/tail/local_energy_stderr": "0.01"},
        {"run_id": "arch-hermite_o3_envelope_norm-N2_lr-1e-3_ch-16_seed-0", "architecture": "hermite_o3_envelope",
         "normalization": "N2", "lr": "1e-3", "channels": "16", "seed": "0", "status": "completed",
         "eval/stratified_geometry/local_energy_mean": "1.95",
         "eval/stratified_geometry/local_energy_stderr": "0.20",
         "eval/tail/local_energy_mean": "1.70",
         "eval/tail/local_energy_stderr": "0.01"},
        {"run_id": "arch-raw_envelope_norm-N0_lr-1e-3_ch-8_seed-0", "architecture": "raw_envelope",
         "normalization": "N0", "lr": "1e-3", "channels": "8", "seed": "0", "status": "completed",
         "eval/stratified_geometry/local_energy_mean": "3.0",
         "eval/stratified_geometry/local_energy_stderr": "0.01"},
    ]
    _write_csv(collection / "summary.csv", rows)

    result = select_champions.select(results_root=results_root, collection_attempt_id="C1", select_attempt_id="S1")
    attempt = Path(result["attempt_dir"])
    assert attempt == results_root / "04_select" / "S1"

    source = json.loads((attempt / "source_collection_attempt.json").read_text())
    assert source["collection_attempt_id"] == "C1"

    champions = {row["architecture"]: row for row in _read_csv(attempt / "champions.csv")}
    # Stratified geometry overlaps, so tail local energy breaks the tie.
    assert champions["hermite_o3_envelope"]["run_id"].endswith("norm-N2_lr-1e-3_ch-16_seed-0")
    assert champions["hermite_o3_envelope"]["metric"] == "eval/tail/local_energy_mean"
    report = json.loads((attempt / "selection_report.json").read_text())
    assert report["metric"] is None
    assert report["energy_task_order"] == ["stratified_geometry", "tail", "cusp", "hooke_orbital"]
    assert report["overall_champion"].endswith("norm-N2_lr-1e-3_ch-16_seed-0")


def test_select_champions_falls_back_to_wall_time_after_energy_ties() -> None:
    rows = [
        {"run_id": "slow", "architecture": "raw_envelope", "status": "completed", "runtime/wall_time_sec": "20.0"},
        {"run_id": "fast", "architecture": "raw_envelope", "status": "completed", "runtime/wall_time_sec": "10.0"},
    ]
    for row in rows:
        for task in select_champions.ENERGY_TASK_ORDER:
            row[f"eval/{task}/local_energy_mean"] = "2.0"
            row[f"eval/{task}/local_energy_stderr"] = "0.1"

    selection = select_champions.select_champions(rows)

    assert selection["overall_champion"] == "fast"
    assert selection["overall_metric"] == "wall_time_sec"
    champion = selection["champions"][0]
    assert champion["run_id"] == "fast"
    assert champion["metric"] == "wall_time_sec"


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
