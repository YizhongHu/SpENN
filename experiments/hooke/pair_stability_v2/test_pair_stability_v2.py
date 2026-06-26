"""Study-level tests for the pair-stability V2 major/minor grid."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import pytest
from omegaconf import OmegaConf

STUDY_DIR = Path(__file__).resolve().parent
CONFIGS = STUDY_DIR / "configs"
GRID = CONFIGS / "grid.yaml"

while str(STUDY_DIR) in sys.path:
    sys.path.remove(str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR))
for module_name in list(sys.modules):
    if module_name == "utils" or module_name.startswith("utils."):
        del sys.modules[module_name]


def _load_script(name: str, *, bind_direct: bool = False) -> ModuleType:
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"pair_stability_v2_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if bind_direct:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


from utils import io as json_io  # noqa: E402
from utils import layout  # noqa: E402
launch = _load_script("launch")
plan = _load_script("plan")
train = _load_script("train")
collect = _load_script("collect")
select_champions = _load_script("select_champions")
final_plan = _load_script("final_plan")
final_train = _load_script("final_train", bind_direct=True)
final_eval = _load_script("final_eval")
final_collect = _load_script("final_collect")
validate = _load_script("validate")


ATTEMPT = "20260623T120000-0400"
ROOT = STUDY_DIR.parents[2]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _planned_results(tmp_path: Path) -> Path:
    results_root = tmp_path / "results"
    code = plan.main(["--grid", str(GRID), "--results-root", str(results_root), "--attempt-id", ATTEMPT])
    assert code == 0
    return results_root


def _write_checkpoint_pointer(results_root: Path, run_id: str, attempt_id: str) -> Path:
    checkpoint_dir = layout.train_attempt_dir(results_root, run_id, attempt_id) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "latest.json").write_text(json.dumps({"path": "step_000000"}))
    return checkpoint_dir


def _write_final_checkpoint(results_root: Path, final_run_id: str, attempt_id: str) -> Path:
    attempt_dir = layout.final_train_attempt_dir(results_root, final_run_id, attempt_id)
    checkpoint_dir = attempt_dir / "checkpoints" / "step_000000"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "COMPLETE").write_text("")
    (checkpoint_dir / "manifest.json").write_text(json.dumps({"step": 0}) + "\n")
    latest = attempt_dir / "checkpoints" / "latest.json"
    latest.write_text(json.dumps({"checkpoint_dir": "step_000000"}) + "\n")
    (attempt_dir / "selected_checkpoint.json").write_text(
        json.dumps(
            {
                "selection_policy": "latest_checkpoint_pointer",
                "checkpoint_pointer": str(latest),
            }
        )
        + "\n"
    )
    return checkpoint_dir


def test_v2_smoke_slurm_defaults_match_pair_stability() -> None:
    args = types.SimpleNamespace(
        slurm_partition=None,
        slurm_array_parallelism=None,
        slurm_timeout_min=None,
        slurm_mem_gb=None,
        slurm_cpus=None,
        slurm_gpus=None,
    )

    smoke_cpu = launch.slurm_parameters(args, profile="cpu", smoke=True)
    smoke_cuda = launch.slurm_parameters(args, profile="cuda", smoke=True)

    assert smoke_cpu["slurm_partition"] == "test"
    assert smoke_cuda["slurm_partition"] == "gpu_test"
    assert smoke_cpu["timeout_min"] == 15
    assert smoke_cuda["timeout_min"] == 15
    assert smoke_cpu["mem_gb"] == 128
    assert smoke_cpu["cpus_per_task"] == 16
    assert smoke_cuda["cpus_per_task"] == 4
    assert smoke_cpu["slurm_array_parallelism"] == 2
    assert smoke_cuda["slurm_array_parallelism"] == 2
    assert smoke_cuda["gpus_per_node"] == 1


def test_v2_mixed_device_prepares_cpu_and_cuda_commands() -> None:
    args = train.parse_args(
        [
            "--backend",
            "submitit",
            "--device",
            "cpu,cuda",
            "--slurm-cpu-partition",
            "test",
            "--slurm-cuda-partition",
            "gpu_test",
            "--slurm-cpu-timeout-min",
            "60",
            "--slurm-cuda-timeout-min",
            "30",
        ]
    )

    command_sets = launch.environment_command_sets(
        [["python", "-u", "run.py", "--config", "cfg.yaml", "runtime.device=cpu"]],
        args=args,
        repo_root=ROOT,
    )

    assert tuple(command_sets) == ("cpu", "cuda")
    cpu_script = command_sets["cpu"][0][-1]
    cuda_script = command_sets["cuda"][0][-1]
    assert "export UV_PROJECT_ENVIRONMENT=.venv" in cpu_script
    assert "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}" in cpu_script
    assert "uv sync --extra cpu" in cpu_script
    assert "runtime.device=cpu" in cpu_script
    assert "export UV_PROJECT_ENVIRONMENT=.venv-gpu" in cuda_script
    assert "OMP_NUM_THREADS" not in cuda_script
    assert "uv sync --extra cu126" in cuda_script
    assert "runtime.device=cuda" in cuda_script

    cpu_slurm = launch.slurm_parameters(args, profile="cpu")
    assert cpu_slurm["slurm_partition"] == "test"
    assert cpu_slurm["mem_gb"] == 128
    assert cpu_slurm["cpus_per_task"] == 16
    cuda_slurm = launch.slurm_parameters(args, profile="cuda")
    assert cuda_slurm["slurm_partition"] == "gpu_test"
    assert cuda_slurm["cpus_per_task"] == 8
    assert cuda_slurm["gpus_per_node"] == 1


def test_v2_mixed_submitit_submits_separate_claimed_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_parameters: list[dict[str, Any]] = []
    captured_calls: list[tuple[Any, tuple[Any, ...]]] = []

    class FakeExecutor:
        def __init__(self, folder: str):
            self.folder = folder

        def update_parameters(self, **kwargs: Any) -> None:
            captured_parameters.append(kwargs)

        def map_array(self, fn: Any, *args: Any) -> list[types.SimpleNamespace]:
            captured_calls.append((fn, args))
            commands = args[0]
            return [
                types.SimpleNamespace(job_id=f"{self.folder}-job-{index}")
                for index, _command in enumerate(commands)
            ]

    monkeypatch.setitem(sys.modules, "submitit", types.SimpleNamespace(AutoExecutor=FakeExecutor))
    args = train.parse_args(
        [
            "--backend",
            "submitit",
            "--device",
            "cpu,cuda",
            "--slurm-cpu-partition",
            "test",
            "--slurm-cuda-partition",
            "gpu_test",
            "--slurm-cpu-timeout-min",
            "60",
            "--slurm-cuda-timeout-min",
            "30",
        ]
    )
    command_sets = {
        "cpu": [["bash", "-lc", "cpu"]],
        "cuda": [["bash", "-lc", "cuda"]],
    }
    row_status = tmp_path / "run" / "launcher_status.json"

    job_ids = launch.submit_command_sets(
        command_sets,
        args=args,
        backend="submitit",
        repo_root=ROOT,
        log_dir=tmp_path / "logs",
        job_name="mixed",
        smoke=False,
        row_status_paths=[row_status],
        chunk_status_dir=tmp_path / "chunks",
    )

    assert len(captured_parameters) == 2
    assert captured_parameters[0]["slurm_partition"] == "test"
    assert captured_parameters[0]["timeout_min"] == 60
    assert captured_parameters[0]["mem_gb"] == 128
    assert captured_parameters[0]["cpus_per_task"] == 16
    assert captured_parameters[1]["slurm_partition"] == "gpu_test"
    assert captured_parameters[1]["timeout_min"] == 30
    assert captured_parameters[1]["mem_gb"] == 80
    assert captured_parameters[1]["cpus_per_task"] == 8
    assert captured_parameters[1]["gpus_per_node"] == 1
    assert captured_calls[0][0] is launch.run_command_chunk
    assert captured_calls[0][1][5] == [[row_status.with_name("launcher_claim.json")]]
    assert captured_calls[0][1][6] == ["cpu"]
    assert captured_calls[1][1][5] == [[row_status.with_name("launcher_claim.json")]]
    assert captured_calls[1][1][6] == ["cuda"]
    assert job_ids[0].startswith("cpu:")
    assert ",cuda:" in job_ids[0]


def test_v2_local_claim_mode_uses_claim_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_submit_local(commands: Sequence[Sequence[str]], **kwargs: Any) -> list[str]:
        captured["commands"] = commands
        captured["kwargs"] = kwargs
        return ["local-0"]

    monkeypatch.setattr(launch, "submit_local", fake_submit_local)
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1782579321")
    args = train.parse_args(["--backend", "local", "--device", "cpu"])
    row_status = tmp_path / "run" / "launcher_status.json"

    job_ids = launch.submit_command_sets(
        {"cpu": [["bash", "-lc", "cpu"]]},
        args=args,
        backend="local",
        repo_root=ROOT,
        log_dir=tmp_path / "logs",
        job_name="local",
        smoke=False,
        row_status_paths=[row_status],
        chunk_status_dir=tmp_path / "chunks",
        claim_rows=True,
    )

    assert job_ids == ["local-0"]
    assert captured["kwargs"]["claim_paths"] == [row_status.with_name("launcher_claim.json")]
    assert captured["kwargs"]["claim_label"] == "local-cpu"
    assert captured["kwargs"]["claim_deadline_unix"] == 1782579321.0
    assert captured["kwargs"]["claim_deadline_guard_min"] == 60


def test_v2_run_command_chunk_deadline_guard_skips_unclaimed_row(tmp_path: Path) -> None:
    row_status = tmp_path / "run" / "launcher_status.json"
    claim_path = row_status.with_name("launcher_claim.json")

    result = launch.run_command_chunk(
        [["bash", "-lc", "false"]],
        row_status_paths=[row_status],
        claim_paths=[claim_path],
        claim_label="local-cpu",
        claim_deadline_unix=0.0,
        claim_deadline_guard_min=60,
    )

    assert result["status"] == "deadline_guard"
    assert result["rows"][0]["status"] == "skipped_deadline_guard"
    assert not claim_path.exists()
    assert not row_status.exists()


def test_v2_run_command_chunk_reclaims_failed_row_claim(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "run"
    attempt_dir.mkdir()
    row_status = attempt_dir / "launcher_status.json"
    claim_path = attempt_dir / "launcher_claim.json"
    row_status.write_text(json.dumps({"status": "running"}) + "\n")
    (attempt_dir / "status.json").write_text(json.dumps({"status": "failed"}) + "\n")
    claim_path.write_text(json.dumps({"status": "claimed", "claim_label": "cuda"}) + "\n")

    result = launch.run_command_chunk(
        [["bash", "-lc", "true"]],
        row_status_paths=[row_status],
        claim_paths=[claim_path],
    )

    assert result["rows"][0]["status"] == "success"
    claim = json.loads(claim_path.read_text())
    assert claim["reclaimed"] is True
    assert claim["reclaim_reason"] == "failed"
    assert claim["previous_claim"]["claim_label"] == "cuda"


def test_v2_submitit_launcher_reexec_uses_dedicated_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_execvpe(file: str, args: list[str], env: dict[str, str]) -> None:
        captured["file"] = file
        captured["args"] = args
        captured["env"] = env
        raise RuntimeError("execvpe")

    monkeypatch.setattr(launch, "_python_in_environment", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(launch.os, "execvpe", fake_execvpe)

    args = types.SimpleNamespace(backend="submitit")
    with pytest.raises(RuntimeError, match="execvpe"):
        launch.ensure_submitit_launcher_environment(
            args,
            script_path=STUDY_DIR / "validate.py",
            argv=["--backend=submitit", "--wait-job=123"],
            repo_root=ROOT,
        )

    assert captured["file"] == "uv"
    assert captured["env"]["UV_PROJECT_ENVIRONMENT"] == ".venv-submitit"
    assert captured["env"]["SPENN_SUBMITIT_LAUNCHER_REEXEC"] == "1"
    assert captured["args"][:5] == ["uv", "run", "--extra", "submitit", "python"]
    assert "--wait-job=123" in captured["args"]


def test_v2_latest_attempt_id_prefers_pointer_with_sorted_fallback(tmp_path: Path) -> None:
    parent = tmp_path / "stage"
    (parent / "zzz").mkdir(parents=True)
    (parent / "aaa").mkdir()
    layout.write_latest(parent, "aaa")

    assert layout.latest_attempt_id(parent) == "aaa"

    layout.write_latest(parent, "diagnostic", smoke=True)
    assert layout.latest_attempt_id(parent) == "aaa"
    assert layout.latest_attempt_id(parent, smoke=False) == "aaa"
    assert layout.latest_attempt_id(parent, smoke=True) == "diagnostic"
    assert layout.latest_attempt_id(parent / "missing") is None


def test_v2_smoke_final_workflow_falls_back_to_full_upstream_attempts(tmp_path: Path) -> None:
    results_root = tmp_path / "results"

    select_stage = layout.stage_dir(results_root, layout.STAGE_SELECT)
    (select_stage / "select-full").mkdir(parents=True)
    layout.write_latest(select_stage, "select-full", smoke=False)

    final_grid_stage = layout.stage_dir(results_root, layout.STAGE_FINAL_GRID)
    (final_grid_stage / "final-grid-full").mkdir(parents=True)
    layout.write_latest(final_grid_stage, "final-grid-full", smoke=False)

    assert final_plan._resolve_selection_attempt(results_root, None, smoke=True) == "select-full"
    assert final_train._resolve_final_grid_attempt_id(results_root, None, smoke=True) == "final-grid-full"
    assert final_eval._resolve_final_grid_attempt_id(results_root, None, smoke=True) == "final-grid-full"


def test_v2_train_and_validation_default_through_latest_pointers(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    job = manifest["jobs"][0]
    run_id = str(job["run_id"])

    row_status_paths = train.write_train_launch_provenance(
        [job],
        manifest=manifest,
        results_root=results_root,
        grid_attempt_id=ATTEMPT,
        repo_root=ROOT,
        submitted_commands=[["python", "run.py"]],
    )
    _write_checkpoint_pointer(results_root, run_id, ATTEMPT)
    _write_checkpoint_pointer(results_root, run_id, "zzz")

    assert row_status_paths == [layout.train_attempt_dir(results_root, run_id, ATTEMPT) / "launcher_status.json"]
    assert validate.latest_train_attempt_id(results_root, run_id, smoke=False) == ATTEMPT

    scalar_axes = validate._scalar_axes(manifest)
    args = types.SimpleNamespace(smoke=False, train_attempt_id=None, attempt_id="manual-validation")
    planned, skipped = validate.plan_validation_jobs(
        [job],
        args=args,
        study="pair_stability_v2",
        results_root=results_root,
        grid_attempt_id=ATTEMPT,
        validation_config="validation.yaml",
        scalar_axes=scalar_axes,
        override_paths=validate._axis_override_paths(manifest, scalar_axes),
        seed_axis=str(manifest["scan_seed_axis"]),
        smoke_overrides={},
        seed_policy=manifest.get("seed_overrides"),
    )

    assert skipped == []
    assert planned[0]["train_attempt_id"] == ATTEMPT
    latest_validation = json.loads((layout.validation_run_dir(results_root, run_id) / "latest.json").read_text())
    assert latest_validation["attempt_id"] == "manual-validation"


def test_v2_real_validation_uses_non_smoke_train_attempts(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    job = manifest["jobs"][0]
    run_id = str(job["run_id"])
    smoke_attempt = "diagnostic-train"
    _write_checkpoint_pointer(results_root, run_id, ATTEMPT)
    _write_checkpoint_pointer(results_root, run_id, smoke_attempt)
    layout.write_latest(layout.train_run_dir(results_root, run_id), smoke_attempt, smoke=True)

    assert validate.latest_train_attempt_id(results_root, run_id, smoke=False) == ATTEMPT
    assert validate.latest_train_attempt_id(results_root, run_id, smoke=True) == smoke_attempt

    args = types.SimpleNamespace(smoke=False, train_attempt_id=None, attempt_id="real-validation")
    scalar_axes = validate._scalar_axes(manifest)
    planned, skipped = validate.plan_validation_jobs(
        [job],
        args=args,
        study="pair_stability_v2",
        results_root=results_root,
        grid_attempt_id=ATTEMPT,
        validation_config="validation.yaml",
        scalar_axes=scalar_axes,
        override_paths=validate._axis_override_paths(manifest, scalar_axes),
        seed_axis=str(manifest["scan_seed_axis"]),
        smoke_overrides={},
        seed_policy=manifest.get("seed_overrides"),
    )
    assert skipped == []
    assert planned[0]["train_attempt_id"] == ATTEMPT

    args.train_attempt_id = smoke_attempt
    with pytest.raises(ValueError, match="refuses a smoke train attempt"):
        validate.plan_validation_jobs(
            [job],
            args=args,
            study="pair_stability_v2",
            results_root=results_root,
            grid_attempt_id=ATTEMPT,
            validation_config="validation.yaml",
            scalar_axes=scalar_axes,
            override_paths=validate._axis_override_paths(manifest, scalar_axes),
            seed_axis=str(manifest["scan_seed_axis"]),
            smoke_overrides={},
            seed_policy=manifest.get("seed_overrides"),
        )


def test_v2_wait_job_submits_dependent_launcher(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> types.SimpleNamespace:
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="88888;cluster\n", stderr="")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)

    submitted = launch.submit_dependent_launcher(
        "24211558",
        script_path=STUDY_DIR / "validate.py",
        argv=[
            "--backend=submitit",
            "--cuda",
            "--wait-job=24211558",
            "--chunk-size",
            "32",
        ],
        repo_root=ROOT,
        log_dir=tmp_path / "logs",
        job_name="pair-stability-v2-validate-launcher",
        partition="test",
        timeout_min=19,
        study="pair_stability_v2",
    )

    command, kwargs = calls[0]
    assert submitted == "88888"
    assert "--dependency=afterany:24211558" in command
    assert "--partition=test" in command
    assert "--time=00:19:00" in command
    assert "--output=" + str(tmp_path / "logs" / "%x-%j.out") in command
    script = str(kwargs["input"])
    assert "UV_PROJECT_ENVIRONMENT=.venv-submitit" in script
    assert "uv run --extra submitit python -u" in script
    assert "--wait-job" not in script
    assert "--backend=submitit" in script


def test_v2_blinding_is_reproducible_by_seed(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    for attempt, seed in (("SAME1", 811), ("SAME2", 811), ("DIFF", 812)):
        code = plan.main(
            [
                "--grid",
                str(GRID),
                "--results-root",
                str(results_root),
                "--attempt-id",
                attempt,
                "--blind",
                "--blind-seed",
                str(seed),
            ]
        )
        assert code == 0

    same1 = json.loads((results_root / "00_grid" / "SAME1" / "unblind.json").read_text())
    same2 = json.loads((results_root / "00_grid" / "SAME2" / "unblind.json").read_text())
    diff = json.loads((results_root / "00_grid" / "DIFF" / "unblind.json").read_text())

    assert same1["axes"] == same2["axes"]
    assert same1["axes"] != diff["axes"]


def test_v2_plan_records_major_minor_scan_manifest(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    grid_attempt = layout.grid_attempt_dir(results_root, ATTEMPT)
    manifest = json.loads((grid_attempt / "manifest.json").read_text())

    assert manifest["study"] == "pair_stability_v2"
    assert manifest["config_snapshots"] == {
        "train": "train_config.yaml",
        "validation": "validation_config.yaml",
        "smoke": "smoke_config.yaml",
    }
    assert (grid_attempt / "train_config.yaml").is_file()
    assert (grid_attempt / "validation_config.yaml").is_file()
    assert (grid_attempt / "smoke_config.yaml").is_file()
    assert not (grid_attempt / "pair_stability.yaml").exists()
    assert not (grid_attempt / "pair_validation.yaml").exists()
    assert manifest["grid_schema"] == "major_minor_scan"
    assert manifest["major_axes"] == ["basis", "mechanism"]
    assert manifest["minor_axes"] == ["lr", "channels"]
    assert manifest["scan_seed_axis"] == "seed"
    assert manifest["axis_id_labels"] == {
        "basis": "b",
        "mechanism": "m",
        "lr": "lr",
        "channels": "ch",
        "seed": "seed",
    }
    assert manifest["axis_overrides"] == {
        "basis": "run_parameters.basis_slot",
        "mechanism": "run_parameters.mechanism_slot",
        "lr": "run_parameters.lr",
        "channels": "run_parameters.channels",
    }
    assert manifest["choice_validation"]["basis"]["choices_path"] == "choices.basis"
    assert manifest["choice_validation"]["mechanism"]["choices_path"] == "choices.mechanism"
    assert [champion["name"] for champion in manifest["champions"]] == ["energy"]
    assert manifest["champion_kinds"] == ["energy"]
    assert manifest["champions"][0]["selector"] == "metric_ladder"
    assert manifest["seed_overrides"]["scan_train"] == {
        "run_parameters.seed": "scan_seed",
        "runtime.seed": "scan_seed",
        "sampler.seed": "scan_seed",
    }
    assert manifest["seed_overrides"]["validation"] == {
        "run_parameters.seed": "scan_seed",
        "runtime.seed": "scan_seed",
        "evaluation.seed": "scan_seed",
    }
    assert manifest["final_seed_sequences"] == {
        "final_train_sampler_seed": {"start": 101, "step": 1},
        "final_train_model_seed": {"start": 1001, "step": 1},
        "final_eval_seed": {"start": 10001, "step": 1},
    }
    assert manifest["final_replicates"] == 9
    assert manifest["n_jobs"] == 270
    assert manifest["blinding"]["enabled"] is True
    assert manifest["blinding"]["blind_seed"] == 0

    unblind = json.loads((grid_attempt / "unblind.json").read_text())
    assert set(unblind["axes"]) == {"basis", "mechanism"}
    assert set(unblind["axes"]["basis"]["slot_to_value"].values()) == set(OmegaConf.load(GRID).major_grid.basis)
    assert set(unblind["axes"]["mechanism"]["slot_to_value"].values()) == set(OmegaConf.load(GRID).major_grid.mechanism)

    grid = OmegaConf.load(GRID)
    jobs = manifest["jobs"]
    assert {job["choices"]["basis"] for job in jobs} == set(unblind["axes"]["basis"]["slot_to_value"])
    assert {job["choices"]["mechanism"] for job in jobs} == set(unblind["axes"]["mechanism"]["slot_to_value"])
    assert {float(job["choices"]["lr"]) for job in jobs} == {float(value) for value in grid.minor_grid.lr}
    assert {job["choices"]["channels"] for job in jobs} == {int(value) for value in grid.minor_grid.channels}
    assert {job["choices"]["seed"] for job in jobs} == {int(value) for value in grid.scan_seeds}

    job = jobs[0]
    assert job["run_id"].startswith("b-")
    assert "_m-" in job["run_id"]
    assert job["minor_id"].startswith("lr-")
    assert job["minor_choices"]["channels"] == 8
    assert job["scan_seed"] in {0, 1, 2}
    assert job["seed_overrides"]["scan_train"] == {
        "run_parameters.seed": job["scan_seed"],
        "runtime.seed": job["scan_seed"],
        "sampler.seed": job["scan_seed"],
    }
    assert "study.name=pair_stability_v2" in job["overrides"]
    assert "experiment.name=pair_stability_v2" in job["overrides"]
    assert "experiment.run_name=pair_stability_v2_train" in job["overrides"]
    assert f"runtime.seed={job['scan_seed']}" in job["overrides"]
    assert f"sampler.seed={job['scan_seed']}" in job["overrides"]
    assert any(str(override).startswith("run_parameters.basis_slot=B") for override in job["overrides"])
    assert any(str(override).startswith("run_parameters.mechanism_slot=A") for override in job["overrides"])


def test_v2_validation_config_resolves_from_manifest_snapshot(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)

    resolved = validate._validation_config_from_grid(
        results_root=results_root,
        grid_attempt_id=ATTEMPT,
        requested_config=None,
    )

    assert resolved == str(results_root / "00_grid" / ATTEMPT / "validation_config.yaml")


def test_v2_collect_uses_status_for_required_train_wall_time(tmp_path: Path) -> None:
    train_attempt = tmp_path / "01_train" / "run-a" / "T1"
    train_attempt.mkdir(parents=True)
    (train_attempt / "status.json").write_text(
        json.dumps(
            {
                "start_time": "2026-06-24T10:00:00+00:00",
                "end_time": "2026-06-24T10:02:03+00:00",
            }
        )
        + "\n"
    )
    # This file is intentionally invalid. Wall time should come from
    # status.json without forcing collection to parse large train metrics.
    (train_attempt / "metrics.jsonl").write_text("{not-json}\n")

    metrics = collect._train_metrics(
        {"train_attempt_dir": str(train_attempt)},
        required_metrics={collect.TRAIN_WALL_TIME_METRIC},
    )

    assert metrics == {collect.TRAIN_WALL_TIME_METRIC: 123.0}


def test_v2_collect_prefers_grid_job_choices_over_resolved_config(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "02_validation" / "run-a" / "V1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "resolved_config.yaml").write_text("run_parameters: [not-a-mapping\n")
    (attempt_dir / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")
    (attempt_dir / "metrics.jsonl").write_text("")
    axis_metadata = {
        "major_axes": ("basis", "mechanism"),
        "minor_axes": ("lr", "channels"),
        "config_axes": ("basis", "mechanism", "lr", "channels"),
        "run_axes": ("basis", "mechanism", "lr", "channels", "seed"),
        "axis_id_labels": {"basis": "b", "mechanism": "m", "lr": "lr", "channels": "ch", "seed": "seed"},
    }
    grid_job = {
        "choices": {"basis": "B00", "mechanism": "A00", "lr": 1.0e-3, "channels": 8, "seed": 0},
        "major_id": "b-B00_m-A00",
        "minor_id": "lr-1e-3_ch-8",
        "config_id": "b-B00_m-A00_lr-1e-3_ch-8",
    }

    row = collect.collect_validation_attempt(
        "run-a",
        "V1",
        attempt_dir,
        grid_job=grid_job,
        axis_metadata=axis_metadata,
        required_train_metrics=set(),
    )

    assert row["basis"] == "B00"
    assert row["mechanism"] == "A00"
    assert row["major_id"] == "b-B00_m-A00"


def test_v2_validate_main_consumes_planned_manifest_snapshot(tmp_path: Path, monkeypatch) -> None:
    results_root = _planned_results(tmp_path)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    job = manifest["jobs"][0]
    _write_checkpoint_pointer(results_root, str(job["run_id"]), ATTEMPT)
    submitted_commands: list[list[str]] = []

    def fake_submit_local(commands: Sequence[Sequence[str]], **kwargs: Any) -> list[str]:
        submitted_commands.extend([list(command) for command in commands])
        assert len(kwargs["row_status_paths"]) == len(commands)
        assert kwargs["chunk_status_dir"] == results_root / "02_validation" / "chunk_status" / "V1"
        return [f"local-validation-{index}" for index, _ in enumerate(commands)]

    # The script under test imports a direct ``launch`` module when executed as
    # a file; bind the v2 module explicitly so this test remains isolated from
    # the legacy pair_stability test module imports.
    monkeypatch.setattr(validate, "launch", launch)
    monkeypatch.setattr(validate.launch, "submit_local", fake_submit_local)

    code = validate.main(
        [
            "--results-root",
            str(results_root),
            "--grid-attempt-id",
            ATTEMPT,
            "--train-attempt-id",
            ATTEMPT,
            "--attempt-id",
            "V1",
            "--backend",
            "local",
        ]
    )

    assert code == 0
    assert len(submitted_commands) == 1
    script = submitted_commands[0][-1]
    assert str(results_root / "00_grid" / ATTEMPT / "validation_config.yaml") in script
    assert "run_parameters.basis_slot=" in script
    assert "run_parameters.mechanism_slot=" in script
    assert "load.path=" in script
    assert "study.name=pair_stability_v2" in script

    validation_attempt = results_root / "02_validation" / str(job["run_id"]) / "V1"
    source_train = json.loads((validation_attempt / "source_train_attempt.json").read_text())
    source_grid = json.loads((validation_attempt / "source_grid_attempt.json").read_text())
    submission = json.loads((validation_attempt / "submission.json").read_text())
    assert source_train["grid_attempt_id"] == ATTEMPT
    assert source_train["train_attempt_id"] == ATTEMPT
    assert source_grid["grid_attempt_id"] == ATTEMPT
    assert submission["launcher_job_id"] == "local-validation-0"
    assert "validation_config.yaml" in submission["submitted_command"]


def _write_collection_summary(results_root: Path) -> None:
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    rows = []
    for job in manifest["jobs"]:
        point = dict(job["choices"])
        lr = float(point["lr"])
        seed = int(point["seed"])
        energy = 2.0 + (0.0 if lr == 3.0e-4 else 0.2)
        feature = 0.01 if lr == 1.0e-3 else 0.03
        rows.append(
            {
                "run_id": job["run_id"],
                "status": "completed",
                **{key: str(value) for key, value in point.items()},
                "major_id": job["major_id"],
                "minor_id": job["minor_id"],
                "config_id": job["config_id"],
                "eval/stratified_geometry/local_energy_mean": str(energy + 0.001 * seed),
                "eval/feature_trace_stability/feature_rms_q95": str(feature + 0.001 * seed),
            }
        )
    collect_dir = results_root / "03_collect" / "C1"
    _write_csv(collect_dir / "summary.csv", rows)
    (collect_dir / "source_grid_attempt.json").write_text(json.dumps({"grid_attempt_id": ATTEMPT}) + "\n")
    layout.write_latest(results_root / "03_collect", "C1")


def test_v2_collect_traces_grid_from_latest_validation_attempts(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    job = manifest["jobs"][0]
    validation_dir = results_root / "02_validation" / job["run_id"] / "V1"
    validation_dir.mkdir(parents=True)
    (validation_dir / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")
    (validation_dir / "source_grid_attempt.json").write_text(
        json.dumps(
            {
                "grid_attempt_id": ATTEMPT,
                "grid_attempt_dir": str(results_root / "00_grid" / ATTEMPT),
                "manifest_path": str(results_root / "00_grid" / ATTEMPT / "manifest.json"),
            }
        )
        + "\n"
    )
    (validation_dir / "source_train_attempt.json").write_text(
        json.dumps(
            {
                "run_id": job["run_id"],
                "grid_attempt_id": ATTEMPT,
                "train_attempt_id": ATTEMPT,
            }
        )
        + "\n"
    )
    (validation_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "namespace": "eval/stratified_geometry",
                "step": 0,
                "metrics": {"local_energy_mean": 2.0},
            }
        )
        + "\n"
    )

    result = collect.collect(results_root=results_root, collect_attempt_id="C0")
    report = result["report"]
    source = json.loads((results_root / "03_collect" / "C0" / "source_grid_attempt.json").read_text())
    latest = json.loads((results_root / "03_collect" / "latest.json").read_text())

    assert report["grid_attempt_id"] == ATTEMPT
    assert latest["attempt_id"] == "C0"
    assert source["grid_attempt_id"] == ATTEMPT
    assert source["manifest_path"].endswith("/00_grid/20260623T120000-0400/manifest.json")
    assert len(result["rows"]) == 1
    assert result["rows"][0]["basis"].startswith("B")
    assert result["rows"][0]["mechanism"].startswith("A")


def test_v2_selects_energy_champions_per_major_and_plans_nine_final_seeds_by_default(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    _write_collection_summary(results_root)

    result = select_champions.select(
        results_root=results_root,
        select_attempt_id="S1",
    )
    report = result["report"]
    latest = json.loads((results_root / "04_select" / "latest.json").read_text())
    assert report["champion_kinds"] == ["energy"]
    assert latest["attempt_id"] == "S1"
    assert [spec["selector"] for spec in report["champion_specs"]] == ["metric_ladder"]
    assert report["group_by"] == ["basis", "mechanism"]
    assert report["n_champions"] == 30

    champions = _read_csv(results_root / "04_select" / "S1" / "champions.csv")
    assert len(Counter((row["basis"], row["mechanism"]) for row in champions)) == 30
    assert set(Counter((row["basis"], row["mechanism"]) for row in champions).values()) == {1}
    assert {row["winner_kind"] for row in champions} == {"energy"}
    assert {row["minor_id"] for row in champions} == {"lr-3e-4_ch-8"}
    true_grid = OmegaConf.load(GRID)
    assert not ({row["basis"] for row in champions} & set(true_grid.major_grid.basis))
    assert not ({row["mechanism"] for row in champions} & set(true_grid.major_grid.mechanism))
    assert {row["basis"][0] for row in champions} == {"B"}
    assert {row["mechanism"][0] for row in champions} == {"A"}

    code = final_plan.main(
        [
            "--results-root",
            str(results_root),
            "--attempt-id",
            "F1",
        ]
    )
    assert code == 0

    final_dir = results_root / "05_final_grid" / "F1"
    manifest = json.loads((final_dir / "manifest.json").read_text())
    jobs = [json.loads(path.read_text()) for path in sorted((final_dir / "jobs").glob("*.json"))]
    assert manifest["study"] == "pair_stability_v2"
    assert manifest["final_replicates"] == 9
    assert manifest["n_jobs"] == 270
    assert manifest["axis_overrides"] == {
        "basis": "run_parameters.basis_slot",
        "mechanism": "run_parameters.mechanism_slot",
        "lr": "run_parameters.lr",
        "channels": "run_parameters.channels",
    }
    assert len(jobs) == 270
    assert set(Counter(job["source_champion_id"] for job in jobs).values()) == {9}
    assert {int(job["replicate_index"]) for job in jobs} == set(range(9))

    code = final_plan.main(
        [
            "--results-root",
            str(results_root),
            "--attempt-id",
            "F2",
            "--replicates",
            "1",
            "--limit-champions",
            "1",
        ]
    )
    assert code == 0
    final_job = json.loads(next((results_root / "05_final_grid" / "F2" / "jobs").glob("*.json")).read_text())
    assert final_job["basis"].startswith("B")
    assert final_job["mechanism"].startswith("A")
    assert final_job["choices"]["basis"] == final_job["basis"]
    assert final_job["choices"]["mechanism"] == final_job["mechanism"]
    assert final_job["basis"] not in set(true_grid.major_grid.basis)
    assert final_job["mechanism"] not in set(true_grid.major_grid.mechanism)


def test_v2_final_plan_rejects_zero_configured_replicates_without_override(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    _write_collection_summary(results_root)
    select_champions.select(results_root=results_root, select_attempt_id="S1")

    manifest_path = results_root / "00_grid" / ATTEMPT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["final_replicates"] = 0
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="final_replicates must be >= 1"):
        final_plan.main(
            [
                "--results-root",
                str(results_root),
                "--selection-attempt-id",
                "S1",
                "--attempt-id",
                "F0",
            ]
        )

    code = final_plan.main(
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
    assert code == 0
    planned = _read_csv(results_root / "05_final_grid" / "F1" / "final_jobs.csv")
    assert len(planned) == 1


def test_v2_final_train_rejects_empty_final_grid(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    attempt = results_root / "05_final_grid" / "F0"
    attempt.mkdir(parents=True)
    (attempt / "final_jobs.csv").write_text("final_run_id\n", encoding="utf-8")
    json_io.write_json(
        attempt / "manifest.json",
        {
            "study": "pair_stability_v2",
            "stage": layout.STAGE_FINAL_GRID,
            "attempt_id": "F0",
            "train_config": str(CONFIGS / "pair_stability.yaml"),
            "major_axes": [],
            "minor_axes": [],
            "axis_overrides": {},
        },
    )

    with pytest.raises(ValueError, match="final grid attempt F0 has no jobs"):
        final_train.main(
            [
                "--results-root",
                str(results_root),
                "--final-grid-attempt-id",
                "F0",
                "--backend",
                "local",
            ]
        )


def test_v2_final_train_excludes_completed_and_resumes_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    final_grid_id = "F0"
    final_grid_dir = results_root / "05_final_grid" / final_grid_id
    final_grid_dir.mkdir(parents=True)
    _write_csv(
        final_grid_dir / "final_jobs.csv",
        [
            {
                "final_run_id": "done",
                "source_champion_id": "champion-0",
                "final_train_model_seed": 1001,
                "final_train_sampler_seed": 101,
            },
            {
                "final_run_id": "partial",
                "source_champion_id": "champion-1",
                "final_train_model_seed": 1002,
                "final_train_sampler_seed": 102,
            },
        ],
    )
    json_io.write_json(
        final_grid_dir / "manifest.json",
        {
            "study": "pair_stability_v2",
            "stage": layout.STAGE_FINAL_GRID,
            "attempt_id": final_grid_id,
            "train_config": str(CONFIGS / "pair_stability.yaml"),
            "major_axes": [],
            "minor_axes": [],
            "axis_overrides": {},
        },
    )

    done_attempt = layout.final_train_attempt_dir(results_root, "done", final_grid_id)
    _write_final_checkpoint(results_root, "done", final_grid_id)
    (done_attempt / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")

    partial_attempt = layout.final_train_attempt_dir(results_root, "partial", final_grid_id)
    checkpoint = partial_attempt / "checkpoints" / "step_000003"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").write_text("")
    (checkpoint / "manifest.json").write_text(json.dumps({"step": 3}) + "\n")

    captured: dict[str, Any] = {}

    def fake_submit_command_sets(command_sets: dict[str, list[list[str]]], **kwargs: Any) -> list[str]:
        captured["command_sets"] = command_sets
        captured["kwargs"] = kwargs
        return ["job-0"]

    monkeypatch.setattr(final_train.launch, "submit_command_sets", fake_submit_command_sets)

    code = final_train.main(
        [
            "--results-root",
            str(results_root),
            "--final-grid-attempt-id",
            final_grid_id,
            "--backend",
            "local",
            "--device",
            "cpu",
        ]
    )

    assert code == 0
    cpu_commands = captured["command_sets"]["cpu"]
    assert len(cpu_commands) == 1
    script = cpu_commands[0][-1]
    assert "run.run_id=partial/F0" in script
    assert "run.run_id=done/F0" not in script
    assert f"load.path={checkpoint}" in script
    assert "load.mode=train_resume" in script


def test_v2_final_stage_defaults_use_latest_pointers(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    final_grid_stage = results_root / "05_final_grid"
    (final_grid_stage / "zzz").mkdir(parents=True)
    (final_grid_stage / "aaa").mkdir()
    layout.write_latest(final_grid_stage, "aaa")
    layout.write_latest(final_grid_stage, "diagnostic-final-grid", smoke=True)

    assert final_train._resolve_final_grid_attempt_id(results_root, None, smoke=False) == "aaa"
    assert final_eval._resolve_final_grid_attempt_id(results_root, None, smoke=False) == "aaa"
    assert final_train._resolve_final_grid_attempt_id(results_root, None, smoke=True) == "diagnostic-final-grid"

    final_run_id = "final-run-0"
    _write_final_checkpoint(results_root, final_run_id, "zzz")
    _write_final_checkpoint(results_root, final_run_id, "aaa")
    layout.write_latest(layout.final_train_run_dir(results_root, final_run_id), "aaa")

    assert final_eval.latest_final_train_attempt_id(results_root, final_run_id, smoke=False) == "aaa"
    assert final_eval._latest_ready_final_train_attempt_id(results_root, final_run_id, smoke=False) == "aaa"

    eval_run_dir = layout.final_eval_run_dir(results_root, final_run_id)
    (eval_run_dir / "zzz").mkdir(parents=True)
    (eval_run_dir / "aaa").mkdir()
    layout.write_latest(eval_run_dir, "aaa")

    assert final_collect._iter_final_eval_attempts(results_root, None) == [
        (final_run_id, "aaa", eval_run_dir / "aaa")
    ]


def test_v2_real_final_eval_uses_non_smoke_final_train_attempts(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    final_run_id = "final-run-0"
    final_grid_attempt_id = "FG0"
    smoke_attempt = "diagnostic-final-train"
    _write_final_checkpoint(results_root, final_run_id, final_grid_attempt_id)
    _write_final_checkpoint(results_root, final_run_id, smoke_attempt)
    layout.write_latest(layout.final_train_run_dir(results_root, final_run_id), smoke_attempt, smoke=True)

    assert final_eval.latest_final_train_attempt_id(results_root, final_run_id, smoke=False) == final_grid_attempt_id
    assert final_eval.latest_final_train_attempt_id(results_root, final_run_id, smoke=True) == smoke_attempt
    assert (
        final_eval._latest_ready_final_train_attempt_id(results_root, final_run_id, smoke=False)
        == final_grid_attempt_id
    )

    args = types.SimpleNamespace(
        smoke=False,
        final_train_attempt_id=smoke_attempt,
        allow_production_final_train=False,
    )
    with pytest.raises(ValueError, match="refuses a smoke final-train attempt"):
        final_eval._final_train_attempt_id_for_job(
            args=args,
            results_root=results_root,
            final_run_id=final_run_id,
        )
