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

from omegaconf import OmegaConf

STUDY_DIR = Path(__file__).resolve().parent
CONFIGS = STUDY_DIR / "configs"
GRID = CONFIGS / "grid.yaml"

if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))


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


run_utils = _load_script("run_utils", bind_direct=True)
launch = _load_script("launch")
plan = _load_script("plan")
collect = _load_script("collect")
select_champions = _load_script("select_champions")
final_plan = _load_script("final_plan")
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
    checkpoint_dir = run_utils.train_attempt_dir(results_root, run_id, attempt_id) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "latest.json").write_text(json.dumps({"path": "step_000000"}))
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
    assert smoke_cpu["slurm_array_parallelism"] == 2
    assert smoke_cuda["slurm_array_parallelism"] == 2
    assert smoke_cuda["gpus_per_node"] == 1


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
    assert "UV_PROJECT_ENVIRONMENT=.venv" in script
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
    grid_attempt = run_utils.grid_attempt_dir(results_root, ATTEMPT)
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
    assert manifest["final_replicates"] == 0
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

    assert report["grid_attempt_id"] == ATTEMPT
    assert source["grid_attempt_id"] == ATTEMPT
    assert source["manifest_path"].endswith("/00_grid/20260623T120000-0400/manifest.json")
    assert len(result["rows"]) == 1
    assert result["rows"][0]["basis"].startswith("B")
    assert result["rows"][0]["mechanism"].startswith("A")


def test_v2_selects_energy_champions_per_major_and_skips_final_jobs_by_default(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    _write_collection_summary(results_root)

    result = select_champions.select(
        results_root=results_root,
        collection_attempt_id="C1",
        select_attempt_id="S1",
    )
    report = result["report"]
    assert report["champion_kinds"] == ["energy"]
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
            "--selection-attempt-id",
            "S1",
            "--attempt-id",
            "F1",
        ]
    )
    assert code == 0

    final_dir = results_root / "05_final_grid" / "F1"
    manifest = json.loads((final_dir / "manifest.json").read_text())
    jobs = [json.loads(path.read_text()) for path in sorted((final_dir / "jobs").glob("*.json"))]
    assert manifest["study"] == "pair_stability_v2"
    assert manifest["final_replicates"] == 0
    assert manifest["n_jobs"] == 0
    assert manifest["axis_overrides"] == {
        "basis": "run_parameters.basis_slot",
        "mechanism": "run_parameters.mechanism_slot",
        "lr": "run_parameters.lr",
        "channels": "run_parameters.channels",
    }
    assert jobs == []

    code = final_plan.main(
        [
            "--results-root",
            str(results_root),
            "--selection-attempt-id",
            "S1",
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
