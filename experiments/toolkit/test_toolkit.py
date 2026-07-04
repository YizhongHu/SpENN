"""Tests for reusable experiment toolkit primitives."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.toolkit import (
    CompletionSpec,
    ExecutionRecord,
    ResourceSpec,
    StagePlan,
    TaskSpec,
    execution_records_from_submission,
    task_id_from_parts,
    write_execution_records,
)


def _task(
    tmp_path: Path,
    *,
    stage: str = "01_train",
    attempt_id: str = "A1",
    run_id: str = "run-a",
    task_id: str | None = None,
    command: tuple[str, ...] = ("python", "run.py"),
    resources: ResourceSpec | None = None,
    completion: CompletionSpec | None = None,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id or task_id_from_parts(stage=stage, run_id=run_id, attempt_id=attempt_id),
        stage=stage,
        attempt_id=attempt_id,
        run_id=run_id,
        command=command,
        result_dir=str(tmp_path / run_id / attempt_id),
        logs=(str(tmp_path / run_id / attempt_id / "launcher_status.json"),),
        resources=resources or ResourceSpec(profile="cpu", device="cpu", threads=1),
        completion=completion or CompletionSpec(policy="none"),
    )


def _plan(tmp_path: Path, *tasks: TaskSpec, **overrides: object) -> StagePlan:
    if not tasks:
        tasks = (_task(tmp_path),)
    values = {
        "study": "study",
        "stage": "01_train",
        "attempt_id": "A1",
        "results_root": str(tmp_path / "results"),
        "source_attempts": {"grid": "G1"},
        "tasks": tasks,
    }
    values.update(overrides)
    return StagePlan(**values)


def test_stage_plan_round_trip(tmp_path: Path) -> None:
    task = _task(
        tmp_path,
        resources=ResourceSpec(
            profile="cuda",
            device="cuda",
            partition="gpu_test",
            threads=4,
            mem_gb=16,
            gpus=1,
            timeout_min=15,
            uv_environment=".venv-gpu",
            uv_extras=("cu126",),
        ),
        completion=CompletionSpec(
            policy="status_completed_with_checkpoint",
            status_path=str(tmp_path / "run-a" / "A1" / "status.json"),
            checkpoint_path=str(tmp_path / "run-a" / "A1" / "checkpoints" / "latest.json"),
        ),
    )
    plan = _plan(tmp_path, task)

    plan.write(tmp_path / "plan")
    restored = StagePlan.read(tmp_path / "plan")

    assert restored.study == "study"
    assert restored.n_tasks == 1
    assert restored.tasks[0].task_id == task.task_id
    assert restored.tasks[0].resources.partition == "gpu_test"
    assert restored.tasks[0].completion.policy == "status_completed_with_checkpoint"


def test_stage_plan_read_rejects_corrupt_task_count(tmp_path: Path) -> None:
    plan_dir = _plan(tmp_path).write(tmp_path / "plan")
    manifest_path = plan_dir / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["n_tasks"] = 2
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="n_tasks"):
        StagePlan.read(plan_dir)


def test_stage_plan_read_rejects_incomplete_task_rows(tmp_path: Path) -> None:
    plan_dir = _plan(tmp_path).write(tmp_path / "plan")
    tasks_path = plan_dir / "tasks.jsonl"
    task = json.loads(tasks_path.read_text().splitlines()[0])
    del task["command"]
    tasks_path.write_text(json.dumps(task) + "\n")

    with pytest.raises(ValueError, match="command"):
        StagePlan.read(plan_dir)


def test_stage_plan_read_rejects_scalar_sequence_fields(tmp_path: Path) -> None:
    plan_dir = _plan(tmp_path).write(tmp_path / "plan")
    tasks_path = plan_dir / "tasks.jsonl"
    task = json.loads(tasks_path.read_text().splitlines()[0])
    task["command"] = "python run.py"
    tasks_path.write_text(json.dumps(task) + "\n")

    with pytest.raises(ValueError, match="command must be a sequence"):
        StagePlan.read(plan_dir)


def test_stage_plan_rejects_schema_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _plan(tmp_path, schema_version="experiment-toolkit/v0").write(tmp_path / "plan")


def test_stage_plan_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    task_a = _task(tmp_path, run_id="run-a")
    task_b = _task(tmp_path, run_id="run-a")

    with pytest.raises(ValueError, match="duplicate task_id"):
        _plan(tmp_path, task_a, task_b).write(tmp_path / "plan")


def test_stage_plan_rejects_task_stage_or_attempt_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match plan stage"):
        _plan(tmp_path, _task(tmp_path, stage="02_validation")).write(tmp_path / "plan-stage")

    with pytest.raises(ValueError, match="does not match plan attempt_id"):
        _plan(tmp_path, _task(tmp_path, attempt_id="B1")).write(tmp_path / "plan-attempt")


def test_task_spec_rejects_missing_required_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="command"):
        _task(tmp_path, command=()).validate()

    with pytest.raises(ValueError, match="deterministic id"):
        _task(tmp_path, task_id="custom").validate()

    with pytest.raises(ValueError, match="result_dir"):
        replace(_task(tmp_path), result_dir="").validate()


def test_resource_spec_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="profile"):
        ResourceSpec(profile="", device="cpu").validate()

    with pytest.raises(ValueError, match="threads"):
        ResourceSpec(profile="cpu", device="cpu", threads=0).validate()

    with pytest.raises(ValueError, match="gpus"):
        ResourceSpec.from_dict({"profile": "cuda", "device": "cuda", "gpus": -1})

    with pytest.raises(ValueError, match="uv_extras must be a sequence"):
        ResourceSpec.from_dict({"profile": "cpu", "device": "cpu", "uv_extras": "cpu"})


def test_completion_spec_rejects_invalid_policies() -> None:
    with pytest.raises(ValueError, match="unknown completion policy"):
        CompletionSpec(policy="done").validate()

    with pytest.raises(ValueError, match="requires checkpoint_path"):
        CompletionSpec(policy="checkpoint_exists").validate()


def test_execution_record_round_trip_and_validation(tmp_path: Path) -> None:
    task = _task(tmp_path)
    records = execution_records_from_submission(
        tasks=(task,),
        backend="local",
        job_ids=("local-0",),
        submitted_commands=(("python", "run.py"),),
        claim_paths=(tmp_path / "claim.json",),
    )

    path = write_execution_records(tmp_path / "plan", records)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    restored = ExecutionRecord.from_dict(rows[0])

    assert restored.task_id == task.task_id
    assert restored.claim_path == str(tmp_path / "claim.json")
    assert restored.submitted_command == ("python", "run.py")

    with pytest.raises(ValueError, match="launcher_job_id"):
        replace(restored, launcher_job_id="").validate()

    row = rows[0]
    row["submitted_command"] = "python run.py"
    with pytest.raises(ValueError, match="submitted_command must be a sequence"):
        ExecutionRecord.from_dict(row)


def test_completion_specs_check_files(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    checkpoint = tmp_path / "checkpoints" / "latest.json"
    completion = CompletionSpec(
        policy="status_completed_with_checkpoint",
        status_path=str(status),
        checkpoint_path=str(checkpoint),
    )

    assert completion.is_complete() is False
    status.write_text('{"status": "completed"}\n')
    assert completion.is_complete() is False
    checkpoint.parent.mkdir()
    checkpoint.write_text("{}\n")
    assert completion.is_complete() is True
