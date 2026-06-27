"""Tests for reusable experiment toolkit primitives."""

from __future__ import annotations

from pathlib import Path

from experiments.toolkit import CompletionSpec, ResourceSpec, StagePlan, TaskSpec


def test_stage_plan_round_trip(tmp_path: Path) -> None:
    task = TaskSpec(
        task_id="train:run-a:A1",
        stage="01_train",
        attempt_id="A1",
        run_id="run-a",
        command=("python", "run.py"),
        result_dir=str(tmp_path / "run-a" / "A1"),
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
    plan = StagePlan(
        study="study",
        stage="01_train",
        attempt_id="A1",
        results_root=str(tmp_path / "results"),
        source_attempts={"grid": "G1"},
        tasks=(task,),
    )

    plan.write(tmp_path / "plan")
    restored = StagePlan.read(tmp_path / "plan")

    assert restored.study == "study"
    assert restored.n_tasks == 1
    assert restored.tasks[0].task_id == task.task_id
    assert restored.tasks[0].resources.partition == "gpu_test"
    assert restored.tasks[0].completion.policy == "status_completed_with_checkpoint"


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
