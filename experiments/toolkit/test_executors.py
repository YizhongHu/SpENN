"""Tests for executor adapters around existing launcher paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from experiments.toolkit import (
    CompletionSpec,
    ExecutorOptions,
    LocalExecutor,
    ResourceSpec,
    StagePlan,
    SubmissionRequest,
    SubmititExecutor,
    TaskSpec,
    task_id_from_parts,
)


def _task(tmp_path: Path, run_id: str, *, command: tuple[str, ...] | None = None) -> TaskSpec:
    attempt_id = "A1"
    stage = "01_train"
    return TaskSpec(
        task_id=task_id_from_parts(stage=stage, run_id=run_id, attempt_id=attempt_id),
        stage=stage,
        attempt_id=attempt_id,
        run_id=run_id,
        command=command or ("python", "run.py", run_id),
        result_dir=str(tmp_path / "results" / stage / run_id / attempt_id),
        logs=(str(tmp_path / "results" / stage / run_id / attempt_id / "launcher_status.json"),),
        resources=ResourceSpec(profile="cpu", device="cpu", threads=1),
        completion=CompletionSpec(policy="none"),
    )


def _plan(tmp_path: Path, tasks: Sequence[TaskSpec]) -> StagePlan:
    return StagePlan(
        study="study",
        stage="01_train",
        attempt_id="A1",
        results_root=str(tmp_path / "results"),
        tasks=tuple(tasks),
    )


def _options(tmp_path: Path, *, backend: str, **overrides: Any) -> ExecutorOptions:
    values = {
        "backend": backend,
        "args": SimpleNamespace(slurm_partition=None),
        "repo_root": tmp_path / "repo",
        "log_dir": tmp_path / "logs",
        "job_name": "study-train",
        "smoke": False,
        "chunk_size": 2,
    }
    values.update(overrides)
    return ExecutorOptions(**values)


def test_local_executor_submits_single_profile_and_returns_records(tmp_path: Path) -> None:
    tasks = (_task(tmp_path, "run-a"), _task(tmp_path, "run-b"))
    plan = _plan(tmp_path, tasks)
    captured: dict[str, Any] = {}

    def fake_submit(command_sets: dict[str, list[list[str]]], **kwargs: Any) -> list[str]:
        captured["command_sets"] = command_sets
        captured["kwargs"] = kwargs
        return ["local-0", "local-1"]

    request = SubmissionRequest(
        command_sets={"cpu": [task.command for task in tasks]},
        submitted_commands=[task.command for task in tasks],
    )
    records = LocalExecutor(
        submit_command_sets=fake_submit,
        options=_options(tmp_path, backend="local"),
    ).submit(plan, tasks, request)

    assert captured["command_sets"] == {
        "cpu": [["python", "run.py", "run-a"], ["python", "run.py", "run-b"]]
    }
    assert captured["kwargs"]["backend"] == "local"
    assert captured["kwargs"]["row_status_paths"] == tuple(task.logs[0] for task in tasks)
    assert captured["kwargs"]["claim_rows"] is False
    assert [record.launcher_job_id for record in records] == ["local-0", "local-1"]
    assert records[0].status_path == tasks[0].logs[0]
    assert records[0].claim_path is None


def test_submitit_executor_submits_cuda_profile_options(tmp_path: Path) -> None:
    tasks = (_task(tmp_path, "run-a"),)
    plan = _plan(tmp_path, tasks)
    captured: dict[str, Any] = {}

    def fake_submit(command_sets: dict[str, list[list[str]]], **kwargs: Any) -> list[str]:
        captured["command_sets"] = command_sets
        captured["kwargs"] = kwargs
        return ["12345"]

    request = SubmissionRequest(
        command_sets={"cuda": [["bash", "-lc", "cuda"]]},
        submitted_commands=[["bash", "-lc", "cuda"]],
    )
    records = SubmititExecutor(
        submit_command_sets=fake_submit,
        options=_options(
            tmp_path,
            backend="submitit",
            smoke=True,
            allow_partial_failures=True,
            chunk_status_dir=tmp_path / "chunks",
        ),
    ).submit(plan, tasks, request)

    assert captured["command_sets"] == {"cuda": [["bash", "-lc", "cuda"]]}
    assert captured["kwargs"]["backend"] == "submitit"
    assert captured["kwargs"]["smoke"] is True
    assert captured["kwargs"]["allow_partial_failures"] is True
    assert captured["kwargs"]["chunk_status_dir"] == tmp_path / "chunks"
    assert records[0].backend == "submitit"
    assert records[0].launcher_job_id == "12345"


def test_submitit_executor_records_mixed_profile_claim_paths(tmp_path: Path) -> None:
    tasks = (_task(tmp_path, "run-a"),)
    plan = _plan(tmp_path, tasks)
    captured: dict[str, Any] = {}

    def fake_submit(command_sets: dict[str, list[list[str]]], **kwargs: Any) -> list[str]:
        captured["command_sets"] = command_sets
        captured["kwargs"] = kwargs
        return ["cpu:111,cuda:222"]

    request = SubmissionRequest(
        command_sets={
            "cpu": [["bash", "-lc", "cpu"]],
            "cuda": [["bash", "-lc", "cuda"]],
        },
        submitted_commands=[["device-candidates", "cpu=...", "cuda=..."]],
    )
    records = SubmititExecutor(
        submit_command_sets=fake_submit,
        options=_options(tmp_path, backend="submitit"),
    ).submit(plan, tasks, request)

    assert tuple(captured["command_sets"]) == ("cpu", "cuda")
    assert captured["kwargs"]["row_status_paths"] == tuple(task.logs[0] for task in tasks)
    assert records[0].launcher_job_id == "cpu:111,cuda:222"
    assert records[0].claim_path == str(Path(tasks[0].logs[0]).with_name("launcher_claim.json"))


def test_executor_rejects_misaligned_request_lengths(tmp_path: Path) -> None:
    tasks = (_task(tmp_path, "run-a"),)
    plan = _plan(tmp_path, tasks)

    def fake_submit(command_sets: dict[str, list[list[str]]], **kwargs: Any) -> list[str]:
        raise AssertionError("submitter should not be called")

    request = SubmissionRequest(
        command_sets={"cpu": [["python", "run.py"], ["python", "run.py"]]},
        submitted_commands=[["python", "run.py"]],
    )

    with pytest.raises(ValueError, match="has 2 commands for 1 tasks"):
        LocalExecutor(
            submit_command_sets=fake_submit,
            options=_options(tmp_path, backend="local"),
        ).submit(plan, tasks, request)
