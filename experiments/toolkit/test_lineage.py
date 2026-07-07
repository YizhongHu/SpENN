"""Tests for task-lineage sidecars."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.toolkit import (
    CompletionSpec,
    ResourceSpec,
    StagePlan,
    TaskLineageRow,
    TaskSpec,
    read_task_lineage,
    stage_plan_task_ids,
    synthesized_task_id,
    task_id_from_parts,
    write_task_lineage,
)


def _stage_plan(tmp_path: Path, *, stage: str, attempt_id: str, run_ids: list[str]) -> StagePlan:
    tasks = tuple(
        TaskSpec(
            task_id=task_id_from_parts(stage=stage, run_id=run_id, attempt_id=attempt_id),
            stage=stage,
            attempt_id=attempt_id,
            run_id=run_id,
            command=("python", "run.py"),
            result_dir=str(tmp_path / run_id / attempt_id),
            resources=ResourceSpec(profile="cpu", device="cpu"),
            completion=CompletionSpec(policy="none"),
        )
        for run_id in run_ids
    )
    return StagePlan(study="study", stage=stage, attempt_id=attempt_id, results_root=str(tmp_path), tasks=tasks)


def test_write_and_read_task_lineage_roundtrip(tmp_path: Path) -> None:
    rows = [
        TaskLineageRow(row_id="run-a", task_ids={"validation": "02_validation:run-a:A1"}),
        TaskLineageRow(row_id="run-b", task_ids={"validation": "02_validation:run-b:A1", "train": "01_train:run-b:T1"}),
    ]
    write_task_lineage(tmp_path, rows)
    lineage = read_task_lineage(tmp_path)
    assert set(lineage) == {"run-a", "run-b"}
    assert lineage["run-b"].task_ids == {"validation": "02_validation:run-b:A1", "train": "01_train:run-b:T1"}


def test_read_task_lineage_missing_sidecar_returns_empty(tmp_path: Path) -> None:
    assert read_task_lineage(tmp_path) == {}


def test_stage_plan_task_ids_missing_directory_returns_none(tmp_path: Path) -> None:
    assert stage_plan_task_ids(tmp_path / "does-not-exist") is None


def test_stage_plan_task_ids_reads_real_plan(tmp_path: Path) -> None:
    plan = _stage_plan(tmp_path, stage="02_validation", attempt_id="A1", run_ids=["run-a", "run-b"])
    plan_dir = tmp_path / "stage_plans" / "A1"
    plan.write(plan_dir)
    assert stage_plan_task_ids(plan_dir) == {
        "02_validation:run-a:A1",
        "02_validation:run-b:A1",
    }


def test_synthesized_task_id_matches_deterministic_formula() -> None:
    task_id = synthesized_task_id(stage="02_validation", run_id="run-a", attempt_id="A1", known_task_ids=None)
    assert task_id == "02_validation:run-a:A1"


def test_synthesized_task_id_verifies_against_known_task_ids() -> None:
    known = frozenset({"02_validation:run-a:A1"})
    assert synthesized_task_id(stage="02_validation", run_id="run-a", attempt_id="A1", known_task_ids=known) == (
        "02_validation:run-a:A1"
    )
    with pytest.raises(ValueError, match="not a known task id"):
        synthesized_task_id(stage="02_validation", run_id="run-b", attempt_id="A1", known_task_ids=known)
