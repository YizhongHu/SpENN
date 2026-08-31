"""Laptop-only contract tests for the baseline plan builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.baselines.plan import build_plan
from experiments.toolkit.specs import CompletionSpec


MATRIX = {
    "code": ("ferminet", "deepqmc"),
    "ansatz": ("default",),
    "system": ("he", "li"),
    "seed": (0, 7),
    "steps": (100,),
    "batch_size": (32,),
}


def test_matrix_cartesian_product_and_injected_argv(tmp_path: Path) -> None:
    plan = build_plan(
        MATRIX,
        command=("python", "-m", "baseline_runner", "--literal", "line\\ntext"),
        results_root=tmp_path,
        plan_id="plan-1",
    )

    assert plan.n_tasks == 8
    assert all(task.command == ("python", "-m", "baseline_runner", "--literal", "line\\ntext") for task in plan.tasks)
    assert all(task.resources.gpus == 1 and task.resources.device == "cuda" for task in plan.tasks)


def test_ids_are_deterministic_and_result_dirs_pairwise_distinct(tmp_path: Path) -> None:
    first = build_plan(MATRIX, command=("runner",), results_root=tmp_path, plan_id="plan-1")
    second = build_plan(MATRIX, command=("runner",), results_root=tmp_path, plan_id="plan-1")

    assert [task.logical_task_id for task in first.tasks] == [task.logical_task_id for task in second.tasks]
    result_dirs = [task.result_dir for task in first.tasks]
    assert len(result_dirs) == len(set(result_dirs))


def test_completion_spec_skips_finished_row(tmp_path: Path) -> None:
    plan = build_plan(MATRIX, command=("runner",), results_root=tmp_path, plan_id="plan-1")
    complete = plan.tasks[0]
    unfinished = plan.tasks[1]
    Path(complete.completion.status_path).parent.mkdir(parents=True)
    Path(complete.completion.status_path).write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    assert isinstance(complete.completion, CompletionSpec)
    assert complete.completion.is_complete()
    assert not unfinished.completion.is_complete()


def test_empty_dimension_is_an_empty_plan(tmp_path: Path) -> None:
    empty = {**MATRIX, "system": ()}
    plan = build_plan(empty, command=("runner",), results_root=tmp_path, plan_id="plan-1")
    assert plan.tasks == ()


@pytest.mark.parametrize("field", ("code", "ansatz", "system", "seed", "steps", "batch_size"))
def test_none_or_duplicate_dimension_is_rejected(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        build_plan({**MATRIX, field: (None,)}, command=("runner",), results_root=tmp_path, plan_id="plan-1")
    with pytest.raises(ValueError, match=field):
        build_plan({**MATRIX, field: (0, 0)}, command=("runner",), results_root=tmp_path, plan_id="plan-1")


def test_duplicate_system_seed_rows_are_not_collapsed(tmp_path: Path) -> None:
    matrix = {**MATRIX, "code": ("ferminet", "deepqmc"), "system": ("he",), "seed": (0,)}
    plan = build_plan(matrix, command=("runner",), results_root=tmp_path, plan_id="plan-1")
    assert plan.n_tasks == 2
    assert len({task.result_dir for task in plan.tasks}) == 2


def test_cpu_resource_is_rejected(tmp_path: Path) -> None:
    from experiments.toolkit.resources import ResourceSpec

    with pytest.raises(ValueError, match="CUDA"):
        build_plan(MATRIX, command=("runner",), results_root=tmp_path, plan_id="plan-1", resources=ResourceSpec(profile="cpu", device="cpu"))
