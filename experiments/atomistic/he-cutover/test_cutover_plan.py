from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import cutover_plan


GRID = Path(__file__).with_name("smoke_grid.yaml")


def test_strict_grid_rejects_unknown_key(tmp_path: Path) -> None:
    text = GRID.read_text(encoding="utf-8") + "unknown: true\n"
    path = tmp_path / "grid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(cutover_plan.PlanError, match="keys mismatch"):
        cutover_plan.load_grid(path)


@pytest.mark.parametrize("facility", ["cannon", "polaris"])
def test_plan_has_one_train_two_eval_and_exact_completion_specs(tmp_path: Path, facility: str) -> None:
    train, evaluation, manifest = cutover_plan.build_plans(cutover_plan.load_grid(GRID), facility=facility, results_root=tmp_path, plan_id="plan-1")
    assert [task.run_id for task in train.tasks] == ["seed-000"]
    assert [task.run_id for task in evaluation.tasks] == ["seed-000-chain-00", "seed-000-chain-01"]
    assert train.tasks[0].completion.policy == "status_completed_with_checkpoint"
    assert train.tasks[0].completion.checkpoint_path.endswith("step_000025/COMPLETE")
    assert {task.completion.policy for task in evaluation.tasks} == {"status_completed"}
    assert all(task.dependencies == (train.tasks[0].logical_task_id,) for task in evaluation.tasks)
    assert len(manifest["rows"]) == 3
