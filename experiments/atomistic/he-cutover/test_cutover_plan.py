from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import cutover_plan
import run_eval_row
import run_train_row


GRID = Path(__file__).with_name("smoke_grid.yaml")


def _override_value(overrides: list[str], key: str) -> str:
    prefix = f"{key}="
    values = [override.removeprefix(prefix) for override in overrides if override.startswith(prefix)]
    assert len(values) == 1
    return values[0]


def _assert_runner_matches_plan(row: dict[str, object], overrides: list[str]) -> None:
    assert _override_value(overrides, "run.layout") == "flat"
    instructed_dir = Path(_override_value(overrides, "run.root")) / _override_value(overrides, "run.run_id")
    assert str(instructed_dir) == str(row["result_dir"])


def test_strict_grid_rejects_unknown_key(tmp_path: Path) -> None:
    text = GRID.read_text(encoding="utf-8") + "unknown: true\n"
    path = tmp_path / "grid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(cutover_plan.PlanError, match="keys mismatch"):
        cutover_plan.load_grid(path)


def test_strict_grid_rejects_invalid_facility_placement(tmp_path: Path) -> None:
    payload = cutover_plan.load_grid(GRID)
    payload["facilities"]["polaris"]["partition"] = "not-debug"
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="debug/a100_40gb"):
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


def test_plan_writer_emits_rows_and_both_v2_stage_tables(tmp_path: Path) -> None:
    plans = cutover_plan.build_plans(cutover_plan.load_grid(GRID), facility="cannon", results_root=tmp_path / "results", plan_id="plan-1")
    output = cutover_plan.write_plans(tmp_path / "plan", *plans)
    assert (output / "rows.csv").read_text().splitlines()[0] == "stage,kind,row_id,facility,runtime,result_dir,checkpoint_dir"
    assert (output / "02_train/tasks.jsonl").is_file()
    assert (output / "03_eval/tasks.jsonl").is_file()


def test_train_and_eval_runner_paths_are_identical_to_the_plan(tmp_path: Path) -> None:
    train, evaluation, manifest = cutover_plan.build_plans(
        cutover_plan.load_grid(GRID), facility="cannon", results_root=tmp_path / "results", plan_id="plan-1"
    )
    train_row, eval_row = manifest["rows"][0], manifest["rows"][1]

    _assert_runner_matches_plan(train_row, run_train_row.output_overrides(train_row))
    assert train.tasks[0].completion.status_path == str(Path(str(train_row["result_dir"])) / "status.json")
    assert train.tasks[0].completion.checkpoint_path == str(Path(str(train_row["checkpoint_dir"])) / "COMPLETE")
    assert Path(str(train_row["checkpoint_dir"])).is_relative_to(Path(str(train_row["result_dir"])))

    _assert_runner_matches_plan(eval_row, run_eval_row.output_overrides(eval_row))
    assert evaluation.tasks[0].completion.status_path == str(Path(str(eval_row["result_dir"])) / "status.json")
    assert evaluation.tasks[0].completion.checkpoint_path is None
    assert eval_row["checkpoint_dir"] == train_row["checkpoint_dir"]
