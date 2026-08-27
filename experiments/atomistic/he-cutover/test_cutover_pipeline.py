from __future__ import annotations

import json
from pathlib import Path

import pytest

import cutover_plan
import pipeline
from experiments.toolkit.dispatch import DispatchRecord


class RecordingExecutor:
    def __init__(self, fail_stage=None):
        self.stages = []
        self.fail_stage = fail_stage

    def dispatch(self, dispatches, *, context):
        stage = dispatches[0].stage
        self.stages.append(stage)
        if stage == self.fail_stage:
            raise RuntimeError("injected failure")
        records = []
        for dispatch in dispatches:
            if stage == "02_train":
                status = Path(dispatch.completion.status_path)
                status.parent.mkdir(parents=True, exist_ok=True)
                status.write_text('{"status":"completed"}\n')
                checkpoint = Path(dispatch.completion.checkpoint_path)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text("complete\n")
                checkpoint_dir = checkpoint.parent
                (checkpoint_dir / "manifest.json").write_text("{}\n")
            elif stage == "03_eval":
                status = Path(dispatch.completion.status_path)
                status.parent.mkdir(parents=True, exist_ok=True)
                status.write_text('{"status":"completed"}\n')
            records.append(DispatchRecord.accepted(dispatch, backend="fake", launcher_job_id=context.allocation_id, submitted_command=dispatch.argv))
        return tuple(records)


def _plans(tmp_path):
    grid = cutover_plan.load_grid(Path(__file__).with_name("smoke_grid.yaml"))
    return cutover_plan.build_plans(grid, facility="cannon", results_root=tmp_path / "results", plan_id="p")[:2]


def test_facility_binding_branches_are_exercised_end_to_end(tmp_path: Path) -> None:
    cannon = pipeline.allocation_context(facility="cannon", run_root=tmp_path / "cannon", environ={"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "MIG-owned-by-slurm"})
    polaris = pipeline.allocation_context(facility="polaris", run_root=tmp_path / "polaris", environ={"PBS_JOBID": "2.server"})
    assert cannon.visibility_values == ()
    assert polaris.visibility_values == ("0", "1", "2", "3")


@pytest.mark.parametrize(
    ("facility", "scheduler_env"),
    [("cannon", {"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "MIG-owned-by-slurm"}), ("polaris", {"PBS_JOBID": "2.server"})],
)
def test_pipeline_orders_probe_train_barrier_eval_and_verification_matches_exit(tmp_path: Path, facility: str, scheduler_env: dict[str, str]) -> None:
    grid = cutover_plan.load_grid(Path(__file__).with_name("smoke_grid.yaml"))
    train, evaluation = cutover_plan.build_plans(grid, facility=facility, results_root=tmp_path / "results", plan_id="p")[:2]
    executor = RecordingExecutor()
    code = pipeline.run_pipeline(train_plan=train, eval_plan=evaluation, facility=facility, launch_dir=tmp_path / "launch", admission_id="a", executor=executor, environ=scheduler_env)
    verification = json.loads((tmp_path / "launch/verification.json").read_text())
    assert executor.stages == ["01_preflight", "02_train", "03_eval"]
    assert (code, verification["exit_code"], verification["complete"]) == (0, 0, True)
    assert (tmp_path / "launch/02_train/dispatch_specs.jsonl").is_file()
    assert (tmp_path / "launch/03_eval/dispatch_specs.jsonl").is_file()


def test_preflight_failure_prevents_all_science_and_is_truthful(tmp_path: Path) -> None:
    train, evaluation = _plans(tmp_path)
    executor = RecordingExecutor(fail_stage="01_preflight")
    code = pipeline.run_pipeline(train_plan=train, eval_plan=evaluation, facility="cannon", launch_dir=tmp_path / "launch", admission_id="a", executor=executor, environ={"SLURM_JOB_ID": "1"})
    verification = json.loads((tmp_path / "launch/verification.json").read_text())
    assert executor.stages == ["01_preflight"]
    assert (code, verification["exit_code"], verification["complete"]) == (1, 1, False)
