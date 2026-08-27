from __future__ import annotations

import json
from pathlib import Path

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
            elif stage == "03_eval":
                status = Path(dispatch.completion.status_path)
                status.parent.mkdir(parents=True, exist_ok=True)
                status.write_text('{"status":"completed"}\n')
            records.append(DispatchRecord.accepted(dispatch, backend="fake", launcher_job_id=context.allocation_id, submitted_command=dispatch.argv))
        return tuple(records)


def _plans(tmp_path):
    grid = cutover_plan.load_grid(Path(__file__).with_name("smoke_grid.yaml"))
    return cutover_plan.build_plans(grid, facility="cannon", results_root=tmp_path / "results", plan_id="p")[:2]


def test_pipeline_orders_probe_train_barrier_eval_and_verification_matches_exit(tmp_path: Path) -> None:
    train, evaluation = _plans(tmp_path)
    executor = RecordingExecutor()
    code = pipeline.run_pipeline(train_plan=train, eval_plan=evaluation, facility="cannon", launch_dir=tmp_path / "launch", admission_id="a", executor=executor, environ={"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "MIG-1"})
    verification = json.loads((tmp_path / "launch/verification.json").read_text())
    assert executor.stages == ["01_preflight", "02_train", "03_eval"]
    assert (code, verification["exit_code"], verification["complete"]) == (0, 0, True)


def test_preflight_failure_prevents_all_science_and_is_truthful(tmp_path: Path) -> None:
    train, evaluation = _plans(tmp_path)
    executor = RecordingExecutor(fail_stage="01_preflight")
    code = pipeline.run_pipeline(train_plan=train, eval_plan=evaluation, facility="cannon", launch_dir=tmp_path / "launch", admission_id="a", executor=executor, environ={"SLURM_JOB_ID": "1"})
    verification = json.loads((tmp_path / "launch/verification.json").read_text())
    assert executor.stages == ["01_preflight"]
    assert (code, verification["exit_code"], verification["complete"]) == (1, 1, False)
