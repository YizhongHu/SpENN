"""Cluster-free tests for the baselines Parsl dispatch seam."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.baselines import pipeline
from experiments.toolkit.dispatch import DispatchRecord, LogicalTaskSpec, StagePlanV2
from experiments.toolkit.resources import ResourceSpec
from experiments.toolkit.specs import CompletionSpec


class RecordingExecutor:
    def __init__(self) -> None:
        self.dispatches = ()
        self.context = None

    def dispatch(self, dispatches: Any, *, context: Any) -> tuple[DispatchRecord, ...]:
        self.dispatches = tuple(dispatches)
        self.context = context
        return tuple(
            DispatchRecord.accepted(
                dispatch,
                backend="fake",
                launcher_job_id=context.allocation_id,
                submitted_command=dispatch.argv,
            )
            for dispatch in self.dispatches
        )


def _plan(tmp_path: Path, gpu_counts: tuple[int, ...]) -> StagePlanV2:
    tasks = tuple(
        LogicalTaskSpec(
            logical_task_id=f"row-{index}",
            stage="01_baseline",
            run_id=f"run-{index}",
            command=("{python}", "-c", "print('batch=4096')"),
            result_dir=str(tmp_path / f"result-{index}"),
            logs=(str(tmp_path / f"result-{index}" / "status.json"),),
            params={"total_batch_size": 4096},
            resources=ResourceSpec(profile="cuda", device="cuda", gpus=gpus),
            completion=CompletionSpec(policy="none"),
            metadata={"runtime": "ferminet"},
        )
        for index, gpus in enumerate(gpu_counts)
    )
    return StagePlanV2(
        study="baselines",
        stage="01_baseline",
        plan_id="plan-1",
        results_root=str(tmp_path / "results"),
        tasks=tasks,
    ).validate()


def test_one_plan_dispatches_uniform_one_gpu_rows_and_writes_records(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    code = pipeline.run_pipeline(
        plan=_plan(tmp_path, (1, 1)),
        facility="polaris",
        launch_dir=tmp_path / "launch",
        admission_id="admission-1",
        executor=executor,
        environ={"PBS_JOBID": "123.server"},
        cwd=tmp_path,
        python="/env/bin/python",
    )

    assert code == 0
    assert [row.environment for row in executor.dispatches] == [{}, {}]
    assert executor.context.visibility_values == ("0", "1", "2", "3")
    assert [row.argv for row in executor.dispatches] == [
        ("/env/bin/python", "-c", "print('batch=4096')")
    ] * 2
    records = (tmp_path / "launch" / "dispatch_records.jsonl").read_text().splitlines()
    assert len(records) == 2
    assert json.loads((tmp_path / "launch" / "verification.json").read_text())["complete"] is True


def test_four_gpu_rows_use_one_parsl_worker_each(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    assert pipeline.run_pipeline(
        plan=_plan(tmp_path, (4, 4, 4)),
        facility="polaris",
        launch_dir=tmp_path / "launch",
        admission_id="admission-2",
        executor=executor,
        environ={"PBS_JOBID": "456.server"},
    ) == 0
    assert executor.context.visibility_values == ("0,1,2,3",)


def test_mixed_gpu_plan_fails_before_dispatch(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    bad = _plan(tmp_path, (1, 4))
    with pytest.raises(ValueError, match="uniform resources.gpus"):
        pipeline.run_pipeline(
            plan=bad,
            facility="polaris",
            launch_dir=tmp_path / "launch",
            admission_id="admission-3",
            executor=executor,
            environ={"PBS_JOBID": "789.server"},
        )
    assert executor.dispatches == ()


def test_cannon_is_not_a_parsl_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only on Polaris"):
        pipeline.allocation_context(
            facility="cannon",
            gpus_per_row=1,
            run_root=tmp_path / "launch",
            environ={"SLURM_JOB_ID": "1"},
        )
