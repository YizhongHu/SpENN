"""Resolve logical He-cutover tasks into immutable dispatch specifications."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from experiments.toolkit.dispatch import DispatchSpec, LogicalTaskSpec, StagePlanV2, admit_tasks
from experiments.toolkit.jsonio import write_jsonl


def _argv(task: LogicalTaskSpec, python: str) -> tuple[str, ...]:
    return tuple(python if token == "{python}" else token for token in task.command)


def admit_plan(plan: StagePlanV2, *, admission_id: str, cwd: str | Path, environment: Mapping[str, str], python: str | None = None) -> tuple[DispatchSpec, ...]:
    """Admit every task for its declared runtime and reject visibility injection."""

    if "CUDA_VISIBLE_DEVICES" in environment:
        raise ValueError("CUDA_VISIBLE_DEVICES belongs to allocation binding, not DispatchSpec.environment")
    executable = str(Path(python or sys.executable).resolve())
    runtimes = {str(task.metadata["runtime"]) for task in plan.tasks}
    argv_by_runtime: dict[str, list[tuple[str, ...]]] = {}
    for runtime in sorted(runtimes):
        selected = [task for task in plan.tasks if task.metadata["runtime"] == runtime]
        if len(selected) != len(plan.tasks):
            raise ValueError("one StagePlanV2 must align to one runtime")
        argv_by_runtime[runtime] = [_argv(task, executable) for task in plan.tasks]
    dispatches = admit_tasks(plan, plan.tasks, admission_id=admission_id, argv_by_runtime=argv_by_runtime, cwd=cwd, environment=environment, admission_ref=plan.plan_id)
    if any("CUDA_VISIBLE_DEVICES" in dispatch.environment for dispatch in dispatches):
        raise ValueError("admission created forbidden CUDA_VISIBLE_DEVICES environment")
    return dispatches


def write_dispatch_specs(path: str | Path, dispatches: Sequence[DispatchSpec]) -> Path:
    """Write the exact admitted rows as JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_jsonl(path, (dispatch.to_dict() for dispatch in dispatches))
