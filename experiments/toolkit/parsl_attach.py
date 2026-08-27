"""Parsl executor that attaches work to an existing allocation.

Parsl is deliberately imported only while dispatching.  The module therefore
remains importable in planning and unit-test environments that do not install
the optional ``parsl`` extra.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .dispatch import AllocationContext, DispatchRecord, DispatchSpec
from .task_state import _deadline_guard_reached


def _run_dispatch_payload(
    argv: tuple[str, ...],
    cwd: str,
    environment: Mapping[str, str],
    output_directory: str,
    attempt_id: str,
    visibility_variable: str | None = None,
    visibility_value: str | None = None,
) -> dict[str, Any]:
    """Run one immutable argv and write only stdlib worker evidence.

    This function is intentionally self-contained: Parsl serializes it by
    value, and a worker must never import this package to execute a row.
    """

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    stdout_path = output_path / "stdout.log"
    stderr_path = output_path / "stderr.log"
    status_path = output_path / "attempt_status.json"
    worker_environment = os.environ | {str(key): str(value) for key, value in environment.items()}
    if visibility_variable is not None and visibility_value is not None:
        worker_environment[visibility_variable] = visibility_value
    started_at = time.time()
    launch_error: str | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=worker_environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
            returncode: int | None = int(completed.returncode)
        except OSError as exc:
            launch_error = repr(exc)
            stderr.write(launch_error + "\n")
            returncode = None
    ended_at = time.time()
    status: dict[str, Any] = {
        "status": "success" if returncode == 0 else "failed",
        "attempt_id": attempt_id,
        "argv": list(argv),
        "cwd": cwd,
        "returncode": returncode,
        "visibility_variable": visibility_variable,
        "visibility_value": visibility_value,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "started_at_unix": started_at,
        "ended_at_unix": ended_at,
        "elapsed_sec": ended_at - started_at,
    }
    if launch_error is not None:
        status["error"] = launch_error
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return {**status, "attempt_status_path": str(status_path)}


AppRunner = Callable[..., Any]


@dataclass(frozen=True)
class ParslAttachExecutor:
    """Dispatch a ready batch concurrently through Parsl HTEX.

    ``app_runner`` is an intentional unit-test seam.  It receives the same
    keyword arguments as the Parsl application and may return either a payload
    mapping or an object with a blocking ``result()`` method.
    """

    app_runner: AppRunner | None = None

    def dispatch(
        self,
        dispatches: Sequence[DispatchSpec],
        *,
        context: AllocationContext,
    ) -> tuple[DispatchRecord, ...]:
        """Run the ready batch and return its acceptance records.

        The caller owns ordering and retries.  This method only starts the
        supplied batch, waits for it, and verifies its declared completion.
        """

        # C1's general allocation validator requires an assigned visibility
        # value, but Cannon inherit mode deliberately has none: Slurm owns the
        # variable and workers must leave it untouched.
        if context.visibility_values:
            context.validate()
        elif not context.allocation_id or not context.visibility_variable:
            raise ValueError("inherit-mode allocation context requires allocation and visibility names")
        ready_dispatches = tuple(dispatch.validate() for dispatch in dispatches)
        if not ready_dispatches:
            return ()
        deadline_unix = context.deadline_unix()
        if _deadline_guard_reached(deadline_unix, context.deadline_guard_min):
            raise RuntimeError("allocation deadline guard reached; refusing new Parsl dispatches")
        if context.run_root is None:
            raise ValueError("ParslAttachExecutor requires context.run_root as the launch attempt directory")

        launch_attempt_dir = Path(context.run_root)
        runner = self.app_runner or _parsl_app_runner(context, launch_attempt_dir)
        submitted: list[tuple[DispatchSpec, Any]] = []
        for index, dispatch in enumerate(ready_dispatches):
            output_directory = launch_attempt_dir / "dispatch" / dispatch.attempt_id
            runner_kwargs: dict[str, Any] = {
                "argv": dispatch.argv,
                "cwd": dispatch.cwd,
                "environment": os.environ | {**context.environment, **dispatch.environment},
                "output_directory": str(output_directory),
                "attempt_id": dispatch.attempt_id,
            }
            if context.visibility_values:
                runner_kwargs.update(
                    visibility_variable=context.visibility_variable,
                    visibility_value=context.visibility_values[index % len(context.visibility_values)],
                )
            submitted.append(
                (
                    dispatch,
                    runner(**runner_kwargs),
                )
            )

        records: list[DispatchRecord] = []
        for dispatch, future in submitted:
            payload = _result_payload(future)
            _check_completed_dispatch(dispatch, payload)
            records.append(
                DispatchRecord.accepted(
                    dispatch,
                    backend="parsl_attach",
                    launcher_job_id=context.allocation_id,
                    submitted_command=dispatch.argv,
                    metadata={"attempt_status_path": payload.get("attempt_status_path")},
                )
            )
        return tuple(records)


def _parsl_app_runner(context: AllocationContext, launch_attempt_dir: Path) -> AppRunner:
    """Build and load the allocation-local Parsl application lazily."""

    import parsl
    from parsl.app.app import python_app
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider

    executor_options: dict[str, Any] = {
        "label": "parsl-attach",
        "provider": LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1),
        "max_workers_per_node": len(context.visibility_values) or 1,
    }
    if context.visibility_values:
        executor_options["available_accelerators"] = context.visibility_values
    config = Config(
        executors=[HighThroughputExecutor(**executor_options)],
        retries=0,
        run_dir=str(launch_attempt_dir / "parsl"),
        usage_tracking=False,
    )
    parsl.load(config)
    return python_app(_run_dispatch_payload)


def _result_payload(value: Any) -> Mapping[str, Any]:
    """Resolve a Parsl future or injectable test result into a mapping."""

    payload = value.result() if hasattr(value, "result") else value
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Parsl app returned non-mapping payload: {payload!r}")
    return payload


def _check_completed_dispatch(dispatch: DispatchSpec, payload: Mapping[str, Any]) -> None:
    """Mirror the legacy post-subprocess completion predicate check."""

    returncode = payload.get("returncode")
    if returncode != 0:
        raise RuntimeError(f"dispatch {dispatch.attempt_id!r} exited with returncode {returncode!r}")
    if dispatch.completion.policy != "none" and not dispatch.completion.is_complete():
        message = (
            f"command exited 0 but completion predicate {dispatch.completion.policy!r} "
            f"was not satisfied for dispatch {dispatch.attempt_id!r}: "
            f"{dispatch.completion.to_dict()!r}"
        )
        status_path = payload.get("attempt_status_path")
        if status_path:
            path = Path(str(status_path))
            status = json.loads(path.read_text(encoding="utf-8"))
            status["status"] = "failed"
            status["completion_error"] = message
            path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(message)
