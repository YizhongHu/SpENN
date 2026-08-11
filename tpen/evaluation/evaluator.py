"""Composable evaluation task runner."""

from __future__ import annotations

import traceback as traceback_module
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from tpen.evaluation.bundle import EvaluationBundle
from tpen.evaluation.events import (
    CalculatorRun,
    ComponentFailed,
    ComponentRun,
    EvaluationTaskRun,
    GeneratorRun,
    SummaryRun,
    component_failure_payload,
    component_payload,
    task_payload,
    task_result_payload,
)
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import ArtifactRecord, EvaluationFailure, EvaluationResult, MetricScalar, TaskResult
from tpen.evaluation.state import EvaluationRunState
from tpen.evaluation.task import ArtifactLevel, EvaluationTask, FailurePolicy, coerce_task


@contextmanager
def _component_span(
    emit: Callable[..., None],
    *,
    task: EvaluationTask,
    component_type: str,
    component_class: type[ComponentRun],
    component_name: str | None,
    output_dir: Path,
    run_context: Any,
    state: EvaluationRunState,
):
    """Bracket one evaluation component with lifecycle events.

    The evaluator owns the component boundaries but no timing policy: it only
    emits ``<component_type>_start``/``<component_type>_end`` events, and
    timing callbacks such as ``EvaluationComponentTiming`` decide whether and
    how to measure them. The ``_end`` event fires on success and failure
    alike; failures additionally emit the existing ``<component_type>_failed``
    events.

    The typed `ComponentRun` scope is nested strictly inside the legacy string
    pair, so the legacy sequence is unchanged and no callback has to move yet.
    ``component_class`` is passed as a type rather than derived from
    ``component_type``: resolving a type from a string would be the routing
    bridge ADR-E002 forecloses.
    """

    emit(
        f"{component_type}_start",
        payload=component_payload(task=task, component_name=component_name, output_dir=output_dir),
    )
    try:
        with run_context.scope(component_class(name=component_name), state=state):
            yield
    finally:
        emit(
            f"{component_type}_end",
            payload=component_payload(task=task, component_name=component_name, output_dir=output_dir),
        )


class Evaluator:
    """Run a sequence of evaluation tasks against one model."""

    def __init__(
        self,
        *,
        tasks: Sequence[EvaluationTask | Mapping[str, object]],
        namespace: str,
        artifact_level: ArtifactLevel = "metrics_only",
        task_failure_policy: FailurePolicy = "continue",
        seed: int | None = None,
    ) -> None:
        self.tasks = tuple(coerce_task(task) for task in tasks)
        self.namespace = str(namespace).strip("/")
        if not self.namespace:
            raise ValueError("Evaluator namespace must be non-empty")
        if artifact_level not in ("metrics_only", "summaries", "records"):
            raise ValueError(f"unsupported artifact_level {artifact_level!r}")
        if task_failure_policy not in ("continue", "fail_fast"):
            raise ValueError(f"unsupported task_failure_policy {task_failure_policy!r}")
        self.artifact_level = artifact_level
        self.task_failure_policy = task_failure_policy
        self.seed = seed

    def evaluate(
        self,
        *,
        model: torch.nn.Module,
        context: Any,
        emit: Callable[..., None],
    ) -> EvaluationResult:
        """Run all configured tasks and return aggregate metrics."""

        # Two unrelated context types meet here: `context` is the run-level
        # `tpen.artifacts.RunContext` that owns typed emission, while
        # `base_context`/`task_context` are `EvaluationContext` values. The
        # alias keeps the two visually distinct wherever both are passed on.
        run_context = context
        base_context = self._context_from_run_context(context)
        # One state object for the whole suite, updated in place: `scope`
        # captures this reference at entry, so its identity must stay stable.
        state = EvaluationRunState()
        task_results: list[TaskResult] = []
        full_metrics: dict[str, MetricScalar] = {}
        failures: list[EvaluationFailure] = []
        artifacts: list[ArtifactRecord] = []

        for task in self.tasks:
            task_output_dir = _resolve_task_output_dir(task.output_dir, run_dir=base_context.run_dir)
            # Materialize the task directory before running so summaries can write
            # task-local artifacts without each re-creating it.
            task_output_dir.mkdir(parents=True, exist_ok=True)
            task_context = replace(
                base_context,
                namespace=task.namespace,
                artifact_level=task.artifact_level or base_context.artifact_level,
                task_output_dir=task_output_dir,
            )
            # Clear the previous task's result before this task's scope opens, so
            # a handler at the `Started` boundary cannot read a stale one.
            state.task_result = None
            task_run = EvaluationTaskRun(
                name=task.name, namespace=task.namespace, output_dir=task_output_dir
            )
            with run_context.scope(task_run, state=state):
                result = self._evaluate_task(
                    model=model,
                    task=task,
                    context=task_context,
                    emit=emit,
                    run_context=run_context,
                    state=state,
                )
                # Written inside the scope so the `Ended` boundary observes it.
                state.task_result = result
            task_results.append(result)
            failures.extend(result.failures)
            artifacts.extend(result.artifacts)
            for key, value in result.metrics.items():
                full_metrics[f"{result.namespace}/{key}"] = value
            if result.status in {"failed", "partial_failed"} and self.task_failure_policy == "fail_fast":
                break

        status = _aggregate_status(task_results)
        return EvaluationResult(
            status=status,
            metrics=full_metrics,
            task_results=tuple(task_results),
            artifacts=tuple(artifacts),
            failures=tuple(failures),
        )

    def _context_from_run_context(self, context: Any) -> EvaluationContext:
        device = _torch_device(getattr(getattr(context, "metadata", None), "device", None))
        dtype = _torch_dtype(getattr(getattr(context, "metadata", None), "dtype", None))
        run_dir = _context_run_dir(context) or Path(".")
        return EvaluationContext(
            namespace=self.namespace,
            artifact_level=self.artifact_level,
            task_failure_policy=self.task_failure_policy,
            device=device,
            dtype=dtype,
            seed=self.seed,
            run_dir=run_dir,
            task_output_dir=run_dir,
            metadata={},
        )

    def _evaluate_task(
        self,
        *,
        model: torch.nn.Module,
        task: EvaluationTask,
        context: EvaluationContext,
        emit: Callable[..., None],
        run_context: Any,
        state: EvaluationRunState,
    ) -> TaskResult:
        output_dir = context.task_output_dir
        emit("task_start", payload=task_payload(task, output_dir=output_dir))
        failures: list[EvaluationFailure] = []
        artifacts: list[ArtifactRecord] = []
        metrics: dict[str, MetricScalar] = {}
        task_failed = False
        partial_failed = False
        bundle: EvaluationBundle | None = None

        try:
            with _component_span(
                emit,
                task=task,
                component_type="generator",
                component_class=GeneratorRun,
                component_name=_component_name(task.generator),
                output_dir=output_dir,
                run_context=run_context,
                state=state,
            ):
                generated = task.generator.generate(model=model, context=context)
            bundle = EvaluationBundle(generated=generated)
        except Exception as exc:
            failure = _failure(context, task=task, component=task.generator, component_type="generator", exc=exc)
            failures.append(failure)
            result = _task_result(task, output_dir, "failed", metrics, artifacts, failures)
            # Deliberately inverted relative to the calculator and summary paths:
            # this path builds the task result first and only then reports the
            # component. `test_generator_failure_emits_task_failed_first` pins it.
            emit("task_failed", payload=task_result_payload(result))
            emit(
                "generator_failed",
                payload=component_failure_payload(
                    task=task,
                    component_name=_component_name(task.generator),
                    failure=failure,
                    output_dir=output_dir,
                ),
            )
            run_context.emit(ComponentFailed(failure=failure), state=state)
            return result

        for calculator in task.calculators:
            try:
                with _component_span(
                    emit,
                    task=task,
                    component_type="calculator",
                    component_class=CalculatorRun,
                    component_name=_component_name(calculator),
                    output_dir=output_dir,
                    run_context=run_context,
                    state=state,
                ):
                    bundle = calculator.calculate(model=model, bundle=bundle, context=context)
            except Exception as exc:
                failure = _failure(context, task=task, component=calculator, component_type="calculator", exc=exc)
                failures.append(failure)
                task_failed = True
                emit(
                    "calculator_failed",
                    payload=component_failure_payload(
                        task=task,
                        component_name=_component_name(calculator),
                        failure=failure,
                        output_dir=output_dir,
                    ),
                )
                run_context.emit(ComponentFailed(failure=failure), state=state)
                if context.task_failure_policy == "fail_fast":
                    break

        if bundle is not None:
            for summary in task.summaries:
                if not _summary_dependencies_present(summary, bundle):
                    if task_failed:
                        continue
                    failure = _missing_dependency_failure(context, task=task, summary=summary)
                    failures.append(failure)
                    partial_failed = True
                    emit(
                        "summary_failed",
                        payload=component_failure_payload(
                            task=task,
                            component_name=_component_name(summary),
                            failure=failure,
                            output_dir=output_dir,
                        ),
                    )
                    run_context.emit(ComponentFailed(failure=failure), state=state)
                    continue
                try:
                    with _component_span(
                        emit,
                        task=task,
                        component_type="summary",
                        component_class=SummaryRun,
                        component_name=_component_name(summary),
                        output_dir=output_dir,
                        run_context=run_context,
                        state=state,
                    ):
                        result = summary.summarize(bundle=bundle, context=context, namespace=task.namespace)
                    _merge_metrics(metrics, result.metrics, component_name=_component_name(summary))
                except Exception as exc:
                    failure = _failure(context, task=task, component=summary, component_type="summary", exc=exc)
                    failures.append(failure)
                    partial_failed = True
                    emit(
                        "summary_failed",
                        payload=component_failure_payload(
                            task=task,
                            component_name=_component_name(summary),
                            failure=failure,
                            output_dir=output_dir,
                        ),
                    )
                    run_context.emit(ComponentFailed(failure=failure), state=state)
                    continue
                artifacts.extend(result.artifacts)

        if task_failed:
            status = "failed"
        elif partial_failed:
            status = "partial_failed"
        else:
            status = "success"
        task_result = _task_result(task, output_dir, status, metrics, artifacts, failures)
        event_name = "task_failed" if status in {"failed", "partial_failed"} else "task_end"
        emit(event_name, payload=task_result_payload(task_result))
        return task_result


def _task_result(
    task: EvaluationTask,
    output_dir: Path,
    status: str,
    metrics: Mapping[str, MetricScalar],
    artifacts: Sequence[ArtifactRecord],
    failures: Sequence[EvaluationFailure],
) -> TaskResult:
    return TaskResult(
        name=task.name,
        namespace=task.namespace,
        output_dir=output_dir,
        status=status,  # type: ignore[arg-type]
        metrics=dict(metrics),
        artifacts=tuple(artifacts),
        failures=tuple(failures),
    )


def _summary_dependencies_present(summary: object, bundle: EvaluationBundle) -> bool:
    required = getattr(summary, "required_fields", frozenset())
    return all(getattr(bundle, field, None) is not None for field in required)


def _missing_dependency_failure(context: EvaluationContext, *, task: EvaluationTask, summary: object) -> EvaluationFailure:
    required = sorted(getattr(summary, "required_fields", frozenset()))
    return EvaluationFailure(
        task=task.name,
        component=_component_name(summary),
        component_type="summary",
        error_type="MissingBundleField",
        message=f"summary requires missing bundle field(s): {required}",
        traceback=None,
    )


def _failure(
    context: EvaluationContext,
    *,
    task: EvaluationTask,
    component: object,
    component_type: str,
    exc: Exception,
) -> EvaluationFailure:
    return EvaluationFailure(
        task=task.name,
        component=_component_name(component),
        component_type=component_type,  # type: ignore[arg-type]
        error_type=type(exc).__name__,
        message=str(exc),
        traceback=traceback_module.format_exc(),
    )


def _component_name(component: object | None) -> str | None:
    if component is None:
        return None
    name = getattr(component, "name", None)
    return str(name) if name is not None else type(component).__name__


def _merge_metrics(target: dict[str, MetricScalar], values: Mapping[str, MetricScalar], *, component_name: str | None) -> None:
    for key, value in values.items():
        if key in target:
            raise ValueError(f"metric key collision for {key!r} from {component_name or 'summary'}")
        target[key] = value


def _resolve_task_output_dir(value: Path | str, *, run_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(run_dir) / path


def _aggregate_status(task_results: Sequence[TaskResult]) -> str:
    # Any configured task failure (full or partial) fails the suite; there is no
    # `required` flag to downgrade it. `success_with_warnings` is reserved for
    # non-task-critical issues (e.g. skipped tasks), not broken evaluation tasks.
    if any(task.status in {"failed", "partial_failed"} for task in task_results):
        return "failed"
    if any(task.status == "skipped" for task in task_results):
        return "success_with_warnings"
    return "success"


def _torch_device(value: object) -> torch.device | None:
    if value in (None, ""):
        return None
    return torch.device(str(value))


def _torch_dtype(value: object) -> torch.dtype | None:
    if value in (None, ""):
        return None
    dtype = getattr(torch, str(value))
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported evaluation dtype {value!r}")
    return dtype


def _context_run_dir(context: Any) -> Path | None:
    try:
        return context.run_dir
    except AttributeError:
        return None


__all__ = ["Evaluator"]
