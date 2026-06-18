"""Composable evaluation task runner."""

from __future__ import annotations

import traceback as traceback_module
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from spenn.evaluation.bundle import EvaluationBundle
from spenn.evaluation.events import component_failure_payload, task_payload, task_result_payload
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.results import ArtifactRecord, EvaluationFailure, EvaluationResult, MetricScalar, TaskResult
from spenn.evaluation.task import ArtifactLevel, EvaluationTask, FailurePolicy, coerce_task


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

        base_context = self._context_from_run_context(context)
        task_results: list[TaskResult] = []
        full_metrics: dict[str, MetricScalar] = {}
        failures: list[EvaluationFailure] = []
        artifacts: list[ArtifactRecord] = []

        run_dir = _context_run_dir(context)
        for task in self.tasks:
            task_output_dir = task.output_dir if task.output_dir is not None else (
                (run_dir / task.name) if run_dir is not None else Path(task.name)
            )
            task_context = replace(
                base_context,
                namespace=task.namespace,
                artifact_level=task.artifact_level or base_context.artifact_level,
                task_output_dir=task_output_dir,
            )
            result = self._evaluate_task(model=model, task=task, context=task_context, emit=emit)
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
        run_dir = _context_run_dir(context)
        output_dir = run_dir / "diagnostics" if run_dir is not None else Path("diagnostics")
        return EvaluationContext(
            namespace=self.namespace,
            artifact_level=self.artifact_level,
            task_failure_policy=self.task_failure_policy,
            device=device,
            dtype=dtype,
            seed=self.seed,
            output_dir=output_dir,
            task_output_dir=output_dir,
            metadata={},
        )

    def _evaluate_task(
        self,
        *,
        model: torch.nn.Module,
        task: EvaluationTask,
        context: EvaluationContext,
        emit: Callable[..., None],
    ) -> TaskResult:
        emit("task_start", payload=task_payload(task))
        failures: list[EvaluationFailure] = []
        artifacts: list[ArtifactRecord] = []
        metrics: dict[str, MetricScalar] = {}
        task_failed = False
        partial_failed = False
        bundle: EvaluationBundle | None = None

        try:
            generated = task.generator.generate(model=model, context=context)
            bundle = EvaluationBundle(generated=generated)
        except Exception as exc:
            failure = _failure(context, task=task, component=task.generator, component_type="generator", exc=exc)
            failures.append(failure)
            emit("task_failed", payload={**task_result_payload(_task_result(task, "failed", metrics, artifacts, failures))})
            emit("generator_failed", payload=component_failure_payload(task=task, component_name=_component_name(task.generator), failure=failure))
            return _task_result(task, "failed", metrics, artifacts, failures)

        for calculator in task.calculators:
            try:
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
                    ),
                )
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
                        ),
                    )
                    continue
                try:
                    result = summary.summarize(bundle=bundle, context=context, namespace=task.namespace)
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
                        ),
                    )
                    continue
                _merge_metrics(metrics, result.metrics, component_name=_component_name(summary))
                artifacts.extend(result.artifacts)

        if task_failed:
            status = "failed"
        elif partial_failed:
            status = "partial_failed"
        else:
            status = "success"
        task_result = _task_result(task, status, metrics, artifacts, failures)
        event_name = "task_failed" if status in {"failed", "partial_failed"} else "task_end"
        emit(event_name, payload=task_result_payload(task_result))
        return task_result


def _task_result(
    task: EvaluationTask,
    status: str,
    metrics: Mapping[str, MetricScalar],
    artifacts: Sequence[ArtifactRecord],
    failures: Sequence[EvaluationFailure],
) -> TaskResult:
    return TaskResult(
        name=task.name,
        namespace=task.namespace,
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


def _aggregate_status(task_results: Sequence[TaskResult]) -> str:
    if any(task.status == "failed" for task in task_results):
        return "failed"
    if any(task.status in {"partial_failed", "skipped"} for task in task_results):
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
