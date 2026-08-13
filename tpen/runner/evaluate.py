"""Evaluation runner target."""

from __future__ import annotations

from typing import Any

from tpen.artifacts import RunContext, RunResult
from tpen.checkpoint import CheckpointRestored, restore_checkpoint_with_events
from tpen.dependencies import require_torch
from tpen.evaluation import EvaluationResult, Evaluator
from tpen.evaluation.events import EvaluationCompleted, EvaluationStarted

from .base import (
    Runner,
    _assert_eager_initialized,
    _is_torch_module,
    _place_module_for_runtime,
)

torch = require_torch(feature="evaluation runner")


class Evaluate(Runner):
    """Generic evaluation runner that delegates task execution to `Evaluator`.

    Parameters
    ----------
    model : callable
        Wavefunction model returning ``WavefunctionOutput``.
    load : object or None, optional
        Checkpoint restore config. Evaluation accepts ``mode: model_only`` and
        rejects training-resume restores.
    evaluator : Evaluator
        Composable task evaluator. It owns generators, calculators, summaries,
        and task failure policy.
    """

    def __init__(
        self,
        model,
        load=None,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.model = model
        self.load = load
        if evaluator is None:
            raise ValueError("Evaluate requires an evaluator")
        self.evaluator = evaluator

    def run(self, context: RunContext) -> RunResult:
        """Prepare the model, delegate evaluation, and log task metrics."""


        if _is_torch_module(self.model):
            _place_module_for_runtime(self.model, context)
            self.model.eval()
            _assert_eager_initialized(self.model)

        mode = _load_mode(self.load)
        if mode == "train_resume":
            raise ValueError("Evaluate rejects load.mode='train_resume'; use model_only")
        if mode == "model_only":
            report = restore_checkpoint_with_events(
                load=self.load,
                model=self.model,
                context=context,
                emit=context.emit,
            )
            # Typed counterpart, carrying the report itself rather than a
            # flattened mapping. No callback subscribes it; its consumer is the
            # durable occurrence record (D3).
            context.emit(CheckpointRestored(report=report))
            if _is_torch_module(self.model):
                self.model.eval()

        # A point event rather than a scope: a scope's `Ended` fires from a
        # `finally`, which would pre-empt the run-level `exception` event that
        # `EvaluationTiming` turns into `eval/perf {failed: True}`.
        context.emit(EvaluationStarted())
        result = self.evaluator.evaluate(model=self.model, context=context)
        _log_result(context, result, namespace=self.evaluator.namespace)

        # The existing completion moment carries the evaluator's aggregate
        # verdict so timing consumers can close it truthfully.
        context.emit(EvaluationCompleted(status=result.status))
        return RunResult(status="completed" if result.status != "failed" else "failed")


def _load_mode(load) -> str:
    if load is None:
        return "none"
    if hasattr(load, "get"):
        return str(load.get("mode", "none"))
    return "none"


def _log_result(context: RunContext, result: EvaluationResult, *, namespace: str) -> None:
    """Log task metrics in their task namespaces."""

    context.log(
        {
            "suite_success": result.status == "success",
            "suite_failed": result.status == "failed",
        },
        step=0,
        namespace=f"{namespace}/status",
    )
    for task in result.task_results:
        status_metrics: dict[str, Any] = {
            "task_success": task.status == "success",
            "task_failed": task.failed,
        }
        if task.metrics:
            context.log(dict(task.metrics), step=0, namespace=task.namespace)
        context.log(status_metrics, step=0, namespace=f"{task.namespace}/status")


__all__ = ["Evaluate"]
