"""Evaluation runner target."""

from __future__ import annotations

from typing import Any

from spenn.artifacts import RunContext, RunResult
from spenn.checkpoint import restore_checkpoint_with_events
from spenn.dependencies import require_torch
from spenn.evaluation import EvaluationResult, Evaluator

from .base import Runner, _assert_eager_initialized, _is_torch_module, _place_module_for_runtime

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
    construction_seed : int or None, optional
        Optional seed applied before model materialization checks.
    """

    def __init__(
        self,
        model,
        load=None,
        evaluator: Evaluator | None = None,
        construction_seed: int | None = None,
    ) -> None:
        self.model = model
        self.load = load
        if evaluator is None:
            raise ValueError("Evaluate requires an evaluator")
        self.evaluator = evaluator
        self.construction_seed = None if construction_seed is None else int(construction_seed)

    def run(self, context: RunContext) -> RunResult:
        """Prepare the model, delegate evaluation, and log task metrics."""

        self.emit("run_start", context)

        if self.construction_seed is not None:
            torch.manual_seed(self.construction_seed)

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
                emit=self.emit,
            )
            self.emit("checkpoint_restored", context, payload={"restore_report": report.to_dict()})
            if _is_torch_module(self.model):
                self.model.eval()

        self.emit("evaluate_start", context)
        result = self.evaluator.evaluate(
            model=self.model,
            context=context,
            emit=lambda name, *, payload=None: self.emit(name, context, payload=payload),
        )
        _log_result(context, result)

        self.emit("evaluate_end", context, payload=result.to_payload())
        self.emit("run_end", context)
        return RunResult(status="completed" if result.status != "failed" else "failed")


def _load_mode(load) -> str:
    if load is None:
        return "none"
    if hasattr(load, "get"):
        return str(load.get("mode", "none"))
    return "none"


def _log_result(context: RunContext, result: EvaluationResult) -> None:
    """Log task metrics in their task namespaces."""

    for task in result.task_results:
        status_metrics: dict[str, Any] = {
            "task_success": task.status == "success",
            "task_failed": task.status in {"failed", "partial_failed"},
            "task_required": task.required,
        }
        if task.metrics:
            context.log(dict(task.metrics), step=0, namespace=task.namespace)
        context.log(status_metrics, step=0, namespace=f"{task.namespace}/status")


__all__ = ["Evaluate"]
