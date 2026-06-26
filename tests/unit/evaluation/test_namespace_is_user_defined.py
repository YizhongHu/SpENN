"""Tests asserting namespace is purely user-defined — no phase inference."""

from __future__ import annotations

import dataclasses

import pytest

from spenn.evaluation import Evaluator
from spenn.evaluation.protocols import EvaluationContext
from spenn.evaluation.task import EvaluationTask


def test_evaluation_context_has_no_phase_field() -> None:
    fields = {f.name for f in dataclasses.fields(EvaluationContext)}
    assert "phase" not in fields


def test_evaluator_accepts_arbitrary_namespace() -> None:
    evaluator = Evaluator(namespace="my_custom_run/experiment_A", tasks=[])
    assert evaluator.namespace == "my_custom_run/experiment_A"


def test_evaluator_does_not_require_validation_or_eval_prefix() -> None:
    # Arbitrary namespaces must not raise
    for ns in ("staging", "sweep/run_042", "debug", "custom/nested/deep"):
        evaluator = Evaluator(namespace=ns, tasks=[])
        assert evaluator.namespace == ns


def test_evaluator_has_no_phase_attribute() -> None:
    evaluator = Evaluator(namespace="eval", tasks=[])
    assert not hasattr(evaluator, "phase")


def test_reference_energy_summary_is_not_phase_gated() -> None:
    from spenn.evaluation.summaries import ReferenceEnergySummary
    import inspect
    sig = inspect.signature(ReferenceEnergySummary.__init__)
    assert "allow_phase" not in sig.parameters
    assert "phase" not in sig.parameters
