"""Tests asserting EvaluationTask and TaskResult have no required field."""

from __future__ import annotations

import dataclasses

import pytest

from spenn.evaluation.results import TaskResult
from spenn.evaluation.task import EvaluationTask, coerce_task


def test_evaluation_task_has_no_required_field() -> None:
    fields = {f.name for f in dataclasses.fields(EvaluationTask)}
    assert "required" not in fields


def test_task_result_has_no_required_field() -> None:
    fields = {f.name for f in dataclasses.fields(TaskResult)}
    assert "required" not in fields


def test_coerce_task_rejects_required_key() -> None:
    with pytest.raises(ValueError, match="required"):
        coerce_task(
            {
                "name": "energy",
                "namespace": "eval/energy",
                "generator": object(),
                "required": True,  # removed key — must be rejected, not ignored
            }
        )


def test_coerce_task_rejects_phase_key() -> None:
    with pytest.raises(ValueError, match="phase"):
        coerce_task(
            {
                "name": "energy",
                "namespace": "eval/energy",
                "generator": object(),
                "phase": "eval",
            }
        )


def test_coerce_task_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown"):
        coerce_task(
            {
                "name": "energy",
                "namespace": "eval/energy",
                "generator": object(),
                "bogus": 1,
            }
        )


def test_evaluation_task_constructor_rejects_required_kwarg() -> None:
    with pytest.raises(TypeError):
        EvaluationTask(  # type: ignore[call-arg]
            name="energy",
            namespace="eval/energy",
            generator=object(),
            calculators=[],
            summaries=[],
            required=True,
        )
