"""Tests for task-local output directory contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
import pytest
from torch import nn

from tpen.artifacts import RunContext
from tpen.callback import Callback, SubscriptionGroup
from tpen.checkpoint import CheckpointReplaySemantics, CuspDistanceSemantics
from tpen.evaluation import Evaluator, EvaluationTask
from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations
from tpen.data.batch import ElectronBatch
from tpen.evaluation.events import ComponentFailed
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import SummaryResult
from tpen.events import Subscription
from tests.helpers.run_context import make_run_context


def _replay_semantics() -> CheckpointReplaySemantics:
    return CheckpointReplaySemantics(
        source_git_sha="a" * 40,
        source_tpen_version="0.3.1",
        checkpoint_schema_version=2,
        checkpoint_kind="tpen.checkpoint",
        checkpoint_model_sha256="b" * 64,
        evaluation_config_sha256="c" * 64,
        runtime_dtype="float64",
        cusp_distance=CuspDistanceSemantics(
            electron_electron_distance_form="sqrt_squared_distance_plus_eps_squared",
            electron_electron_distance_eps=1.0e-12,
            electron_electron_range_offset_form="softplus_plus_eps",
            electron_electron_range_offset_eps=1.0e-12,
            electron_nucleus_coulomb_distance_form="euclidean_norm_clamp_min_eps",
            electron_nucleus_coulomb_distance_eps=0.0,
        ),
    )


class _NullGenerator:
    name = "null"

    def generate(self, *, model: nn.Module | None, context: EvaluationContext) -> GeneratedConfigurations:
        batch = ElectronBatch(
            positions=torch.zeros(1, 2, 3, dtype=torch.float64),
            spins=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        )
        return GeneratedConfigurations(batch=batch, metadata={})


class _FailingGenerator:
    name = "broken"

    def generate(
        self, *, model: nn.Module | None, context: EvaluationContext
    ) -> GeneratedConfigurations:
        raise RuntimeError("generator boom")


class _RecordingOutputDirSummary:
    name = "output_dir_recorder"
    required_fields: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.recorded_task_output_dir: Path | None = None

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        self.recorded_task_output_dir = context.task_output_dir
        return SummaryResult(metrics={})


class _MetricSummary:
    name = "metric"
    required_fields: frozenset[str] = frozenset()

    def __init__(self, key: str, value: float) -> None:
        self.key = key
        self.value = value

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        return SummaryResult(metrics={self.key: self.value})


def _run_context(tmp_path: Path) -> RunContext:
    """Return a real `RunContext` rooted at ``tmp_path``.

    The evaluator now emits typed occurrences through the context itself, so a
    `types.SimpleNamespace` stand-in no longer suffices: `RunContext.scope` needs
    the occurrence counters and `write_occurrence_artifact` needs the artifact
    manager and the run clock. Its ``run_dir`` is a real run directory *under*
    ``tmp_path`` rather than ``tmp_path`` itself, which is what a relative task
    output dir resolves against.
    """

    return make_run_context(tmp_path)


def test_evaluator_requires_explicit_task_output_dir() -> None:
    with pytest.raises(ValueError, match="output_dir"):
        Evaluator(
            namespace="eval",
            tasks=[
                {
                    "name": "energy",
                    "namespace": "eval/energy",
                    "generator": _NullGenerator(),
                    "calculators": [],
                    "summaries": [],
                }
            ],
        )


def test_task_output_dir_is_respected(tmp_path: Path) -> None:
    explicit_dir = tmp_path / "energy"
    recorder = _RecordingOutputDirSummary()
    evaluator = Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name="energy",
                namespace="eval/energy",
                output_dir=explicit_dir,
                generator=_NullGenerator(),
                calculators=[],
                summaries=[recorder],
            )
        ],
    )
    evaluator.evaluate(model=nn.Linear(1, 1), context=_run_context(tmp_path))
    assert recorder.recorded_task_output_dir == explicit_dir


def test_task_output_dir_override_is_respected(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_energy_output"
    recorder = _RecordingOutputDirSummary()
    evaluator = Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name="energy",
                namespace="eval/energy",
                output_dir=custom_dir,
                generator=_NullGenerator(),
                calculators=[],
                summaries=[recorder],
            )
        ],
    )
    evaluator.evaluate(model=nn.Linear(1, 1), context=_run_context(tmp_path))
    assert recorder.recorded_task_output_dir == custom_dir


def test_relative_task_output_dir_is_resolved_against_run_dir(tmp_path: Path) -> None:
    recorder = _RecordingOutputDirSummary()
    evaluator = Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name="energy",
                namespace="eval/energy",
                output_dir="energy",
                generator=_NullGenerator(),
                calculators=[],
                summaries=[recorder],
            )
        ],
    )

    context = _run_context(tmp_path)

    result = evaluator.evaluate(model=nn.Linear(1, 1), context=context)

    # Resolved against the context's real run directory, which is what the
    # evaluator reads; ``tmp_path`` is only the root that directory sits under.
    assert recorder.recorded_task_output_dir == context.run_dir / "energy"
    assert result.task_results[0].output_dir == context.run_dir / "energy"


def test_duplicate_summary_metrics_are_structured_task_failures(tmp_path: Path) -> None:
    failures: list[ComponentFailed] = []

    class _FailureRecorder(Callback):
        def __init__(self) -> None:
            super().__init__(
                typed_groups=(
                    SubscriptionGroup(selectors=(Subscription.of(ComponentFailed),)),
                )
            )

        def handle_occurrence_impl(self, occurrence, context) -> None:
            failures.append(occurrence.event)

    evaluator = Evaluator(
        namespace="eval",
        tasks=[
            EvaluationTask(
                name="energy",
                namespace="eval/energy",
                output_dir=tmp_path / "energy",
                generator=_NullGenerator(),
                calculators=[],
                summaries=[_MetricSummary("duplicate", 1.0), _MetricSummary("duplicate", 2.0)],
            )
        ],
    )

    result = evaluator.evaluate(
        model=nn.Linear(1, 1),
        context=make_run_context(tmp_path, callbacks=[_FailureRecorder()]),
    )

    assert result.status == "failed"
    assert result.task_results[0].status == "partial_failed"
    assert result.task_results[0].failures[0].error_type == "ValueError"
    # The metric-key collision is reported as a summary component failure, which
    # is what `FailureLog` writes to ``diagnostics/failures.jsonl``.
    assert [event.failure.component_type for event in failures] == ["summary"]


def test_replay_semantics_reaches_every_task_result_and_the_run_sidecar(
    tmp_path: Path,
) -> None:
    """Even a generator failure carries the typed replay identity."""

    semantics = _replay_semantics()
    context = _run_context(tmp_path)
    result = _evaluator(
        output_dir=tmp_path / "energy", generator=_FailingGenerator()
    ).evaluate(
        model=nn.Linear(1, 1), context=context, replay_semantics=semantics
    )

    assert result.replay_semantics == semantics
    assert result.task_results[0].replay_semantics == semantics
    assert result.task_results[0].status == "failed"
    sidecar = context.run_dir / "checkpoint_replay_semantics.json"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == semantics.to_dict()
    assert result.artifacts[0].path == sidecar
    assert result.to_payload()["replay_semantics"] == semantics.to_dict()
    assert result.task_results[0].to_payload()["replay_semantics"] == semantics.to_dict()


def test_replay_sidecar_refuses_a_different_semantic_identity_before_a_task_starts(
    tmp_path: Path,
) -> None:
    """The immutable sidecar gate has a negative arm, not only a write path."""

    context = _run_context(tmp_path)
    semantics = _replay_semantics()
    evaluator = _evaluator(output_dir=tmp_path / "energy", generator=_NullGenerator())
    evaluator.evaluate(model=nn.Linear(1, 1), context=context, replay_semantics=semantics)

    with pytest.raises(FileExistsError, match="different checkpoint replay semantics"):
        evaluator.evaluate(
            model=nn.Linear(1, 1),
            context=context,
            replay_semantics=replace(semantics, source_git_sha="d" * 40),
        )
