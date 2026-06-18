"""Evaluation task specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

ArtifactLevel: TypeAlias = Literal["metrics_only", "summaries", "records"]
FailurePolicy: TypeAlias = Literal["continue", "fail_fast"]
EvaluationPhase: TypeAlias = Literal["validation", "eval"]


@dataclass(frozen=True)
class EvaluationTask:
    """Thin spec object for one generator/calculator/summary pipeline."""

    name: str
    namespace: str
    generator: object
    calculators: Sequence[object]
    summaries: Sequence[object]
    required: bool = True
    artifact_level: ArtifactLevel | None = None


def coerce_task(spec: EvaluationTask | Mapping[str, object]) -> EvaluationTask:
    """Coerce a Hydra-style mapping into an `EvaluationTask`."""

    if isinstance(spec, EvaluationTask):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError(f"evaluation tasks must be EvaluationTask or mapping, got {type(spec)!r}")
    name = str(spec.get("name", "")).strip()
    namespace = str(spec.get("namespace", "")).strip("/")
    if not name:
        raise ValueError("evaluation task requires a non-empty name")
    if not namespace:
        raise ValueError(f"evaluation task {name!r} requires a non-empty namespace")
    generator = spec.get("generator")
    if generator is None:
        raise ValueError(f"evaluation task {name!r} requires a generator")
    calculators = tuple(spec.get("calculators", ()) or ())
    summaries = tuple(spec.get("summaries", ()) or ())
    return EvaluationTask(
        name=name,
        namespace=namespace,
        generator=generator,
        calculators=calculators,
        summaries=summaries,
        required=bool(spec.get("required", True)),
        artifact_level=spec.get("artifact_level"),  # type: ignore[arg-type]
    )


__all__ = ["ArtifactLevel", "EvaluationPhase", "EvaluationTask", "FailurePolicy", "coerce_task"]
