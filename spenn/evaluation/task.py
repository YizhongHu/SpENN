"""Evaluation task specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

ArtifactLevel: TypeAlias = Literal["metrics_only", "summaries", "records"]
FailurePolicy: TypeAlias = Literal["continue", "fail_fast"]


@dataclass(frozen=True)
class EvaluationTask:
    """Thin spec object for one generator/calculator/summary pipeline."""

    name: str
    namespace: str
    output_dir: Path | str
    generator: object
    calculators: Sequence[object]
    summaries: Sequence[object]
    artifact_level: ArtifactLevel | None = None


_ALLOWED_TASK_KEYS = frozenset(
    {"name", "namespace", "generator", "calculators", "summaries", "output_dir", "artifact_level"}
)
# Keys from the removed phase/required compatibility layer: reject loudly so
# stale configs surface during migration instead of carrying dead semantics.
_FORBIDDEN_TASK_KEYS = frozenset({"required", "phase"})


def coerce_task(spec: EvaluationTask | Mapping[str, object]) -> EvaluationTask:
    """Coerce a Hydra-style mapping into an `EvaluationTask`.

    Unknown task keys are rejected. ``required`` and ``phase`` belonged to the
    removed compatibility layer and are rejected explicitly so stale configs
    fail loudly rather than silently preserving dead semantics.
    """

    if isinstance(spec, EvaluationTask):
        _validate_output_dir(spec.name, spec.output_dir)
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError(f"evaluation tasks must be EvaluationTask or mapping, got {type(spec)!r}")
    keys = {str(key) for key in spec}
    forbidden = sorted(keys & _FORBIDDEN_TASK_KEYS)
    if forbidden:
        raise ValueError(f"evaluation task must not define removed key(s): {forbidden}")
    unknown = sorted(keys - _ALLOWED_TASK_KEYS)
    if unknown:
        raise ValueError(f"evaluation task has unknown key(s): {unknown}")
    name = str(spec.get("name", "")).strip()
    namespace = str(spec.get("namespace", "")).strip("/")
    if not name:
        raise ValueError("evaluation task requires a non-empty name")
    if not namespace:
        raise ValueError(f"evaluation task {name!r} requires a non-empty namespace")
    output_dir_raw = spec.get("output_dir")
    output_dir = _validate_output_dir(name, output_dir_raw)
    generator = spec.get("generator")
    if generator is None:
        raise ValueError(f"evaluation task {name!r} requires a generator")
    calculators = tuple(spec.get("calculators", ()) or ())
    summaries = tuple(spec.get("summaries", ()) or ())
    return EvaluationTask(
        name=name,
        namespace=namespace,
        output_dir=output_dir,
        generator=generator,
        calculators=calculators,
        summaries=summaries,
        artifact_level=spec.get("artifact_level"),  # type: ignore[arg-type]
    )


def _validate_output_dir(name: str, value: object) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"evaluation task {name!r} requires output_dir")
    return Path(str(value))


__all__ = ["ArtifactLevel", "EvaluationTask", "FailurePolicy", "coerce_task"]
