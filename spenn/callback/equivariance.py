"""Runtime equivariance checking callback."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from spenn.artifacts import write_json
from spenn.naming import camel_to_snake

from .base import Callback, Event


class RuntimeEquivariance(Callback):
    """Schedule one or more runtime equivariance checkers.

    The callback owns *when* to run (triggers, ``every_n_steps``,
    ``probability``), assigns each checker a stable log name, logs its metrics
    under ``checks/equivariance/<name>``, persists any failure artifact under
    ``artifact_dir``, and raises in ``fail_fast`` mode. Each injected checker
    owns *how* to check (permutation selection, comparison) and returns a
    `spenn.equivariance.checks.EquivarianceCheckResult`.

    The callback ``seed`` controls probabilistic *scheduling*; each checker's own
    ``seed`` controls *which permutations* it selects — separate random streams.

    Parameters
    ----------
    triggers : iterable of str
        Event names that trigger the check (typically ``step_end``).
    checkers : sequence
        Checker objects exposing ``run(state) -> EquivarianceCheckResult``.
    fail_fast : bool, optional
        Raise when any checker fails.
    artifact_dir : str or pathlib.Path or None, optional
        Root directory for failure artifacts. When ``None``, artifacts are not
        written even if a checker returns one.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        *,
        checkers: "Sequence[Any]",
        fail_fast: bool = True,
        artifact_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.checkers = list(checkers)
        self.fail_fast = bool(fail_fast)
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self._checker_log_names = _assign_checker_log_names(self.checkers)

    def on_step_end(self, event: Event) -> None:
        """Run every checker against the current state and log/persist results."""

        state = event.state
        for checker, log_name in zip(self.checkers, self._checker_log_names, strict=True):
            result = checker.run(state)

            metrics = dict(result.metrics)
            metrics["passed"] = bool(result.passed)
            metrics["checker_class"] = type(checker).__name__

            if result.artifact is not None and self.artifact_dir is not None:
                artifact_path = _write_equivariance_artifact(
                    root=self.artifact_dir,
                    checker_name=log_name,
                    step=state.step,
                    artifact=result.artifact,
                )
                metrics["artifact_path"] = str(artifact_path)

            event.context.log(metrics, step=state.step, namespace=f"checks/equivariance/{log_name}")

            if self.fail_fast and not result.passed:
                raise RuntimeError(
                    f"RuntimeEquivariance checker {log_name!r} failed at step {state.step}: "
                    f"{result.failures or metrics}"
                )


_DEFAULT_CHECKER_NAMES = {
    "FullModelEquivarianceChecker": "full_model",
    "TraceEquivarianceChecker": "trace",
}


def _checker_base_name(checker: object) -> str:
    """Return a readable base log name for a checker."""

    class_name = type(checker).__name__
    if class_name in _DEFAULT_CHECKER_NAMES:
        return _DEFAULT_CHECKER_NAMES[class_name]
    explicit = getattr(checker, "name", None)
    if explicit:
        return str(explicit)
    snake = camel_to_snake(class_name)
    for suffix in ("_equivariance_checker", "_checker"):
        if snake.endswith(suffix):
            return snake[: -len(suffix)]
    return snake or class_name.lower()


def _assign_checker_log_names(checkers: Sequence[object]) -> list[str]:
    """Assign stable, de-duplicated log names; warn (do not fail) on duplicates.

    The first instance of a base name keeps it; later duplicates get ``_1``,
    ``_2`` suffixes.
    """

    seen: dict[str, int] = {}
    names: list[str] = []
    for checker in checkers:
        base = _checker_base_name(checker)
        count = seen.get(base, 0)
        if count == 0:
            assigned = base
        else:
            assigned = f"{base}_{count}"
            warnings.warn(
                f"RuntimeEquivariance received duplicate checker name {base!r}; "
                f"using {assigned!r} for the duplicate.",
                stacklevel=3,
            )
        seen[base] = count + 1
        names.append(assigned)
    return names


def _write_equivariance_artifact(
    *,
    root: Path,
    checker_name: str,
    step: int,
    artifact: Mapping[str, Any],
) -> Path:
    """Write a failure artifact under ``root/<checker_name>/step_<step>/failure.json``."""

    path = root / checker_name / f"step_{int(step):06d}" / "failure.json"
    write_json(path, dict(artifact))
    return path



__all__ = ["RuntimeEquivariance"]
