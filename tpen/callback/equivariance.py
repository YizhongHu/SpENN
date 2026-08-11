"""Runtime equivariance checking callback."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from tpen.artifacts import RunContext, write_json
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.naming import camel_to_snake
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence


class RuntimeEquivariance(StatefulCallback[TrainerState]):
    """Schedule one or more runtime equivariance checkers.

    The callback owns *when* to run (``every_n_steps``, ``probability``),
    assigns each checker a stable log name, logs its metrics under
    ``checks/equivariance/<name>``, persists any failure artifact under
    ``artifact_dir``, and raises in ``fail_fast`` mode. Each injected checker
    owns *how* to check (permutation selection, comparison) and returns a
    `tpen.equivariance.checks.EquivarianceCheckResult`.

    The callback ``seed`` controls probabilistic *scheduling*; each checker's own
    ``seed`` controls *which permutations* it selects — separate random streams.

    Parameters
    ----------
    checkers : sequence
        Checker objects exposing ``run(state) -> EquivarianceCheckResult``.
    fail_fast : bool, optional
        Raise when any checker fails.
    artifact_dir : str or pathlib.Path or None, optional
        Root directory for failure artifacts. When ``None``, artifacts are not
        written even if a checker returns one.

    Notes
    -----
    This is the only migrated callback whose step coordinate reaches a durable
    path name: `_write_equivariance_artifact` formats ``step_{step:06d}``. It is
    read from the typed event so a future boundary change cannot turn it into
    ``step_-00001``. Each checker separately derives its permutation selection
    from the step it reads off the state; at this boundary the two coordinates
    are equal, so no permutation selection moves.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(
        self,
        *,
        checkers: "Sequence[Any]",
        fail_fast: bool = True,
        artifact_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        cadence = pop_step_cadence(kwargs)
        super().__init__(
            # Subscriptions are class-owned, never configured: this callback
            # observes completed training iterations and nothing else.
            typed_groups=(
                SubscriptionGroup(
                    selectors=(Subscription.of(TrainingIterationCompleted),)
                ),
            ),
            **kwargs,
        )
        # `cadence=None` on the group above is deliberate: a group `Cadence`
        # gates on the run-local occurrence count, which restarts after a
        # restore. This is the most expensive check in the stack, so a schedule
        # that silently shifted phase on a resumed run would be visible.
        self._steps = StepCadenceGate(cadence)
        self.checkers = list(checkers)
        self.fail_fast = bool(fail_fast)
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self._checker_log_names = _assign_checker_log_names(self.checkers)

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        """Run every checker against the current state and log/persist results."""

        event = occurrence.event
        if not isinstance(event, TrainingIterationCompleted):
            return
        # The coordinate rides the typed event, never `state.step`; see the note
        # on `tpen.training.state.TrainerState`'s value fields in the trainer.
        step = int(event.iteration.step)
        if not self._steps.should_run(step):
            return

        for checker, log_name in zip(self.checkers, self._checker_log_names, strict=True):
            result = checker.run(state)

            metrics = dict(result.metrics)
            metrics["passed"] = bool(result.passed)
            metrics["checker_class"] = type(checker).__name__

            if result.artifact is not None and self.artifact_dir is not None:
                artifact_path = _write_equivariance_artifact(
                    root=self.artifact_dir,
                    checker_name=log_name,
                    step=step,
                    artifact=result.artifact,
                )
                metrics["artifact_path"] = str(artifact_path)

            context.log(metrics, step=step, namespace=f"checks/equivariance/{log_name}")

            if self.fail_fast and not result.passed:
                raise RuntimeError(
                    f"RuntimeEquivariance checker {log_name!r} failed at step {step}: "
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
