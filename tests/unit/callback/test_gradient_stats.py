"""Tests for the GradientStats runtime-check callback."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tpen.callback import GradientStats
from tests.unit.callback.support import (
    RecordingContext,
    deliver_completed_iteration,
    training_state,
)

STEP = 4


class TwoParamModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(2, dtype=torch.float64))


def _handle(
    callback: GradientStats, model: nn.Module, *, optimizer_step: bool = True
) -> RecordingContext:
    """Deliver one completed iteration to `callback`.

    Parameters
    ----------
    optimizer_step : bool, optional
        Whether this iteration applied an optimizer update, i.e. whether it had
        gradients to produce. It defaults to ``True`` because that is what an
        ordinary iteration does; a test of the vacuum-skip path must say
        ``False`` explicitly, since that is the whole distinction under test.
    """

    context = RecordingContext()
    # ``state.step`` is left at its constructor default on purpose: the logged
    # coordinate must come from the typed event, so a state carrying a
    # different value would expose a callback that read the wrong one.
    deliver_completed_iteration(
        callback,
        context,
        training_state(model=model, optimizer_step=optimizer_step),
        step=STEP,
    )
    return context


def test_logs_global_grad_norm() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)
    model.b.grad = torch.zeros(2, dtype=torch.float64)

    metrics = _handle(GradientStats(), model).latest("checks/gradient")

    assert metrics["global_grad_norm"] == pytest.approx(5.0)
    assert metrics["max_abs_grad"] == pytest.approx(4.0)
    assert metrics["n_grad_tensors"] == 2
    assert metrics["n_grad_elements"] == 5
    assert metrics["nonfinite_grad_fraction"] == 0.0
    assert metrics["passed"] is True


def test_handles_parameters_with_grad_none() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    # model.b.grad stays None

    metrics = _handle(GradientStats(), model).latest("checks/gradient")

    assert metrics["n_grad_tensors"] == 1
    assert metrics["n_grad_elements"] == 3
    assert metrics["global_grad_norm"] == pytest.approx(1.0)


def test_vacuum_skip_with_no_gradients_still_passes() -> None:
    """An iteration that applied no update had nothing to differentiate.

    This is the property the deleted ``test_no_gradients_passes_with_zero_norm``
    was really protecting, and it must survive the fix: the zero-electron vacuum
    has ``loss.requires_grad is False``, runs no backward, and leaves every
    ``.grad`` as ``None``. There were no gradients to look at, so reporting a
    failure here would be a regression, not a catch.
    """

    metrics = _handle(
        GradientStats(), TwoParamModule(), optimizer_step=False
    ).latest("checks/gradient")

    assert metrics["n_grad_tensors"] == 0
    assert metrics["global_grad_norm"] == 0.0
    assert metrics["passed"] is True


def test_vacuum_skip_does_not_raise_under_fail_fast() -> None:
    """The sharp edge of the property above: production sets ``fail_fast: true``.

    ``experiments/hooke/tpen-pair-v1/configs/train.yaml`` runs this callback with
    ``fail_fast: true``, so a fix that failed the vacuum path would not merely
    mislabel a metric -- it would abort a legitimate run.
    """

    metrics = _handle(
        GradientStats(fail_fast=True, check_finite=True),
        TwoParamModule(),
        optimizer_step=False,
    ).latest("checks/gradient")

    assert metrics["passed"] is True


def test_cleared_gradients_on_an_updated_iteration_fail() -> None:
    """An update ran and consumed gradients, yet none are visible: broken.

    Every statistic is well defined over the empty set -- the norm is ``0.0``
    and ``nonfinite_grad_fraction`` is ``0.0``, so the finiteness check passes --
    which is exactly how this state used to report ``passed``. Measured on
    Cannon three times with ``fail_fast: true`` set and never firing.
    """

    metrics = _handle(
        GradientStats(), TwoParamModule(), optimizer_step=True
    ).latest("checks/gradient")

    assert metrics["n_grad_tensors"] == 0
    assert metrics["global_grad_norm"] == 0.0
    assert metrics["passed"] is False


def test_fail_fast_raises_when_an_update_ran_but_no_gradients_were_observed() -> None:
    with pytest.raises(RuntimeError, match="observed no gradients"):
        _handle(
            GradientStats(fail_fast=True, check_finite=True),
            TwoParamModule(),
            optimizer_step=True,
        )


@pytest.mark.parametrize("optimizer_step", [False, True], ids=["vacuum_skip", "cleared"])
def test_the_published_key_set_does_not_depend_on_what_was_observed(
    optimizer_step: bool,
) -> None:
    """No metric key was added to mark the distinction, and none may be dropped.

    ``n_grad_tensors`` and ``train/optimizer_step`` already state between them
    whether anything was observed, so a third spelling would duplicate one fact
    (ADR-E003, ADR-E006). Pinning the exact key set here is what stops the fix
    from quietly growing an ``observed`` flag later, and what stops either
    branch from publishing a step-dependent key set into JSONL/CSV.
    """

    metrics = _handle(
        GradientStats(), TwoParamModule(), optimizer_step=optimizer_step
    ).latest("checks/gradient")

    assert list(metrics) == [
        "n_grad_tensors",
        "n_grad_elements",
        "global_grad_norm",
        "max_abs_grad",
        "mean_abs_grad",
        "nonfinite_grad_fraction",
        "passed",
    ]


def test_detects_nonfinite_gradients_without_raising() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([1.0, float("inf"), 2.0], dtype=torch.float64)

    metrics = _handle(GradientStats(fail_fast=False), model).latest("checks/gradient")

    assert metrics["nonfinite_grad_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["passed"] is False
    assert metrics["global_grad_norm"] == pytest.approx(5.0**0.5)  # finite-only norm


def test_fail_fast_raises_on_nonfinite_gradients() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([float("nan"), 0.0, 0.0], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="nonfinite_grad_fraction"):
        _handle(GradientStats(fail_fast=True, check_finite=True), model)


def test_fail_fast_raises_when_grad_norm_exceeds_max() -> None:
    model = TwoParamModule()
    model.a.grad = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="global_grad_norm"):
        _handle(GradientStats(fail_fast=True, max_global_grad_norm=1.0), model)


def test_logged_step_comes_from_the_event_not_the_state() -> None:
    # `TrainerState` value fields are assigned at the END of the trainer's loop
    # body, so at any earlier boundary they hold the previous iteration's values
    # and, on iteration 0, the constructor default -1. A callback that took its
    # coordinate from the state would publish that stale number. Measured on
    # Cannon: a probe on both iteration boundaries read
    # ``[-1, 0, 0, 1, 1, 2, ...]`` while the metric axis ran a clean 0..9.
    context = RecordingContext()
    state = training_state(model=TwoParamModule(), step=-1)

    deliver_completed_iteration(context=context, callback=GradientStats(), state=state, step=7)

    assert [record["step"] for record in context.by_namespace("checks/gradient")] == [7]


def test_cadence_gates_on_the_durable_step_not_the_occurrence_count() -> None:
    # A typed `Cadence` would gate on ``Occurrence.count``, which is run-local
    # and restarts at 1 after a checkpoint restore. Holding the count fixed
    # while the durable step advances is exactly the situation a resumed run
    # creates, and the schedule must follow the step.
    callback = GradientStats(every_n_steps=5)
    context = RecordingContext()
    state = training_state(model=TwoParamModule())

    for step in range(12):
        deliver_completed_iteration(callback, context, state, step=step, count=1)

    assert [record["step"] for record in context.by_namespace("checks/gradient")] == [0, 5, 10]
