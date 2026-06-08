"""Tests for base Callback scheduling, including probabilistic gating."""

from __future__ import annotations

import pytest

from spenn.callback import Callback, Event


class Recorder(Callback):
    """Records the step of every event it actually handles."""

    def __init__(self, **kwargs) -> None:
        super().__init__(["step_end"], **kwargs)
        self.handled: list[int] = []

    def on_step_end(self, event: Event) -> None:
        self.handled.append(event.step)


def _drive(callback: Callback, steps) -> None:
    for step in steps:
        callback.handle(Event(name="step_end", context=None, state=None, payload={"step": step}))


def test_every_n_steps_filters_by_step() -> None:
    callback = Recorder(every_n_steps=2)

    _drive(callback, range(0, 6))

    assert callback.handled == [0, 2, 4]


def test_start_step_delays_first_run() -> None:
    callback = Recorder(every_n_steps=1, start_step=3)

    _drive(callback, range(0, 6))

    assert callback.handled == [3, 4, 5]


def test_max_calls_counts_actual_executions() -> None:
    callback = Recorder(every_n_steps=1, max_calls=2)

    _drive(callback, range(1, 10))

    assert callback.handled == [1, 2]
    assert callback.num_calls == 2


def test_probability_zero_never_runs() -> None:
    callback = Recorder(probability=0.0)

    _drive(callback, range(1, 21))

    assert callback.handled == []
    assert callback.num_calls == 0


def test_probability_one_always_runs_when_scheduled() -> None:
    callback = Recorder(every_n_steps=2, probability=1.0)

    _drive(callback, range(0, 6))

    assert callback.handled == [0, 2, 4]


def test_probability_is_deterministic_with_seed() -> None:
    first = Recorder(probability=0.5, seed=1234)
    second = Recorder(probability=0.5, seed=1234)

    _drive(first, range(1, 51))
    _drive(second, range(1, 51))

    assert first.handled == second.handled
    assert 0 < len(first.handled) < 50  # neither always nor never


def test_probability_is_applied_after_step_filters() -> None:
    # With probability < 1 the RNG is only consumed on scheduled steps, so a
    # seeded callback fires only on a subset of the every_n_steps schedule.
    callback = Recorder(every_n_steps=2, probability=0.5, seed=7)

    _drive(callback, range(0, 40))

    assert all(step % 2 == 0 for step in callback.handled)
    assert len(callback.handled) < 20


def test_invalid_probability_raises() -> None:
    with pytest.raises(ValueError, match="probability"):
        Recorder(probability=1.5)
