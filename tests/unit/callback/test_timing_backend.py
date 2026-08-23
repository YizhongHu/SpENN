"""Deterministic contract tests for the callback timing backends."""

from __future__ import annotations

from tpen.callback.timing.base import DeviceTimingBackend, TimingSource


class _Timer:
    def __init__(self, elapsed: float) -> None:
        self.elapsed_value = elapsed
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> float:
        assert self.started
        return self.elapsed_value


def test_host_source_uses_emitter_stamps_without_calling_clock() -> None:
    calls: list[str] = []
    source = TimingSource(clock=lambda: calls.append("clock") or 99.0)

    marker = source.start(10.0)
    elapsed = source.elapsed(marker, 12.5)

    assert elapsed.host == 2.5
    assert elapsed.device is None
    assert calls == []


def test_optional_device_source_has_distinct_elapsed_value() -> None:
    timers: list[_Timer] = []

    def factory() -> _Timer:
        timer = _Timer(0.125)
        timers.append(timer)
        return timer

    source = TimingSource(
        clock=lambda: 0.0,
        device_backend=DeviceTimingBackend(factory=factory),
    )
    marker = source.start(4.0)
    elapsed = source.elapsed(marker, 5.0)

    assert elapsed.host == 1.0
    assert elapsed.device == 0.125
    assert len(timers) == 1 and timers[0].started
