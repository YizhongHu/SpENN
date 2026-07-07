"""Tests for the run-level resource-usage callback."""

from __future__ import annotations

import pytest

from spenn.callback import Event, ResourceUsage
from spenn.callback import resource_usage as resource_usage_module
from tests.unit.callback.support import RecordingContext


def _fake_torch(*, available: bool, calls: list[str] | None = None):
    calls = [] if calls is None else calls
    cuda = type(
        "FakeCuda",
        (),
        {
            "is_available": staticmethod(lambda: available),
            "reset_peak_memory_stats": staticmethod(lambda: calls.append("reset")),
            "max_memory_allocated": staticmethod(lambda: 3 * 1024 * 1024),
            "max_memory_reserved": staticmethod(lambda: 8 * 1024 * 1024),
            "device_count": staticmethod(lambda: 2),
        },
    )()
    return type("FakeTorch", (), {"cuda": cuda})()


def test_resource_usage_logs_peak_rss_at_run_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resource_usage_module, "require_torch", lambda *, feature: _fake_torch(available=False)
    )
    context = RecordingContext()
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 512.0)

    callback.handle(Event(name="run_start", context=context))
    callback.handle(Event(name="run_end", context=context))

    assert context.by_namespace("runtime") == [
        {"metrics": {"peak_memory_mb": 512.0}, "step": 0, "namespace": "runtime", "event": None}
    ]


def test_resource_usage_resets_and_logs_cuda_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        resource_usage_module, "require_torch", lambda *, feature: _fake_torch(available=True, calls=calls)
    )
    context = RecordingContext()
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 512.0)

    callback.handle(Event(name="run_start", context=context))
    callback.handle(Event(name="run_end", context=context))

    assert calls == ["reset"]
    assert context.latest("runtime") == {
        "peak_memory_mb": 512.0,
        "cuda_max_memory_allocated_mb": 3.0,
        "cuda_max_memory_reserved_mb": 8.0,
        "cuda_device_count": 2,
    }


def test_resource_usage_logs_on_exception_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resource_usage_module, "require_torch", lambda *, feature: _fake_torch(available=False)
    )
    context = RecordingContext()
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 100.5)

    callback.handle(Event(name="exception", context=context, payload={"exception": RuntimeError("boom")}))

    assert context.latest("runtime") == {"peak_memory_mb": 100.5}


def test_resource_usage_omits_metrics_from_failing_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_reader() -> float:
        raise OSError("getrusage unavailable")

    monkeypatch.setattr(
        resource_usage_module, "require_torch", lambda *, feature: _fake_torch(available=False)
    )
    context = RecordingContext()
    callback = ResourceUsage(peak_rss_mb_reader=broken_reader)

    callback.handle(Event(name="run_end", context=context))

    assert context.records == []


def test_resource_usage_default_reader_returns_positive_mib() -> None:
    assert resource_usage_module._default_peak_rss_mb() > 0.0
