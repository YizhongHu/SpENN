"""Tests for the run-level resource-usage callback.

Migrated off the ``run_start``/``run_end``/``exception``/``run_failed`` strings
onto `tpen.run_events` (item ``39eacd99``). These call ``handle_occurrence``
directly, which is this callback's own contract. Delivery of those occurrences
by the real dispatcher, and their emission by `tpen.run`, are covered in
``test_typed_run_lifecycle.py``.
"""

from __future__ import annotations

import pytest

from tpen.callback import ResourceUsage
from tpen.callback import resource_usage as resource_usage_module
from tpen.events import Occurrence
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tests.unit.callback.support import RecordingContext


def _deliver(callback: ResourceUsage, context: RecordingContext, event: object) -> None:
    """Hand one run-lifecycle occurrence to the callback."""

    callback.handle_occurrence(Occurrence(event=event, count=1), context)


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


def test_resource_usage_logs_peak_rss_at_run_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resource_usage_module, "require_torch", lambda *, feature: _fake_torch(available=False)
    )
    context = RecordingContext()
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 512.0)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    assert context.by_namespace("runtime") == [
        {"metrics": {"peak_memory_mb": 512.0}, "step": 0, "namespace": "runtime"}
    ]


def test_resource_usage_resets_and_logs_cuda_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        resource_usage_module, "require_torch", lambda *, feature: _fake_torch(available=True, calls=calls)
    )
    context = RecordingContext()
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 512.0)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    assert calls == ["reset"]
    assert context.latest("runtime") == {
        "peak_memory_mb": 512.0,
        "cuda_max_memory_allocated_mb": 3.0,
        "cuda_max_memory_reserved_mb": 8.0,
        "cuda_device_count": 2,
    }


def test_resource_usage_logs_at_the_failure_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """One typed `RunFailed` replaces the ``run_failed``/``exception`` pair.

    Both strings carried the same payload and this callback answered both, so a
    failed run logged ``runtime`` twice. The single record here is the observable
    difference.
    """

    monkeypatch.setattr(
        resource_usage_module, "require_torch", lambda *, feature: _fake_torch(available=False)
    )
    context = RecordingContext()
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 100.5)

    _deliver(callback, context, RunFailed(exception_type="RuntimeError", exception_message="boom"))

    assert context.by_namespace("runtime") == [
        {"metrics": {"peak_memory_mb": 100.5}, "step": 0, "namespace": "runtime"}
    ]


def test_resource_usage_omits_metrics_from_failing_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_reader() -> float:
        raise OSError("getrusage unavailable")

    monkeypatch.setattr(
        resource_usage_module, "require_torch", lambda *, feature: _fake_torch(available=False)
    )
    context = RecordingContext()
    callback = ResourceUsage(peak_rss_mb_reader=broken_reader)

    _deliver(callback, context, RunCompleted())

    assert context.records == []


def test_resource_usage_default_reader_returns_positive_mib() -> None:
    assert resource_usage_module._default_peak_rss_mb() > 0.0
