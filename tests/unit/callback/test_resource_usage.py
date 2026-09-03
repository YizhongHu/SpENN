"""Tests for the run-level resource-usage callback.

Migrated off the ``run_start``/``run_end``/``exception``/``run_failed`` strings
onto `tpen.run_events` (item ``39eacd99``). These call ``handle_occurrence``
directly, which is this callback's own contract. Delivery of those occurrences
by the real dispatcher, and their emission by `tpen.run`, are covered in
``test_typed_run_lifecycle.py``.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from tpen.callback import ResourceUsage
from tpen.callback import resource_usage as resource_usage_module
from tpen.events import Occurrence
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tpen.process_resources import (
    ProcessRUsageProbe,
    ProcessResourceBaseline,
    ProcessResourceResult,
    ResourceScope,
    ResourceUnavailable,
)
from tpen import process_resources as process_resources_module
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


def test_process_probe_reports_counter_deltas_and_linux_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    readings = iter(
        (
            SimpleNamespace(
                ru_utime=1.25,
                ru_stime=0.5,
                ru_inblock=4,
                ru_oublock=6,
                ru_nvcsw=8,
                ru_nivcsw=10,
                ru_maxrss=1024,
            ),
            SimpleNamespace(
                ru_utime=2.0,
                ru_stime=0.75,
                ru_inblock=9,
                ru_oublock=7,
                ru_nvcsw=11,
                ru_nivcsw=14,
                ru_maxrss=2048,
            ),
        )
    )
    monkeypatch.setattr(process_resources_module.resource, "getrusage", lambda scope: next(readings))
    monkeypatch.setattr(process_resources_module.sys, "platform", "linux")

    probe = ProcessRUsageProbe(ResourceScope.PROCESS)
    baseline = probe.read()
    result = probe.result(baseline)

    assert result.user_cpu_seconds == pytest.approx(0.75)
    assert result.system_cpu_seconds == pytest.approx(0.25)
    assert result.read_block_operations == 5
    assert result.write_block_operations == 1
    assert result.voluntary_context_switches == 3
    assert result.involuntary_context_switches == 4
    assert result.peak_rss_mb == pytest.approx(2.0)


def test_process_probe_preserves_unavailable_counter_evidence_when_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(scope):
        raise OSError("getrusage unavailable")

    monkeypatch.setattr(process_resources_module.resource, "getrusage", fail)

    result = ProcessRUsageProbe().read()

    assert all(
        isinstance(value, ResourceUnavailable)
        for value in (
            result.user_cpu_seconds,
            result.system_cpu_seconds,
            result.read_block_operations,
            result.write_block_operations,
            result.voluntary_context_switches,
            result.involuntary_context_switches,
            result.peak_rss_mb,
        )
    )


def test_process_probe_normalizes_macos_peak_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        process_resources_module.resource,
        "getrusage",
        lambda scope: SimpleNamespace(
            ru_utime=0.0,
            ru_stime=0.0,
            ru_inblock=0,
            ru_oublock=0,
            ru_nvcsw=0,
            ru_nivcsw=0,
            ru_maxrss=2 * 1024 * 1024,
        ),
    )
    monkeypatch.setattr(process_resources_module.sys, "platform", "darwin")

    assert ProcessRUsageProbe().read().peak_rss_mb == pytest.approx(2.0)


class _FixedProbe(ProcessRUsageProbe):
    """Probe stand-in that makes callback boundary assertions deterministic."""

    def __init__(self, result: ProcessResourceResult) -> None:
        super().__init__()
        self.result_value = result

    def read(self) -> ProcessResourceBaseline:
        return ProcessResourceBaseline(0, 0, 0, 0, 0, 0, 0)

    def result(self, baseline: ProcessResourceBaseline) -> ProcessResourceResult:
        return self.result_value


def _fixed_result(*, unavailable: bool = False) -> ProcessResourceResult:
    value = ResourceUnavailable("probe failed") if unavailable else 1
    return ProcessResourceResult(value, value, value, value, value, value, value)


def test_resource_usage_logs_process_metrics_at_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resource_usage_module, "require_torch", lambda *, feature: _fake_torch(available=False)
    )
    context = RecordingContext()
    callback = ResourceUsage(process_probe=_FixedProbe(_fixed_result()), peak_rss_mb_reader=lambda: 4.0)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    assert context.latest("process") == {
        "process_user_cpu_seconds": 1.0,
        "process_system_cpu_seconds": 1.0,
        "process_read_block_operations": 1.0,
        "process_write_block_operations": 1.0,
        "process_voluntary_context_switches": 1.0,
        "process_involuntary_context_switches": 1.0,
    }
    assert context.latest("runtime")["peak_memory_mb"] == 4.0


def test_resource_usage_logs_process_receipt_at_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resource_usage_module, "require_torch", lambda *, feature: _fake_torch(available=False)
    )
    context = RecordingContext()
    callback = ResourceUsage(process_probe=_FixedProbe(_fixed_result()), peak_rss_mb_reader=lambda: 4.0)

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunFailed(exception_type="RuntimeError", exception_message="boom"))

    assert len(context.by_namespace("process")) == 1
    assert len(context.by_namespace("runtime")) == 1


def test_resource_usage_projects_unavailable_process_readings_as_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resource_usage_module, "require_torch", lambda *, feature: _fake_torch(available=False)
    )
    context = RecordingContext()
    callback = ResourceUsage(process_probe=_FixedProbe(_fixed_result(unavailable=True)))

    _deliver(callback, context, RunStarted())
    _deliver(callback, context, RunCompleted())

    process = context.latest("process")
    assert process["process_user_cpu_seconds_unavailable"] is True
    assert process["process_peak_rss_unavailable"] is True
