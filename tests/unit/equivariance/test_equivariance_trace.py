"""Tests for the passive equivariance trace recorder."""

from __future__ import annotations

import warnings

import pytest

from spenn.equivariance.trace import (
    EquivarianceTrace,
    EquivarianceTraceWarning,
    trace_equivariant,
)


class _NamedProducer:
    def __init__(self, trace_name: str | None = None) -> None:
        self.trace_name = trace_name


def test_trace_equivariant_is_noop_when_inactive() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", EquivarianceTraceWarning)
        trace_equivariant("value", slot="output", producer=_NamedProducer())  # no active trace


def test_record_uses_trace_name_then_slot() -> None:
    with EquivarianceTrace.capture() as trace:
        trace.record("v", slot="output", producer=_NamedProducer(trace_name="toy"))

    assert trace.keys() == ("toy/output",)
    assert trace["toy/output"].value == "v"


def test_explicit_key_overrides_trace_name() -> None:
    with EquivarianceTrace.capture() as trace:
        trace.record("v", slot="output", producer=_NamedProducer(trace_name="toy"), key="explicit")

    assert trace.keys() == ("explicit",)


def test_entry_indices_increase_in_recording_order() -> None:
    with EquivarianceTrace.capture() as trace:
        trace.record("a", producer=_NamedProducer(trace_name="a"))
        trace.record("b", producer=_NamedProducer(trace_name="b"))
        trace.record("c", producer=_NamedProducer(trace_name="c"))

    assert [entry.index for entry in trace] == [0, 1, 2]
    assert trace.keys() == ("a/output", "b/output", "c/output")


def test_fallback_names_are_per_class_indexed_and_warn() -> None:
    first = _NamedProducer()
    second = _NamedProducer()
    with pytest.warns(EquivarianceTraceWarning):
        with EquivarianceTrace.capture() as trace:
            trace.record("v0", producer=first)
            trace.record("v1", producer=second)

    assert trace.keys() == ("_NamedProducer0/output", "_NamedProducer1/output")
    assert trace["_NamedProducer0/output"].fallback_name is True


def test_fallback_name_is_stable_across_slots_for_one_producer() -> None:
    producer = _NamedProducer()
    with pytest.warns(EquivarianceTraceWarning):
        with EquivarianceTrace.capture() as trace:
            trace.record("h", slot="hidden", producer=producer)
            trace.record("o", slot="output", producer=producer)

    assert trace.keys() == ("_NamedProducer0/hidden", "_NamedProducer0/output")


def test_fallback_warns_once_per_producer() -> None:
    producer = _NamedProducer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", EquivarianceTraceWarning)
        with EquivarianceTrace.capture() as trace:
            trace.record("h", slot="hidden", producer=producer)
            trace.record("o", slot="output", producer=producer)

    assert len([w for w in caught if issubclass(w.category, EquivarianceTraceWarning)]) == 1


def test_duplicate_keys_warn_and_are_suffixed() -> None:
    with pytest.warns(EquivarianceTraceWarning):
        with EquivarianceTrace.capture() as trace:
            trace.record("a", key="layers.0.update/output")
            trace.record("b", key="layers.0.update/output")
            trace.record("c", key="layers.0.update/output")

    assert trace.keys() == (
        "layers.0.update/output",
        "layers.0.update/output#1",
        "layers.0.update/output#2",
    )
