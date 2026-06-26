"""Tests for the passive trace recorder."""

from __future__ import annotations

import warnings

import pytest

from spenn.trace import Trace, TraceWarning, trace_value


class _NamedProducer:
    def __init__(self, trace_name: str | None = None) -> None:
        self.trace_name = trace_name


def test_trace_value_is_noop_when_inactive() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", TraceWarning)
        trace_value("value", slot="output", producer=_NamedProducer())  # no active trace


def test_record_uses_trace_name_then_slot() -> None:
    with Trace.capture() as trace:
        trace.record(value="v", slot="output", producer=_NamedProducer(trace_name="toy"))

    assert trace.keys() == ("toy/output",)
    assert trace["toy/output"].value == "v"


def test_explicit_key_overrides_trace_name() -> None:
    with Trace.capture() as trace:
        trace.record(value="v", slot="output", producer=_NamedProducer(trace_name="toy"), key="explicit")

    assert trace.keys() == ("explicit",)


def test_entry_indices_increase_in_recording_order() -> None:
    with Trace.capture() as trace:
        trace.record(value="a", producer=_NamedProducer(trace_name="a"))
        trace.record(value="b", producer=_NamedProducer(trace_name="b"))
        trace.record(value="c", producer=_NamedProducer(trace_name="c"))

    assert [entry.index for entry in trace] == [0, 1, 2]
    assert trace.keys() == ("a/output", "b/output", "c/output")


def test_fallback_names_are_per_class_indexed_and_warn() -> None:
    first = _NamedProducer()
    second = _NamedProducer()
    with pytest.warns(TraceWarning):
        with Trace.capture() as trace:
            trace.record(value="v0", producer=first)
            trace.record(value="v1", producer=second)

    assert trace.keys() == ("_NamedProducer0/output", "_NamedProducer1/output")
    assert trace["_NamedProducer0/output"].fallback_name is True


def test_fallback_name_is_stable_across_slots_for_one_producer() -> None:
    producer = _NamedProducer()
    with pytest.warns(TraceWarning):
        with Trace.capture() as trace:
            trace.record(value="h", slot="hidden", producer=producer)
            trace.record(value="o", slot="output", producer=producer)

    assert trace.keys() == ("_NamedProducer0/hidden", "_NamedProducer0/output")


def test_fallback_warns_once_per_producer() -> None:
    producer = _NamedProducer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", TraceWarning)
        with Trace.capture() as trace:
            trace.record(value="h", slot="hidden", producer=producer)
            trace.record(value="o", slot="output", producer=producer)

    assert len([w for w in caught if issubclass(w.category, TraceWarning)]) == 1


def test_duplicate_keys_warn_and_are_suffixed() -> None:
    with pytest.warns(TraceWarning):
        with Trace.capture() as trace:
            trace.record(value="a", key="layers.0.update/output")
            trace.record(value="b", key="layers.0.update/output")
            trace.record(value="c", key="layers.0.update/output")

    assert trace.keys() == (
        "layers.0.update/output",
        "layers.0.update/output#1",
        "layers.0.update/output#2",
    )
