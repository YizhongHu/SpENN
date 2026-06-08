"""Tests for passive output tracing in EquivariantMap.forward."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

from spenn.equivariance import EquivariantMap
from spenn.equivariance.trace import EquivarianceTrace, EquivarianceTraceWarning


class ToyMap(EquivariantMap):
    def forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class HiddenMap(EquivariantMap):
    def forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x + 1
        self.trace("hidden", hidden)
        return hidden * 2


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = ToyMap()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def _x() -> torch.Tensor:
    return torch.arange(3, dtype=torch.float64)


def test_forward_is_normal_and_silent_without_active_trace() -> None:
    module = ToyMap(trace_name="toy")
    with warnings.catch_warnings():
        warnings.simplefilter("error", EquivarianceTraceWarning)
        output = module(_x())

    torch.testing.assert_close(output, _x() * 2)


def test_active_trace_records_named_output() -> None:
    module = ToyMap(trace_name="toy")
    with EquivarianceTrace.capture() as trace:
        module(_x())

    assert trace.keys() == ("toy/output",)
    torch.testing.assert_close(trace["toy/output"].value, _x() * 2)


def test_trace_output_false_records_nothing() -> None:
    module = ToyMap(trace_name="toy", trace_output=False)
    with EquivarianceTrace.capture() as trace:
        module(_x())

    assert len(trace) == 0


def test_capture_with_model_uses_module_path() -> None:
    model = ToyModel()
    with EquivarianceTrace.capture(model=model) as trace:
        model(_x())

    assert trace.keys() == ("layer/output",)


def test_explicit_trace_name_overrides_module_path() -> None:
    model = ToyModel()
    model.layer.trace_name = "custom_layer"
    with EquivarianceTrace.capture(model=model) as trace:
        model(_x())

    assert "custom_layer/output" in trace
    assert "layer/output" not in trace


def test_unnamed_map_uses_fallback_name_and_warns() -> None:
    first = ToyMap()
    second = ToyMap()
    with pytest.warns(EquivarianceTraceWarning):
        with EquivarianceTrace.capture() as trace:
            first(_x())
            second(_x())

    assert trace.keys() == ("ToyMap0/output", "ToyMap1/output")


def test_manual_internal_trace_records_hidden_and_output() -> None:
    model = nn.Module()
    model.add_module("layer", HiddenMap())
    with EquivarianceTrace.capture(model=model) as trace:
        model.layer(_x())

    assert trace.keys() == ("layer/hidden", "layer/output")
    assert [entry.index for entry in trace] == [0, 1]
