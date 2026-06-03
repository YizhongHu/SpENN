"""Tests for reusable production-sample statistics."""

from __future__ import annotations

import math

import pytest
import torch

from spenn.diagnostics.statistics import (
    autocorrelation_by_lag,
    effective_sample_size,
    integrated_autocorrelation_time,
)


def test_autocorrelation_by_lag_handles_one_dimensional_samples() -> None:
    samples = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)

    correlation = autocorrelation_by_lag(samples, max_lag=2)

    assert correlation.shape == (3,)
    assert correlation.dtype == samples.dtype
    assert correlation.device == samples.device
    assert torch.isclose(correlation[0], torch.tensor(1.0, dtype=samples.dtype))


def test_integrated_autocorrelation_time_stops_at_negative_lag() -> None:
    samples = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)

    tau = integrated_autocorrelation_time(samples)
    ess = effective_sample_size(samples)

    assert tau == 1.0
    assert ess == 4.0


def test_effective_sample_size_supports_chain_axis() -> None:
    samples = torch.tensor(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        dtype=torch.float64,
    )

    ess = effective_sample_size(samples)

    assert ess == 8.0


def test_constant_samples_return_nan_statistics() -> None:
    samples = torch.ones(4, 2, dtype=torch.float64)

    correlation = autocorrelation_by_lag(samples)

    assert torch.isnan(correlation).all()
    assert math.isnan(integrated_autocorrelation_time(samples))
    assert math.isnan(effective_sample_size(samples))


def test_statistics_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        autocorrelation_by_lag(torch.zeros(2, 2, 2))
