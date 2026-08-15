"""Tests for the typed observable trajectory container."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from tpen.statistics.trajectory import ObservableTrajectory


def test_construction_exposes_shape_metadata_and_detaches_float64_values() -> None:
    """Preserve draw/walker axes while normalising storage for statistics."""
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)

    trajectory = ObservableTrajectory("energy", values, draw_stride=3, burn_in_draws=2)

    assert trajectory.n_draws == 2
    assert trajectory.n_walkers == 2
    assert trajectory.total_draws == 4
    assert trajectory.values.dtype == torch.float64
    assert trajectory.values.requires_grad is False
    torch.testing.assert_close(trajectory.values, values.detach().to(torch.float64))


@pytest.mark.parametrize(
    ("observable", "values", "draw_stride", "burn_in_draws", "match"),
    [
        ("   ", torch.ones(1, 1), 1, 0, "non-empty"),
        ("energy", torch.ones(3), 1, 0, "two-dimensional"),
        ("energy", torch.ones(1, 1, 1), 1, 0, "two-dimensional"),
        ("energy", torch.ones(0, 1), 1, 0, "at least one"),
        ("energy", torch.ones(1, 0), 1, 0, "at least one"),
        ("energy", torch.ones(1, 1), 0, 0, "at least 1"),
        ("energy", torch.ones(1, 1), 1, -1, "non-negative"),
    ],
)
def test_validation_rejects_invalid_trajectory_metadata_or_shape(
    observable: str,
    values: torch.Tensor,
    draw_stride: int,
    burn_in_draws: int,
    match: str,
) -> None:
    """Reject malformed trajectories before they can lose chain semantics."""
    with pytest.raises(ValueError, match=match):
        ObservableTrajectory(observable, values, draw_stride, burn_in_draws)


def test_one_dimensional_error_explains_why_flattening_is_forbidden() -> None:
    """Make the walker-boundary guardrail visible to callers."""
    with pytest.raises(ValueError, match="flattened.*walker boundaries|walker boundaries.*lags"):
        ObservableTrajectory("energy", torch.ones(4), 1, 0)


def test_non_tensor_values_raise_type_error() -> None:
    """Require the typed tensor contract instead of coercing arbitrary containers."""
    with pytest.raises(TypeError, match="torch.Tensor"):
        ObservableTrajectory("energy", [[1.0]], 1, 0)


def test_observable_name_is_stripped() -> None:
    """Canonicalise identity strings at the boundary."""
    trajectory = ObservableTrajectory("  local_energy  ", torch.ones(1, 1), 1, 0)

    assert trajectory.observable == "local_energy"


def test_nonfinite_count_counts_nan_and_infinity() -> None:
    """Report contamination without dropping entries from the time series."""
    contaminated = ObservableTrajectory("energy", torch.tensor([[float("nan"), float("inf")]]), 1, 0)
    clean = ObservableTrajectory("energy", torch.ones(1, 2), 1, 0)

    assert contaminated.nonfinite_count == 2
    assert clean.nonfinite_count == 0


def test_content_sha256_is_stable_equal_and_lowercase_hex() -> None:
    """Equal trajectory identities must address the same durable content."""
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    first = ObservableTrajectory("energy", values, 2, 1)
    second = ObservableTrajectory("energy", values.clone(), 2, 1)

    assert first.content_sha256 == first.content_sha256
    assert first.content_sha256 == second.content_sha256
    assert len(first.content_sha256) == 64
    assert first.content_sha256 == first.content_sha256.lower()
    assert all(character in "0123456789abcdef" for character in first.content_sha256)


@pytest.mark.parametrize(
    "change",
    [
        lambda values: ObservableTrajectory("other", values, 2, 1),
        lambda values: ObservableTrajectory("energy", values, 3, 1),
        lambda values: ObservableTrajectory("energy", values, 2, 2),
        lambda values: ObservableTrajectory("energy", values + torch.tensor([[0.0, 1.0], [0.0, 0.0]]), 2, 1),
    ],
)
def test_content_sha256_changes_for_each_identity_or_value_change(change) -> None:
    """Prevent distinct measurements from silently sharing receipt content."""
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    baseline = ObservableTrajectory("energy", values, 2, 1)

    assert change(values).content_sha256 != baseline.content_sha256


def test_content_sha256_is_independent_of_tensor_layout() -> None:
    """Canonical bytes must not depend on whether the input view is contiguous."""
    contiguous = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    noncontiguous = torch.tensor([[1.0, 3.0], [2.0, 4.0]]).transpose(0, 1)
    assert noncontiguous.is_contiguous() is False

    first = ObservableTrajectory("energy", contiguous, 1, 0)
    second = ObservableTrajectory("energy", noncontiguous, 1, 0)

    assert first.content_sha256 == second.content_sha256


def test_trajectory_is_frozen() -> None:
    """Prevent post-construction mutation of durable trajectory identity."""
    trajectory = ObservableTrajectory("energy", torch.ones(1, 1), 1, 0)

    with pytest.raises(FrozenInstanceError):
        trajectory.observable = "changed"
