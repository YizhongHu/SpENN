"""Tests for split-chain mixing diagnostics."""

from __future__ import annotations

import pytest
import torch

from tpen.statistics.mixing import MixingDiagnostics, split_r_hat


def _assert_unresolved(result: MixingDiagnostics) -> str:
    """Assert that an unresolved diagnostic carries a reason, not an R-hat.

    Parameters
    ----------
    result : MixingDiagnostics
        Result expected to be unresolved.

    Returns
    -------
    str
        The non-empty unresolved reason for path-specific checks.
    """

    assert result.r_hat is None
    assert result.reason is not None
    assert result.reason.strip()
    return result.reason


def test_well_mixed_independent_chains_have_unit_split_r_hat() -> None:
    """Keep split-Rhat near one for independent chains from one law."""
    generator = torch.Generator().manual_seed(2701)
    values = torch.randn((4096, 8), generator=generator, dtype=torch.float64)

    result = split_r_hat(values)

    assert result.r_hat is not None
    assert 0.99 < result.r_hat < 1.05


def test_separated_chain_means_produce_large_split_r_hat() -> None:
    """Expose between-chain disagreement through a large split-Rhat."""
    generator = torch.Generator().manual_seed(2702)
    noise = torch.randn((2048, 4), generator=generator, dtype=torch.float64)
    values = noise + torch.tensor([-5.0, -5.0, 5.0, 5.0], dtype=torch.float64)

    result = split_r_hat(values)

    assert result.r_hat is not None
    # Per-chain autocorrelation should ignore these offsets, but R-hat owns the
    # between-chain disagreement and must expose it strongly.
    assert result.r_hat > 4.0


def test_single_walker_step_change_is_detected_by_splitting() -> None:
    """Detect non-stationarity between the halves of a single walker."""
    generator = torch.Generator().manual_seed(2703)
    first = -4.0 + torch.randn((1024, 1), generator=generator, dtype=torch.float64)
    second = 4.0 + torch.randn((1024, 1), generator=generator, dtype=torch.float64)
    values = torch.cat((first, second), dim=0)

    result = split_r_hat(values)

    assert values.shape[1] == 1
    assert result.n_split_chains == 2
    assert result.r_hat is not None
    assert result.r_hat > 4.0


@pytest.mark.parametrize(
    ("n_draws", "n_walkers"),
    [pytest.param(20, 3, id="even"), pytest.param(21, 4, id="odd")],
)
def test_split_metadata_matches_input_axes(n_draws: int, n_walkers: int) -> None:
    """Report two half-chains per walker and floor-divided draw counts."""
    generator = torch.Generator().manual_seed(2704)
    values = torch.randn((n_draws, n_walkers), generator=generator, dtype=torch.float64)

    result = split_r_hat(values)

    assert result.n_split_chains == 2 * n_walkers
    assert result.draws_per_split_chain == n_draws // 2


def test_odd_draw_count_drops_the_oldest_draw() -> None:
    """Drop the oldest unmatched draw before splitting an odd trajectory."""
    # Dropping the oldest 999 leaves identical halves [0, 1] and [0, 1]. Each
    # has unbiased variance 1/2 and their between-chain variance is zero, so
    # split-Rhat is exactly sqrt((half - 1) / half) = sqrt(1/2). Dropping the
    # newest draw instead would retain the outlier and fail this invariant.
    values = torch.tensor([999.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float64).reshape(-1, 1)

    result = split_r_hat(values)

    assert result.n_split_chains == 2
    assert result.draws_per_split_chain == 2
    assert result.r_hat == pytest.approx(2.0**-0.5, rel=0.0, abs=1.0e-14)


def test_fewer_than_four_draws_is_unresolved() -> None:
    """Withhold split-Rhat when each half would have fewer than two draws."""
    generator = torch.Generator().manual_seed(2705)
    values = torch.randn((3, 2), generator=generator, dtype=torch.float64)

    reason = _assert_unresolved(split_r_hat(values))

    assert "insufficient draws" in reason


def test_all_constant_chains_are_unresolved() -> None:
    """Withhold split-Rhat when every half-chain has zero variance."""
    reason = _assert_unresolved(split_r_hat(torch.ones((16, 3))))

    assert "constant" in reason.lower()


@pytest.mark.parametrize(
    "nonfinite",
    [pytest.param(float("nan"), id="nan"), pytest.param(float("inf"), id="inf")],
)
def test_nonfinite_values_are_unresolved(nonfinite: float) -> None:
    """Withhold split-Rhat rather than dropping a non-finite draw."""
    generator = torch.Generator().manual_seed(2706)
    values = torch.randn((16, 3), generator=generator, dtype=torch.float64)
    values[5, 1] = nonfinite

    reason = _assert_unresolved(split_r_hat(values))

    assert "non-finite" in reason


def test_mixing_diagnostics_rejects_both_estimate_and_reason() -> None:
    """Reject a mixing result that claims both resolution and failure."""
    with pytest.raises(ValueError, match="exactly one"):
        MixingDiagnostics(
            r_hat=1.0,
            n_split_chains=4,
            draws_per_split_chain=8,
            reason="contradictory",
        )


def test_mixing_diagnostics_rejects_neither_estimate_nor_reason() -> None:
    """Reject a mixing result that records neither resolution nor failure."""
    with pytest.raises(ValueError, match="exactly one"):
        MixingDiagnostics(
            r_hat=None,
            n_split_chains=4,
            draws_per_split_chain=8,
            reason=None,
        )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5,), id="one-dimensional"),
        pytest.param((0, 2), id="empty-draws"),
        pytest.param((3, 0), id="empty-walkers"),
    ],
)
def test_split_r_hat_rejects_invalid_shapes(shape) -> None:
    """Require non-empty tensors with explicit draw and walker axes."""
    with pytest.raises(ValueError):
        split_r_hat(torch.empty(shape, dtype=torch.float64))
