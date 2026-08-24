"""Typed observable trajectories with explicit ``[draw, walker]`` axes.

The statistics domain owns the draw axis, so the container that carries it
lives here rather than in whichever module happens to build one first. The
sampling-side collector (:mod:`tpen.sampling.trajectory`) imports this type; it
does not define its own.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

__all__ = ["ObservableTrajectory", "ObservableTrajectoryReconciliation"]


@dataclass(frozen=True)
class ObservableTrajectoryReconciliation:
    """Deterministic identity and moments for a ``[draw, walker]`` observable."""

    observable: str
    draw_count: int
    walker_count: int
    draw_stride: int
    burn_in_draws: int
    row_count: int
    finite_count: int
    mean: float
    variance: float
    values_content_id: str

    @property
    def nonfinite_count(self) -> int:
        """Return the number of non-finite observable rows."""

        return self.row_count - self.finite_count


@dataclass(frozen=True)
class ObservableTrajectory:
    """Scalar observable samples laid out as ``[draw, walker]``.

    Walker columns are independent chains. They are stored side by side and are
    never concatenated along the draw axis: chain boundaries are real
    discontinuities, and flattening them fabricates a slowly-decaying time
    series out of independent samples.

    Values are coerced to ``float64`` and detached on construction. Every
    downstream estimator sums many lag products, and ``float32`` loses the tail
    of the autocorrelation function to rounding well before the plateau.

    Parameters
    ----------
    observable : str
        Name of the sampled observable, for example ``"local_energy"``.
        Autocorrelation is observable-specific -- the energy IAT does not bound
        the gradient IAT -- so the name is part of the trajectory identity.
    values : torch.Tensor
        Two-dimensional tensor indexed ``[draw, walker]``.
    draw_stride : int
        Sampler steps advanced between consecutive retained draws. A stride of
        one means every step is a draw.
    burn_in_draws : int
        Draws discarded after the sampler's own burn-in and before ``values``
        begins. Recorded so a consumer can tell a short chain from a heavily
        trimmed one.

    Attributes
    ----------
    n_draws : int
        Draws retained per chain.
    n_walkers : int
        Number of independent chains.
    total_draws : int
        ``n_draws * n_walkers``.
    nonfinite_count : int
        Number of non-finite entries. Non-finite draws are never dropped:
        removing an element from a time series silently re-indexes every lag
        after it, so the producer reports ``unresolved`` instead.

    Raises
    ------
    ValueError
        If the observable name is blank, the tensor is not two-dimensional,
        either axis is empty, the stride is below one, or the burn-in count is
        negative.
    """

    observable: str
    values: torch.Tensor
    draw_stride: int
    burn_in_draws: int

    def __post_init__(self) -> None:
        observable = str(self.observable).strip()
        if not observable:
            raise ValueError("observable must be a non-empty name")

        if not isinstance(self.values, torch.Tensor):
            raise TypeError(f"values must be a torch.Tensor, got {type(self.values).__name__}")
        if self.values.ndim != 2:
            raise ValueError(
                "values must be two-dimensional [draw, walker]; got shape "
                f"{tuple(self.values.shape)}. Trajectories are never flattened: "
                "a flat vector cannot distinguish walker boundaries from lags."
            )
        n_draws, n_walkers = self.values.shape
        if n_draws < 1 or n_walkers < 1:
            raise ValueError(f"values must have at least one draw and one walker; got {(n_draws, n_walkers)}")

        draw_stride = int(self.draw_stride)
        if draw_stride < 1:
            raise ValueError(f"draw_stride must be at least 1, got {draw_stride}")
        burn_in_draws = int(self.burn_in_draws)
        if burn_in_draws < 0:
            raise ValueError(f"burn_in_draws must be non-negative, got {burn_in_draws}")

        # Coerce once, here, so every consumer sees the same dtype regardless of
        # whether the trajectory came from a sampler or a test fixture.
        object.__setattr__(self, "observable", observable)
        object.__setattr__(self, "values", self.values.detach().to(torch.float64))
        object.__setattr__(self, "draw_stride", draw_stride)
        object.__setattr__(self, "burn_in_draws", burn_in_draws)

    @property
    def n_draws(self) -> int:
        """Return the number of draws retained per chain."""
        return int(self.values.shape[0])

    @property
    def n_walkers(self) -> int:
        """Return the number of independent chains."""
        return int(self.values.shape[1])

    @property
    def total_draws(self) -> int:
        """Return the total number of samples across all chains."""
        return self.n_draws * self.n_walkers

    @property
    def nonfinite_count(self) -> int:
        """Return the number of non-finite entries in the trajectory."""
        return int((~torch.isfinite(self.values)).sum().item())

    def reconciliation(self) -> ObservableTrajectoryReconciliation:
        """Return shape, moments, finite count, and content identity.

        The moment reduction is explicitly draw-major. A streaming record
        writer can reproduce it from one retained draw at a time without
        holding the full row artifact in memory.
        """

        values = self.values.detach().to(device="cpu", dtype=torch.float64)
        draw_sums = values.sum(dim=1)
        draw_square_sums = values.square().sum(dim=1)
        total = draw_sums.sum()
        total_square = draw_square_sums.sum()
        row_count = self.total_draws
        mean = float((total / row_count).item())
        variance = float((total_square / row_count - (total / row_count).square()).item())
        if math.isfinite(variance) and variance < 0.0:
            # Roundoff can make an exactly constant trajectory microscopically
            # negative under the sum/sum-of-squares identity.
            variance = 0.0
        return ObservableTrajectoryReconciliation(
            observable=self.observable,
            draw_count=self.n_draws,
            walker_count=self.n_walkers,
            draw_stride=self.draw_stride,
            burn_in_draws=self.burn_in_draws,
            row_count=row_count,
            finite_count=row_count - self.nonfinite_count,
            mean=mean,
            variance=variance,
            values_content_id=_values_content_id(values),
        )


def _values_content_id(values: torch.Tensor) -> str:
    """Hash canonical float64 values in draw-major, walker-minor order."""

    canonical = values.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(canonical.shape)).encode("utf-8"))
    digest.update(canonical.numpy().astype("<f8", copy=False).tobytes())
    return digest.hexdigest()
