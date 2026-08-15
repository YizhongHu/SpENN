"""Typed observable trajectories with explicit ``[draw, walker]`` axes.

The statistics domain owns the draw axis, so the container that carries it
lives here rather than in whichever module happens to build one first. The
sampling-side collector (:mod:`tpen.sampling.trajectory`) imports this type; it
does not define its own.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

__all__ = ["ObservableTrajectory"]


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
    content_sha256 : str
        Content address of the exact numbers these statistics were computed
        from, used as ``source_artifact_sha256`` on the emitted receipt.

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

    @property
    def content_sha256(self) -> str:
        """Return a sha256 over the trajectory's identity and exact values.

        The digest covers the observable name, shape, stride and burn-in as
        well as the raw ``float64`` bytes, so two trajectories collide only if
        they carry the same numbers under the same layout.
        """

        digest = hashlib.sha256()
        header = (
            f"observable={self.observable}\n"
            f"n_draws={self.n_draws}\n"
            f"n_walkers={self.n_walkers}\n"
            f"draw_stride={self.draw_stride}\n"
            f"burn_in_draws={self.burn_in_draws}\n"
        )
        digest.update(header.encode("utf-8"))
        # Little-endian float64 on CPU: a fixed wire layout, so the digest does
        # not change when the same trajectory is produced on a different device.
        contiguous = self.values.cpu().contiguous().numpy()
        digest.update(contiguous.astype("<f8", copy=False).tobytes())
        return digest.hexdigest()
