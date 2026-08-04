"""Batch-dependent coordinate envelopes for real feature/update states."""

from __future__ import annotations

from spenn.data.batch import ElectronBatch
from spenn.data.real import Feature
from spenn.dependencies import require_torch, require_torch_nn
from spenn.equivariance import EquivariantMap
from spenn.nn.context import TPENForwardContext

torch = require_torch(feature="SpENN coordinate envelopes")
nn = require_torch_nn(feature="SpENN coordinate envelopes")


class GaussianDecayGate(nn.Module):
    """Return ``exp(-x / (2 sigma**2))`` elementwise."""

    def __init__(self, *, sigma: float = 1.0) -> None:
        super().__init__()
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = float(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the Gaussian decay gate without density normalization."""

        scale = 2.0 * self.sigma * self.sigma
        return torch.exp(-x / scale)


class CoordinateEnvelope(EquivariantMap):
    """Batch-derived multiplicative envelope that owns its application.

    Per decision D14, an envelope is a multiplicative factor ``env(r)`` on the
    feature/update it modifies, and the envelope object owns the
    multiplication — there is no separate producer/applier split. Subclasses
    implement :meth:`scalar`; this base class broadcast-multiplies the
    invariant scalar onto every real block, which preserves permutation
    equivariance because the gate is invariant and shared over particles.

    The former per-forward context cache was dropped with the collapse: it was
    keyed by a class-level ``cache_key`` rather than by envelope parameters,
    so two envelopes with different widths could silently share one cached
    gate. The scalar is cheap to recompute per call.
    """

    def scalar(self, batch: ElectronBatch) -> torch.Tensor:
        """Return an invariant scalar with shape ``[batch]``."""

        raise NotImplementedError(f"{type(self).__name__}.scalar is not implemented")

    def forward_impl(self, features: Feature, context: TPENForwardContext) -> Feature:
        """Scale real blocks by the batch-dependent invariant gate."""

        gate = self.scalar(context.batch.flatten_samples())
        if gate.ndim != 1:
            raise ValueError(f"coordinate envelope must have shape [batch], got {tuple(gate.shape)}")
        blocks = []
        for order, block in features.items():
            if block.shape[0] != gate.shape[0]:
                raise ValueError(
                    "coordinate envelope batch size does not match real block: "
                    f"{gate.shape[0]} vs {block.shape[0]}"
                )
            view = gate.reshape(gate.shape[0], 1, *([1] * order)).to(
                device=block.device,
                dtype=block.dtype,
            )
            blocks.append(block * view)
        return type(features)(blocks)


class GaussianCoordinateEnvelope(CoordinateEnvelope):
    """Gaussian coordinate envelope ``exp(-sum_i |r_i|^2 / (2 sigma**2))``."""

    def __init__(self, *, sigma: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gate = GaussianDecayGate(sigma=sigma)

    @property
    def sigma(self) -> float:
        """Return the Gaussian width."""

        return self.gate.sigma

    def scalar(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the batch-level Gaussian coordinate gate."""

        radius_squared = batch.positions.square().sum(dim=(1, 2))
        return self.gate(radius_squared)


__all__ = ["CoordinateEnvelope", "GaussianCoordinateEnvelope", "GaussianDecayGate"]
