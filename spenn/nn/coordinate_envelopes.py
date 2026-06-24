"""Batch-dependent coordinate envelopes for real feature/update states."""

from __future__ import annotations

from spenn.data.batch import ElectronBatch
from spenn.data.real import RealFeature
from spenn.dependencies import require_torch, require_torch_nn
from spenn.equivariance import EquivariantMap
from spenn.nn.context import SpENNForwardContext
from spenn.nn.scalar_gates import GaussianDecayGate

torch = require_torch(feature="SpENN coordinate envelopes")
nn = require_torch_nn(feature="SpENN coordinate envelopes")


class CoordinateEnvelope(nn.Module):
    """Base class for coordinate envelopes derived from ``ElectronBatch``."""

    cache_key = "coordinate_envelope"

    def scalar(self, batch: ElectronBatch) -> torch.Tensor:
        """Return an invariant scalar with shape ``[batch]``."""

        raise NotImplementedError(f"{type(self).__name__}.scalar is not implemented")

    def forward(self, context: SpENNForwardContext) -> torch.Tensor:
        """Return a cached or newly computed coordinate gate."""

        cached = context.coordinate_envelope(self.cache_key)
        if cached is not None:
            return cached
        value = self.scalar(context.batch.flatten_samples())
        context.coordinate_envelopes[self.cache_key] = value
        return value


class GaussianCoordinateEnvelope(CoordinateEnvelope):
    """Gaussian coordinate envelope ``exp(-sum_i |r_i|^2 / (2 sigma**2))``."""

    cache_key = "gaussian"

    def __init__(self, *, sigma: float = 1.0, cache_key: str | None = None) -> None:
        super().__init__()
        self.gate = GaussianDecayGate(sigma=sigma)
        if cache_key is not None:
            self.cache_key = str(cache_key)

    @property
    def sigma(self) -> float:
        """Return the Gaussian width."""

        return self.gate.sigma

    def scalar(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the batch-level Gaussian coordinate gate."""

        radius_squared = batch.positions.square().sum(dim=(1, 2))
        return self.gate(radius_squared)


class RealCoordinateEnvelope(EquivariantMap):
    """Apply a coordinate envelope to each real feature/update block."""

    def __init__(self, envelope: CoordinateEnvelope, **kwargs) -> None:
        super().__init__(**kwargs)
        self.envelope = envelope

    def forward_impl(self, features: RealFeature, context: SpENNForwardContext) -> RealFeature:
        """Scale real blocks by a batch-dependent invariant gate."""

        gate = self.envelope(context)
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


__all__ = ["CoordinateEnvelope", "GaussianCoordinateEnvelope", "RealCoordinateEnvelope"]
