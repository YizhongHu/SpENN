"""Real feature/update gates that preserve typed real-state semantics."""

from __future__ import annotations

from spenn.data.real import RealFeature
from spenn.dependencies import require_torch_nn
from spenn.equivariance import EquivariantMap
from spenn.nn.scalar_gates import GaussianDecayGate, RMSInverseGate

nn = require_torch_nn(feature="SpENN real gates")


class RealNormGate(EquivariantMap):
    """Gate each real block by a scalar function of its channel norm.

    The statistic is the per-position channel mean square with shape
    ``[batch, 1, tuple...]``. The output preserves the concrete input type,
    including ``RealFeature`` versus ``RealUpdate``.
    """

    def __init__(self, gate: nn.Module, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gate = gate

    def forward_impl(self, features: RealFeature) -> RealFeature:
        """Apply the configured norm gate blockwise."""

        blocks = []
        for _order, block in features.items():
            if block.shape[1] == 0:
                blocks.append(block.clone())
                continue
            statistic = block.square().mean(dim=1, keepdim=True)
            gate = self.gate(statistic)
            if tuple(gate.shape) != tuple(statistic.shape):
                raise ValueError(
                    "RealNormGate gate must preserve statistic shape "
                    f"{tuple(statistic.shape)}, got {tuple(gate.shape)}"
                )
            blocks.append(block * gate)
        return type(features)(blocks)


class RealRMSGate(RealNormGate):
    """Real norm gate using ``(s + eps)^-1/2``."""

    def __init__(self, *, eps: float = 1.0e-8, **kwargs) -> None:
        super().__init__(RMSInverseGate(eps=eps), **kwargs)


class RealGaussianNormGate(RealNormGate):
    """Real norm gate using ``exp(-s / (2 sigma**2))``."""

    def __init__(self, *, sigma: float = 1.0, **kwargs) -> None:
        super().__init__(GaussianDecayGate(sigma=sigma), **kwargs)


__all__ = ["RealGaussianNormGate", "RealNormGate", "RealRMSGate"]
