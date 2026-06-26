"""Scalar gates used by irrep, real, and coordinate scale controls."""

from __future__ import annotations

from spenn.dependencies import require_torch, require_torch_nn

torch = require_torch(feature="SpENN scalar gates")
nn = require_torch_nn(feature="SpENN scalar gates")


class ScalarGate(nn.Module):
    """Base class for scalar gate modules."""


class RMSInverseGate(ScalarGate):
    """Return ``(x + eps)^-1/2`` elementwise."""

    def __init__(self, *, eps: float = 1.0e-8) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the RMS inverse gate."""

        return torch.rsqrt(x + self.eps)


class GaussianDecayGate(ScalarGate):
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


class SigmoidGate(ScalarGate):
    """Apply ``torch.sigmoid`` elementwise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the sigmoid gate."""

        return torch.sigmoid(x)


class TanhGate(ScalarGate):
    """Apply ``torch.tanh`` elementwise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the tanh gate."""

        return torch.tanh(x)


__all__ = [
    "GaussianDecayGate",
    "RMSInverseGate",
    "ScalarGate",
    "SigmoidGate",
    "TanhGate",
]
