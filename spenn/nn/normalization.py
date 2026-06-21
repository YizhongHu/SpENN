"""Equivariant feature-scale normalization for real tuple features.

Feature normalization controls the magnitude of the per-tuple feature vectors
without mixing tuple positions, so it preserves particle-permutation
equivariance. It is deliberately separate from the basis: the basis computes
features; normalization controls their scale.

:class:`IrrepRMSNorm` is a parameter-free root-mean-square norm over the channel
axis of each block. Because the scale at a tuple position depends only on that
position's own channel values, permuting tuple positions permutes the scale
identically, so ``norm(pi x) == pi norm(x)``.

The normalization sites used by the pair-stability study are wired by
:class:`spenn.nn.SpENNWaveFunction`:

``none`` (N0)
    No normalization module is inserted.
``post_embedding`` (N1)
    ``h = norm(embedding(features))``.
``post_feature_layer`` (N2)
    ``h = norm(layer(h))`` after each feature layer.
``update`` (N3)
    ``delta = norm(update(h)); h = h + delta`` inside each layer.
``pre_readout`` (N4)
    ``output = readout(norm(h))``.
"""

from __future__ import annotations

from spenn.data.real import RealFeature
from spenn.dependencies import require_torch, require_torch_nn
from spenn.equivariance import EquivariantMap

torch = require_torch(feature="SpENN normalization modules")
nn = require_torch_nn(feature="SpENN normalization modules")

# Supported feature-normalization modes, scanned one at a time in PR8.8.
FEATURE_NORMALIZATION_MODES = (
    "none",
    "post_embedding",
    "post_feature_layer",
    "update",
    "pre_readout",
)


class IrrepRMSNorm(EquivariantMap):
    """Root-mean-square normalize each block over its channel axis.

    For every positive-order block ``x`` with shape ``[batch, channels,
    indices...]`` the output is ``x * rsqrt(mean_c x^2 + eps)`` where the mean is
    taken over the channel axis. The zero-order block (zero channels) and any
    empty block are passed through unchanged. The norm carries no learnable
    parameters, so the same module can be reused at every normalization site.

    Parameters
    ----------
    eps : float, optional
        Positive constant added to the mean square for numerical stability.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(self, *, eps: float = 1.0e-8, **kwargs) -> None:
        super().__init__(**kwargs)
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = float(eps)

    def forward_impl(self, features: RealFeature) -> RealFeature:
        """Return a per-position channel RMS normalization of every block.

        Parameters
        ----------
        features : RealFeature
            Real tuple features (or a :class:`spenn.data.real.RealUpdate`).

        Returns
        -------
        RealFeature
            A new state of the same concrete type with normalized blocks.
        """

        blocks = []
        for _order, block in features.items():
            if block.shape[1] == 0:
                # Zero-channel blocks (the order-0 block) carry no scale.
                blocks.append(block.clone())
                continue
            mean_square = block.square().mean(dim=1, keepdim=True)
            blocks.append(block * torch.rsqrt(mean_square + self.eps))
        # Preserve the concrete type (RealFeature vs RealUpdate).
        return type(features)(blocks)


class FeatureNormalization(nn.Module):
    """Typed feature-normalization choice bundling a mode and a norm module.

    This is the object :class:`spenn.nn.SpENNWaveFunction` consumes to decide
    where to insert normalization. It is a thin holder so a single
    config-selected choice carries both the insertion ``mode`` and the norm
    module, which keeps the model's public override surface scalar.

    Parameters
    ----------
    mode : str, optional
        One of :data:`FEATURE_NORMALIZATION_MODES`. ``none`` inserts no module.
    norm : torch.nn.Module or None, optional
        The normalization module applied at the selected site. Required for
        every mode other than ``none``.
    """

    def __init__(self, *, mode: str = "none", norm: nn.Module | None = None) -> None:
        super().__init__()
        mode = str(mode)
        if mode not in FEATURE_NORMALIZATION_MODES:
            raise ValueError(
                f"feature normalization mode must be one of {FEATURE_NORMALIZATION_MODES}, got {mode!r}"
            )
        if mode != "none" and norm is None:
            raise ValueError(f"feature normalization mode {mode!r} requires a norm module")
        self.mode = mode
        self.norm = norm

    def applies_at(self, site: str) -> bool:
        """Return whether normalization should run at the named site."""

        return self.mode == site

    def apply_norm(self, features: RealFeature) -> RealFeature:
        """Apply the configured norm module to a real feature state."""

        if self.norm is None:
            return features
        return self.norm(features)


__all__ = ["FEATURE_NORMALIZATION_MODES", "FeatureNormalization", "IrrepRMSNorm"]
