"""Pointwise activation functions owned by the TPEN layer stages.

This module owns the ``Gamma`` / ``Gamma_c`` pointwise functions that
:class:`tpen.nn.EquivariantMixing`, :class:`tpen.nn.PathAggregation`, and
:class:`tpen.nn.MLP` accept through their ``activation`` argument. Per decision
D2 activation is an arbitrary function on the index axes owned by those stages,
and there is no gated activation anywhere in TPEN. Per decision D3 the first
activation form is elementwise with ``C_out == C_in``; channel mixing stays in
the mixing weights ``W``.

The D-series decision log lives in Task Orchestrator notes on the TPEN project
root (``decision-log-D1-D9`` and ``decision-log-D10-D18``); the former
``TPEN-MIGRATION.md`` was folded into them and removed on 2026-08-14.

Modules here are therefore plain :class:`torch.nn.Module` pointwise maps, *not*
:class:`tpen.equivariance.EquivariantMap` subclasses: they carry no typed
real-state contract of their own, and the equivariant module is the stage that
owns them.
"""

from __future__ import annotations

import math

from tpen.dependencies import require_torch, require_torch_nn

torch = require_torch(feature="TPEN activations")
nn = require_torch_nn(feature="TPEN activations")


class GaussianActivation(nn.Module):
    """Elementwise Gaussian activation ``f(x) = exp(-x**2 / (2 sigma**2))``.

    A shape-preserving pointwise map: every entry is transformed independently,
    so ``C_out == C_in`` and no channel, path, or particle axis is mixed (D3).
    That is what makes it safe to pass as the ``activation`` of an equivariant
    stage — an elementwise map commutes with particle permutation.

    This is deliberately *not*
    :class:`tpen.nn.coordinate_envelopes.GaussianDecayGate`, which is
    ``exp(-x / (2 sigma**2))`` and is only ever fed non-negative squared radii;
    that form diverges for negative inputs and so cannot be used on signed
    feature blocks.

    ``sigma`` is a fixed constructor float held as a plain attribute, not a
    :class:`torch.nn.Parameter`: the module has no learnable state, and the
    width is architecture metadata rather than something training moves.

    Parameters
    ----------
    sigma : float, optional
        Positive, finite Gaussian width. Defaults to ``1.0``.

    Raises
    ------
    ValueError
        If `sigma` is not positive and finite.

    Notes
    -----
    ``f(0) == 1``, so this activation is one of the ``Gamma(0) != 0`` choices
    described in :class:`tpen.nn.EquivariantMixing`: applied to a full mixing
    block it writes the invariant constant ``1`` onto the non-distinct tuple
    entries that mixing never writes.

    The mathematical range is ``(0, 1]``, but ``exp`` underflows to exactly
    ``0.0`` once ``x**2 / (2 sigma**2)`` exceeds the floating-point exponent
    range (``|x| > ~38.6`` at ``sigma == 1`` in ``float64``). Values stay
    finite and in ``[0, 1]`` for every representable input whose square is
    finite.
    """

    def __init__(self, *, sigma: float = 1.0) -> None:
        super().__init__()
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma must be positive and finite, got {sigma}")
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Gaussian bump elementwise, preserving shape, dtype, device.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            ``exp(-x**2 / (2 sigma**2))`` with the same shape, dtype, and
            device as `x`.
        """

        # 2 sigma^2 is the full denominator of the Gaussian exponent; folding
        # the factor 2 in here keeps a single division on the hot path.
        scale = 2.0 * self.sigma * self.sigma
        return torch.exp(-x.square() / scale)

    def extra_repr(self) -> str:
        """Return the width in ``repr`` so configured modules are readable."""

        return f"sigma={self.sigma}"


__all__ = ["GaussianActivation"]
