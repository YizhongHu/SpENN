"""Coordinate envelopes applied to a real update proposal.

An update envelope sits at :class:`~tpen.nn.TPENLayer`'s ``update_envelope``
seam: it receives the aggregated real update and the per-forward context, and
returns a modified update, before the updater combines it with the persistent
features. The seam already existed and had no implementations; this module adds
the first.

Why this is an envelope and not an updater
------------------------------------------
The A8 feature-update coordinate has three levels: ``x + u``, ``u``, and
``x + g(R) u``. The first two are pure functions of the features and the
update, and are :class:`~tpen.nn.update.Updater` implementations already. The
third needs the electron COORDINATES, which an updater never receives and
should not: giving every updater a coordinate argument to serve one arm would
change a shared interface for a single case.

``update_envelope`` already carries the context, so the gate goes there and the
Gaussian arm is spelled as the residual updater plus this envelope. Nothing in
the landed primitives is reauthored.

What this gate is NOT
---------------------
The distinction matters because several nearby things would look similar in a
plot and are different arms, or not arms at all:

- NOT ``exp(-u^2)``. The gate reads coordinates, never the update's own
  magnitude.
- NOT a channel norm and NOT RMS normalization. Those are functions of the
  update; ``tpen.nn.update.NormGatedUpdater`` is such a rule and is
  deliberately not exported, precisely so it cannot stand in for this.
- NOT a per-electron gate. It is ONE configuration-level scalar per sample,
  shared across every body order, channel and tuple.
- NOT a Gaussian multiplier on the whole wavefunction. It scales the update
  inside a layer; the ungated skip path and the Coulomb envelope are untouched.
"""

from __future__ import annotations

from tpen.data.real import Update
from tpen.dependencies import require_torch, require_torch_nn
from tpen.equivariance import EquivariantMap
from tpen.nn.context import TPENForwardContext

torch = require_torch(feature="TPEN update envelopes")
nn = require_torch_nn(feature="TPEN update envelopes")

__all__ = ["GaussianCoordinateGate", "UpdateEnvelope"]


class UpdateEnvelope(EquivariantMap):
    """Base class for coordinate-aware maps applied to a real update proposal.

    Subclasses receive the aggregated :class:`~tpen.data.real.Update` and the
    :class:`~tpen.nn.context.TPENForwardContext` for the current forward pass,
    and return a new update.
    """


class GaussianCoordinateGate(UpdateEnvelope):
    r"""Scale a real update by a Gaussian in the electron-nucleus distances.

    Implements the A8 "Gaussian residual" arm's gate

    .. math::

        g(R) = \exp\!\left[-\frac{\sum_i \lVert r_i - R_\text{nuc}\rVert^2}
                                 {2\sigma^2}\right],

    one scalar per sample, applied to every block of the update. Combined with
    :class:`~tpen.nn.update.ResidualUpdater` this gives
    ``x_next = x + g(R) u``.

    It asks one question: should the learned correction fade when the electrons
    are far from the nucleus?

    Parameters
    ----------
    sigma : float, optional
        Gate width in bohr. The study fixes this at 1.0; it is a constructor
        argument rather than a literal so a test can vary it, not because it is
        a scanned coordinate.

    Notes
    -----
    **Distances are measured from the nucleus, not from the origin.** For the
    helium control these coincide, because the literal control protocol places
    the nucleus at the origin. Measuring from the nucleus anyway is what makes
    the gate translation-covariant: a study that moved the atom would otherwise
    silently change the model rather than translate it.

    **Equivariance.** The exponent is a sum over electrons of a per-electron
    scalar, so it is invariant under electron permutation. Multiplying every
    block by one per-sample scalar therefore preserves whatever equivariance
    the update already had, rather than establishing it.

    **Single nucleus only.** The authority defines this gate for a
    single-nucleus atom, as ``r1^2 + r2^2``. It does not say what the exponent
    should be for several nuclei -- nearest nucleus, every nucleus, or a
    charge-weighted sum are all defensible and would be different models. A
    batch carrying more than one nucleus is REFUSED rather than silently
    assigned one of those conventions, because picking one here would be
    inventing science in a utility class.
    """

    def __init__(self, sigma: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        sigma = float(sigma)
        if not sigma > 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = sigma

    def gate(self, context: TPENForwardContext) -> "torch.Tensor":
        """Return the per-sample gate value.

        Parameters
        ----------
        context : TPENForwardContext
            The per-forward context, whose batch supplies electron and nuclear
            coordinates.

        Returns
        -------
        torch.Tensor
            Gate values with shape ``[batch]``.
        """

        batch = context.batch
        positions = batch.positions
        if positions.ndim < 2:
            raise ValueError(
                f"electron positions must have shape [*sample, n_electrons, dim], "
                f"got {tuple(positions.shape)}"
            )

        origin = self._nucleus(context, positions)
        # [*sample, n_electrons, dim] -> [*sample]; sum over BOTH the electron
        # axis and the spatial axis, which is the r1^2 + r2^2 of the definition
        # written out for an arbitrary electron count.
        squared = (positions - origin).square().sum(dim=(-1, -2))
        return torch.exp(-squared / (2.0 * self.sigma**2))

    def _nucleus(self, context: TPENForwardContext, positions: "torch.Tensor") -> "torch.Tensor":
        """Return the single nuclear position to measure distances from."""

        nuclear = context.batch.nuclear_positions
        if nuclear is None:
            raise ValueError(
                "GaussianCoordinateGate requires nuclear_positions on the batch; the gate "
                "is defined by electron-nucleus distances and there is no defensible "
                "default nucleus"
            )
        if nuclear.shape[-2] != 1:
            raise ValueError(
                f"GaussianCoordinateGate is defined for a single nucleus and this batch "
                f"carries {nuclear.shape[-2]}. The authority writes the gate as r1^2 + r2^2 "
                "for a one-nucleus atom and does not say whether several nuclei should use "
                "the nearest, all of them, or a charge-weighted sum -- those are different "
                "models. Refusing rather than choosing one silently"
            )
        # [..., 1, dim] already broadcasts against [*sample, n_electrons, dim].
        return nuclear.to(dtype=positions.dtype, device=positions.device)

    def forward_impl(self, u: Update, context: TPENForwardContext) -> Update:
        """Return ``u`` with every block scaled by the per-sample gate."""

        gate = self.gate(context)
        blocks = []
        for block in u.blocks:
            if block.shape[1] == 0:
                # The order-0 block is [batch, 0] and carries no channels.
                # Scaling it is a no-op that would still allocate; skip it and
                # keep the object identity contract of the other branches.
                blocks.append(block.clone())
                continue
            # Broadcast [batch] against [batch, channels, indices...].
            shaped = gate.reshape(gate.shape + (1,) * (block.ndim - gate.ndim))
            blocks.append(shaped * block)
        return Update(blocks)
