"""Real tuple update state and shape validators."""

from __future__ import annotations

from spenn.data.real.feature import RealFeature


class RealUpdate(RealFeature):
    """Store real-space tuple update proposal blocks.

    `RealUpdate` has the same tensor layout as :class:`RealFeature`, but its
    semantic role is distinct: it is an update proposal consumed by
    :class:`spenn.nn.Update`.
    """


def validate_matching_real_blocks(feature: RealFeature, update: RealUpdate) -> None:
    """Validate that a real update can be applied blockwise to features.

    Parameters
    ----------
    feature : RealFeature
        Persistent real tuple features.
    update : RealUpdate
        Real tuple update proposal.

    Raises
    ------
    ValueError
        If the states do not have the same body-order blocks or block shapes.
    """

    feature.validate_matching_update(update)


def validate_real_update_geometry(feature: RealFeature, update: RealUpdate) -> None:
    """Validate real feature/update geometry while allowing channel maps.

    Parameters
    ----------
    feature : RealFeature
        Persistent real tuple features.
    update : RealUpdate
        Real tuple update proposal.

    Raises
    ------
    ValueError
        If the states do not share body-order blocks, batch dimensions, or
        tuple-index geometry. Channel dimensions may differ.
    """

    feature.validate_update_geometry(update)


__all__ = [
    "RealUpdate",
    "validate_matching_real_blocks",
    "validate_real_update_geometry",
]
