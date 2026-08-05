"""Real tuple update state and shape validators."""

from __future__ import annotations

from tpen.data.real.feature import Feature


class Update(Feature):
    """Store real-space tuple update proposal blocks.

    `Update` has the same tensor layout as :class:`Feature`, but its
    semantic role is distinct: it is an update proposal consumed by
    :class:`tpen.nn.Updater`.
    """


def validate_matching_real_blocks(feature: Feature, update: Update) -> None:
    """Validate that a real update can be applied blockwise to features.

    Parameters
    ----------
    feature : Feature
        Persistent real tuple features.
    update : Update
        Real tuple update proposal.

    Raises
    ------
    ValueError
        If the states do not have the same body-order blocks or block shapes.
    """

    feature.validate_matching_update(update)


def validate_real_update_geometry(feature: Feature, update: Update) -> None:
    """Validate real feature/update geometry while allowing channel maps.

    Parameters
    ----------
    feature : Feature
        Persistent real tuple features.
    update : Update
        Real tuple update proposal.

    Raises
    ------
    ValueError
        If the states do not share body-order blocks, batch dimensions, or
        tuple-index geometry. Channel dimensions may differ.
    """

    feature.validate_update_geometry(update)


__all__ = [
    "Update",
    "validate_matching_real_blocks",
    "validate_real_update_geometry",
]
