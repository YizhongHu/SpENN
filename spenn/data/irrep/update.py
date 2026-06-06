"""Irrep tensor update state."""

from __future__ import annotations

from spenn.data.irrep.feature import IrrepFeature


class IrrepUpdate(IrrepFeature):
    """Store irrep-space update proposals.

    `IrrepUpdate` currently has the same layout and behavior as
    :class:`IrrepFeature`; the distinct name marks its role in future
    update-specific maps.
    """


__all__ = ["IrrepUpdate"]
