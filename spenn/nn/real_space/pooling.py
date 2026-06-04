"""Real-space message pooling scaffold."""

from __future__ import annotations

from typing import Any

from spenn.data.base import EquivariantMap
from spenn.data.real_features import RealFeature, RealMessage


class Pooling(EquivariantMap):
    """Pool real-space messages into feature proposals.

    Parameters
    ----------
    pool_same_order : bool, optional
        Whether same-order message blocks should be copied into feature
        proposals in the placeholder implementation.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, pool_same_order: bool = True, **_: Any) -> None:
        super().__init__()
        self.pool_same_order = bool(pool_same_order)

    def forward(self, messages: RealMessage) -> RealFeature:
        """Return pooled real-space feature proposals.

        Parameters
        ----------
        messages : RealMessage
            Real-space message blocks.

        Returns
        -------
        RealFeature
            Feature proposals. The scaffold copies message blocks with matching
            partition labels when :attr:`pool_same_order` is ``True`` and
            otherwise returns an empty container.
        """

        if not self.pool_same_order:
            return RealFeature([tensor.new_empty(tensor.shape[0], 0, *tensor.shape[2:]) for tensor in messages.data])
        return RealFeature([tensor.clone() for tensor in messages.data])


__all__ = ["Pooling"]
