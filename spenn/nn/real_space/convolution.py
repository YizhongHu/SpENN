"""Real-space Specht convolution scaffold."""

from __future__ import annotations

from typing import Any

from torch import nn

from spenn.data.real_features import RealFeature, RealMessage


class Convolution(nn.Module):
    """Build real-space messages from persistent features.

    Parameters
    ----------
    max_order : int or None, optional
        Maximum retained feature order.
    channels : object or None, optional
        Channel specification reserved for the future implementation.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, max_order: int | None = None, channels: object | None = None, **_: Any) -> None:
        super().__init__()
        self.max_order = max_order
        self.channels = channels

    def forward(self, features: RealFeature) -> RealMessage:
        """Return real-space messages from persistent features.

        Parameters
        ----------
        features : RealFeature
            Persistent real-space feature blocks.

        Returns
        -------
        RealMessage
            Real-space message blocks.

        Raises
        ------
        NotImplementedError
            Always raised until the real-space convolution math lands.
        """

        raise NotImplementedError("Convolution.forward is not implemented yet")


__all__ = ["Convolution"]
