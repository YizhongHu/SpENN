"""Real-space Specht activation scaffolds."""

from __future__ import annotations

from torch import nn

from spenn.data.base import EquivariantMap
from spenn.data.real_features import RealFeature, RealMessage


class SpechtMessageActivation(EquivariantMap):
    """Apply an optional activation to real-space messages.

    Parameters
    ----------
    activation : torch.nn.Module or None, optional
        Activation module. If ``None``, messages are cloned unchanged.
    """

    def __init__(self, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, messages: RealMessage) -> RealMessage:
        """Return activated real-space messages.

        Parameters
        ----------
        messages : RealMessage
            Message blocks to activate.

        Returns
        -------
        RealMessage
            Activated message blocks.

        Raises
        ------
        TypeError
            If the wrapped activation does not return a `RealMessage`.
        """

        if self.activation is None:
            return messages.clone()
        activated = self.activation(messages)
        if not isinstance(activated, RealMessage):
            raise TypeError("SpechtMessageActivation activation must return a RealMessage")
        return activated


class SpechtFeatureActivation(EquivariantMap):
    """Apply an optional activation to real-space features.

    Parameters
    ----------
    activation : torch.nn.Module or None, optional
        Activation module. If ``None``, features are cloned unchanged.
    """

    def __init__(self, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, features: RealFeature) -> RealFeature:
        """Return activated real-space features.

        Parameters
        ----------
        features : RealFeature
            Feature blocks to activate.

        Returns
        -------
        RealFeature
            Activated feature blocks.

        Raises
        ------
        TypeError
            If the wrapped activation does not return a `RealFeature`.
        """

        if self.activation is None:
            return features.clone()
        activated = self.activation(features)
        if not isinstance(activated, RealFeature):
            raise TypeError("SpechtFeatureActivation activation must return a RealFeature")
        return activated


__all__ = ["SpechtFeatureActivation", "SpechtMessageActivation"]
