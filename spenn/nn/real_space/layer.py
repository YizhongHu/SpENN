"""Real-space SpechtMP layer scaffold."""

from __future__ import annotations

from typing import Any

from torch import nn

from spenn.data.real_features import ConcatenatedState, RealFeature, RealMessage


class RealSpechtMPLayer(nn.Module):
    """Compose one real-space SpechtMP message-passing layer.

    Parameters
    ----------
    convolution : torch.nn.Module or None, optional
        Module mapping :class:`RealFeature` to :class:`RealMessage`.
    pooling : torch.nn.Module or None, optional
        Module mapping :class:`RealMessage` to :class:`RealFeature`.
    message_activation : torch.nn.Module or None, optional
        Module mapping :class:`RealMessage` to :class:`RealMessage`.
    feature_activation : torch.nn.Module or None, optional
        Module mapping :class:`RealFeature` to :class:`RealFeature`.
    message_update : torch.nn.Module or None, optional
        Module taking ``(old, proposal)`` real-space messages.
    feature_update : torch.nn.Module or None, optional
        Module taking ``(old, proposal)`` real-space features.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        convolution: nn.Module | None = None,
        pooling: nn.Module | None = None,
        message_activation: nn.Module | None = None,
        feature_activation: nn.Module | None = None,
        message_update: nn.Module | None = None,
        feature_update: nn.Module | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.convolution = convolution
        self.pooling = pooling
        self.message_activation = message_activation
        self.feature_activation = feature_activation
        self.message_update = message_update
        self.feature_update = feature_update

    def forward(self, state: ConcatenatedState) -> ConcatenatedState:
        """Return one real-space SpechtMP state update.

        Parameters
        ----------
        state : ConcatenatedState
            Real-space feature/message state entering the layer.

        Returns
        -------
        ConcatenatedState
            Updated state with real-space features and messages replaced.

        Raises
        ------
        NotImplementedError
            If any required component has not been injected.
        TypeError
            If an injected component returns the wrong scaffold container.
        """

        convolution = self._required("convolution")
        message_proposal = convolution(state.features)
        if not isinstance(message_proposal, RealMessage):
            raise TypeError("RealSpechtMPLayer convolution must return a RealMessage")

        message_activation = self._required("message_activation")
        activated_messages = message_activation(message_proposal)
        if not isinstance(activated_messages, RealMessage):
            raise TypeError("RealSpechtMPLayer message_activation must return a RealMessage")

        message_update = self._required("message_update")
        messages = message_update(state.messages, activated_messages)
        if not isinstance(messages, RealMessage):
            raise TypeError("RealSpechtMPLayer message_update must return a RealMessage")

        pooling = self._required("pooling")
        feature_proposal = pooling(messages)
        if not isinstance(feature_proposal, RealFeature):
            raise TypeError("RealSpechtMPLayer pooling must return a RealFeature")

        feature_activation = self._required("feature_activation")
        activated_features = feature_activation(feature_proposal)
        if not isinstance(activated_features, RealFeature):
            raise TypeError("RealSpechtMPLayer feature_activation must return a RealFeature")

        feature_update = self._required("feature_update")
        features = feature_update(state.features, activated_features)
        if not isinstance(features, RealFeature):
            raise TypeError("RealSpechtMPLayer feature_update must return a RealFeature")

        return ConcatenatedState(features=features, messages=messages)

    def _required(self, name: str) -> nn.Module:
        component = getattr(self, name)
        if component is None:
            raise NotImplementedError(f"RealSpechtMPLayer.forward requires {name}")
        return component


__all__ = ["RealSpechtMPLayer"]
