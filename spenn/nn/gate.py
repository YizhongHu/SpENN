"""Gate modules used by update and activation rules."""

from __future__ import annotations

from torch import nn

from spenn.data.feature_dict import FeatureDict
from spenn.data.partitions import Par


class GateUpdate(nn.Module):
    """Template for update gates.

    Parameters
    ----------
    activation : torch.nn.Module
        Tensor activation used to transform the gate signal.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, activation: nn.Module, **_: object) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, features: FeatureDict, updates: FeatureDict) -> FeatureDict:
        """Compute gates for proposed feature updates.

        Parameters
        ----------
        features : FeatureDict
            Current persistent feature state.
        updates : FeatureDict
            Proposed feature updates.

        Returns
        -------
        FeatureDict
            Gate tensors keyed like the updates they influence.

        Raises
        ------
        NotImplementedError
            Always raised by the template class.
        """

        raise NotImplementedError("GateUpdate.forward must be implemented by subclasses")


class GateActivate(nn.Module):
    """Template for feature-activation gates.

    Parameters
    ----------
    activation : torch.nn.Module
        Tensor activation used to transform the gate signal.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, activation: nn.Module, **_: object) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, features: FeatureDict) -> FeatureDict:
        """Compute gates for feature activations.

        Parameters
        ----------
        features : FeatureDict
            Feature blocks to gate.

        Returns
        -------
        FeatureDict
            Gate tensors keyed like the features they influence.

        Raises
        ------
        NotImplementedError
            Always raised by the template class.
        """

        raise NotImplementedError("GateActivate.forward must be implemented by subclasses")


class ScalarGateUpdate(GateUpdate):
    """Gate order-1 scalar updates using the scalar update signal.

    The configured activation is applied to the order-1 ``(1)`` update tensor,
    and the returned :class:`FeatureDict` contains only the order-1 ``(1)``
    gate. The current feature state is accepted for interface compatibility and
    is not used by this scalar rule.

    Parameters
    ----------
    activation : torch.nn.Module
        Tensor activation applied to ``dx_(1)``.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def forward(self, features: FeatureDict, updates: FeatureDict) -> FeatureDict:
        """Return the activated scalar update gate.

        Parameters
        ----------
        features : FeatureDict
            Current persistent feature state.
        updates : FeatureDict
            Proposed feature updates.

        Returns
        -------
        FeatureDict
            Feature dictionary containing the scalar update gate.

        Raises
        ------
        KeyError
            If the scalar update component is absent.
        """

        update = _scalar_component(updates, "updates")
        return FeatureDict({Par("H"): self.activation(update)})


class ScalarGateActivate(GateActivate):
    """Gate order-1 scalar features using the scalar feature signal.

    The configured activation is applied to the order-1 ``(1)`` feature tensor,
    and the returned :class:`FeatureDict` contains only the order-1 ``(1)``
    gate.

    Parameters
    ----------
    activation : torch.nn.Module
        Tensor activation applied to ``x_(1)``.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def forward(self, features: FeatureDict) -> FeatureDict:
        """Return the activated scalar feature gate.

        Parameters
        ----------
        features : FeatureDict
            Feature blocks to gate.

        Returns
        -------
        FeatureDict
            Feature dictionary containing the scalar feature gate.

        Raises
        ------
        KeyError
            If the scalar component is absent.
        """

        feature = _scalar_component(features, "features")
        return FeatureDict({Par("H"): self.activation(feature)})


def _scalar_component(features: FeatureDict, name: str):
    tensor = features.get(Par("H"))
    if tensor is None:
        raise KeyError(f"Missing scalar (1) component in {name}")
    return tensor


__all__ = [
    "GateActivate",
    "GateUpdate",
    "ScalarGateActivate",
    "ScalarGateUpdate",
]
