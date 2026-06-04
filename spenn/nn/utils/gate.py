"""Gate modules used by update and activation rules."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.irrep_features import FeatureDict, MessageDict
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

    def forward(self, features: FeatureDict | MessageDict) -> FeatureDict:
        """Compute gates for feature activations.

        Parameters
        ----------
        features : FeatureDict or MessageDict
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

    def forward(self, features: FeatureDict | MessageDict) -> FeatureDict:
        """Return the activated scalar feature gate.

        Parameters
        ----------
        features : FeatureDict or MessageDict
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


class NormGateActivate(GateActivate):
    """Gate features by smooth functions of irrep-coordinate norms.

    The norm is computed over the transforming irrep-coordinate ``alpha`` axis
    while preserving batch, channel, ordered-tuple, and multiplicity/Fourier
    ``beta`` axes. The returned gate has shape ``[..., 1, beta]`` for each
    feature block and is projected onto the full permutation-safe block by
    :class:`spenn.nn.utils.activations.GatedActivation`.

    Parameters
    ----------
    activation : torch.nn.Module
        Scalar activation applied to each local norm.
    eps : float, optional
        Positive lower bound used when ``normalize=True``.
    normalize : bool, optional
        If ``True``, return ``activation(norm) / max(norm, eps)``. If
        ``False``, return ``activation(norm)``. For smoothness at zero with
        ``normalize=False``, use an activation satisfying ``activation(0) == 0``.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        activation: nn.Module,
        eps: float = 1.0e-12,
        normalize: bool = True,
        **_: object,
    ) -> None:
        super().__init__(activation)
        self.eps = float(eps)
        self.normalize = bool(normalize)

    def forward(self, features: FeatureDict | MessageDict) -> FeatureDict:
        """Return norm-derived gates for every feature block.

        Parameters
        ----------
        features : FeatureDict or MessageDict
            Feature or message blocks to gate.

        Returns
        -------
        FeatureDict
            Gate tensors keyed like `features`.
        """

        gates = FeatureDict()
        for partition, tensor in features.flat_items():
            norm = torch.linalg.vector_norm(tensor, dim=-2, keepdim=True)
            gate = self.activation(norm)
            if gate.shape != norm.shape:
                raise ValueError(
                    f"Norm gate for partition {partition} must preserve norm shape {tuple(norm.shape)}, "
                    f"got {tuple(gate.shape)}"
                )
            if self.normalize:
                gate = gate / norm.clamp_min(self.eps)
            gates.set(partition, gate)
        return gates


def _scalar_component(features: FeatureDict | MessageDict, name: str):
    tensor = features.get(Par("H"))
    if tensor is None:
        raise KeyError(f"Missing scalar (1) component in {name}")
    return tensor


__all__ = [
    "GateActivate",
    "GateUpdate",
    "NormGateActivate",
    "ScalarGateActivate",
    "ScalarGateUpdate",
]
