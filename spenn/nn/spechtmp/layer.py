"""SpechtMP layer and stack scaffolds."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

from torch import nn

from spenn.data.feature_dict import FeatureDict
from spenn.nn.spechtmp.message_head import MessageHead
from spenn.nn.spechtmp.update_head import UpdateHead
from spenn.nn.utils.update import Update
from spenn.reps.branch import BranchMap
from spenn.reps.fusion import FusionMap


_DEPRECATION_MESSAGE = (
    "Legacy SpechtMP components are deprecated by the PR #3 real-space scaffold; "
    "use spenn.nn.real_space.RealSpechtMPLayer instead."
)


class SpechtMPLayer(nn.Module):
    """Compose fixed fusion, message aggregation, branching, and updating.

    Parameters
    ----------
    fusion_map : FusionMap
        Fixed tensor-product map.
    message_head : MessageHead
        Trainable message aggregation module.
    branch_map : BranchMap
        Fixed branching map.
    update_head : UpdateHead
        Trainable update aggregation module.
    update : Update
        Feature-state update module.
    norm : torch.nn.Module or None, optional
        Optional normalization applied after update construction.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        fusion_map: FusionMap,
        message_head: MessageHead,
        branch_map: BranchMap,
        update_head: UpdateHead,
        update: Update,
        norm: nn.Module | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        self.fusion_map = fusion_map
        self.message_head = message_head
        self.branch_map = branch_map
        self.update_head = update_head
        self.update = update
        self.norm = norm

    def forward(self, features: FeatureDict) -> FeatureDict:
        """Return one SpechtMP feature update.

        Parameters
        ----------
        features : FeatureDict
            Persistent feature blocks entering the layer.

        Returns
        -------
        FeatureDict
            Updated persistent feature blocks.
        """

        products = self.fusion_map(features)
        messages = self.message_head(products, features=features)
        branches = self.branch_map(messages)
        updates = self.update_head(branches, features=features)
        output = self.update(features, updates)
        if self.norm is not None:
            output = self.norm(output)
        return output


class SpechtMP(nn.Module):
    """Stack hard-coded M=2 SpechtMP layers.

    Parameters
    ----------
    layers : sequence of SpechtMPLayer
        Explicit layers to apply in order.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        layers: Sequence[SpechtMPLayer],
        **_: Any,
    ) -> None:
        super().__init__()
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        self.layers = nn.ModuleList(layers)

    def forward(self, features: FeatureDict) -> FeatureDict:
        """Apply every SpechtMP layer to a feature dictionary.

        Parameters
        ----------
        features : FeatureDict
            Persistent feature blocks entering the stack.

        Returns
        -------
        FeatureDict
            Persistent feature blocks after all layers.
        """

        for layer in self.layers:
            features = layer(features)
        return features


__all__ = ["SpechtMP", "SpechtMPLayer"]
