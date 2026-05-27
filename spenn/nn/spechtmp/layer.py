"""SpechtMP layer and stack scaffolds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

from spenn.data.feature_dict import FeatureDict
from spenn.nn.spechtmp.message_head import MessageHead
from spenn.nn.spechtmp.update_head import UpdateHead
from spenn.nn.update import RawUpdate, Update
from spenn.reps.branch import BranchMap
from spenn.reps.fusion import FusionMap


class SpechtMPLayer(nn.Module):
    """Compose fixed fusion, message aggregation, branching, and updating.

    Parameters
    ----------
    fusion_map : FusionMap or None, optional
        Fixed tensor-product map. If ``None``, a default M=2 scaffold is used.
    message_head : MessageHead or None, optional
        Trainable message aggregation module. If ``None``, a default M=2
        scaffold is used.
    branch_map : BranchMap or None, optional
        Fixed branching map. If ``None``, a default M=2 scaffold is used.
    update_head : UpdateHead or None, optional
        Trainable update aggregation module. If ``None``, a default M=2
        scaffold is used.
    update : Update or None, optional
        Feature-state update module. If ``None``, raw updates replace the
        incoming features.
    norm : torch.nn.Module or None, optional
        Optional normalization applied after update construction.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        fusion_map: FusionMap | None = None,
        message_head: MessageHead | None = None,
        branch_map: BranchMap | None = None,
        update_head: UpdateHead | None = None,
        update: Update | None = None,
        norm: nn.Module | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.fusion_map = fusion_map or FusionMap()
        self.message_head = message_head or MessageHead()
        self.branch_map = branch_map or BranchMap()
        self.update_head = update_head or UpdateHead()
        self.update = update or RawUpdate()
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
    M : int, optional
        Maximum retained feature order. Only values up to ``2`` are accepted in
        this scaffold.
    M_virtual : int, optional
        Maximum virtual tensor-product order. Only values up to ``2`` are
        accepted in this scaffold.
    num_layers : int, optional
        Number of SpechtMP layers to create when `layers` is ``None``.
    layers : sequence of SpechtMPLayer or None, optional
        Explicit layers to use instead of constructing defaults.
    update_head : UpdateHead or None, optional
        Trainable update aggregation module passed to default layers. If
        ``None``, a default M=2 scaffold is used.
    update : Update or None, optional
        Feature-state update module passed to default layers. If ``None``, raw
        updates replace incoming features.
    activation : torch.nn.Module or None, optional
        Activation passed to default message heads.
    normalization : torch.nn.Module or None, optional
        Normalization passed to default layers.
    channels : mapping or None, optional
        Channel specification passed to default message heads.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        M: int = 2,
        M_virtual: int = 2,
        num_layers: int = 1,
        layers: Sequence[SpechtMPLayer] | None = None,
        update_head: UpdateHead | None = None,
        update: Update | None = None,
        activation: nn.Module | None = None,
        normalization: nn.Module | None = None,
        channels: object | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if M > 2 or M_virtual > 2:
            raise ValueError("SpechtMP scaffold only supports M <= 2 and M_virtual <= 2")
        self.M = M
        self.M_virtual = M_virtual
        self.num_layers = num_layers
        self.channels = channels
        if layers is None:
            layers = [
                SpechtMPLayer(
                    fusion_map=FusionMap(M=M, M_virtual=M_virtual),
                    message_head=MessageHead(
                        M=M,
                        M_virtual=M_virtual,
                        channels=channels,
                        activation=activation,
                    ),
                    branch_map=BranchMap(M=M, M_virtual=M_virtual),
                    update_head=update_head
                    if update_head is not None
                    else UpdateHead(M=M, channels=channels, activation=activation),
                    update=update,
                    norm=normalization,
                )
                for _ in range(num_layers)
            ]
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
