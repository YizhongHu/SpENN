"""SpechtMP layer and stack classes."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from spenn.data_structures.feature_dict import FeatureDict
from spenn.nn.activations import TensorProductActivation
from spenn.nn.channel_mixing import FeatureChannelMixer
from spenn.nn.spechtmp.brancher import SpechtBrancher
from spenn.nn.spechtmp.fuser import SpechtFuser


class SpechtMPLayer(nn.Module):
    """One hard-coded phase-1 SpechtMP layer."""

    def __init__(
        self,
        fuser: SpechtFuser | None = None,
        brancher: SpechtBrancher | None = None,
        activation: nn.Module | None = None,
        residual: bool = True,
        norm: nn.Module | None = None,
        intertwiner: SpechtFuser | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        if fuser is not None and intertwiner is not None:
            raise ValueError("Pass only one of fuser or intertwiner")
        self.fuser = fuser or intertwiner or SpechtFuser()
        self.brancher = brancher or SpechtBrancher()
        self.activation = activation or TensorProductActivation()
        self.residual = residual
        self.norm = norm
        self.mixer = FeatureChannelMixer()

    def forward(self, features: FeatureDict) -> FeatureDict:
        messages = self.fuser(features)
        features = self.brancher(messages, residual=features if self.residual else None)
        features = self.activation(features)
        if self.norm is not None:
            features = self.norm(features)
        return self.mixer(features)


class SpechtMP(nn.Module):
    """A stack of hard-coded phase-1 SpechtMP layers."""

    def __init__(
        self,
        M: int = 2,
        M_virtual: int = 2,
        num_layers: int = 1,
        layers: list[SpechtMPLayer] | None = None,
        residual: bool = True,
        activation: nn.Module | None = None,
        normalization: nn.Module | None = None,
        channels: Mapping | None = None,
        fixed_maps: object | None = None,
        branch_maps: object | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        if M > 2 or M_virtual > 2:
            raise ValueError("Phase 1 SpechtMP only supports M <= 2 and M_virtual <= 2")
        self.M = M
        self.M_virtual = M_virtual
        self.num_layers = num_layers
        self.channels = channels
        self.fixed_maps = fixed_maps
        self.branch_maps = branch_maps
        if layers is None:
            layers = [
                SpechtMPLayer(
                    fuser=SpechtFuser(M=M, M_virtual=M_virtual, channels=channels, fixed_maps=fixed_maps),
                    brancher=SpechtBrancher(M=M, M_virtual=M_virtual, channels=channels, branch_maps=branch_maps),
                    activation=activation or TensorProductActivation(),
                    residual=residual,
                    norm=normalization,
                )
                for _ in range(num_layers)
            ]
        self.layers = nn.ModuleList(layers)

    def forward(self, features: FeatureDict) -> FeatureDict:
        for layer in self.layers:
            features = layer(features)
        return features
