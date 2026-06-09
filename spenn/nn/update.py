"""Real-feature update modules."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from spenn.data.real import (
    RealFeature,
    RealUpdate,
    common_real_dtype,
    validate_matching_real_blocks,
    validate_real_update_geometry,
)
from spenn.equivariance import EquivariantMap


class Update(EquivariantMap):
    """Base class for real-feature update rules.

    Subclasses map a persistent :class:`RealFeature` and a proposed
    :class:`RealUpdate` to the next persistent feature state.
    """


class ReplaceUpdate(Update):
    """Replace persistent real features with a real update proposal."""

    def forward_impl(self, x: RealFeature, u: RealUpdate) -> RealFeature:
        """Return the update proposal as the next real feature state."""

        validate_matching_real_blocks(x, u)
        return RealFeature([tensor.clone() for tensor in u.blocks])


class ResidualUpdate(Update):
    """Add a scaled real update proposal to persistent features."""

    def __init__(self, step: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.step = float(step)

    def forward_impl(self, x: RealFeature, u: RealUpdate) -> RealFeature:
        """Return ``x + step * u`` blockwise."""

        validate_matching_real_blocks(x, u)
        return RealFeature([left + self.step * right for left, right in zip(x.blocks, u.blocks)])


class NormGatedUpdate(Update):
    """Gate a residual update by an equivariant per-tuple update norm."""

    def __init__(self, step: float = 1.0, eps: float = 1.0e-12, **kwargs) -> None:
        super().__init__(**kwargs)
        self.step = float(step)
        self.eps = float(eps)

    def forward_impl(self, x: RealFeature, u: RealUpdate) -> RealFeature:
        """Return a norm-gated residual update."""

        validate_matching_real_blocks(x, u)
        output = []
        for feature, update in zip(x.blocks, u.blocks):
            if update.shape[1] == 0:
                output.append(feature.clone())
                continue
            norm = update.square().mean(dim=1, keepdim=True).clamp_min(self.eps).sqrt()
            gate = torch.sigmoid(norm)
            output.append(feature + self.step * gate * update)
        return RealFeature(output)


class ChannelMappedUpdate(Update):
    """Add a channel-mapped real update proposal to persistent features.

    The learned map is shared across all tuple positions within each body
    order. This preserves particle equivariance because only channel axes are
    mixed.

    Parameters
    ----------
    step : float, optional
        Scalar multiplier for the mapped update.
    max_order : int
        Maximum positive body order to initialize.
    channels : int or mapping
        Persistent feature channels per body order.
    update_channels : int, mapping, or None, optional
        Real update channels per body order. If ``None``, uses `channels`.
    initial_weight : float, optional
        Initial value for non-identity channel maps.
    identity_init : bool, optional
        If ``True``, same-size channel maps start as identity matrices.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        step: float = 1.0,
        *,
        max_order: int,
        channels: int | Mapping[int, int],
        update_channels: int | Mapping[int, int] | None = None,
        initial_weight: float = 0.0,
        identity_init: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.step = float(step)
        self.max_order = int(max_order)
        if self.max_order <= 0:
            raise ValueError(f"max_order must be positive, got {self.max_order}")
        self.channels_by_order = _normalize_positive_channels(channels, max_order=self.max_order, name="channels")
        self.update_channels_by_order = _normalize_positive_channels(
            channels if update_channels is None else update_channels,
            max_order=self.max_order,
            name="update_channels",
        )
        self.initial_weight = float(initial_weight)
        self.identity_init = bool(identity_init)
        self.channel_maps = nn.ParameterDict()
        self._initialize_channel_maps()

    def forward_impl(self, x: RealFeature, u: RealUpdate) -> RealFeature:
        """Return ``x + step * W_m u_m`` for every body order ``m``."""

        validate_real_update_geometry(x, u)
        common_real_dtype(x, u)
        output = []
        for order, (feature, update) in enumerate(zip(x.blocks, u.blocks)):
            if feature.shape[1] == 0:
                output.append(feature.clone())
                continue
            weight = self._weight_for_order(
                order,
                out_channels=int(feature.shape[1]),
                in_channels=int(update.shape[1]),
                device=feature.device,
                dtype=feature.dtype,
            )
            mapped = torch.einsum("oc,bc...->bo...", weight, update)
            output.append(feature + self.step * mapped)
        return RealFeature(output)

    def _weight_for_order(
        self,
        order: int,
        *,
        out_channels: int,
        in_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = str(order)
        shape = (out_channels, in_channels)
        if key not in self.channel_maps:
            raise RuntimeError(f"Missing eager ChannelMappedUpdate map for order {order}")
        weight = self.channel_maps[key]
        if tuple(weight.shape) != shape:
            raise ValueError(f"Order-{order} channel map shape {tuple(weight.shape)} does not match {shape}")
        return weight.to(device=device, dtype=dtype)

    def _initialize_channel_maps(self) -> None:
        for order in range(1, self.max_order + 1):
            shape = (self.channels_by_order[order], self.update_channels_by_order[order])
            initial = torch.full(shape, self.initial_weight)
            if self.identity_init and shape[0] == shape[1]:
                initial = torch.eye(shape[0])
            self.channel_maps[str(order)] = nn.Parameter(initial)


def _normalize_positive_channels(
    value: int | Mapping[int, int],
    *,
    max_order: int,
    name: str,
) -> dict[int, int]:
    if isinstance(value, Mapping):
        channels = {int(order): int(count) for order, count in value.items()}
        missing = [order for order in range(1, max_order + 1) if order not in channels]
        if missing:
            raise ValueError(f"{name} is missing orders {missing}")
    else:
        count = int(value)
        channels = {order: count for order in range(1, max_order + 1)}
    for order, count in channels.items():
        if order < 1 or order > max_order:
            raise ValueError(f"{name} contains order {order} outside [1, {max_order}]")
        if count <= 0:
            raise ValueError(f"{name}[{order}] must be positive, got {count}")
    return dict(sorted(channels.items()))


__all__ = ["ChannelMappedUpdate", "NormGatedUpdate", "ReplaceUpdate", "ResidualUpdate", "Update"]
