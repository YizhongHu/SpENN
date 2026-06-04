"""Order-1 and order-2 electron feature encoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spenn.data.batch import ElectronBatch
from spenn.data.irrep_features import FeatureDict
from spenn.data.partitions import Par, Partition
from spenn.nn.encoding.base import BaseEncoder
from spenn.nn.utils.mlp import MLP


class ElectronPairEncoder(BaseEncoder):
    """Encode learned particle-tuple scalars into order-1 and order-2 features.

    Parameters
    ----------
    max_order : int, optional
        Maximum initial feature order. Only values up to ``2`` are accepted.
    channels : sequence of int or None, optional
        Channel counts by body order. ``channels[0]`` is ignored and treated
        as zero, ``channels[1]`` controls order-1 scalar tuple channels, and
        ``channels[2]`` controls order-2 scalar tuple channels.
    name : str, optional
        Human-readable encoder name.
    hidden_channels : int, optional
        Width of hidden layers in each tuple MLP.
    num_hidden_layers : int, optional
        Number of hidden layers in each tuple MLP.
    activation : torch.nn.Module or None, optional
        Activation module copied between hidden layers. If ``None``, SiLU is
        used.
    include_spins : bool, optional
        Whether to append ``batch.spins[..., None]`` to particle descriptors
        when spin labels are available.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        max_order: int = 2,
        channels: object | None = None,
        name: str = "basic",
        hidden_channels: int = 64,
        num_hidden_layers: int = 2,
        activation: nn.Module | None = None,
        include_spins: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        if max_order > 2:
            raise ValueError("ElectronPairEncoder only supports max_order <= 2")
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be nonnegative")
        self.max_order = max_order
        self.channels = channels
        self.name = name
        self.hidden_channels = hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.include_spins = include_spins

        self.order_channels = _channel_counts(channels, max_order=max_order)
        self.order1_channels = self.order_channels[1] if max_order >= 1 else 0
        self.order2_channels = self.order_channels[2] if max_order >= 2 else 0

        self.order1_encoder = self._make_tuple_mlp(self.order1_channels)
        self.order2_encoder = self._make_tuple_mlp(self.order2_channels if max_order >= 2 else 0)

    def output_keys(self) -> tuple[Partition, ...]:
        """Return the feature keys produced by the encoder.

        Returns
        -------
        tuple of Partition
            Partition keys for active order-1 and order-2 channels.
        """

        keys: list[Partition] = []
        if self.order1_channels and self.order1_channels > 0:
            keys.append(Par("H"))
        if self.max_order >= 2 and self.order2_channels > 0:
            keys.append(Par("S"))
            keys.append(Par("A"))
        return tuple(keys)

    def raw_features(self, batch: ElectronBatch) -> list[torch.Tensor]:
        """Return learned scalar tuple tensors ``q``.

        Parameters
        ----------
        batch : ElectronBatch
            Electron positions and optional metadata.

        Returns
        -------
        list of torch.Tensor
            Tuple scalar tensors indexed by order. ``q[0]`` has shape
            ``[batch, 0]``, ``q[1]`` has shape ``[batch, channel, n]``, and
            ``q[2]`` has shape ``[batch, channel, n, n]`` when ``max_order`` is
            at least two.
        """

        batch = batch.flatten_samples()
        descriptors = self.particle_descriptors(batch)
        q = [descriptors.new_empty(batch.batch_size, 0)]
        if self.max_order >= 1:
            q.append(self.encode_order1(descriptors))
        if self.max_order >= 2:
            q.append(self.encode_order2(descriptors))
        assert q[0].shape == (batch.batch_size, 0)
        if self.max_order >= 1:
            assert q[1].shape == (batch.batch_size, self.order1_channels, batch.n_electrons)
        if self.max_order >= 2:
            assert q[2].shape == (batch.batch_size, self.order2_channels, batch.n_electrons, batch.n_electrons)
        return q

    def forward(self, batch: ElectronBatch) -> FeatureDict:
        """Encode an electron batch into order-1 and order-2 features.

        Parameters
        ----------
        batch : ElectronBatch
            Electron positions and optional metadata.

        Returns
        -------
        FeatureDict
            Ordered-tuple feature blocks.
        """

        batch = batch.flatten_samples()
        q = self.raw_features(batch)
        features = FeatureDict()
        if self.order1_channels and self.order1_channels > 0:
            features.set(Par("H"), q[1].unsqueeze(-1).unsqueeze(-1))
        if self.max_order >= 2 and self.order2_channels > 0:
            pair_symmetric = 0.5 * (q[2] + q[2].transpose(2, 3))
            pair_antisymmetric = 0.5 * (q[2] - q[2].transpose(2, 3))
            features.set(Par("S"), pair_symmetric.unsqueeze(-1).unsqueeze(-1))
            features.set(Par("A"), pair_antisymmetric.unsqueeze(-1).unsqueeze(-1))
        features.validate(batch_size=batch.batch_size, n_electrons=batch.n_electrons, supported=self.output_keys())
        return features

    def particle_descriptors(self, batch: ElectronBatch) -> torch.Tensor:
        """Return per-particle encoder descriptors.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch.

        Returns
        -------
        torch.Tensor
            Particle descriptors with shape ``[batch, n_electrons, features]``.
        """

        descriptors = batch.positions
        if self.include_spins and batch.spins is not None:
            descriptors = torch.cat((descriptors, batch.spins.to(device=batch.device, dtype=batch.dtype).unsqueeze(-1)), dim=-1)
        assert descriptors.shape[:2] == (batch.batch_size, batch.n_electrons)
        return descriptors

    def encode_order1(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode order-1 tuple scalars.

        Parameters
        ----------
        positions : torch.Tensor
            Electron positions with shape ``[batch, n_electrons, spatial_dim]``.

        Returns
        -------
        torch.Tensor
            Order-1 scalar tuple tensor with shape
            ``[batch, channels[1], n_electrons]``.
        """

        if self.order1_encoder is None or self.order1_channels <= 0:
            return positions.new_empty(*positions.shape[:2], 0).movedim(-1, 1)
        encoded = self.apply_tuple_mlp(self.order1_encoder, positions)
        output = encoded.movedim(-1, 1)
        assert output.shape == (positions.shape[0], self.order1_channels, positions.shape[1])
        return output

    def encode_order2(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode order-2 tuple scalars.

        Parameters
        ----------
        positions : torch.Tensor
            Electron positions with shape ``[batch, n_electrons, spatial_dim]``.

        Returns
        -------
        torch.Tensor
            Order-2 scalar tuple tensor with shape
            ``[batch, channels[2], n_electrons, n_electrons]``.
        """

        batch_size, n_electrons, spatial_dim = positions.shape
        if self.order2_encoder is None or self.order2_channels <= 0:
            return positions.new_empty(batch_size, 0, n_electrons, n_electrons)
        left = positions.unsqueeze(2).expand(batch_size, n_electrons, n_electrons, spatial_dim)
        right = positions.unsqueeze(1).expand(batch_size, n_electrons, n_electrons, spatial_dim)
        tuple_inputs = torch.cat((left, right), dim=-1)
        encoded = self.apply_tuple_mlp(self.order2_encoder, tuple_inputs).movedim(-1, 1)
        diagonal = torch.eye(n_electrons, device=positions.device, dtype=torch.bool)
        output = encoded.masked_fill(diagonal.view(1, 1, n_electrons, n_electrons), 0)
        assert output.shape == (batch_size, self.order2_channels, n_electrons, n_electrons)
        assert torch.all(output.diagonal(dim1=2, dim2=3) == 0)
        return output

    def apply_tuple_mlp(self, encoder: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Apply a tuple MLP on the input dtype and device.

        Parameters
        ----------
        encoder : torch.nn.Module
            Tuple MLP to apply.
        inputs : torch.Tensor
            Tuple input tensor.

        Returns
        -------
        torch.Tensor
            Encoded tuple tensor.
        """

        encoder = encoder.to(device=inputs.device, dtype=inputs.dtype)
        output = encoder(inputs)
        assert output.shape[:-1] == inputs.shape[:-1]
        return output

    def _make_tuple_mlp(self, channels: int) -> MLP | None:
        if channels <= 0:
            return None
        return MLP(
            channels,
            hidden_channels=self.hidden_channels,
            num_hidden_layers=self.num_hidden_layers,
            activation=self.activation,
        )


def _channel_counts(channels: object | None, *, max_order: int) -> list[int]:
    if channels is None:
        values = [0] + [32] * max_order
    else:
        if isinstance(channels, dict):
            raise TypeError("ElectronPairEncoder.channels must be a sequence indexed by body order")
        try:
            values = [int(value) for value in channels]
        except (TypeError, ValueError) as exc:
            raise TypeError("ElectronPairEncoder.channels must be a sequence indexed by body order") from exc
    counts = [0] * (max_order + 1)
    for order in range(1, max_order + 1):
        counts[order] = values[order] if order < len(values) else 0
        if counts[order] < 0:
            raise ValueError("ElectronPairEncoder channel counts must be nonnegative")
    return counts


__all__ = ["ElectronPairEncoder"]
