"""Trainable embedding from electron batches to real tuple features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from spenn.data.batch import ElectronBatch
from spenn.data.indices import no_repeated_particle_mask, tuple_particle_inputs
from spenn.data.real import RealFeature, zero_block
from spenn.equivariance import EquivariantMap
from spenn.nn.mlp import MLP


class Embedding(EquivariantMap):
    """Encode raw non-repeating particle-vector tuples with per-order MLPs.

    For each body order ``m``, the input to the order-specific MLP is the
    concatenation ``(v_{i_1}, ..., v_{i_m})`` over ordered non-repeating tuple
    indices. By default ``v_i`` contains the electron coordinate and, when
    present, its spin. Extra per-particle vectors can be appended from
    ``ElectronBatch.aux``. Repeated tuple positions are stored as zeros in the
    dense :class:`RealFeature` output. No handcrafted geometry channels such as
    distances, radii, or inverse distances are constructed here.

    Parameters
    ----------
    max_order : int, optional
        Highest body order to return.
    out_channels : int or mapping, optional
        Output channels per order for generated MLPs.
    hidden_channels : int, optional
        Hidden width for generated MLPs.
    num_hidden_layers : int, optional
        Number of hidden layers for generated MLPs.
    activation : torch.nn.Module or None, optional
        Activation copied into generated MLPs. If ``None``, SiLU is used.
    bias : bool, optional
        Whether generated MLP linear layers include bias terms.
    mlps : mapping of int to torch.nn.Module or None, optional
        Explicit per-order modules. Missing orders are filled with generated
        :class:`MLP` instances.
    include_spins : bool, optional
        If ``True``, append ``ElectronBatch.spins`` to the per-particle vector
        whenever spins are present.
    aux_feature_keys : sequence of str, optional
        Keys in ``ElectronBatch.aux`` whose values are per-particle feature
        tensors with shape ``[*sample_shape, n_electrons, channels]``.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        max_order: int = 3,
        *,
        out_channels: int | Mapping[int, int] = 16,
        hidden_channels: int = 64,
        num_hidden_layers: int = 2,
        activation: nn.Module | None = None,
        bias: bool = True,
        mlps: Mapping[int, nn.Module] | None = None,
        include_spins: bool = True,
        aux_feature_keys: Sequence[str] = (),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if max_order < 1:
            raise ValueError(f"max_order must be positive, got {max_order}")
        self.max_order = int(max_order)
        self.out_channels = {int(order): int(channels) for order, channels in out_channels.items()} if isinstance(out_channels, Mapping) else int(out_channels)
        self.include_spins = bool(include_spins)
        self.aux_feature_keys = tuple(str(key) for key in aux_feature_keys)
        self.order_mlps = nn.ModuleDict()
        supplied = {} if mlps is None else {int(order): module for order, module in mlps.items()}
        for order in range(1, self.max_order + 1):
            module = supplied.get(order)
            if module is None:
                module = MLP(
                    out_channels=self._out_channels(order),
                    hidden_channels=hidden_channels,
                    num_hidden_layers=num_hidden_layers,
                    activation=activation,
                    bias=bias,
                )
            self.order_mlps[str(order)] = module
        unknown = sorted(order for order in supplied if order < 1 or order > self.max_order)
        if unknown:
            raise ValueError(f"mlps contains orders outside [1, {self.max_order}]: {unknown}")

    def forward_impl(self, batch: ElectronBatch) -> RealFeature:
        """Embed an electron batch as persistent real tuple features."""

        flat = batch.flatten_samples()
        if self.max_order > flat.n_electrons:
            raise ValueError(
                f"Embedding max_order={self.max_order} exceeds n_electrons={flat.n_electrons}"
            )
        particle_vectors = _particle_vectors(
            flat,
            include_spins=self.include_spins,
            aux_feature_keys=self.aux_feature_keys,
        )
        blocks = [zero_block(batch_size=flat.batch_size, device=flat.device, dtype=flat.dtype)]
        for order in range(1, self.max_order + 1):
            inputs = tuple_particle_inputs(particle_vectors, order)
            mlp = self.order_mlps[str(order)].to(device=flat.device, dtype=flat.dtype)
            block = mlp(inputs).movedim(-1, 1)
            block = block * no_repeated_particle_mask(flat.n_electrons, order, device=flat.device).reshape(
                1,
                1,
                *((flat.n_electrons,) * order),
            ).to(dtype=block.dtype)
            blocks.append(block)
        return RealFeature(blocks)

    def _out_channels(self, order: int) -> int:
        if isinstance(self.out_channels, dict):
            try:
                channels = self.out_channels[order]
            except KeyError as exc:
                raise KeyError(f"Missing out_channels for order {order}") from exc
        else:
            channels = self.out_channels
        if channels <= 0:
            raise ValueError(f"Embedding out_channels must be positive, got {channels}")
        return int(channels)


def _particle_vectors(
    batch: ElectronBatch,
    *,
    include_spins: bool,
    aux_feature_keys: Sequence[str],
) -> torch.Tensor:
    """Return dense per-particle vectors for an electron batch."""

    features = [batch.positions]
    if include_spins and batch.spins is not None:
        features.append(batch.spins.unsqueeze(-1).to(dtype=batch.positions.dtype))
    for key in aux_feature_keys:
        if key not in batch.aux:
            raise KeyError(f"ElectronBatch.aux is missing particle feature key {key!r}")
        value = batch.aux[key]
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=batch.device)
        tensor = tensor.to(device=batch.device, dtype=batch.dtype)
        if tensor.ndim != 3 or tuple(tensor.shape[:2]) != (batch.batch_size, batch.n_electrons):
            raise ValueError(
                f"ElectronBatch.aux[{key!r}] must have shape [batch, n_electrons, channels], "
                f"got {tuple(tensor.shape)}"
            )
        features.append(tensor)
    return torch.cat(features, dim=-1)


__all__ = ["Embedding"]
