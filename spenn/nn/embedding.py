"""Trainable embedding from electron batches to real tuple features."""

from __future__ import annotations

from collections.abc import Mapping

from spenn.data.batch import ElectronBatch
from spenn.data.indices import no_repeated_particle_mask, tuple_particle_inputs
from spenn.data.real import RealFeature, zero_block
from spenn.nn.basis import ElectronBasisFeatures
from spenn.dependencies import require_torch, require_torch_nn
from spenn.equivariance import EquivariantMap
from spenn.nn.mlp import MLP

torch = require_torch(feature="SpENN embedding modules")
nn = require_torch_nn(feature="SpENN embedding modules")


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
    spatial_dim : int
        Coordinate dimension of each particle vector.
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
        If ``True``, append ``ElectronBatch.spins`` to the per-particle vector.
        Forward requires spins to be present in the batch.
    aux_feature_channels : mapping of str to int, optional
        Per-particle auxiliary feature widths keyed by ``ElectronBatch.aux``.
        Values must have shape ``[*sample_shape, n_electrons, channels]`` with
        the configured channel count.
    in_features : int or None, optional
        Explicit per-particle input width. When set, the order MLPs are sized
        for this width instead of the derived coordinate/spin/aux width; use it
        when an :class:`spenn.nn.ElectronBasis` supplies ``one_body`` features
        whose width is ``basis.out_features``.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        max_order: int = 3,
        *,
        spatial_dim: int,
        out_channels: int | Mapping[int, int] = 16,
        hidden_channels: int = 64,
        num_hidden_layers: int = 2,
        activation: nn.Module | None = None,
        bias: bool = True,
        mlps: Mapping[int, nn.Module] | None = None,
        include_spins: bool = True,
        aux_feature_channels: Mapping[str, int] | None = None,
        in_features: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if max_order < 1:
            raise ValueError(f"max_order must be positive, got {max_order}")
        if spatial_dim <= 0:
            raise ValueError(f"spatial_dim must be positive, got {spatial_dim}")
        self.max_order = int(max_order)
        self.spatial_dim = int(spatial_dim)
        self.out_channels = {int(order): int(channels) for order, channels in out_channels.items()} if isinstance(out_channels, Mapping) else int(out_channels)
        self.include_spins = bool(include_spins)
        aux_feature_channels = {} if aux_feature_channels is None else dict(aux_feature_channels)
        self.aux_feature_channels = _normalize_aux_feature_channels(aux_feature_channels)
        # When an ElectronBasis supplies pre-built per-particle features, the
        # input width is the basis ``out_features`` rather than the raw
        # coordinate/spin/aux width. ``in_features`` overrides the derived width
        # so the order MLPs are sized for the basis path.
        derived_channels = self.spatial_dim + (1 if self.include_spins else 0) + sum(
            self.aux_feature_channels.values()
        )
        if in_features is not None:
            if int(in_features) <= 0:
                raise ValueError(f"in_features must be positive, got {in_features}")
            self.particle_input_channels = int(in_features)
        else:
            self.particle_input_channels = derived_channels
        self.in_features = None if in_features is None else int(in_features)
        self.order_mlps = nn.ModuleDict()
        supplied = {} if mlps is None else {int(order): module for order, module in mlps.items()}
        for order in range(1, self.max_order + 1):
            order_out_channels = self._out_channels(order)
            module = supplied.get(order)
            if module is None:
                module = MLP(
                    in_channels=order * self.particle_input_channels,
                    out_channels=order_out_channels,
                    hidden_channels=hidden_channels,
                    num_hidden_layers=num_hidden_layers,
                    activation=activation,
                    bias=bias,
                )
            self.order_mlps[str(order)] = module
        unknown = sorted(order for order in supplied if order < 1 or order > self.max_order)
        if unknown:
            raise ValueError(f"mlps contains orders outside [1, {self.max_order}]: {unknown}")

    def forward_impl(self, inputs: ElectronBatch | ElectronBasisFeatures) -> RealFeature:
        """Embed electron inputs as persistent real tuple features.

        Accepts either a raw :class:`ElectronBatch` (the per-particle vector is
        built from coordinates, spins, and aux features) or an
        :class:`ElectronBasisFeatures` whose ``one_body`` tensor is used directly
        as the per-particle vector. The tuple construction, per-order MLPs, and
        repeated-particle masking are identical for both paths.
        """

        if isinstance(inputs, ElectronBasisFeatures):
            particle_vectors = inputs.one_body.reshape(-1, inputs.n_electrons, inputs.n_features)
            n_electrons = inputs.n_electrons
        else:
            flat = inputs.flatten_samples()
            particle_vectors = _particle_vectors(
                flat,
                spatial_dim=self.spatial_dim,
                include_spins=self.include_spins,
                aux_feature_channels=self.aux_feature_channels,
            )
            n_electrons = flat.n_electrons
        if particle_vectors.shape[-1] != self.particle_input_channels:
            raise ValueError(
                f"Embedding expected per-particle width {self.particle_input_channels}, "
                f"got {particle_vectors.shape[-1]}"
            )
        batch_size = int(particle_vectors.shape[0])
        device = particle_vectors.device
        dtype = particle_vectors.dtype
        blocks = [zero_block(batch_size=batch_size, device=device, dtype=dtype)]
        for order in range(1, self.max_order + 1):
            tuple_inputs = tuple_particle_inputs(particle_vectors, order)
            block = self.order_mlps[str(order)](tuple_inputs).movedim(-1, 1)
            block = block * no_repeated_particle_mask(n_electrons, order, device=device).reshape(
                1,
                1,
                *((n_electrons,) * order),
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
    spatial_dim: int,
    include_spins: bool,
    aux_feature_channels: Mapping[str, int],
) -> torch.Tensor:
    """Return dense per-particle vectors for an electron batch."""

    if batch.spatial_dim != spatial_dim:
        raise ValueError(f"ElectronBatch spatial_dim={batch.spatial_dim} disagrees with Embedding spatial_dim={spatial_dim}")
    features = [batch.positions]
    if include_spins and batch.spins is None:
        raise ValueError("Embedding include_spins=True requires ElectronBatch.spins")
    if include_spins:
        features.append(batch.spins.unsqueeze(-1).to(dtype=batch.positions.dtype))
    for key, channels in aux_feature_channels.items():
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
        if int(tensor.shape[-1]) != channels:
            raise ValueError(
                f"ElectronBatch.aux[{key!r}] has {tensor.shape[-1]} channels, expected {channels}"
            )
        features.append(tensor)
    return torch.cat(features, dim=-1)


def _normalize_aux_feature_channels(value: Mapping[str, int]) -> dict[str, int]:
    normalized = {}
    for raw_key, raw_channels in value.items():
        key = str(raw_key)
        channels = int(raw_channels)
        if channels < 0:
            raise ValueError(f"aux_feature_channels[{key!r}] must be nonnegative, got {channels}")
        normalized[key] = channels
    return normalized


__all__ = ["Embedding"]
