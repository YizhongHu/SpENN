"""Real-space equivariant mixing kernels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
from torch import nn

from spenn.data.indices import (
    flatten_tuple_indices,
    ordered_tuple_tensor,
    ordered_tuples,
    select_tuple,
    select_tuple_tensor,
)
from spenn.data.real import (
    RealFeature,
    RealInteraction,
    common_real_batch_size,
    common_real_dtype,
    common_real_particle_count,
    zero_block,
)
from spenn.equivariance import EquivariantMap
from spenn.reps.paths import PathMetadata, VirtualPath, load_default_path_metadata


Aggregation = Literal["sum", "completion_mean"]
MixingImplementation = Literal["slow", "vectorized"]


class EquivariantMixing(EquivariantMap):
    """Bilinear virtual-support real-space mixing module.

    The slow implementation is a literal correctness reference that loops over
    paths and ordered distinct virtual tuples exactly as written in the PR
    plan. The vectorized implementation batches virtual tuples path-by-path and
    is tested against the slow oracle.

    Parameters
    ----------
    max_order : int
        Maximum input/output body order.
    max_virtual_order : int or None, optional
        Maximum virtual support order. Defaults to `max_order`.
    paths : PathMetadata, tuple of VirtualPath, or None, optional
        Precomputed paths. If ``None``, canonical paths are generated.
    output_embedding : {"canonical", "full"}, optional
        Path family used when generating paths.
    aggregation : {"sum", "completion_mean"}, optional
        Whether to sum over completions or average over compatible completions
        for each output tuple and path.
    channels : int or mapping
        Input feature channels per body order. This is architecture metadata
        and is independent of particle count.
    left_channels, right_channels : int, mapping, or None, optional
        Input channels for asymmetric two-input mixing. If omitted, `channels`
        is used for both sides.
    out_channels : int, mapping, or None, optional
        Output channels per target order. ``None`` preserves `channels`.
    initial_weight : float, optional
        Initial value for each path weight.
    implementation : {"slow", "vectorized"}, optional
        Mixing kernel implementation. ``"slow"`` keeps the literal loop oracle;
        ``"vectorized"`` batches virtual tuples path-by-path and should match
        the slow reference exactly.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        max_order: int,
        *,
        max_virtual_order: int | None = None,
        paths: PathMetadata | tuple[VirtualPath, ...] | None = None,
        output_embedding: Literal["canonical", "full"] = "canonical",
        aggregation: Aggregation = "sum",
        channels: int | Mapping[int, int],
        left_channels: int | Mapping[int, int] | None = None,
        right_channels: int | Mapping[int, int] | None = None,
        out_channels: int | Mapping[int, int] | None = None,
        initial_weight: float = 1.0,
        implementation: MixingImplementation = "slow",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_order = int(max_order)
        self.max_virtual_order = self.max_order if max_virtual_order is None else int(max_virtual_order)
        if self.max_order <= 0:
            raise ValueError(f"max_order must be positive, got {self.max_order}")
        if self.max_virtual_order <= 0:
            raise ValueError(f"max_virtual_order must be positive, got {self.max_virtual_order}")
        if aggregation not in {"sum", "completion_mean"}:
            raise ValueError(f"Unsupported aggregation {aggregation!r}")
        if implementation not in {"slow", "vectorized"}:
            raise ValueError(f"Unsupported mixing implementation {implementation!r}")
        self.aggregation: Aggregation = aggregation
        self.implementation: MixingImplementation = implementation
        self.output_embedding = output_embedding
        self.initial_weight = float(initial_weight)
        self.left_channels = _normalize_channels(
            channels if left_channels is None else left_channels,
            max_order=self.max_order,
            name="left_channels",
        )
        self.right_channels = _normalize_channels(
            channels if right_channels is None else right_channels,
            max_order=self.max_order,
            name="right_channels",
        )
        self.out_channels = _normalize_channels(
            channels if out_channels is None else out_channels,
            max_order=self.max_order,
            name="out_channels",
        )
        if isinstance(paths, PathMetadata):
            self.paths = self._paths_from_metadata(paths)
        elif paths is None:
            metadata = load_default_path_metadata(output_embedding)
            if metadata.max_order < self.max_order or metadata.max_virtual_order < self.max_virtual_order:
                raise ValueError(
                    "Saved path metadata only covers "
                    f"max_order={metadata.max_order}, max_virtual_order={metadata.max_virtual_order}; "
                    "pass explicit PathMetadata for larger generated path families"
                )
            self.paths = self._paths_from_metadata(metadata)
        else:
            self.paths = tuple(paths)
        self.weights = nn.ParameterDict()
        self._initialize_weights()

    def forward_impl(self, x1: RealFeature, x2: RealFeature | None = None) -> RealInteraction:
        """Mix one or two real feature states into path-resolved interactions."""

        x2 = x1 if x2 is None else x2
        x1.validate()
        x2.validate()
        n_particles = common_real_particle_count(x1, x2)
        batch_size = common_real_batch_size(x1, x2)
        dtype = common_real_dtype(x1, x2)
        device = x1.blocks[0].device if x1.blocks else None
        output_blocks: list[torch.Tensor] = [
            zero_block(batch_size=batch_size, paths=0, device=device, dtype=dtype)
        ]
        for order in range(1, self.max_order + 1):
            active_paths = self._paths_for_order(order, x1=x1, x2=x2)
            out_channels = self._out_channels(order)
            block = torch.zeros(
                (batch_size, out_channels, len(active_paths), *((n_particles,) * order)),
                device=device,
                dtype=dtype,
            )
            counts = (
                torch.zeros((len(active_paths), *((n_particles,) * order)), device=device, dtype=dtype)
                if self.aggregation == "completion_mean"
                else None
            )
            if self.implementation == "slow":
                self._mix_order_slow(
                    block,
                    counts,
                    active_paths,
                    x1=x1,
                    x2=x2,
                    n_particles=n_particles,
                    out_channels=out_channels,
                )
            else:
                self._mix_order_vectorized(
                    block,
                    counts,
                    active_paths,
                    x1=x1,
                    x2=x2,
                    n_particles=n_particles,
                    out_channels=out_channels,
                )
            if counts is not None:
                block = block / counts.clamp_min(1).unsqueeze(0).unsqueeze(0)
            output_blocks.append(block)
        return RealInteraction(output_blocks)

    def _mix_order_slow(
        self,
        block: torch.Tensor,
        counts: torch.Tensor | None,
        active_paths: list[VirtualPath],
        *,
        x1: RealFeature,
        x2: RealFeature,
        n_particles: int,
        out_channels: int,
    ) -> None:
        for path_index, path in enumerate(active_paths):
            weight = self._weight_for(path, x1=x1, x2=x2, out_channels=out_channels)
            left = x1.blocks[path.m1]
            right = x2.blocks[path.m2]
            for virtual_tuple in ordered_tuples(n_particles, path.s, distinct=True):
                output_tuple = select_tuple(virtual_tuple, path.tau)
                left_tuple = select_tuple(virtual_tuple, path.tau1)
                right_tuple = select_tuple(virtual_tuple, path.tau2)
                left_value = left[(slice(None), slice(None), *left_tuple)]
                right_value = right[(slice(None), slice(None), *right_tuple)]
                contribution = torch.einsum("ocd,bc,bd->bo", weight, left_value, right_value)
                block[(slice(None), slice(None), path_index, *output_tuple)] += contribution
                if counts is not None:
                    counts[(path_index, *output_tuple)] += 1

    def _mix_order_vectorized(
        self,
        block: torch.Tensor,
        counts: torch.Tensor | None,
        active_paths: list[VirtualPath],
        *,
        x1: RealFeature,
        x2: RealFeature,
        n_particles: int,
        out_channels: int,
    ) -> None:
        block_flat = block.reshape(*block.shape[:3], -1)
        for path_index, path in enumerate(active_paths):
            weight = self._weight_for(path, x1=x1, x2=x2, out_channels=out_channels)
            virtual_tuples = ordered_tuple_tensor(n_particles, path.s, distinct=True, device=block.device)
            output_indices = select_tuple_tensor(virtual_tuples, path.tau)
            left_indices = select_tuple_tensor(virtual_tuples, path.tau1)
            right_indices = select_tuple_tensor(virtual_tuples, path.tau2)

            left = x1.blocks[path.m1][(slice(None), slice(None), *left_indices.unbind(dim=1))]
            right = x2.blocks[path.m2][(slice(None), slice(None), *right_indices.unbind(dim=1))]
            contribution = torch.einsum("ocd,bcv,bdv->bov", weight, left, right)
            flat_output_indices = flatten_tuple_indices(output_indices, n_particles)
            scatter_index = flat_output_indices.reshape(1, 1, -1).expand(
                block.shape[0],
                out_channels,
                -1,
            )
            block_flat[:, :, path_index].scatter_add_(2, scatter_index, contribution)

            if counts is not None:
                counts_flat = counts[path_index].reshape(-1)
                counts_flat.scatter_add_(
                    0,
                    flat_output_indices,
                    torch.ones_like(flat_output_indices, dtype=counts.dtype),
                )

    def _paths_for_order(self, order: int, *, x1: RealFeature, x2: RealFeature) -> list[VirtualPath]:
        return [
            path
            for path in self.paths
            if path.m == order and path.m1 < len(x1.blocks) and path.m2 < len(x2.blocks)
        ]

    def _paths_from_metadata(self, metadata: PathMetadata) -> tuple[VirtualPath, ...]:
        return tuple(
            path
            for path in metadata.all_paths()
            if path.s <= self.max_virtual_order
            and path.m <= self.max_order
            and path.m1 <= self.max_order
            and path.m2 <= self.max_order
        )

    def _out_channels(self, order: int) -> int:
        return self.out_channels[order]

    def _initialize_weights(self) -> None:
        for path in self.paths:
            key = f"g{path.global_id}"
            if key in self.weights:
                continue
            shape = (
                self.out_channels[path.m],
                self.left_channels[path.m1],
                self.right_channels[path.m2],
            )
            self.weights[key] = nn.Parameter(torch.full(shape, self.initial_weight))

    def _weight_for(
        self,
        path: VirtualPath,
        *,
        x1: RealFeature,
        x2: RealFeature,
        out_channels: int,
    ) -> torch.Tensor:
        left_channels = self.left_channels[path.m1]
        right_channels = self.right_channels[path.m2]
        _validate_feature_channels(x1, path.m1, left_channels, name="left input")
        _validate_feature_channels(x2, path.m2, right_channels, name="right input")
        shape = (out_channels, left_channels, right_channels)
        key = f"g{path.global_id}"
        if key not in self.weights:
            raise RuntimeError(f"Missing eager EquivariantMixing weight for path {path.global_id}")
        weight = self.weights[key]
        if tuple(weight.shape) != shape:
            raise ValueError(f"Path {path.global_id} weight shape {tuple(weight.shape)} does not match {shape}")
        return weight


def _normalize_channels(value: int | Mapping[int, int], *, max_order: int, name: str) -> dict[int, int]:
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


def _validate_feature_channels(feature: RealFeature, order: int, expected: int, *, name: str) -> None:
    if order >= len(feature.blocks):
        raise ValueError(f"{name} has no order-{order} block")
    actual = int(feature.blocks[order].shape[1])
    if actual != expected:
        raise ValueError(f"{name} order-{order} channels {actual} do not match configured {expected}")


__all__ = ["EquivariantMixing"]
