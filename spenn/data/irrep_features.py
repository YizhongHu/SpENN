"""Irrep-space tensor containers for temporary Specht coordinates."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from copy import deepcopy
from typing import Any

import torch

from spenn.data.irrep_tensor import validate_irrep_tensor
from spenn.data.partitions import Partition
from spenn.data.permutation import Permutation


class IrrepTensors(MutableMapping[Partition, torch.Tensor]):
    """Store irrep-keyed tensor blocks.

    Entries use shape ``[batch, channel, n..., d_lambda, d_lambda]`` where the
    number of particle axes is determined by the partition order.

    Parameters
    ----------
    data : mapping or None, optional
        Optional mapping from partition keys to tensors.
    """

    def __init__(self, data: Mapping[Partition, torch.Tensor] | None = None) -> None:
        self._data: dict[Partition, torch.Tensor] = {}
        if data is not None:
            self.update(data)
        self._validate_common_channel_count()

    @property
    def data(self) -> Mapping[Partition, torch.Tensor]:
        """Return the underlying partition-to-tensor mapping."""

        return self._data

    def __getitem__(self, irrep: Partition) -> torch.Tensor:
        return self._data[irrep]

    def __setitem__(self, irrep: Partition, value: torch.Tensor) -> None:
        self._data[irrep] = value
        self._validate_common_channel_count()

    def __delitem__(self, irrep: Partition) -> None:
        del self._data[irrep]

    def __iter__(self) -> Iterator[Partition]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"{type(self).__name__}({{{', '.join(str(p) for p in self._data)}}})"

    def set(self, irrep: Partition, value: torch.Tensor) -> None:
        """Store one irrep tensor block."""

        self[irrep] = value

    def get(self, irrep: Partition, default: Any = None):
        """Return one irrep tensor block or `default`."""

        return self._data.get(irrep, default)

    def has(self, irrep: Partition) -> bool:
        """Return whether `irrep` is present."""

        return irrep in self._data

    def flat_items(self) -> Iterator[tuple[Partition, torch.Tensor]]:
        """Iterate over ``(partition, tensor)`` entries."""

        yield from self._data.items()

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "IrrepTensors":
        """Move every tensor block to a new device or dtype."""

        return type(self)({partition: tensor.to(device=device, dtype=dtype) for partition, tensor in self._data.items()})

    def clone(self) -> "IrrepTensors":
        """Clone all tensor blocks."""

        return type(self)({partition: tensor.clone() for partition, tensor in self._data.items()})

    def add(self, other: "IrrepTensors") -> "IrrepTensors":
        """Return the keywise sum with another irrep container."""

        output = self.clone()
        for partition, tensor in other.flat_items():
            existing = output.get(partition)
            output.set(partition, tensor.clone() if existing is None else existing + tensor)
        return output

    def __add__(self, other: "IrrepTensors") -> "IrrepTensors":
        """Return the keywise tensor sum with another container."""

        return self.add(other)

    def permute(self, permutation: Permutation) -> "IrrepTensors":
        """Return a copy with every ordered-particle axis permuted."""

        return type(self)(
            {
                partition: _permute_irrep_tensor(tensor, partition=partition, permutation=permutation)
                for partition, tensor in self._data.items()
            }
        )

    def validate(
        self,
        *,
        batch_size: int | None = None,
        n_electrons: int | None = None,
        supported: Iterable[Partition] | None = None,
        min_channel_dim: int = 1,
    ) -> None:
        """Validate stored tensor shapes and optional supported keys."""

        supported_set = None if supported is None else set(supported)
        for partition, tensor in self._data.items():
            if supported_set is not None and partition not in supported_set:
                raise KeyError(f"Unsupported irrep key {partition}")
            validate_irrep_tensor(
                tensor,
                order=partition.order,
                irrep=partition,
                batch_size=batch_size,
                n_electrons=n_electrons,
                min_channel_dim=min_channel_dim,
            )

    def to_dict(self) -> dict[Partition, torch.Tensor]:
        """Return a deep copy of the underlying mapping."""

        return deepcopy(self._data)

    def _validate_common_channel_count(self) -> None:
        channels: int | None = None
        for tensor in self._data.values():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("IrrepTensors values must be torch.Tensor objects")
            current = int(tensor.shape[1])
            if channels is None:
                channels = current
            elif current != channels:
                raise ValueError(
                    "IrrepTensors entries must share one channel count, "
                    f"got {channels} and {current}"
                )


def _permute_irrep_tensor(
    tensor: torch.Tensor,
    *,
    partition: Partition,
    permutation: Permutation,
) -> torch.Tensor:
    """Permute the ordered-particle axes of one irrep tensor block."""

    validate_irrep_tensor(tensor, order=partition.order, irrep=partition)
    if partition.order == 0:
        return tensor.clone()
    if tensor.shape[2] != len(permutation):
        raise ValueError(
            f"Permutation of size {len(permutation)} is incompatible with "
            f"partition {partition.parts} tensor axes of length {tensor.shape[2]}"
        )
    index = torch.tensor(permutation.inverse().image, device=tensor.device, dtype=torch.long)
    output = tensor
    for axis in range(2, 2 + partition.order):
        output = output.index_select(axis, index)
    return output


class IrrepFeature(IrrepTensors):
    """Store temporary or persistent irrep-space feature blocks."""


class IrrepMessage(IrrepTensors):
    """Store temporary irrep-space message blocks."""


__all__ = ["IrrepFeature", "IrrepMessage", "IrrepTensors"]
