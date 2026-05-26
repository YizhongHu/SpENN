"""Structured feature container used between encoder, SpechtMP, and readout."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from copy import deepcopy
from typing import Any

import torch

from spenn.data_structures.irrep_tensor import validate_irrep_tensor
from spenn.data_structures.partitions import Partition, PartitionLike, normalize_partition


class FeatureDict(MutableMapping[int, dict[Partition, torch.Tensor]]):
    """Store feature tensors by logical order and partition key.

    The public access pattern is ``features.get(order, partition)`` or
    ``features.set(order, partition, tensor)``. Tuple-like partition specs are
    accepted at API boundaries and canonicalized to :class:`Partition` keys.

    Parameters
    ----------
    data : dict or None, optional
        Optional nested mapping from order to partition keys to tensors.
    """

    def __init__(self, data: Mapping[int, Mapping[PartitionLike, torch.Tensor]] | None = None) -> None:
        self._data: dict[int, dict[Partition, torch.Tensor]] = {}
        if data is not None:
            self.update(data)

    def __getitem__(self, order: int) -> dict[Partition, torch.Tensor]:
        return self._data[order]

    def __setitem__(self, order: int, value: Mapping[PartitionLike, torch.Tensor]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("FeatureDict values must be mappings from partitions to tensors")
        normalized: dict[Partition, torch.Tensor] = {}
        normalized_order = int(order)
        for partition_like, tensor in value.items():
            partition = normalize_partition(order, partition_like)
            normalized_order = partition.order
            normalized[partition] = tensor
        self._data[normalized_order] = normalized

    def __delitem__(self, order: int) -> None:
        del self._data[order]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        keys = ", ".join(f"{order}: {list(block)}" for order, block in self._data.items())
        return f"FeatureDict({{{keys}}})"

    def set(self, order: int, irrep: PartitionLike, value: torch.Tensor) -> None:
        """Store a tensor under an order and partition key.

        Parameters
        ----------
        order : int
            Logical tensor order.
        irrep : PartitionLike
            Partition specifier accepted by `normalize_partition`.
        value : torch.Tensor
            Feature tensor to store.
        """

        partition = normalize_partition(order, irrep)
        self._data.setdefault(partition.order, {})[partition] = value

    def get(self, order: int, irrep: PartitionLike | None = None, default: Any = None):
        """Return a feature block or nested partition entry.

        Parameters
        ----------
        order : int
            Logical tensor order.
        irrep : PartitionLike or None, optional
            Partition specifier accepted by `normalize_partition`. If ``None``, the
            full feature block for `order` is returned.
        default : object, optional
            Value returned when the requested entry is absent.

        Returns
        -------
        object
            Feature block, tensor entry, or `default`.
        """

        if irrep is None:
            return self._data.get(order, default)
        partition = normalize_partition(order, irrep)
        return self._data.get(partition.order, {}).get(partition, default)

    def has(self, order: int, irrep: PartitionLike) -> bool:
        """Return whether an order and partition entry exists.

        Parameters
        ----------
        order : int
            Logical tensor order.
        irrep : PartitionLike
            Partition specifier accepted by `normalize_partition`.

        Returns
        -------
        bool
            ``True`` if the requested feature tensor is present.
        """

        partition = normalize_partition(order, irrep)
        return partition in self._data.get(partition.order, {})

    def items(self):
        """Return top-level order and block pairs.

        Returns
        -------
        dict_items
            View over ``(order, block)`` pairs.
        """

        return self._data.items()

    def flat_items(self):
        """Iterate over flattened feature entries.

        Yields
        ------
        tuple
            ``(order, partition, tensor)`` triples.
        """

        for order, block in self._data.items():
            for partition, tensor in block.items():
                yield order, partition, tensor

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "FeatureDict":
        """Move every feature tensor to a new device or dtype.

        Parameters
        ----------
        device : torch.device, str, or None, optional
            Target device. If ``None``, the current device is preserved.
        dtype : torch.dtype or None, optional
            Target dtype. If ``None``, the current dtype is preserved.

        Returns
        -------
        FeatureDict
            Feature container with all tensors moved to the requested device or
            dtype.
        """

        return FeatureDict(
            {
                order: {partition: tensor.to(device=device, dtype=dtype) for partition, tensor in block.items()}
                for order, block in self._data.items()
            }
        )

    def clone(self) -> "FeatureDict":
        """Clone all stored feature tensors.

        Returns
        -------
        FeatureDict
            Feature container with cloned tensors.
        """

        return FeatureDict(
            {order: {partition: tensor.clone() for partition, tensor in block.items()} for order, block in self._data.items()}
        )

    def validate(
        self,
        *,
        batch_size: int | None = None,
        n_electrons: int | None = None,
        supported: Iterable[tuple[int, PartitionLike]] | None = None,
        min_channel_dim: int = 1,
    ) -> None:
        """Validate feature tensor shapes and supported keys.

        Parameters
        ----------
        batch_size : int or None, optional
            Expected leading batch size.
        n_electrons : int or None, optional
            Expected size of each particle axis.
        supported : iterable of tuple or None, optional
            Optional set of allowed ``(order, partition)`` keys.
        min_channel_dim : int, optional
            Minimum allowed size for the channel axis.

        Raises
        ------
        KeyError
            If a feature key is not in `supported`.
        ValueError
            If any tensor violates the expected shape conventions.
        """

        supported_set = None if supported is None else _normalize_supported(supported)
        for order, block in self._data.items():
            for partition, tensor in block.items():
                if supported_set is not None and (order, partition) not in supported_set:
                    raise KeyError(f"Unsupported feature key {(order, partition)}")
                validate_irrep_tensor(
                    tensor,
                    order=order,
                    irrep=partition,
                    batch_size=batch_size,
                    n_electrons=n_electrons,
                    min_channel_dim=min_channel_dim,
                )

    def to_dict(self) -> dict[int, dict[Partition, torch.Tensor]]:
        """Return a plain nested dictionary copy.

        Returns
        -------
        dict
            Deep copy of the underlying nested mapping.
        """

        return deepcopy(self._data)


def _normalize_supported(supported: Iterable[tuple[int, PartitionLike]]) -> set[tuple[int, Partition]]:
    return {(partition.order, partition) for order, spec in supported for partition in [normalize_partition(order, spec)]}
