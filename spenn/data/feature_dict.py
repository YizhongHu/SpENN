"""Structured feature containers used by encoder, SpechtMP, and readout."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from copy import deepcopy
from typing import Any

import torch

from spenn.data.irrep_features import IrrepFeature, IrrepMessage
from spenn.data.irrep_tensor import validate_branch_tensor, validate_irrep_tensor, validate_tensor_product_tensor
from spenn.data.partitions import Partition


class FeatureDict(IrrepFeature):
    """Store persistent Specht feature tensors.

    `FeatureDict` represents the persistent layer state ``x``. The key is a
    :class:`Partition` labelling the local Specht irrep, which also encodes
    the feature order.

    Tensor entries use shape ``[batch, channel, n..., a, a]``. The first axis is
    batch, the second is learned feature channel, the next `order` axes are
    ordered electron-label axes of length ``n_electrons``, and the final two
    axes are irrep-coordinate and multiplicity/Fourier-column axes. Scalar
    irreps use final shape ``[1, 1]``; the mixed order-3 irrep ``(2, 1)`` uses
    ``[2, 2]``.

    Parameters
    ----------
    data : mapping or None, optional
        Optional mapping from :class:`Partition` keys to tensors.
    """

    def __init__(self, data: Mapping[Partition, torch.Tensor] | None = None) -> None:
        self._data: dict[Partition, torch.Tensor] = {}
        if data is not None:
            self.update(data)

    def __getitem__(self, irrep: Partition) -> torch.Tensor:
        return self._data[irrep]

    def __setitem__(self, irrep: Partition, value: torch.Tensor) -> None:
        self._data[irrep] = value

    def __delitem__(self, irrep: Partition) -> None:
        del self._data[irrep]

    def __iter__(self) -> Iterator[Partition]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"FeatureDict({{{', '.join(str(p) for p in self._data)}}})"

    def set(self, irrep: Partition, value: torch.Tensor) -> None:
        """Store a persistent feature tensor.

        Parameters
        ----------
        irrep : Partition
            Specht irrep partition.
        value : torch.Tensor
            Feature tensor with shape ``[batch, channel, n..., a, a]``.
        """

        self._data[irrep] = value

    def get(self, irrep: Partition, default: Any = None):
        """Return a feature tensor.

        Parameters
        ----------
        irrep : Partition
            Specht irrep partition.
        default : object, optional
            Value returned when the requested entry is absent.

        Returns
        -------
        object
            Feature tensor or `default`.
        """

        return self._data.get(irrep, default)

    def has(self, irrep: Partition) -> bool:
        """Return whether a feature tensor exists.

        Parameters
        ----------
        irrep : Partition
            Specht irrep partition.

        Returns
        -------
        bool
            ``True`` if the requested feature tensor is present.
        """

        return irrep in self._data

    def items(self):
        """Return partition and tensor pairs.

        Returns
        -------
        dict_items
            View over ``(partition, tensor)`` pairs.
        """

        return self._data.items()

    def flat_items(self) -> Iterator[tuple[Partition, torch.Tensor]]:
        """Iterate over feature entries.

        Yields
        ------
        tuple
            ``(partition, tensor)`` pairs.
        """

        for partition, tensor in self._data.items():
            yield partition, tensor

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
            {partition: tensor.to(device=device, dtype=dtype) for partition, tensor in self._data.items()}
        )

    def clone(self) -> "FeatureDict":
        """Clone all stored feature tensors.

        Returns
        -------
        FeatureDict
            Feature container with cloned tensors.
        """

        return FeatureDict({partition: tensor.clone() for partition, tensor in self._data.items()})

    def add(self, other: "FeatureDict") -> "FeatureDict":
        """Return the keywise sum with another feature dictionary.

        Parameters
        ----------
        other : FeatureDict
            Feature updates to add. Entries absent from `self` are copied into
            the returned dictionary.

        Returns
        -------
        FeatureDict
            New feature dictionary containing the keywise tensor sum.

        """

        output = self.clone()
        for partition, tensor in other.flat_items():
            existing = output.get(partition)
            output.set(partition, tensor.clone() if existing is None else existing + tensor)
        return output

    def __add__(self, other: "FeatureDict") -> "FeatureDict":
        """Return the keywise sum with another feature dictionary."""

        return self.add(other)

    def validate(
        self,
        *,
        batch_size: int | None = None,
        n_electrons: int | None = None,
        supported: Iterable[Partition] | None = None,
        min_channel_dim: int = 1,
    ) -> None:
        """Validate feature tensor shapes and supported keys.

        Parameters
        ----------
        batch_size : int or None, optional
            Expected leading batch size.
        n_electrons : int or None, optional
            Expected size of each ordered electron-label axis.
        supported : iterable of Partition or None, optional
            Optional set of allowed partitions.
        min_channel_dim : int, optional
            Minimum allowed size for the channel axis.

        Raises
        ------
        KeyError
            If a feature key is not in `supported`.
        ValueError
            If any tensor violates ``[batch, channel, n..., a, a]``.
        """

        supported_set = None if supported is None else set(supported)
        for partition, tensor in self._data.items():
            if supported_set is not None and partition not in supported_set:
                raise KeyError(f"Unsupported feature key {partition}")
            validate_irrep_tensor(
                tensor,
                order=partition.order,
                irrep=partition,
                batch_size=batch_size,
                n_electrons=n_electrons,
                min_channel_dim=min_channel_dim,
            )

    def to_dict(self) -> dict[Partition, torch.Tensor]:
        """Return a plain dictionary copy.

        Returns
        -------
        dict
            Deep copy of the underlying mapping.
        """

        return deepcopy(self._data)


class TensorProductDict(MutableMapping[Partition, dict[Partition, dict[Partition, torch.Tensor]]]):
    """Store fixed tensor-product features.

    `TensorProductDict` represents the exact tensor-product state ``z``. The
    dictionary convention is ``[lambda] -> [lambda1] -> [lambda2]`` where each
    key is a :class:`Partition`.

    Tensor entries use shape ``[batch, channel, p, I..., I1..., I2...,
    alpha, beta]``. The `p` axis indexes fixed tensor-product paths. Ordered
    tuple axes correspond first to ``I`` for ``lambda``, then ``I1`` for
    ``lambda1``, then ``I2`` for ``lambda2``. The final two axes are target
    irrep-coordinate and multiplicity/Fourier-column axes; scalar target
    irreps use ``[1, 1]``.

    Parameters
    ----------
    data : mapping or None, optional
        Optional nested mapping from target partitions to left source
        partitions to right source partitions to tensors.
    """

    def __init__(
        self,
        data: Mapping[Partition, Mapping[Partition, Mapping[Partition, torch.Tensor]]] | None = None,
    ) -> None:
        self._data: dict[Partition, dict[Partition, dict[Partition, torch.Tensor]]] = {}
        if data is not None:
            self.update(data)

    def __getitem__(self, target: Partition) -> dict[Partition, dict[Partition, torch.Tensor]]:
        return self._data[target]

    def __setitem__(self, target: Partition, value: Mapping[Partition, Mapping[Partition, torch.Tensor]]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("TensorProductDict values must be nested mappings")
        normalized: dict[Partition, dict[Partition, torch.Tensor]] = {}
        for left, right_block in value.items():
            if not isinstance(right_block, Mapping):
                raise TypeError("TensorProductDict left-source values must be mappings")
            normalized[left] = {}
            for right, tensor in right_block.items():
                normalized[left][right] = tensor
        self._data[target] = normalized

    def __delitem__(self, target: Partition) -> None:
        del self._data[target]

    def __iter__(self) -> Iterator[Partition]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        keys = ", ".join(
            f"{target}: {[(left, list(right_block)) for left, right_block in left_block.items()]}"
            for target, left_block in self._data.items()
        )
        return f"TensorProductDict({{{keys}}})"

    def set(self, target: Partition, left: Partition, right: Partition, value: torch.Tensor) -> None:
        """Store a tensor-product feature block.

        Parameters
        ----------
        target : Partition
            Target Specht irrep partition.
        left : Partition
            Left source Specht irrep partition.
        right : Partition
            Right source Specht irrep partition.
        value : torch.Tensor
            Tensor-product tensor with shape ``[batch, channel, p, I...,
            I1..., I2..., alpha, beta]``.
        """

        self._data.setdefault(target, {}).setdefault(left, {})[right] = value

    def get(
        self,
        target: Partition,
        left: Partition | None = None,
        right: Partition | None = None,
        default: Any = None,
    ):
        """Return a tensor-product block or nested mapping.

        Parameters
        ----------
        target : Partition
            Target Specht irrep partition.
        left : Partition or None, optional
            Left source Specht irrep partition. If ``None``, return the full
            target block.
        right : Partition or None, optional
            Right source Specht irrep partition. If ``None``, return the full
            left-source block.
        default : object, optional
            Value returned when the requested entry is absent.

        Returns
        -------
        object
            Tensor-product tensor, nested mapping, or `default`.
        """

        target_block = self._data.get(target)
        if target_block is None:
            return default
        if left is None:
            return target_block
        left_block = target_block.get(left)
        if left_block is None:
            return default
        if right is None:
            return left_block
        return left_block.get(right, default)

    def has(self, target: Partition, left: Partition, right: Partition) -> bool:
        """Return whether a tensor-product entry exists.

        Parameters
        ----------
        target : Partition
            Target Specht irrep partition.
        left : Partition
            Left source Specht irrep partition.
        right : Partition
            Right source Specht irrep partition.

        Returns
        -------
        bool
            ``True`` if the requested tensor-product block is present.
        """

        return self.get(target, left, right, default=None) is not None

    def flat_items(self) -> Iterator[tuple[Partition, Partition, Partition, torch.Tensor]]:
        """Iterate over flattened tensor-product entries.

        Yields
        ------
        tuple
            ``(target, left, right, tensor)`` entries.
        """

        for target, left_block in self._data.items():
            for left, right_block in left_block.items():
                for right, tensor in right_block.items():
                    yield target, left, right, tensor

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "TensorProductDict":
        """Move every tensor-product block to a new device or dtype.

        Parameters
        ----------
        device : torch.device, str, or None, optional
            Target device. If ``None``, the current device is preserved.
        dtype : torch.dtype or None, optional
            Target dtype. If ``None``, the current dtype is preserved.

        Returns
        -------
        TensorProductDict
            Tensor-product container with tensors moved to the requested
            device or dtype.
        """

        return TensorProductDict(
            {
                target: {
                    left: {right: tensor.to(device=device, dtype=dtype) for right, tensor in right_block.items()}
                    for left, right_block in left_block.items()
                }
                for target, left_block in self._data.items()
            }
        )

    def clone(self) -> "TensorProductDict":
        """Clone all stored tensor-product blocks.

        Returns
        -------
        TensorProductDict
            Tensor-product container with cloned tensors.
        """

        return TensorProductDict(
            {
                target: {
                    left: {right: tensor.clone() for right, tensor in right_block.items()}
                    for left, right_block in left_block.items()
                }
                for target, left_block in self._data.items()
            }
        )

    def validate(self, *, batch_size: int | None = None, n_electrons: int | None = None) -> None:
        """Validate tensor-product shape conventions.

        Parameters
        ----------
        batch_size : int or None, optional
            Expected leading batch size.
        n_electrons : int or None, optional
            Expected size of each ordered tuple axis.

        Raises
        ------
        ValueError
            If any tensor-product block violates its expected scaffold shape.
        """

        for target, left, right, tensor in self.flat_items():
            validate_tensor_product_tensor(
                tensor,
                target=target,
                left=left,
                right=right,
                batch_size=batch_size,
                n_electrons=n_electrons,
            )

    def to_dict(self) -> dict[Partition, dict[Partition, dict[Partition, torch.Tensor]]]:
        """Return a plain nested dictionary copy.

        Returns
        -------
        dict
            Deep copy of the underlying nested mapping.
        """

        return deepcopy(self._data)


class BranchDict(MutableMapping[Partition, dict[Partition, torch.Tensor]]):
    """Store branched intermediate tensors.

    `BranchDict` represents the intermediate state ``y`` produced by fixed
    branching and consumed by the trainable update head. The dictionary
    convention is ``[lambda] -> [mu]`` where each key is a :class:`Partition`.

    Tensor entries use shape ``[batch, channel, q, I..., J..., alpha, beta]``.
    The `q` axis indexes fixed branch paths. Ordered tuple axes correspond
    first to target tuple ``I`` for ``lambda``, then source tuple ``J`` for
    ``mu``. The final two axes are target irrep-coordinate and
    multiplicity/Fourier-column axes; scalar target irreps use ``[1, 1]``.

    Parameters
    ----------
    data : mapping or None, optional
        Optional nested mapping from target partitions to source partitions to
        branched tensors.
    """

    def __init__(self, data: Mapping[Partition, Mapping[Partition, torch.Tensor]] | None = None) -> None:
        self._data: dict[Partition, dict[Partition, torch.Tensor]] = {}
        if data is not None:
            self.update(data)

    def __getitem__(self, target: Partition) -> dict[Partition, torch.Tensor]:
        return self._data[target]

    def __setitem__(self, target: Partition, value: Mapping[Partition, torch.Tensor]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("BranchDict values must be mappings from Partition to tensor")
        normalized: dict[Partition, torch.Tensor] = {}
        for source, tensor in value.items():
            normalized[source] = tensor
        self._data[target] = normalized

    def __delitem__(self, target: Partition) -> None:
        del self._data[target]

    def __iter__(self) -> Iterator[Partition]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        keys = ", ".join(f"{target}: {list(source_block)}" for target, source_block in self._data.items())
        return f"BranchDict({{{keys}}})"

    def set(self, target: Partition, source: Partition, value: torch.Tensor) -> None:
        """Store a branched intermediate tensor.

        Parameters
        ----------
        target : Partition
            Target feature irrep partition ``lambda``.
        source : Partition
            Source message irrep partition ``mu``.
        value : torch.Tensor
            Branched tensor with shape ``[batch, channel, q, I..., J...,
            alpha, beta]``.
        """

        self._data.setdefault(target, {})[source] = value

    def get(self, target: Partition, source: Partition | None = None, default: Any = None):
        """Return a branched tensor or nested mapping.

        Parameters
        ----------
        target : Partition
            Target feature irrep partition ``lambda``.
        source : Partition or None, optional
            Source message irrep partition ``mu``. If ``None``, return the full
            target block.
        default : object, optional
            Value returned when the requested entry is absent.

        Returns
        -------
        object
            Branched tensor, nested mapping, or `default`.
        """

        target_block = self._data.get(target)
        if target_block is None:
            return default
        if source is None:
            return target_block
        return target_block.get(source, default)

    def has(self, target: Partition, source: Partition) -> bool:
        """Return whether a branched tensor exists.

        Parameters
        ----------
        target : Partition
            Target feature irrep partition ``lambda``.
        source : Partition
            Source message irrep partition ``mu``.

        Returns
        -------
        bool
            ``True`` if the requested branched tensor is present.
        """

        return self.get(target, source, default=None) is not None

    def flat_items(self) -> Iterator[tuple[Partition, Partition, torch.Tensor]]:
        """Iterate over flattened branched entries.

        Yields
        ------
        tuple
            ``(target, source, tensor)`` entries.
        """

        for target, source_block in self._data.items():
            for source, tensor in source_block.items():
                yield target, source, tensor

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "BranchDict":
        """Move every branched tensor to a new device or dtype.

        Parameters
        ----------
        device : torch.device, str, or None, optional
            Target device. If ``None``, the current device is preserved.
        dtype : torch.dtype or None, optional
            Target dtype. If ``None``, the current dtype is preserved.

        Returns
        -------
        BranchDict
            Branched container with tensors moved to the requested device or
            dtype.
        """

        return BranchDict(
            {
                target: {source: tensor.to(device=device, dtype=dtype) for source, tensor in source_block.items()}
                for target, source_block in self._data.items()
            }
        )

    def clone(self) -> "BranchDict":
        """Clone all stored branched tensors.

        Returns
        -------
        BranchDict
            Branched container with cloned tensors.
        """

        return BranchDict(
            {target: {source: tensor.clone() for source, tensor in source_block.items()} for target, source_block in self._data.items()}
        )

    def validate(self, *, batch_size: int | None = None, n_electrons: int | None = None) -> None:
        """Validate branched tensor shape conventions.

        Parameters
        ----------
        batch_size : int or None, optional
            Expected leading batch size.
        n_electrons : int or None, optional
            Expected size of each ordered tuple axis.

        Raises
        ------
        ValueError
            If any branched tensor violates its expected scaffold shape.
        """

        for target, source, tensor in self.flat_items():
            validate_branch_tensor(
                tensor,
                target=target,
                source=source,
                batch_size=batch_size,
                n_electrons=n_electrons,
            )

    def to_dict(self) -> dict[Partition, dict[Partition, torch.Tensor]]:
        """Return a plain nested dictionary copy.

        Returns
        -------
        dict
            Deep copy of the underlying nested mapping.
        """

        return deepcopy(self._data)


class MessageDict(IrrepMessage):
    """Store aggregated Specht messages.

    `MessageDict` represents the aggregated message state ``m`` passed from the
    trainable message head to fixed branching. The dictionary key is the target
    :class:`Partition` for each message block.

    Tensor entries use shape ``[batch, channel, n..., a, a]``. The first axis is
    batch, the second is message channel, the next `partition.order` axes are
    ordered electron-label axes, and the final two axes are target
    irrep-coordinate and multiplicity/Fourier-column axes. Scalar target
    irreps use final shape ``[1, 1]``.

    Parameters
    ----------
    data : mapping or None, optional
        Optional mapping from :class:`Partition` keys to message tensors.
    """

    def __init__(self, data: Mapping[Partition, torch.Tensor] | None = None) -> None:
        self._data: dict[Partition, torch.Tensor] = {}
        if data is not None:
            self.update(data)

    def __getitem__(self, irrep: Partition) -> torch.Tensor:
        return self._data[irrep]

    def __setitem__(self, irrep: Partition, value: torch.Tensor) -> None:
        self._data[irrep] = value

    def __delitem__(self, irrep: Partition) -> None:
        del self._data[irrep]

    def __iter__(self) -> Iterator[Partition]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"MessageDict({{{', '.join(str(partition) for partition in self._data)}}})"

    def set(self, irrep: Partition, value: torch.Tensor) -> None:
        """Store a message tensor.

        Parameters
        ----------
        irrep : Partition
            Target Specht irrep partition.
        value : torch.Tensor
            Message tensor with shape ``[batch, channel, n..., a, a]``.
        """

        self[irrep] = value

    def get(self, irrep: Partition, default: Any = None):
        """Return a message tensor.

        Parameters
        ----------
        irrep : Partition
            Target Specht irrep partition.
        default : object, optional
            Value returned when the requested entry is absent.

        Returns
        -------
        object
            Message tensor or `default`.
        """

        return self._data.get(irrep, default)

    def has(self, irrep: Partition) -> bool:
        """Return whether an irrep-keyed message exists.

        Parameters
        ----------
        irrep : Partition
            Target Specht irrep partition.

        Returns
        -------
        bool
            ``True`` if the requested message tensor is present.
        """

        return irrep in self._data

    def flat_items(self) -> Iterator[tuple[Partition, torch.Tensor]]:
        """Iterate over flattened message entries.

        Yields
        ------
        tuple
            ``(partition, tensor)`` entries.
        """

        yield from self._data.items()

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "MessageDict":
        """Move every message tensor to a new device or dtype.

        Parameters
        ----------
        device : torch.device, str, or None, optional
            Target device. If ``None``, the current device is preserved.
        dtype : torch.dtype or None, optional
            Target dtype. If ``None``, the current dtype is preserved.

        Returns
        -------
        MessageDict
            Message container with tensors moved to the requested device or
            dtype.
        """

        return MessageDict({partition: tensor.to(device=device, dtype=dtype) for partition, tensor in self._data.items()})

    def clone(self) -> "MessageDict":
        """Clone all stored message tensors.

        Returns
        -------
        MessageDict
            Message container with cloned tensors.
        """

        return MessageDict({partition: tensor.clone() for partition, tensor in self._data.items()})

    def validate(self, *, batch_size: int | None = None, n_electrons: int | None = None) -> None:
        """Validate message tensor shape conventions.

        Parameters
        ----------
        batch_size : int or None, optional
            Expected leading batch size.
        n_electrons : int or None, optional
            Expected size of each ordered electron-label axis.

        Raises
        ------
        ValueError
            If any message tensor violates ``[batch, channel, n..., a, a]``.
        """

        for partition, tensor in self._data.items():
            validate_irrep_tensor(tensor, order=partition.order, irrep=partition, batch_size=batch_size, n_electrons=n_electrons)

    def to_dict(self) -> dict[Partition, torch.Tensor]:
        """Return a plain dictionary copy.

        Returns
        -------
        dict
            Deep copy of the underlying mapping.
        """

        return deepcopy(self._data)


__all__ = ["BranchDict", "FeatureDict", "MessageDict", "TensorProductDict"]
