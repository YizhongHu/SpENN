"""Real-space tensor containers for the SpechtMP scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from spenn.data.base import ConcatenatedState as BaseConcatenatedState
from spenn.data.base import SpechtMPState
from spenn.data.permutation import Permutation


def _validate_real_tensor(tensor: torch.Tensor, *, order: int) -> None:
    """Validate one real tensor block.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor block to validate.
    order : int
        Body order associated with the block.

    Raises
    ------
    TypeError
        If `tensor` is not a :class:`torch.Tensor`.
    ValueError
        If `tensor` does not have shape ``[batch, channels, n, ...]`` with
        exactly `order` particle axes.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Real tensor block {order} must be a torch.Tensor")
    expected_ndim = order + 2
    if tensor.ndim != expected_ndim:
        raise ValueError(
            f"Expected order-{order} tensor with {expected_ndim} dimensions "
            f"[batch, channels, n, ...], got shape {tuple(tensor.shape)}"
        )
    if order > 0:
        n_particles = tensor.shape[2]
        for axis in range(2, 2 + order):
            if tensor.shape[axis] != n_particles:
                raise ValueError(
                    f"Expected every particle axis in order-{order} tensor to have "
                    f"length {n_particles}, got axis {axis} length {tensor.shape[axis]}"
                )


def _validate_common_particle_count(data: list[torch.Tensor]) -> None:
    """Validate that non-scalar tensor blocks share one particle count."""

    n_particles: int | None = None
    for order, tensor in enumerate(data):
        if order == 0:
            continue
        current = int(tensor.shape[2])
        if n_particles is None:
            n_particles = current
        elif current != n_particles:
            raise ValueError(
                "Real tensor blocks must share one particle-axis length, "
                f"got {n_particles} and {current}"
            )


def _permute_tensor(tensor: torch.Tensor, *, order: int, permutation: Permutation) -> torch.Tensor:
    """Permute one real tensor block."""

    _validate_real_tensor(tensor, order=order)
    if order == 0:
        return tensor.clone()
    if tensor.shape[2] != len(permutation):
        raise ValueError(
            f"Permutation of size {len(permutation)} is incompatible with "
            f"order-{order} tensor particle axes of length {tensor.shape[2]}"
        )
    index = torch.tensor(permutation.inverse().image, device=tensor.device, dtype=torch.long)
    output = tensor
    for axis in range(2, 2 + order):
        output = output.index_select(axis, index)
    return output


@dataclass(frozen=True)
class RealTensors:
    """Store real-space tensor blocks by body order.

    Tensor blocks use shape ``[batch, channels, n, ...]``. The list index is
    the body order, so ``data[k]`` has exactly `k` particle-label axes.

    Parameters
    ----------
    data : list of torch.Tensor
        Tensor blocks ordered by body order.
    """

    data: list[torch.Tensor] = field(default_factory=list)

    def __post_init__(self) -> None:
        data = list(self.data)
        for order, tensor in enumerate(data):
            _validate_real_tensor(tensor, order=order)
        _validate_common_particle_count(data)
        object.__setattr__(self, "data", data)

    def __len__(self) -> int:
        """Return the number of stored body-order blocks.

        Returns
        -------
        int
            Length of `data`.
        """

        return len(self.data)

    def __getitem__(self, order: int) -> torch.Tensor:
        """Return a tensor block by body order.

        Parameters
        ----------
        order : int
            Body order.

        Returns
        -------
        torch.Tensor
            Tensor block for `order`.
        """

        return self.data[order]

    def clone(self) -> "RealTensors":
        """Clone all tensor blocks.

        Returns
        -------
        RealTensors
            New container of the same concrete type with cloned tensor
            storage.
        """

        return type(self)([tensor.clone() for tensor in self.data])

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "RealTensors":
        """Move every tensor block to a new device or dtype.

        Parameters
        ----------
        device : torch.device, str, or None, optional
            Target device. If ``None``, the current device is preserved.
        dtype : torch.dtype or None, optional
            Target dtype. If ``None``, the current dtype is preserved.

        Returns
        -------
        RealTensors
            New container of the same concrete type with moved tensor blocks.
        """

        return type(self)([tensor.to(device=device, dtype=dtype) for tensor in self.data])

    def add(self, other: "RealTensors") -> "RealTensors":
        """Return the blockwise tensor sum with another container.

        Parameters
        ----------
        other : RealTensors
            Tensor updates to add.

        Returns
        -------
        RealTensors
            New container containing blockwise sums.

        Raises
        ------
        ValueError
            If the containers do not store the same body orders.
        """

        if len(self.data) != len(other.data):
            raise ValueError(f"Cannot add real tensor lists of lengths {len(self.data)} and {len(other.data)}")
        return type(self)([left + right for left, right in zip(self.data, other.data)])

    def __add__(self, other: "RealTensors") -> "RealTensors":
        """Return the blockwise tensor sum with another container."""

        return self.add(other)

    def permute(self, permutation: Permutation) -> "RealTensors":
        """Return a copy transformed by an active permutation.

        Parameters
        ----------
        permutation : Permutation
            Permutation applied to every particle-label axis by indexing with
            ``permutation.image``.

        Returns
        -------
        RealTensors
            New container with permuted tensor blocks.
        """

        return type(self)(
            [
                _permute_tensor(tensor, order=order, permutation=permutation)
                for order, tensor in enumerate(self.data)
            ]
        )


@dataclass(frozen=True)
class RealFeature(RealTensors):
    """Store persistent real-space feature tensor blocks."""


@dataclass(frozen=True)
class RealMessage(RealTensors):
    """Store real-space message tensor blocks."""


class RealConcatenatedState(BaseConcatenatedState):
    """Store real-space feature and optional message states together.

    Parameters
    ----------
    features : RealFeature
        Persistent feature state.
    messages : RealMessage or None, optional
        Optional message state.
    """

    def __init__(self, features: RealFeature | None = None, messages: RealMessage | None = None) -> None:
        features = RealFeature() if features is None else features
        if not isinstance(features, RealFeature):
            raise TypeError("ConcatenatedState.features must be a RealFeature")
        if messages is not None and not isinstance(messages, RealMessage):
            raise TypeError("ConcatenatedState.messages must be a RealMessage or None")
        data: tuple[SpechtMPState, ...]
        if messages is None:
            data = (features,)
        else:
            data = (features, messages)
        super().__init__(data=data)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "messages", messages)

    def permute(self, permutation: Permutation) -> "RealConcatenatedState":
        """Return a copy transformed by an active permutation.

        Parameters
        ----------
        permutation : Permutation
            Permutation applied to every particle-label axis.

        Returns
        -------
        RealConcatenatedState
            New state with permuted feature and message blocks.
        """

        messages = None if self.messages is None else self.messages.permute(permutation)
        return RealConcatenatedState(features=self.features.permute(permutation), messages=messages)


__all__ = [
    "RealConcatenatedState",
    "RealFeature",
    "RealMessage",
    "RealTensors",
]
