"""Equivariance-preserving activation module scaffolds."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations

import torch
from torch import nn

from spenn.data.irrep_features import FeatureDict, MessageDict
from spenn.data.partitions import Partition


class Activation(nn.Module):
    """Template for equivariant feature activations.

    Activation modules transform irrep-keyed feature or message blocks while
    preserving their partition keys and tensor shape contracts.
    """

    def forward(self, features: FeatureDict | MessageDict) -> FeatureDict:
        """Apply an equivariant activation to feature blocks.

        Parameters
        ----------
        features : FeatureDict or MessageDict
            Feature or message blocks to activate.

        Returns
        -------
        FeatureDict
            Activated feature blocks.

        Raises
        ------
        NotImplementedError
            Always raised by the template class.
        """

        raise NotImplementedError("Activation.forward must be implemented by subclasses")


class ActivationByType(Activation):
    """Route feature activations by Specht partition type.

    Parameters
    ----------
    symmetric : Activation
        Equivariant activation module for one-row partitions ``(order,)``.
    antisymmetric : Activation
        Equivariant activation module for sign partitions ``(1, ..., 1)`` with
        order greater than one.
    tensor : Activation
        Equivariant activation module for all remaining tensor irreps.
    **_ : object
        Ignored compatibility keyword arguments.

    Notes
    -----
    Routed modules must preserve equivariance. In practice they should usually
    be :class:`GatedActivation` instances rather than elementwise tensor
    nonlinearities.
    """

    def __init__(
        self,
        symmetric: Activation,
        antisymmetric: Activation,
        tensor: Activation,
        **_: object,
    ) -> None:
        super().__init__()
        self.symmetric = symmetric
        self.antisymmetric = antisymmetric
        self.tensor = tensor

    def forward(self, features: FeatureDict | MessageDict) -> FeatureDict:
        """Apply type-specific equivariant activations to each irrep.

        Parameters
        ----------
        features : FeatureDict or MessageDict
            Feature or message blocks to activate.

        Returns
        -------
        FeatureDict
            Activated feature blocks.
        """

        output = FeatureDict()
        for partition, tensor in features.flat_items():
            module = self._module_for(partition)
            value = module(FeatureDict({partition: tensor}))
            _merge_single(output, value)
        return output

    def _module_for(self, partition: Partition) -> Activation:
        if partition.parts == (partition.order,):
            return self.symmetric
        if partition.order > 1 and partition.parts == (1,) * partition.order:
            return self.antisymmetric
        return self.tensor


class ActivationByIrrep(Activation):
    """Route feature activations by exact partition key.

    Parameters
    ----------
    activations_by_irrep : mapping
        Mapping from :class:`Partition` keys to equivariant activation modules.
    **_ : object
        Ignored compatibility keyword arguments.

    Notes
    -----
    Routed modules must preserve equivariance. In practice they should usually
    be :class:`GatedActivation` instances rather than elementwise tensor
    nonlinearities.
    """

    def __init__(self, activations_by_irrep: Mapping[Partition, Activation], **_: object) -> None:
        super().__init__()
        modules: dict[Partition, Activation] = {}
        for partition, activation in activations_by_irrep.items():
            modules[partition] = activation
        self.activations_by_irrep = nn.ModuleDict(
            {_irrep_key(partition): activation for partition, activation in modules.items()}
        )
        self._key_by_partition = {partition: _irrep_key(partition) for partition in modules}

    def forward(self, features: FeatureDict | MessageDict) -> FeatureDict:
        """Apply irrep-specific activation modules.

        Parameters
        ----------
        features : FeatureDict or MessageDict
            Feature or message blocks to activate.

        Returns
        -------
        FeatureDict
            Activated feature blocks.

        Raises
        ------
        KeyError
            If no activation module is registered for a feature partition.
        """

        output = FeatureDict()
        for partition, tensor in features.flat_items():
            module_key = self._key_by_partition.get(partition)
            if module_key is None:
                raise KeyError(f"Missing activation module for partition {partition}")
            value = self.activations_by_irrep[module_key](FeatureDict({partition: tensor}))
            _merge_single(output, value)
        return output


class GatedActivation(Activation):
    """Apply a permutation-block-safe feature gate.

    Parameters
    ----------
    gate : torch.nn.Module
        Module called as ``gate(features)``. It must return a
        :class:`FeatureDict` with the same keys as `features` and tensors
        broadcast-compatible with the corresponding feature tensors.
    **_ : object
        Ignored compatibility keyword arguments.

    Notes
    -----
    The gate is projected before multiplication: ordered particle axes are
    averaged over their permutation block, and the transforming irrep-coordinate
    axis is averaged while preserving independent multiplicity/Fourier-column
    ``beta`` gates.
    """

    def __init__(self, gate: nn.Module, **_: object) -> None:
        super().__init__()
        self.gate = gate

    def forward(self, features: FeatureDict | MessageDict) -> FeatureDict:
        """Return keywise projected ``gate(features) * features``.

        Parameters
        ----------
        features : FeatureDict or MessageDict
            Feature or message blocks to gate.

        Returns
        -------
        FeatureDict
            Gated feature blocks.

        Raises
        ------
        KeyError
            If the gate omits a feature key.
        ValueError
            If a gate tensor cannot broadcast to its feature tensor.
        """

        gates = self.gate(features)
        output = FeatureDict()
        for partition, tensor in features.flat_items():
            gate_tensor = gates.get(partition)
            if gate_tensor is None:
                raise KeyError(f"Missing gate for feature key {partition}")
            projected_gate = _project_gate_to_permutation_block(partition, gate_tensor, tensor)
            output.set(partition, projected_gate * tensor)
        return output


def _project_gate_to_permutation_block(
    partition: Partition,
    gate: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Project a gate onto the block that can multiply an irrep tensor."""

    try:
        expanded = torch.broadcast_to(gate, tensor.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Gate for partition {partition} with shape {tuple(gate.shape)} must broadcast to "
            f"feature shape {tuple(tensor.shape)}"
        ) from exc

    projected = _average_over_ordered_tuple_permutations(partition, expanded)
    projected = _average_over_irrep_coordinate(projected)
    return projected.expand_as(tensor)


def _average_over_ordered_tuple_permutations(partition: Partition, tensor: torch.Tensor) -> torch.Tensor:
    """Average over permutations of ordered particle-label axes."""

    order = partition.order
    if order <= 1:
        return tensor

    tuple_axes = tuple(range(2, 2 + order))
    projected = torch.zeros_like(tensor)
    for axis_permutation in permutations(tuple_axes):
        dims = list(range(tensor.ndim))
        for destination, source in zip(tuple_axes, axis_permutation, strict=True):
            dims[destination] = source
        projected = projected + tensor.permute(dims)
    return projected / float(_factorial(order))


def _average_over_irrep_coordinate(tensor: torch.Tensor) -> torch.Tensor:
    """Average over the transforming irrep-coordinate axis only."""

    return tensor.mean(dim=-2, keepdim=True)


def _factorial(value: int) -> int:
    result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result


def _merge_single(output: FeatureDict, value: FeatureDict) -> None:
    for partition, tensor in value.flat_items():
        output.set(partition, tensor)


def _irrep_key(partition: Partition) -> str:
    return f"{partition.order}:{','.join(str(part) for part in partition.parts)}"


__all__ = [
    "Activation",
    "ActivationByIrrep",
    "ActivationByType",
    "GatedActivation",
]
