"""Equivariance-preserving activation module scaffolds."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from spenn.data.feature_dict import FeatureDict
from spenn.data.partitions import Partition


class TensorProductActivation(nn.Module):
    """Scaffold scalar and normed-tensor activations for Specht features.

    Parameters
    ----------
    scalar_activation : torch.nn.Module or None, optional
        Activation applied to scalar irrep channels.
    tensor_activation : torch.nn.Module or None, optional
        Activation applied to non-scalar tensor channels after the future
        normed-activation decomposition.
    **_ : object
        Ignored compatibility keyword arguments.

    Notes
    -----
    The concrete equivariant activation rule is intentionally not implemented
    in this scaffold. The future implementation should route scalar irreps
    through `scalar_activation` and normed non-scalar irreps through
    `tensor_activation`.
    """

    def __init__(
        self,
        scalar_activation: nn.Module | None = None,
        tensor_activation: nn.Module | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.scalar_activation = scalar_activation
        self.tensor_activation = tensor_activation

    def forward(self, features: FeatureDict) -> FeatureDict:
        """Apply the future tensor-product activation rule.

        Parameters
        ----------
        features : FeatureDict
            Feature blocks to activate.

        Returns
        -------
        FeatureDict
            Activated feature blocks.

        Raises
        ------
        NotImplementedError
            Always raised until the normed activation rule is implemented.
        """

        raise NotImplementedError("TensorProductActivation.forward is a scaffold pending normed activation math")


class ActivationByType(nn.Module):
    """Route feature activations by Specht partition type.

    Parameters
    ----------
    symmetric : torch.nn.Module
        Activation module for one-row partitions ``(order,)``.
    antisymmetric : torch.nn.Module
        Activation module for sign partitions ``(1, ..., 1)`` with order
        greater than one.
    tensor : torch.nn.Module
        Activation module for all remaining tensor irreps.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        symmetric: nn.Module,
        antisymmetric: nn.Module,
        tensor: nn.Module,
        **_: object,
    ) -> None:
        super().__init__()
        self.symmetric = symmetric
        self.antisymmetric = antisymmetric
        self.tensor = tensor

    def forward(self, features: FeatureDict) -> FeatureDict:
        """Apply type-specific activations to each irrep independently.

        Parameters
        ----------
        features : FeatureDict
            Feature blocks to activate.

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

    def _module_for(self, partition: Partition) -> nn.Module:
        if partition.parts == (partition.order,):
            return self.symmetric
        if partition.order > 1 and partition.parts == (1,) * partition.order:
            return self.antisymmetric
        return self.tensor


class ActivationByIrrep(nn.Module):
    """Route feature activations by exact partition key.

    Parameters
    ----------
    activations_by_irrep : mapping
        Mapping from :class:`Partition` keys to activation modules.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, activations_by_irrep: Mapping[Partition, nn.Module], **_: object) -> None:
        super().__init__()
        modules: dict[Partition, nn.Module] = {}
        for partition, activation in activations_by_irrep.items():
            modules[partition] = activation
        self.activations_by_irrep = nn.ModuleDict(
            {_irrep_key(partition): activation for partition, activation in modules.items()}
        )
        self._key_by_partition = {partition: _irrep_key(partition) for partition in modules}

    def forward(self, features: FeatureDict) -> FeatureDict:
        """Apply irrep-specific activation modules.

        Parameters
        ----------
        features : FeatureDict
            Feature blocks to activate.

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


class GatedActivation(nn.Module):
    """Apply a feature gate before returning activated features.

    Parameters
    ----------
    gate : torch.nn.Module
        Module called as ``gate(features)``. It must return a
        :class:`FeatureDict` with the same keys as `features` and tensors
        broadcast-compatible with the corresponding feature tensors.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, gate: nn.Module, **_: object) -> None:
        super().__init__()
        self.gate = gate

    def forward(self, features: FeatureDict) -> FeatureDict:
        """Return keywise ``gate(features) * features``.

        Parameters
        ----------
        features : FeatureDict
            Feature blocks to gate.

        Returns
        -------
        FeatureDict
            Gated feature blocks.

        Raises
        ------
        KeyError
            If the gate omits a feature key.
        """

        gates = self.gate(features)
        output = FeatureDict()
        for partition, tensor in features.flat_items():
            gate_tensor = gates.get(partition)
            if gate_tensor is None:
                raise KeyError(f"Missing gate for feature key {partition}")
            output.set(partition, gate_tensor * tensor)
        return output


def _merge_single(output: FeatureDict, value: FeatureDict) -> None:
    for partition, tensor in value.flat_items():
        output.set(partition, tensor)


def _irrep_key(partition: Partition) -> str:
    return f"{partition.order}:{','.join(str(part) for part in partition.parts)}"


__all__ = [
    "ActivationByIrrep",
    "ActivationByType",
    "GatedActivation",
    "TensorProductActivation",
]
