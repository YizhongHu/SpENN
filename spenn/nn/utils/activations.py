"""Activation helpers for irrep feature scaffolds."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from spenn.data import IrrepFeature, Partition
from spenn.data.partitions import as_partition
from spenn.nn.equivariant_map import EquivariantMap


class ActivationByType(EquivariantMap):
    """Apply activation modules by partition type.

    Parameters
    ----------
    symmetric_activation : torch.nn.Module or None, optional
        Activation for symmetric irreps with partition ``(m)``.
    antisymmetric_activation : torch.nn.Module or None, optional
        Activation for antisymmetric irreps with partition ``(1, ..., 1)``.
    tensor_activation : torch.nn.Module or None, optional
        Activation for all other irreps.
    **kwargs : object
        Runtime-check options forwarded to :class:`spenn.nn.EquivariantMap`.
    """

    def __init__(
        self,
        *,
        symmetric_activation: nn.Module | None = None,
        antisymmetric_activation: nn.Module | None = None,
        tensor_activation: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.symmetric_activation = symmetric_activation
        self.antisymmetric_activation = antisymmetric_activation
        self.tensor_activation = tensor_activation

    def forward_impl(self, x: IrrepFeature) -> IrrepFeature:
        """Apply the selected activation to each irrep block."""

        return type(x)({partition: self._activation(partition)(tensor) for partition, tensor in x.items()})

    def _activation(self, partition: Partition) -> nn.Module:
        if partition.is_symmetric():
            return self.symmetric_activation if self.symmetric_activation is not None else nn.Identity()
        if partition.is_antisymmetric():
            return self.antisymmetric_activation if self.antisymmetric_activation is not None else nn.Identity()
        return self.tensor_activation if self.tensor_activation is not None else nn.Identity()


class ActivationByIrrep(EquivariantMap):
    """Apply activation modules selected independently for each irrep.

    Parameters
    ----------
    activations : mapping of partition-like to torch.nn.Module
        Per-irrep activation modules.
    default_activation : torch.nn.Module or None, optional
        Activation used for irreps absent from `activations`.
    **kwargs : object
        Runtime-check options forwarded to :class:`spenn.nn.EquivariantMap`.
    """

    def __init__(
        self,
        activations: Mapping[object, nn.Module] | None = None,
        *,
        default_activation: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        activations = {} if activations is None else dict(activations)
        self._activation_keys: dict[Partition, str] = {}
        modules = {}
        for raw_partition, module in activations.items():
            partition = as_partition(raw_partition)
            key = partition.key
            self._activation_keys[partition] = key
            modules[key] = module
        self.activations = nn.ModuleDict(modules)
        self.default_activation = default_activation

    def forward_impl(self, x: IrrepFeature) -> IrrepFeature:
        """Apply each configured irrep activation."""

        blocks = {}
        for partition, tensor in x.items():
            key = self._activation_keys.get(partition)
            if key is None:
                activation = self.default_activation if self.default_activation is not None else nn.Identity()
            else:
                activation = self.activations[key]
            blocks[partition] = activation(tensor)
        return type(x)(blocks)


__all__ = ["ActivationByIrrep", "ActivationByType"]
