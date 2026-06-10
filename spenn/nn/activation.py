"""Irrep activation modules."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from spenn.data.irrep import IrrepFeature, IrrepInteraction
from spenn.data.partition import Partition, as_partition
from spenn.equivariance import EquivariantMap


class Activation(EquivariantMap):
    """Base class for irrep activation maps.

    Activation modules preserve the input irrep state type and tuple/path
    geometry while applying pointwise equivariant transformations to each irrep
    block.
    """


class GatedNormActivation(Activation):
    """Gate every irrep block by a module applied to invariant norms.

    Parameters
    ----------
    gate : torch.nn.Module
        Scalar module applied to alpha-coordinate norms.
    eps : float, optional
        Numerical floor for the norm.
    """

    def __init__(
        self,
        gate: nn.Module,
        *,
        eps: float = 1.0e-12,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gate = gate
        self.eps = float(eps)

    def forward_impl(self, x: IrrepFeature | IrrepInteraction) -> IrrepFeature | IrrepInteraction:
        """Scale each irrep vector by a scalar function of ``||alpha||``."""

        return type(x)({partition: self._apply_gate(tensor) for partition, tensor in x.items()})

    def _apply_gate(self, tensor: torch.Tensor) -> torch.Tensor:
        norm = (tensor.square().sum(dim=-2, keepdim=True) + self.eps).sqrt()
        return tensor * self.gate(norm)


class ActivationByType(Activation):
    """Apply equivariant activation rules by partition type.

    Symmetric and antisymmetric scalar irreps receive their own scalar modules.
    Higher-dimensional irreps are gated by a scalar function of the transforming
    alpha-coordinate norm and then broadcast over alpha coordinates. The input
    state type is preserved, so path-resolved :class:`IrrepInteraction` blocks
    keep their path axis through activation.

    Parameters
    ----------
    symmetric_activation : torch.nn.Module or None, optional
        Activation for symmetric irreps with partition ``(m)``.
    antisymmetric_activation : torch.nn.Module or None, optional
        Activation for antisymmetric irreps with partition ``(1, ..., 1)``.
    tensor_activation : torch.nn.Module or None, optional
        Scalar gate applied to higher-dimensional irrep norms. If ``None``,
        tensor blocks are left unchanged.
    eps : float, optional
        Numerical floor used when computing alpha-coordinate norms.
    antisymmetric_odd_check : bool, optional
        Whether to verify on each antisymmetric block that the supplied
        activation satisfies ``f(-x) = -f(x)`` on the current tensor values.
    odd_check_atol, odd_check_rtol : float, optional
        Tolerances for the antisymmetric oddness check.
    **kwargs : object
        Runtime-check options forwarded to :class:`spenn.data.EquivariantMap`.
    """

    def __init__(
        self,
        *,
        symmetric_activation: nn.Module | None = None,
        antisymmetric_activation: nn.Module | None = None,
        tensor_activation: nn.Module | None = None,
        eps: float = 1.0e-12,
        antisymmetric_odd_check: bool = True,
        odd_check_atol: float = 1.0e-6,
        odd_check_rtol: float = 1.0e-5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.symmetric_activation = symmetric_activation
        self.antisymmetric_activation = antisymmetric_activation
        self.tensor_activation = tensor_activation
        self.eps = float(eps)
        self.antisymmetric_odd_check = bool(antisymmetric_odd_check)
        self.odd_check_atol = float(odd_check_atol)
        self.odd_check_rtol = float(odd_check_rtol)

    def forward_impl(self, x: IrrepFeature | IrrepInteraction) -> IrrepFeature | IrrepInteraction:
        """Apply the selected activation to each irrep block."""

        return type(x)({partition: self._apply_partition(partition, tensor) for partition, tensor in x.items()})

    def _apply_partition(self, partition: Partition, tensor: torch.Tensor) -> torch.Tensor:
        if partition.is_symmetric():
            return tensor if self.symmetric_activation is None else self.symmetric_activation(tensor)
        if partition.is_antisymmetric():
            return self._apply_antisymmetric(tensor)
        if self.tensor_activation is None:
            return tensor
        norm = tensor.square().sum(dim=-2, keepdim=True).clamp_min(self.eps).sqrt()
        return tensor * self.tensor_activation(norm)

    def _apply_antisymmetric(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.antisymmetric_activation is None:
            return tensor
        output = self.antisymmetric_activation(tensor)
        if self.antisymmetric_odd_check:
            reflected = self.antisymmetric_activation(-tensor)
            try:
                torch.testing.assert_close(
                    reflected,
                    -output,
                    atol=self.odd_check_atol,
                    rtol=self.odd_check_rtol,
                )
            except AssertionError as exc:
                raise ValueError(
                    "antisymmetric_activation must be odd on sign-irrep blocks: f(-x) = -f(x)"
                ) from exc
        return output


class ActivationByIrrep(Activation):
    """Apply activation modules selected independently for each irrep.

    Parameters
    ----------
    activations : mapping of partition-like to torch.nn.Module
        Per-irrep activation modules.
    default_activation : torch.nn.Module or None, optional
        Activation used for irreps absent from `activations`.
    **kwargs : object
        Runtime-check options forwarded to :class:`spenn.data.EquivariantMap`.
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

    def forward_impl(self, x: IrrepFeature | IrrepInteraction) -> IrrepFeature | IrrepInteraction:
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


__all__ = ["Activation", "ActivationByIrrep", "ActivationByType", "GatedNormActivation"]
