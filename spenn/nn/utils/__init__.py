"""Utility neural-network modules used by SpENN components."""

from spenn.nn.utils.activations import ActivationByIrrep, ActivationByType
from spenn.nn.utils.mlp import MLP
from spenn.nn.utils.update import NormGatedUpdate, ReplaceUpdate, ResidualUpdate

__all__ = [
    "ActivationByIrrep",
    "ActivationByType",
    "MLP",
    "NormGatedUpdate",
    "ReplaceUpdate",
    "ResidualUpdate",
]
