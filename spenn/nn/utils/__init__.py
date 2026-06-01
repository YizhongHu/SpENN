"""Utility neural-network modules used by SpENN components."""

from spenn.nn.utils.activations import (
    Activation,
    ActivationByIrrep,
    ActivationByType,
    GatedActivation,
)
from spenn.nn.utils.gate import GateActivate, GateUpdate, NormGateActivate, ScalarGateActivate, ScalarGateUpdate
from spenn.nn.utils.mlp import MLP
from spenn.nn.utils.update import CompositeUpdate, GatedUpdate, RawUpdate, ResidualUpdate, Update, UpdateByIrrep, UpdateByType

__all__ = [
    "ActivationByIrrep",
    "ActivationByType",
    "Activation",
    "CompositeUpdate",
    "GateActivate",
    "GateUpdate",
    "GatedActivation",
    "GatedUpdate",
    "MLP",
    "NormGateActivate",
    "RawUpdate",
    "ResidualUpdate",
    "ScalarGateActivate",
    "ScalarGateUpdate",
    "Update",
    "UpdateByIrrep",
    "UpdateByType",
]
