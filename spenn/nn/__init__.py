"""Neural-network component namespace for SpENN."""

from spenn.nn.activation import Activation, ActivationByIrrep, ActivationByType, GatedNormActivation
from spenn.nn.cusp import Cusp, ElectronElectronCusp, NuclearCusp
from spenn.nn.embedding import Embedding
from spenn.nn.equivariant_mixing import EquivariantMixing
from spenn.nn.mlp import MLP
from spenn.nn.path_aggregation import PathAggregation
from spenn.nn.spenn_layer import SpENNLayer
from spenn.nn.spenn_wave_function import SpENNWaveFunction
from spenn.nn.update import (
    ChannelMappedUpdate,
    NormGatedUpdate,
    ReplaceUpdate,
    ResidualUpdate,
    Update,
)

__all__ = [
    "Activation",
    "ActivationByIrrep",
    "ActivationByType",
    "Cusp",
    "ChannelMappedUpdate",
    "ElectronElectronCusp",
    "Embedding",
    "EquivariantMixing",
    "GatedNormActivation",
    "MLP",
    "NormGatedUpdate",
    "NuclearCusp",
    "PathAggregation",
    "ReplaceUpdate",
    "ResidualUpdate",
    "SpENNLayer",
    "SpENNWaveFunction",
    "Update",
]
