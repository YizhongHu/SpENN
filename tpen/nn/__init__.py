"""Neural-network component namespace for TPEN."""

from tpen.nn.activation import GaussianActivation
from tpen.nn.basis import (
    ElectronBasis,
    ElectronBasisFeatures,
    HookeHermiteBasis,
    HookeOrbitalBasis,
    RawCoordinateBasis,
)
from tpen.nn.context import TPENForwardContext
from tpen.nn.coordinate_envelopes import (
    CoordinateEnvelope,
    GaussianCoordinateEnvelope,
    GaussianDecayGate,
)
from tpen.nn.embedding import Embedding
from tpen.nn.envelope import (
    AdditiveCusp,
    AdditiveEnvelope,
    AsymptoticDecay,
    ElectronElectronCusp,
    ElectronNucleusCusp,
    ElectronNucleusCuspLaw,
    Envelope,
    GaussianConfinement,
    HookeGaussianConfinement,
    LinearElectronNucleusCuspLaw,
    LogAmplitudeFactor,
)
from tpen.nn.equivariant_mixing import EquivariantMixing
from tpen.nn.initialization import SeededLinear, TorchInitializer
from tpen.nn.mlp import MLP
from tpen.nn.normalization import RMSNorm
from tpen.nn.path_aggregation import PathAggregation
from tpen.nn.spenn_layer import TPENLayer
from tpen.nn.spenn_wave_function import TPENWaveFunction
from tpen.nn.tpen_stack import TPENStack
from tpen.nn.update import ResidualUpdater, Updater

__all__ = [
    "AdditiveCusp",
    "AdditiveEnvelope",
    "AsymptoticDecay",
    "CoordinateEnvelope",
    "ElectronBasis",
    "ElectronBasisFeatures",
    "ElectronElectronCusp",
    "ElectronNucleusCusp",
    "ElectronNucleusCuspLaw",
    "Embedding",
    "Envelope",
    "EquivariantMixing",
    "GaussianActivation",
    "GaussianCoordinateEnvelope",
    "GaussianDecayGate",
    "GaussianConfinement",
    "HookeGaussianConfinement",
    "HookeHermiteBasis",
    "HookeOrbitalBasis",
    "LinearElectronNucleusCuspLaw",
    "LogAmplitudeFactor",
    "MLP",
    "PathAggregation",
    "RawCoordinateBasis",
    "RMSNorm",
    "ResidualUpdater",
    "SeededLinear",
    "TPENForwardContext",
    "TPENLayer",
    "TPENWaveFunction",
    "TPENStack",
    "TorchInitializer",
    "Updater",
]
