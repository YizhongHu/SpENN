"""Neural-network component namespace for SpENN."""

from spenn.nn.activation import Activation, GaussianActivation, GatedNormActivation
from spenn.nn.basis import (
    ElectronBasis,
    ElectronBasisFeatures,
    HookeHermiteBasis,
    HookeOrbitalBasis,
    RawCoordinateBasis,
)
from spenn.nn.context import SpENNForwardContext
from spenn.nn.coordinate_envelopes import CoordinateEnvelope, GaussianCoordinateEnvelope, RealCoordinateEnvelope
from spenn.nn.embedding import Embedding
from spenn.nn.envelope import (
    AdditiveEnvelope,
    ElectronElectronCusp,
    Envelope,
    HarmonicConfinement,
    HookeGaussianEnvelope,
)
from spenn.nn.equivariant_mixing import EquivariantMixing
from spenn.nn.initialization import SeededLinear, TorchInitializer
from spenn.nn.mlp import MLP
from spenn.nn.normalization import FeatureNormalization, IrrepRMSNorm
from spenn.nn.path_aggregation import PathAggregation
from spenn.nn.real_gates import RealGaussianNormGate, RealNormGate, RealRMSGate
from spenn.nn.scalar_gates import GaussianDecayGate, RMSInverseGate, ScalarGate, SigmoidGate, TanhGate
from spenn.nn.spenn_layer import SpENNLayer
from spenn.nn.spenn_wave_function import SpENNWaveFunction
from spenn.nn.update import ResidualUpdate, Update

__all__ = [
    "Activation",
    "AdditiveEnvelope",
    "CoordinateEnvelope",
    "ElectronBasis",
    "ElectronBasisFeatures",
    "ElectronElectronCusp",
    "Embedding",
    "Envelope",
    "EquivariantMixing",
    "FeatureNormalization",
    "GaussianActivation",
    "GaussianCoordinateEnvelope",
    "GaussianDecayGate",
    "GatedNormActivation",
    "HarmonicConfinement",
    "HookeGaussianEnvelope",
    "HookeHermiteBasis",
    "HookeOrbitalBasis",
    "IrrepRMSNorm",
    "MLP",
    "PathAggregation",
    "RawCoordinateBasis",
    "RMSInverseGate",
    "RealCoordinateEnvelope",
    "RealGaussianNormGate",
    "RealNormGate",
    "RealRMSGate",
    "ResidualUpdate",
    "ScalarGate",
    "SeededLinear",
    "SigmoidGate",
    "SpENNForwardContext",
    "SpENNLayer",
    "SpENNWaveFunction",
    "TanhGate",
    "TorchInitializer",
    "Update",
]
