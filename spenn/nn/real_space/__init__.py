"""Real-space SpechtMP neural scaffolds."""

from spenn.nn.real_space.activation import SpechtFeatureActivation, SpechtMessageActivation
from spenn.nn.real_space.convolution import Convolution
from spenn.nn.real_space.embedding import Embedding
from spenn.nn.real_space.layer import RealSpechtMPLayer
from spenn.nn.real_space.pooling import Pooling
from spenn.nn.real_space.update import (
    FeatureUpdate,
    MessageUpdate,
    RealToIrrepFeatureUpdate,
    RealToIrrepMessageUpdate,
)

__all__ = [
    "Convolution",
    "Embedding",
    "FeatureUpdate",
    "MessageUpdate",
    "Pooling",
    "RealSpechtMPLayer",
    "RealToIrrepFeatureUpdate",
    "RealToIrrepMessageUpdate",
    "SpechtFeatureActivation",
    "SpechtMessageActivation",
]
