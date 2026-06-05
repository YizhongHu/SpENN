"""Representation-theoretic machinery namespace."""

from spenn.reps.fourier import FourierTransform, InverseFourierTransform
from spenn.reps.irreps import IrrepMetadata, SpechtIrrep, irrep_dimension, specht_irrep
from spenn.reps.paths import VirtualPath, enumerate_virtual_paths, validate_virtual_path

__all__ = [
    "FourierTransform",
    "InverseFourierTransform",
    "IrrepMetadata",
    "SpechtIrrep",
    "VirtualPath",
    "enumerate_virtual_paths",
    "irrep_dimension",
    "specht_irrep",
    "validate_virtual_path",
]
