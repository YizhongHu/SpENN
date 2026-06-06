"""Representation-theoretic machinery namespace."""

from spenn.reps.fourier import FourierTransform, InverseFourierTransform
from spenn.reps.irreps import (
    IrrepMetadata,
    SpechtIrrep,
    SpechtIrrepInfo,
    dimension_key,
    generate_irrep_tensor_cache,
    irrep_dimension,
    load_default_irrep_metadata,
    specht_irrep,
)
from spenn.reps.paths import (
    PathMetadata,
    VirtualPath,
    generate_virtual_paths,
    load_default_path_metadata,
    validate_virtual_path,
)
from spenn.reps.specht import specht_representation_matrix

__all__ = [
    "FourierTransform",
    "InverseFourierTransform",
    "IrrepMetadata",
    "PathMetadata",
    "SpechtIrrep",
    "SpechtIrrepInfo",
    "VirtualPath",
    "dimension_key",
    "generate_virtual_paths",
    "generate_irrep_tensor_cache",
    "irrep_dimension",
    "load_default_irrep_metadata",
    "load_default_path_metadata",
    "specht_irrep",
    "specht_representation_matrix",
    "validate_virtual_path",
]
