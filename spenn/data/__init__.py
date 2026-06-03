"""Data-structure namespace for feature and tensor containers."""

from spenn.data.feature_dict import BranchDict, FeatureDict, MessageDict, TensorProductDict
from spenn.data.batch import ElectronBatch, Walkers, WavefunctionOutput
from spenn.data.equivariant_map import EquivariantMap
from spenn.data.irrep_features import IrrepFeature, IrrepMessage
from spenn.data.irrep_tensor import IrrepTensor
from spenn.data.partitions import (
    Par,
    Partition,
    as_partition,
    format_partition,
    integer_partitions,
    normalize_partition,
    normalize_partition_keys,
    partition_size,
    transpose_partition,
    validate_partition,
)
from spenn.data.permutation import Permutation
from spenn.data.real_features import ConcatenatedState, RealFeature, RealMessage, RealTensors, SpechtMPState
from spenn.data.subset_index import (
    all_ordered_tuples,
    all_pairs,
    all_subsets,
    all_triples,
    subset_complement,
    subset_key,
)

__all__ = [
    "ConcatenatedState",
    "ElectronBatch",
    "EquivariantMap",
    "BranchDict",
    "FeatureDict",
    "IrrepFeature",
    "IrrepMessage",
    "IrrepTensor",
    "MessageDict",
    "Par",
    "Partition",
    "Permutation",
    "RealFeature",
    "RealMessage",
    "RealTensors",
    "SpechtMPState",
    "TensorProductDict",
    "Walkers",
    "WavefunctionOutput",
    "all_ordered_tuples",
    "all_pairs",
    "all_subsets",
    "all_triples",
    "as_partition",
    "format_partition",
    "integer_partitions",
    "normalize_partition",
    "normalize_partition_keys",
    "partition_size",
    "subset_complement",
    "subset_key",
    "transpose_partition",
    "validate_partition",
]
