"""Data-structure namespace for feature and tensor containers."""

from spenn.data_structures.feature_dict import FeatureDict
from spenn.data_structures.batch import ElectronBatch, Walkers, WavefunctionOutput
from spenn.data_structures.irrep_tensor import IrrepTensor
from spenn.data_structures.partitions import (
    Partition,
    PartitionLike,
    format_partition,
    integer_partitions,
    normalize_partition,
    partition_size,
    transpose_partition,
    validate_partition,
)
from spenn.data_structures.subset_index import (
    all_ordered_tuples,
    all_pairs,
    all_subsets,
    all_triples,
    subset_complement,
    subset_key,
)

__all__ = [
    "ElectronBatch",
    "FeatureDict",
    "IrrepTensor",
    "Partition",
    "PartitionLike",
    "Walkers",
    "WavefunctionOutput",
    "all_ordered_tuples",
    "all_pairs",
    "all_subsets",
    "all_triples",
    "format_partition",
    "integer_partitions",
    "normalize_partition",
    "partition_size",
    "subset_complement",
    "subset_key",
    "transpose_partition",
    "validate_partition",
]
