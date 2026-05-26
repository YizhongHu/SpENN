"""General utility namespace for SpENN."""

from spenn.utils.index_utils import all_subsets, pair_indices, triple_indices
from spenn.utils.tensor_utils import (
    antisymmetrize_pair_tensor,
    pairwise_displacements,
    pairwise_distances,
    resolve_dtype,
    symmetrize_pair_tensor,
    upper_triangle_indices,
)

__all__ = [
    "all_subsets",
    "antisymmetrize_pair_tensor",
    "pair_indices",
    "pairwise_displacements",
    "pairwise_distances",
    "resolve_dtype",
    "symmetrize_pair_tensor",
    "triple_indices",
    "upper_triangle_indices",
]
