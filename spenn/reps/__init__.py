"""Representation-theoretic machinery namespace.

This package will contain fixed, non-learned Specht and symmetric-group maps.
"""

from spenn.reps.branch import BranchMap
from spenn.reps.fusion import FusionMap

__all__ = ["BranchMap", "FusionMap"]
