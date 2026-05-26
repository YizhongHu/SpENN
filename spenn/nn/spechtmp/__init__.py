"""SpechtMP neural message-passing namespace."""

from spenn.nn.spechtmp.brancher import SpechtBrancher
from spenn.nn.spechtmp.fuser import SpechtFuser
from spenn.nn.spechtmp.layer import SpechtMP, SpechtMPLayer
from spenn.nn.spechtmp.lowrank_virtual import LowRankVirtualBrancher

__all__ = ["LowRankVirtualBrancher", "SpechtBrancher", "SpechtFuser", "SpechtMP", "SpechtMPLayer"]
