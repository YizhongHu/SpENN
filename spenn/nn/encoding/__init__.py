"""Encoding modules for electron and pair features."""

from spenn.nn.encoding.base import BaseEncoder
from spenn.nn.encoding.electron_features import ElectronPairEncoder

__all__ = ["BaseEncoder", "ElectronPairEncoder"]
