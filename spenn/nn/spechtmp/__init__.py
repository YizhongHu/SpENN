"""SpechtMP neural message-passing namespace."""

from spenn.nn.spechtmp.layer import SpechtMP, SpechtMPLayer
from spenn.nn.spechtmp.message_head import MessageHead
from spenn.nn.spechtmp.update_head import UpdateHead

__all__ = ["MessageHead", "SpechtMP", "SpechtMPLayer", "UpdateHead"]
