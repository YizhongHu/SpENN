"""Base encoder interface."""

from __future__ import annotations

from torch import nn

from spenn.data.batch import ElectronBatch
from spenn.data.feature_dict import FeatureDict
from spenn.data.partitions import Partition


class BaseEncoder(nn.Module):
    """Encode electron batches into ordered-tuple feature dictionaries."""

    def output_keys(self) -> tuple[Partition, ...]:
        """Return the feature keys produced by the encoder.

        Returns
        -------
        tuple of Partition
            Partition keys that may appear in the returned feature dictionary.

        Raises
        ------
        NotImplementedError
            Always raised by the abstract base interface.
        """

        raise NotImplementedError("BaseEncoder.output_keys must be implemented by subclasses")

    def forward(self, batch: ElectronBatch) -> FeatureDict:
        """Encode an electron batch into feature blocks.

        Parameters
        ----------
        batch : ElectronBatch
            Electron positions and optional metadata.

        Returns
        -------
        FeatureDict
            Ordered-tuple feature blocks.

        Raises
        ------
        NotImplementedError
            Always raised by the abstract base interface.
        """

        raise NotImplementedError("BaseEncoder.forward must be implemented by subclasses")


__all__ = ["BaseEncoder"]
