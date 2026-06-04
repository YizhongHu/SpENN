"""Real-space Specht update scaffolds."""

from __future__ import annotations

from typing import Literal

from spenn.data.base import EquivariantMap
from spenn.data.real_features import RealFeature, RealMessage


class MessageUpdate(EquivariantMap):
    """Merge real-space message proposals with optional previous messages.

    Parameters
    ----------
    mode : {"residual", "replace"}, optional
        Merge rule used by the placeholder implementation.
    """

    def __init__(self, mode: Literal["residual", "replace"] = "residual") -> None:
        super().__init__()
        if mode not in {"residual", "replace"}:
            raise ValueError("MessageUpdate mode must be 'residual' or 'replace'")
        self.mode = mode

    def forward(self, old: RealMessage | None, proposal: RealMessage) -> RealMessage:
        """Return an updated real-space message container.

        Parameters
        ----------
        old : RealMessage or None
            Previous real-space messages. If ``None``, the proposal is cloned.
        proposal : RealMessage
            Proposed real-space messages.

        Returns
        -------
        RealMessage
            Updated real-space messages.
        """

        if old is None or self.mode == "replace":
            return proposal.clone()
        return old.add(proposal)


class FeatureUpdate(EquivariantMap):
    """Merge real-space feature proposals with persistent features.

    Parameters
    ----------
    mode : {"residual", "replace"}, optional
        Merge rule used by the placeholder implementation.
    """

    def __init__(self, mode: Literal["residual", "replace"] = "residual") -> None:
        super().__init__()
        if mode not in {"residual", "replace"}:
            raise ValueError("FeatureUpdate mode must be 'residual' or 'replace'")
        self.mode = mode

    def forward(self, old: RealFeature, proposal: RealFeature) -> RealFeature:
        """Return an updated real-space feature container.

        Parameters
        ----------
        old : RealFeature
            Previous persistent real-space features.
        proposal : RealFeature
            Proposed real-space feature updates.

        Returns
        -------
        RealFeature
            Updated real-space features.
        """

        if self.mode == "replace":
            return proposal.clone()
        return old.add(proposal)


class RealToIrrepMessageUpdate(EquivariantMap):
    """Run a message update through temporary irrep-space coordinates.

    Parameters
    ----------
    fourier : EquivariantMap or None, optional
        Module mapping real messages to temporary irrep-space messages.
    irrep_update : EquivariantMap or None, optional
        Irrep-space update taking ``(old_hat, proposal_hat)``.
    inverse_fourier : EquivariantMap or None, optional
        Module reconstructing :class:`RealMessage` objects.
    """

    def __init__(
        self,
        fourier: EquivariantMap | None = None,
        irrep_update: EquivariantMap | None = None,
        inverse_fourier: EquivariantMap | None = None,
    ) -> None:
        super().__init__()
        self.fourier = fourier
        self.irrep_update = irrep_update
        self.inverse_fourier = inverse_fourier

    def forward(self, old: RealMessage | None, proposal: RealMessage) -> RealMessage:
        """Return a real-space message update via temporary irrep tensors.

        Parameters
        ----------
        old : RealMessage or None
            Previous real-space messages.
        proposal : RealMessage
            Real-space message proposal.

        Returns
        -------
        RealMessage
            Updated real-space messages.

        Raises
        ------
        NotImplementedError
            If any wrapper component has not been supplied.
        TypeError
            If reconstruction does not return a `RealMessage`.
        """

        if self.fourier is None or self.irrep_update is None or self.inverse_fourier is None:
            raise NotImplementedError("RealToIrrepMessageUpdate.forward requires fourier, irrep_update, and inverse_fourier")
        old_hat = None if old is None else self.fourier(old)
        proposal_hat = self.fourier(proposal)
        updated_hat = self.irrep_update(old_hat, proposal_hat)
        updated = self.inverse_fourier(updated_hat)
        if not isinstance(updated, RealMessage):
            raise TypeError("RealToIrrepMessageUpdate inverse_fourier must return a RealMessage")
        return updated


class RealToIrrepFeatureUpdate(EquivariantMap):
    """Run a feature update through temporary irrep-space coordinates.

    Parameters
    ----------
    fourier : EquivariantMap or None, optional
        Module mapping real features to temporary irrep-space features.
    irrep_update : EquivariantMap or None, optional
        Irrep-space update taking ``(old_hat, proposal_hat)``.
    inverse_fourier : EquivariantMap or None, optional
        Module reconstructing :class:`RealFeature` objects.
    """

    def __init__(
        self,
        fourier: EquivariantMap | None = None,
        irrep_update: EquivariantMap | None = None,
        inverse_fourier: EquivariantMap | None = None,
    ) -> None:
        super().__init__()
        self.fourier = fourier
        self.irrep_update = irrep_update
        self.inverse_fourier = inverse_fourier

    def forward(self, old: RealFeature, proposal: RealFeature) -> RealFeature:
        """Return a real-space feature update via temporary irrep tensors.

        Parameters
        ----------
        old : RealFeature
            Previous real-space features.
        proposal : RealFeature
            Real-space feature proposal.

        Returns
        -------
        RealFeature
            Updated real-space features.

        Raises
        ------
        NotImplementedError
            If any wrapper component has not been supplied.
        TypeError
            If reconstruction does not return a `RealFeature`.
        """

        if self.fourier is None or self.irrep_update is None or self.inverse_fourier is None:
            raise NotImplementedError("RealToIrrepFeatureUpdate.forward requires fourier, irrep_update, and inverse_fourier")
        old_hat = self.fourier(old)
        proposal_hat = self.fourier(proposal)
        updated_hat = self.irrep_update(old_hat, proposal_hat)
        updated = self.inverse_fourier(updated_hat)
        if not isinstance(updated, RealFeature):
            raise TypeError("RealToIrrepFeatureUpdate inverse_fourier must return a RealFeature")
        return updated


__all__ = [
    "FeatureUpdate",
    "MessageUpdate",
    "RealToIrrepFeatureUpdate",
    "RealToIrrepMessageUpdate",
]
