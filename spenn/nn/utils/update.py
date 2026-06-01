"""Feature-state update modules for SpechtMP."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from spenn.data.feature_dict import FeatureDict
from spenn.data.partitions import Partition


class Update(nn.Module):
    """Template for applying proposed feature updates.

    `Update` modules combine an incoming persistent feature state ``x`` with
    proposed branch updates ``dx``.
    """

    def forward(self, features: FeatureDict, updates: FeatureDict) -> FeatureDict:
        """Apply proposed updates to a feature state.

        Parameters
        ----------
        features : FeatureDict
            Current persistent feature state.
        updates : FeatureDict
            Proposed feature updates.

        Returns
        -------
        FeatureDict
            Next persistent feature state.

        Raises
        ------
        NotImplementedError
            Always raised by the template class.
        """

        raise NotImplementedError("Update.forward must be implemented by subclasses")


class RawUpdate(Update):
    """Replace the feature state with proposed updates."""

    def forward(self, features: FeatureDict, updates: FeatureDict) -> FeatureDict:
        """Return proposed updates as the next feature state.

        Parameters
        ----------
        features : FeatureDict
            Current persistent feature state. This argument is ignored.
        updates : FeatureDict
            Proposed feature updates.

        Returns
        -------
        FeatureDict
            The proposed updates.
        """

        return updates


class ResidualUpdate(Update):
    """Add proposed updates to the current feature state."""

    def forward(self, features: FeatureDict, updates: FeatureDict) -> FeatureDict:
        """Return the residual feature update ``features + updates``.

        Parameters
        ----------
        features : FeatureDict
            Current persistent feature state.
        updates : FeatureDict
            Proposed feature updates.

        Returns
        -------
        FeatureDict
            Keywise residual sum.
        """

        return features.add(updates)


class CompositeUpdate(Update):
    """Chain two feature-state update rules.

    Parameters
    ----------
    first : Update
        Outer update rule ``f1``.
    second : Update
        Inner update rule ``f2``.
    **_ : object
        Ignored compatibility keyword arguments.

    Notes
    -----
    The chained rule is ``f(x, dx) = f1(f2(x, dx), dx)``.
    """

    def __init__(self, first: Update, second: Update, **_: object) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, features: FeatureDict, updates: FeatureDict) -> FeatureDict:
        """Apply the inner rule followed by the outer rule.

        Parameters
        ----------
        features : FeatureDict
            Current persistent feature state.
        updates : FeatureDict
            Proposed feature updates.

        Returns
        -------
        FeatureDict
            Result of ``first(second(features, updates), updates)``.
        """

        return self.first(self.second(features, updates), updates)


class GatedUpdate(Update):
    """Apply a gate-delta update rule.

    Parameters
    ----------
    gate : torch.nn.Module
        Module called as ``gate(features, updates)``. It must return a
        :class:`FeatureDict` with the same keys as `updates` and tensors
        broadcast-compatible with the corresponding update tensors.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, gate: nn.Module, **_: object) -> None:
        super().__init__()
        self.gate = gate

    def forward(self, features: FeatureDict, updates: FeatureDict) -> FeatureDict:
        """Return ``features + gate(features, updates) * updates``.

        Parameters
        ----------
        features : FeatureDict
            Current persistent feature state.
        updates : FeatureDict
            Proposed feature updates.

        Returns
        -------
        FeatureDict
            Gated residual feature state.

        Raises
        ------
        KeyError
            If the gate omits an update key.
        """

        gates = self.gate(features, updates)
        gated_updates = FeatureDict()
        for partition, tensor in updates.flat_items():
            gate_tensor = gates.get(partition)
            if gate_tensor is None:
                raise KeyError(f"Missing gate for update key {partition}")
            gated_updates.set(partition, gate_tensor * tensor)
        return features.add(gated_updates)


class UpdateByType(Update):
    """Route update behavior by Specht partition type.

    Parameters
    ----------
    symmetric : Update
        Update module for one-row partitions ``(order,)``.
    antisymmetric : Update
        Update module for sign partitions ``(1, ..., 1)`` with order greater
        than one.
    tensor : Update
        Update module for all remaining tensor irreps.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        symmetric: Update,
        antisymmetric: Update,
        tensor: Update,
        **_: object,
    ) -> None:
        super().__init__()
        self.symmetric = symmetric
        self.antisymmetric = antisymmetric
        self.tensor = tensor

    def forward(self, features: FeatureDict, updates: FeatureDict) -> FeatureDict:
        """Apply type-specific update modules to each irrep independently.

        Parameters
        ----------
        features : FeatureDict
            Current persistent feature state.
        updates : FeatureDict
            Proposed feature updates.

        Returns
        -------
        FeatureDict
            Combined next feature state.
        """

        output = FeatureDict()
        for partition, update_tensor in updates.flat_items():
            module = self._module_for(partition)
            _merge_single(output, module(_single_feature(features, partition), FeatureDict({partition: update_tensor})))
        return output

    def _module_for(self, partition: Partition) -> Update:
        if partition.parts == (partition.order,):
            return self.symmetric
        if partition.order > 1 and partition.parts == (1,) * partition.order:
            return self.antisymmetric
        return self.tensor


class UpdateByIrrep(Update):
    """Route update behavior by exact partition key.

    Parameters
    ----------
    updates_by_irrep : mapping
        Mapping from :class:`Partition` keys to update modules.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, updates_by_irrep: Mapping[Partition, Update], **_: object) -> None:
        super().__init__()
        modules: dict[Partition, Update] = {}
        for partition, update in updates_by_irrep.items():
            modules[partition] = update
        self.updates_by_irrep = nn.ModuleDict({_irrep_key(partition): update for partition, update in modules.items()})
        self._key_by_partition = {partition: _irrep_key(partition) for partition in modules}

    def forward(self, features: FeatureDict, updates: FeatureDict) -> FeatureDict:
        """Apply irrep-specific update modules to each irrep independently.

        Parameters
        ----------
        features : FeatureDict
            Current persistent feature state.
        updates : FeatureDict
            Proposed feature updates.

        Returns
        -------
        FeatureDict
            Combined next feature state.

        Raises
        ------
        KeyError
            If no update module is registered for an update partition.
        """

        output = FeatureDict()
        for partition, update_tensor in updates.flat_items():
            module_key = self._key_by_partition.get(partition)
            if module_key is None:
                raise KeyError(f"Missing update module for partition {partition}")
            module = self.updates_by_irrep[module_key]
            _merge_single(output, module(_single_feature(features, partition), FeatureDict({partition: update_tensor})))
        return output


def _single_feature(features: FeatureDict, partition: Partition) -> FeatureDict:
    tensor = features.get(partition)
    if tensor is None:
        return FeatureDict()
    return FeatureDict({partition: tensor})


def _merge_single(output: FeatureDict, value: FeatureDict) -> None:
    for partition, tensor in value.flat_items():
        output.set(partition, tensor)


def _irrep_key(partition: Partition) -> str:
    return f"{partition.order}:{','.join(str(part) for part in partition.parts)}"


__all__ = [
    "CompositeUpdate",
    "GatedUpdate",
    "RawUpdate",
    "ResidualUpdate",
    "Update",
    "UpdateByIrrep",
    "UpdateByType",
]
