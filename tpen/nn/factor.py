"""Canonical generic post-readout log-amplitude factor interfaces.

`LogAmplitudeFactor` is the new-generation counterpart to
`tpen.nn.envelope.Envelope`: a factor accepts an :class:`ElectronBatch` and
returns one scalar contribution to ``log |psi|`` per flattened configuration.
`TPENWaveFunction.factors` (see `tpen.nn.spenn_wave_function`) is the sole
canonical composition seam for `LogAmplitudeFactor` terms; direct
construction of a `factors` list of `LogAmplitudeFactor` instances is the
target path for new systems.

`AdditiveCusp` is retained here only as a legacy compatibility compositor
that generically sums `LogAmplitudeFactor` components (despite its name, it
is not cusp-specific): do not use it in new configs or docs, and it carries
no runtime deprecation warning in this minor version. `AsymptoticDecay` is a
separate, optional long-range decay interface, independent from both cusp
factors and legacy feature envelopes.
"""

from __future__ import annotations

from collections.abc import Iterable

from tpen.data.batch import ElectronBatch
from tpen.dependencies import require_torch, require_torch_nn

torch = require_torch(feature="TPEN factor modules")
nn = require_torch_nn(feature="TPEN factor modules")


class LogAmplitudeFactor(nn.Module):
    """Template for generic additive post-readout log-amplitude factors.

    This is the new-generation counterpart to `tpen.nn.envelope.Envelope`: a
    factor accepts an `ElectronBatch` and returns one scalar contribution to
    `log |psi|` per flattened configuration, with a value-only forward (no
    auxiliary radial derivative structure). It is deliberately separate from
    `Envelope` so the legacy compatibility surface never has to change shape
    to accommodate generic atom-system consumers.
    """

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        """Return a flattened-batch factor contribution.

        Parameters
        ----------
        batch : ElectronBatch
            Electron batch whose sample axes may be higher rank.

        Returns
        -------
        torch.Tensor
            Factor contribution with shape ``[batch]`` after sample
            flattening.
        """

        flat_batch = batch.flatten_samples()
        output = self.factor_value(flat_batch)
        _check_factor_tensor(output, flat_batch, name=type(self).__name__)
        return output

    def factor_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the factor contribution for a flattened batch.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch.

        Returns
        -------
        torch.Tensor
            Factor contribution with shape ``[batch]``.
        """

        raise NotImplementedError("LogAmplitudeFactor.factor_value must be implemented by subclasses")


class AdditiveCusp(LogAmplitudeFactor):
    """Generic composition summing typed `LogAmplitudeFactor` components.

    This is a legacy compatibility compositor only: do not use it in new
    configs or docs. Despite its name it composes any `LogAmplitudeFactor`
    components, not only cusp factors.

    Parameters
    ----------
    factors : iterable of LogAmplitudeFactor, optional
        Cusp (or other additive) factors whose outputs are summed. Each
        component must be a `LogAmplitudeFactor`; this is a typed-interface
        check, not container traversal or class-name matching.
    """

    def __init__(self, factors: Iterable["LogAmplitudeFactor"] = ()) -> None:
        super().__init__()
        factors = tuple(factors)
        for factor in factors:
            if not isinstance(factor, LogAmplitudeFactor):
                raise TypeError(
                    f"AdditiveCusp components must be LogAmplitudeFactor, got {type(factor).__name__}"
                )
        self.factors = nn.ModuleList(factors)

    def factor_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the sum of all component factor contributions."""

        total = torch.zeros(batch.batch_size, device=batch.device, dtype=batch.dtype)
        for index, factor in enumerate(self.factors):
            value = factor(batch)
            _check_factor_tensor(value, batch, name=f"factors[{index}]")
            total = total + value
        return total


class AsymptoticDecay(nn.Module):
    """Template for an optional long-range log-amplitude decay factor.

    This is a separate, optional capability from cusp factors
    (`LogAmplitudeFactor`/`AdditiveCusp`) and from legacy feature envelopes
    (`tpen.nn.envelope.Envelope`): it exists so a consumer that needs
    asymptotic decay can require this interface explicitly and fail loudly
    when it is absent, instead of a decay term being inferred or silently
    substituted.
    """

    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        """Return a flattened-batch decay contribution.

        Parameters
        ----------
        batch : ElectronBatch
            Electron batch whose sample axes may be higher rank.

        Returns
        -------
        torch.Tensor
            Decay contribution with shape ``[batch]`` after sample
            flattening.
        """

        flat_batch = batch.flatten_samples()
        output = self.decay_value(flat_batch)
        _check_factor_tensor(output, flat_batch, name=type(self).__name__)
        return output

    def decay_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the decay contribution for a flattened batch.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch.

        Returns
        -------
        torch.Tensor
            Decay contribution with shape ``[batch]``.
        """

        raise NotImplementedError("AsymptoticDecay.decay_value must be implemented by subclasses")


def _check_factor_tensor(value: object, batch: ElectronBatch, *, name: str) -> None:
    """Validate the shared additive-factor output value contract."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} output must be a torch.Tensor, got {type(value)!r}")
    expected = (batch.batch_size,)
    if value.shape != expected:
        raise ValueError(f"{name} output must have shape {expected}, got {tuple(value.shape)}")


def _inverse_softplus(value: float) -> "torch.Tensor":
    """Return the softplus inverse used to initialize positive trainable parameters."""

    value = max(value, 1e-12)
    tensor = torch.tensor(value, dtype=torch.float64)
    return torch.log(torch.expm1(tensor))


__all__ = [
    "AdditiveCusp",
    "AsymptoticDecay",
    "LogAmplitudeFactor",
]
