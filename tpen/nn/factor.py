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
no runtime deprecation warning in this minor version.
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

    def scalar_diagnostics(self) -> dict[str, float]:
        """Return this factor's trainable scalars as flat named floats.

        A factor that owns trainable scalars reports them here so a training
        callback can trace them without inspecting parameter containers. The
        contract is explicit and typed: the factor names its own quantities,
        and a consumer reads names rather than guessing which entry of
        ``named_parameters()`` means what.

        Reporting the CONSTRAINED value is the point. A range parameter held
        positive through a softplus reparameterization moves on a raw axis that
        is not the physical one, so a raw trace can show motion where the
        effective parameter has settled, or stillness where it has not. A
        factor that reports a reparameterized scalar should report the
        constrained value, and may report the raw value beside it.

        Returns
        -------
        dict
            Flat scalar mapping, empty for a factor that owns no scalars.
        """

        return {}


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

    def scalar_diagnostics(self) -> dict[str, float]:
        """Return every component's scalars, prefixed by component position.

        The index prefix keeps two components of the same class distinguishable,
        which a bare name would not.
        """

        scalars: dict[str, float] = {}
        for index, factor in enumerate(self.factors):
            for name, value in factor.scalar_diagnostics().items():
                scalars[f"factors.{index}.{name}"] = value
        return scalars


def _check_factor_tensor(value: object, batch: ElectronBatch, *, name: str) -> None:
    """Validate the shared additive-factor output value contract."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} output must be a torch.Tensor, got {type(value)!r}")
    expected = (batch.batch_size,)
    if value.shape != expected:
        raise ValueError(f"{name} output must have shape {expected}, got {tuple(value.shape)}")


#: Input above which the large-``x`` form of the inverse softplus is used.
#: ``expm1(x)`` overflows to ``inf`` for ``x`` above roughly 709.78 in float64,
#: which used to make this function return a non-finite raw parameter that
#: turned every downstream value and gradient into NaN -- silently, because
#: construction still succeeded. The crossover is far below that bound: by
#: ``x = 20`` the large-``x`` form is already sub-ulp accurate, while below it
#: ``expm1`` is what avoids the ``1 - exp(-x)`` cancellation.
_INVERSE_SOFTPLUS_LARGE_INPUT = 20.0


def _inverse_softplus(value: float) -> "torch.Tensor":
    """Return the softplus inverse used to initialize positive trainable parameters.

    Uses the algebraically exact identity ``log(expm1(x)) = x + log1p(-exp(-x))``
    for large ``x``, which is finite for every representable positive input, so
    a legitimate initial value is never rejected and never silently poisoned.

    The branch is taken on the Python float rather than with `torch.where`,
    which evaluates BOTH branches and would therefore still compute the
    overflowing form. That matters only if this is ever called inside a graph;
    today it is called on a Python float in ``__init__``.

    Parameters
    ----------
    value : float
        Positive target value of the parameter being initialized. Values below
        ``1e-12`` are floored, preserving the previous behaviour.

    Returns
    -------
    torch.Tensor
        Scalar float64 raw parameter whose softplus is ``value``.
    """

    value = max(value, 1e-12)
    tensor = torch.tensor(value, dtype=torch.float64)
    if value > _INVERSE_SOFTPLUS_LARGE_INPUT:
        return tensor + torch.log1p(-torch.exp(-tensor))
    return torch.log(torch.expm1(tensor))


__all__ = [
    "AdditiveCusp",
    "LogAmplitudeFactor",
]
