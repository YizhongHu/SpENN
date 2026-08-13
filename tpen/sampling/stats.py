"""Typed sampler diagnostics record owned by the sampling domain.

`SamplerStats` is the value a sampler hands back alongside its walkers. It is a
typed record, not a payload dict: consumers read named fields and never probe it
with ``getattr``. Metric-name composition for the logging edge lives here too,
so the durable ``*/sampler`` and ``checks/sampler`` key sets are spelled in
exactly one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class SamplerStats:
    """Diagnostics describing one production sampling call.

    Every field except `geometry` is a fixed, named sampler property. Walker
    geometry is deliberately different: its key set is computed by
    `tpen.sampling.summarize_walker_geometry` and varies with ``n_electrons``,
    so it stays a constrained mapping *field inside* this typed record rather
    than being flattened into named attributes. That mapping is metric names to
    metric values only; it is not a general payload and must not be probed for
    behavior.

    The record is hashable. `geometry` is stored as an unhashable read-only
    mapping view, so it is excluded from the generated ``__hash__`` via
    ``field(hash=False)`` while still taking part in ``__eq__``. Equal records
    therefore still hash equal, and ``hash(stats)`` cannot raise.

    Parameters
    ----------
    acceptance_rate : float
        Mean Metropolis-Hastings acceptance rate over the production steps.
    n_walkers : int
        Number of walkers in the returned chain state.
    burn_in : int
        Configured number of one-time equilibration steps for the chain.
    n_steps : int
        Number of production MCMC steps taken by this call.
    proposal_scale : float
        Effective proposal scale of the move kernel.
    geometry : Mapping[str, float], optional
        Walker-geometry metrics from `summarize_walker_geometry`. Integer
        counts (``n_electrons``, ``spatial_dim``, ...) are admitted under the
        usual numeric-tower reading of ``float`` and are preserved as ``int``.
    seed : int or None, optional
        Sampler-local Markov-chain seed, when the sampler was seeded.
    """

    acceptance_rate: float
    n_walkers: int
    burn_in: int
    n_steps: int
    proposal_scale: float
    # hash=False: the stored MappingProxyType is unhashable, and a frozen
    # dataclass otherwise generates a __hash__ that raises on every call.
    geometry: Mapping[str, float] = field(default_factory=dict, hash=False)
    seed: int | None = None

    def __post_init__(self) -> None:
        # Coerce here rather than at each construction site, so every record --
        # sampler-built or test-built -- carries plain JSON-safe scalars.
        object.__setattr__(self, "acceptance_rate", float(self.acceptance_rate))
        object.__setattr__(self, "n_walkers", int(self.n_walkers))
        object.__setattr__(self, "burn_in", int(self.burn_in))
        object.__setattr__(self, "n_steps", int(self.n_steps))
        object.__setattr__(self, "proposal_scale", float(self.proposal_scale))
        object.__setattr__(self, "seed", None if self.seed is None else int(self.seed))
        if not isinstance(self.geometry, Mapping):
            raise TypeError(
                f"geometry must be a Mapping of metric names, got {type(self.geometry).__name__}"
            )
        if not all(isinstance(key, str) for key in self.geometry):
            raise TypeError("geometry metric names must be strings")
        # A read-only view keeps the frozen record genuinely immutable while
        # still comparing equal to the plain dict it was built from.
        object.__setattr__(self, "geometry", MappingProxyType(dict(self.geometry)))

    @property
    def reported_n_walkers(self) -> int:
        """Return the walker count that every metric view reports.

        Geometry wins over the `n_walkers` field when it carries the key,
        because geometry is measured on the walkers actually returned while the
        field may record configured capacity. Both `as_metrics` and
        `as_check_metrics` read this one accessor, so ``*/sampler/n_walkers``
        and ``checks/sampler/n_walkers`` can never disagree. The pre-typed flat
        stats dict held a single geometry-overwritten ``n_walkers`` that both
        views read, so this also keeps the durable values unchanged.

        Returns
        -------
        int
            Geometry's ``n_walkers`` when present, otherwise the named field.
        """

        geometry_n_walkers = self.geometry.get("n_walkers")
        return self.n_walkers if geometry_n_walkers is None else int(geometry_n_walkers)

    def as_metrics(self) -> dict[str, float | int]:
        """Return the flat metric mapping logged under a ``*/sampler`` namespace.

        Returns
        -------
        dict
            Named sampler fields followed by the open geometry key set. The key
            spellings and the `reported_n_walkers` precedence reproduce the
            pre-typed flat stats dict exactly, so durable metric identities such
            as ``train/sampler/acceptance_rate`` are unchanged.
        """

        metrics: dict[str, float | int] = {
            "acceptance_rate": self.acceptance_rate,
            "n_walkers": self.reported_n_walkers,
            "burn_in": self.burn_in,
            "n_steps": self.n_steps,
            "proposal_scale": self.proposal_scale,
        }
        if self.seed is not None:
            metrics["seed"] = self.seed
        # Geometry keys are appended verbatim. Its ``n_walkers``, when present,
        # rewrites the slot above with the identical resolved value, so the key
        # keeps its original position and value.
        metrics.update(self.geometry)
        return metrics

    def as_check_metrics(self) -> dict[str, float | int]:
        """Return the sampler-health subset logged under ``checks/sampler``.

        Returns
        -------
        dict
            The four durable check keys in their published order, with the same
            `reported_n_walkers` precedence `as_metrics` uses. The ``passed``
            flag is owned by the health check, not by this record, and geometry
            is deliberately excluded: ``checks/sampler`` is a fixed-width
            runtime check, not a geometry dump.
        """

        return {
            "acceptance_rate": self.acceptance_rate,
            "n_walkers": self.reported_n_walkers,
            "n_steps": self.n_steps,
            "burn_in": self.burn_in,
        }


__all__ = ["SamplerStats"]
