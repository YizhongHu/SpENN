"""MCMC generator that retains a ``[draw, walker]`` observable trajectory.

:class:`tpen.evaluation.generators.MCMCGenerator` takes a single snapshot of the
sampler's walkers. A snapshot has no time axis, so nothing downstream can
estimate an autocorrelation time from it, and the only error bar available is
the IID ``sigma / sqrt(N)`` that
:func:`tpen.evaluation.summaries.local_energy.summarize_values` publishes.

This generator adds the missing draw axis by delegating to
:func:`tpen.sampling.trajectory.collect_observable_trajectory`, which advances
the sampler's persistent chains and evaluates one observable per draw. The
resulting :class:`~tpen.statistics.ObservableTrajectory` is carried in the
generated metadata for
:class:`~tpen.evaluation.summaries.trajectory_statistics.TrajectoryStatisticsSummary`
to hand to the statistics producer.

Autograd stays live throughout. The local energy differentiates ``logabs``
twice with respect to positions, so a ``torch.no_grad()`` wrapper would make the
contract's first observable raise rather than merely run slower. "Fixed model"
is a claim about *parameters*, and the collector enforces it by freezing them
and fingerprinting them before and after -- not by disabling autograd.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from tpen.data.batch.walkers import Walkers
from tpen.evaluation.bundle import GeneratedConfigurations
from tpen.evaluation.calculators.local_energy import (
    evaluate_local_energy_in_chunks,
    slice_flat_batch,
    split_local_energy_result,
)
from tpen.evaluation.protocols import EvaluationContext
from tpen.physics.hamiltonian import HamiltonianTerm, normalize_hamiltonian_terms
from tpen.sampling.trajectory import collect_observable_trajectory

TRAJECTORY_METADATA_KEY = "observable_trajectory"
"""Metadata key under which the collected trajectory is published."""


class TrajectoryMCMCGenerator:
    """Collect a fixed-model ``[draw, walker]`` local-energy trajectory.

    The evaluation batch returned alongside the trajectory is drawn from the
    same continuing chain, so the existing snapshot summaries keep working
    unchanged and their IID standard error remains available for comparison.

    Parameters
    ----------
    sampler : object
        Sampler exposing ``collect_samples(model, *, reset=..., device=...)``.
    hamiltonian_terms : sequence or mapping of HamiltonianTerm
        Terms defining the local energy evaluated once per draw.
    n_draws : int
        Draws retained per walker. Each draw advances the sampler by its own
        configured ``n_steps``, so this multiplies sampling cost.
    discard_draws : int, optional
        Draws collected and discarded before retention begins, on top of the
        sampler's own burn-in.
    observable_name : str, optional
        Name recorded on the trajectory and used as the receipt's ``observable``
        identity field.
    chunk_size : int, optional
        Local-energy evaluation chunk size, forwarded per draw.
    seed : int, optional
        Recorded in metadata for bookkeeping parity with `MCMCGenerator`.
    max_samples : int, optional
        Cap on the returned evaluation batch. It does **not** truncate the
        trajectory: dropping walkers from a trajectory would silently discard
        whole chains and redefine the estimand.
    reset : bool, optional
        Force a fresh, re-burned-in chain before the first draw.
    verify_model_unchanged : bool, optional
        Fingerprint parameters before and after collection. Defaults to ``True``.

    Raises
    ------
    ValueError
        If `n_draws` is below one or `discard_draws` is negative.
    """

    name = "trajectory_mcmc"

    def __init__(
        self,
        *,
        sampler: object,
        hamiltonian_terms: Sequence[HamiltonianTerm] | Mapping[str, HamiltonianTerm],
        n_draws: int,
        discard_draws: int = 0,
        observable_name: str = "local_energy",
        chunk_size: int | None = None,
        seed: int | None = None,
        max_samples: int | None = None,
        reset: bool = False,
        verify_model_unchanged: bool = True,
    ) -> None:
        self.sampler = sampler
        self.hamiltonian_terms = normalize_hamiltonian_terms(hamiltonian_terms)
        self.n_draws = int(n_draws)
        if self.n_draws < 1:
            raise ValueError(f"n_draws must be at least 1, got {self.n_draws}")
        self.discard_draws = int(discard_draws)
        if self.discard_draws < 0:
            raise ValueError(f"discard_draws must be non-negative, got {self.discard_draws}")
        self.observable_name = str(observable_name).strip()
        if not self.observable_name:
            raise ValueError("observable_name must be non-empty")
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.seed = None if seed is None else int(seed)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.reset = bool(reset)
        self.verify_model_unchanged = bool(verify_model_unchanged)

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Collect the trajectory, then one evaluation batch from the same chain."""

        if model is None:
            raise TypeError("TrajectoryMCMCGenerator requires a model to evaluate the observable")
        collect = getattr(self.sampler, "collect_samples", None)
        if not callable(collect):
            raise TypeError(
                "TrajectoryMCMCGenerator sampler must expose "
                "collect_samples(model, *, reset=..., device=...)"
            )

        def observable(walkers: Walkers) -> torch.Tensor:
            return self._local_energy_per_walker(model=model, walkers=walkers)

        trajectory, _ = collect_observable_trajectory(
            self.sampler,
            model,
            observable,
            observable_name=self.observable_name,
            n_draws=self.n_draws,
            discard_draws=self.discard_draws,
            device=context.device,
            reset=self.reset,
            verify_model_unchanged=self.verify_model_unchanged,
        )

        # One further draw supplies the snapshot batch. Taken after the
        # trajectory rather than from inside it because the collector retains
        # only observable values, not walker state, and re-deriving the state
        # from the values is not possible.
        walkers, sampler_stats = collect(model, device=context.device)
        batch = walkers.make_batch().flatten_samples()
        if self.max_samples is not None and self.max_samples >= 0:
            batch = slice_flat_batch(batch, 0, min(self.max_samples, batch.batch_size))
        sample_index = torch.arange(batch.batch_size, device=batch.device)
        metadata: dict[str, Any] = {
            "sample_index": sample_index,
            "walker_index": sample_index,
            "sampler_stats": sampler_stats,
            TRAJECTORY_METADATA_KEY: trajectory,
        }
        if self.seed is not None:
            metadata["seed"] = self.seed
        return GeneratedConfigurations(batch=batch, metadata=metadata)

    def _local_energy_per_walker(
        self,
        *,
        model: torch.nn.Module,
        walkers: Walkers,
    ) -> torch.Tensor:
        """Return one local-energy value per walker for a single draw."""

        batch = walkers.make_batch().flatten_samples()
        result = evaluate_local_energy_in_chunks(
            self.hamiltonian_terms,
            model,
            batch,
            return_terms=False,
            chunk_size=self.chunk_size,
        )
        total, _ = split_local_energy_result(result)
        return total


__all__ = ["TRAJECTORY_METADATA_KEY", "TrajectoryMCMCGenerator"]
