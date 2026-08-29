"""Hamiltonian terms, local-energy results, and typed evaluation.

A Hamiltonian is a collection of `HamiltonianTerm`s, given either as a sequence
or as a ``dict[str, HamiltonianTerm]`` that names each term explicitly. Dict
keys are the public, authoritative term names: they must be non-empty strings,
and the values must expose ``local_energy(wavefunction, batch)``.

Evaluation goes through a typed evaluator seam: `NaiveLocalEnergyEvaluator`
consumes a `NaiveLocalEnergyContext` and owns term normalization (see
`normalize_hamiltonian_terms`), ordered per-term evaluation, validation, and
summation, optionally returning the per-term decomposition keyed by the
resolved term names. The `local_energy` helper is an explicit delegate to the
naive evaluator and remains the single default path; the naive evaluator is
the permanent numerical reference for any future optimized evaluator behind
the `LocalEnergyEvaluator` protocol. Evaluation summaries use canonical flat
metric keys such as ``energy`` and ``energy_term_{name}``; hierarchy belongs
in the logging namespace.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import torch

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.naming import camel_to_snake


@dataclass
class LocalEnergyResult:
    """Container for a decomposed local-energy evaluation.

    Parameters
    ----------
    total : torch.Tensor
        Summed local energy across all contributions, shape ``[batch]``.
    terms : dict[str, torch.Tensor]
        Per-term local energies keyed by the resolved term name. When produced
        by `local_energy`, names come from the ``dict`` key (named form) or the
        snake-case class name (sequence form), and are guaranteed unique.
    wavefunction_output : WavefunctionOutput or None, optional
        Exact wavefunction output used by a term while evaluating this local
        energy. The aggregate evaluator permits at most one producing term so
        consumers can retain signed-log values without a second model pass.
    per_electron_kinetic : torch.Tensor or None, optional
        Explicit per-electron kinetic attribution with shape
        [batch, n_electrons]. The aggregate evaluator permits at most one
        producing term and carries it alongside the wavefunction output so
        diagnostics reuse the same differentiated model evaluation.
    term_provenance : dict[str, tuple[str, ...]], optional
        Source term names represented by each decomposition entry. Fused
        analytic entries name both participant terms explicitly.
    evaluator_id : str, optional
        Additive backend identity; ``"unknown"`` preserves historical results.
    """

    total: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    wavefunction_output: WavefunctionOutput | None = None
    per_electron_kinetic: torch.Tensor | None = None
    term_provenance: dict[str, tuple[str, ...]] = field(default_factory=dict)
    evaluator_id: str = "unknown"


def normalize_hamiltonian_terms(
    terms: Mapping[Any, Any] | Sequence[Any],
) -> dict[str, Any]:
    """Return an ordered ``{name: term}`` mapping from a dict or sequence.

    A ``dict[str, HamiltonianTerm]`` is used directly: its keys are the
    explicit, authoritative term names used in `LocalEnergyResult.terms` and
    downstream metrics. A sequence falls back to the snake-case class name of
    each term, suffixed with the term index when a class name repeats, so the
    resulting names are always unique.

    Names are enforced as non-empty strings and each value must expose a
    callable ``local_energy``; invalid specifications fail loudly here rather
    than later during evaluation.

    Parameters
    ----------
    terms : Mapping or Sequence of HamiltonianTerm
        Configured Hamiltonian terms, named (dict) or unnamed (sequence).

    Returns
    -------
    dict[str, HamiltonianTerm]
        Ordered mapping from resolved name to term.
    """

    if isinstance(terms, Mapping):
        normalized: dict[str, HamiltonianTerm] = {}
        for key, term in terms.items():
            if not isinstance(key, str):
                raise TypeError(f"hamiltonian term names must be strings, got {type(key).__name__}")
            _validate_hamiltonian_term(key, term)
            normalized[key] = term
        return normalized

    sequence = list(terms)
    base_names = [camel_to_snake(type(term).__name__) for term in sequence]
    counts = Counter(base_names)
    normalized = {}
    for index, (term, base) in enumerate(zip(sequence, base_names)):
        name = base if counts[base] == 1 else f"{base}_{index}"
        _validate_hamiltonian_term(name, term)
        normalized[name] = term
    return normalized


def _validate_hamiltonian_term(name: str, term: object) -> None:
    """Fail loudly on an empty term name or an invalid term specification."""

    if not name or not name.strip():
        raise ValueError("hamiltonian term names must be non-empty strings")
    if not callable(getattr(term, "local_energy", None)):
        raise TypeError(
            f"hamiltonian term {name!r} ({type(term).__name__}) must expose a callable "
            "local_energy(wavefunction, batch)"
        )


@runtime_checkable
class HamiltonianTerm(Protocol):
    """Protocol for a single Hamiltonian term.

    A term reports its contribution to the local energy as a
    `LocalEnergyResult` so that decompositions compose under summation.
    """

    name: str

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        """Return this term's local-energy contribution."""
        ...


ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class NaiveLocalEnergyContext:
    """Typed input context for the naive local-energy evaluator.

    Parameters
    ----------
    wavefunction : callable
        Wavefunction model or exact reference returning ``WavefunctionOutput``.
        The narrowest existing callable contract is reused deliberately; no
        ansatz-specific base class exists solely for this interface.
    batch : ElectronBatch
        Electron configuration batch.
    """

    wavefunction: object
    batch: ElectronBatch


@dataclass(frozen=True)
class AnalyticCuspContext:
    """Typed input context for the analytic electron-nucleus evaluator."""

    wavefunction: object
    batch: ElectronBatch


class LocalEnergyEvaluator(Protocol, Generic[ContextT]):
    """Protocol for local-energy evaluators over an open set of terms.

    Evaluators consume an explicit typed context, never an arbitrary mapping.
    Each evaluator declares the narrow context type it requires; the naive
    evaluator below is the permanent numerical reference for every future
    optimized evaluator.
    """

    def evaluate(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: ContextT,
        *,
        return_terms: bool = False,
    ) -> torch.Tensor | LocalEnergyResult:
        """Evaluate the local energy for one typed context."""
        ...


class NaiveLocalEnergyEvaluator(LocalEnergyEvaluator[NaiveLocalEnergyContext]):
    """Reference evaluator: ordered per-term autodiff evaluation and summation.

    Owns the aggregation loop previously inlined in `local_energy`: term
    normalization, per-term evaluation through the open ``HamiltonianTerm``
    protocol, result validation, named decomposition, and summation — with
    unchanged arithmetic and ordering. This path remains the independent
    numerical reference for any later optimized evaluator.
    """

    evaluator_id = "naive/v1"

    def evaluate(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: NaiveLocalEnergyContext,
        *,
        return_terms: bool = False,
    ) -> torch.Tensor | LocalEnergyResult:
        """Evaluate the summed (optionally decomposed) local energy."""

        if not isinstance(context, NaiveLocalEnergyContext):
            raise TypeError(
                "NaiveLocalEnergyEvaluator requires a NaiveLocalEnergyContext, "
                f"got {type(context).__name__}"
            )
        normalized = normalize_hamiltonian_terms(terms)
        batch = context.batch
        flat = batch.flatten_samples()
        batch_size = flat.batch_size
        total: torch.Tensor | None = None
        decomposition: dict[str, torch.Tensor] = {}
        wavefunction_output: WavefunctionOutput | None = None
        per_electron_kinetic: torch.Tensor | None = None
        for name, term in normalized.items():
            result = term.local_energy(context.wavefunction, batch)
            result = _validate_local_energy_result(
                name,
                result,
                batch_size=batch_size,
                n_electrons=flat.n_electrons,
            )
            decomposition[name] = result.total
            total = result.total if total is None else total + result.total
            if result.wavefunction_output is not None:
                if wavefunction_output is not None:
                    raise ValueError(
                        "local-energy evaluation produced more than one wavefunction output; "
                        "trajectory records require one model evaluation source"
                    )
                wavefunction_output = result.wavefunction_output
            if result.per_electron_kinetic is not None:
                if per_electron_kinetic is not None:
                    raise ValueError(
                        "local-energy evaluation produced more than one per-electron "
                        "kinetic attribution"
                    )
                per_electron_kinetic = result.per_electron_kinetic
        if total is None:
            flat = batch.flatten_samples()
            total = torch.zeros(flat.batch_size, device=flat.device, dtype=flat.dtype)
        if return_terms:
            return LocalEnergyResult(
                total=total,
                terms=decomposition,
                wavefunction_output=wavefunction_output,
                per_electron_kinetic=per_electron_kinetic,
                term_provenance={name: (name,) for name in decomposition},
                evaluator_id=self.evaluator_id,
            )
        return total


class AnalyticCuspEvaluator(LocalEnergyEvaluator[AnalyticCuspContext]):
    """Slow reference evaluator that fuses kinetic energy with one cusp term.

    The evaluator deliberately has a separate context and implementation from
    :class:`NaiveLocalEnergyEvaluator`.  A missing analytic capability or an
    invalid domain is an error, never a reason to silently change evaluators.
    """

    fused_term_name = "kinetic_plus_electron_nucleus"
    evaluator_id = "analytic_cusp/v1"

    def validate(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        wavefunction: object,
        *,
        n_electrons: int | None = None,
        spatial_dim: int | None = None,
    ) -> None:
        """Validate configuration-only analytic eligibility before sampling.

        Batch-dependent checks remain in :meth:`evaluate`, where the actual
        typed batch is available.  This method deliberately performs every
        rejection that can be known from the configured model, Hamiltonian,
        and sampler metadata before a runner emits its evaluation-start event.
        """

        from tpen.nn.cusp import ElectronNucleusCusp
        from tpen.physics.kinetic import KineticEnergy
        from tpen.physics.potential import ElectronNucleusPotential

        normalized = normalize_hamiltonian_terms(terms)
        kinetic = [term for term in normalized.values() if isinstance(term, KineticEnergy)]
        potentials = [term for term in normalized.values() if isinstance(term, ElectronNucleusPotential)]
        if len(kinetic) != 1:
            raise ValueError("analytic cusp evaluation requires exactly one KineticEnergy term")
        if len(potentials) != 1:
            raise ValueError("analytic cusp evaluation requires exactly one ElectronNucleusPotential term")
        potential = potentials[0]
        if potential.eps != 0:
            raise ValueError("analytic cusp evaluation requires ElectronNucleusPotential.eps == 0")
        provider = getattr(wavefunction, "analytic_cusp_provider", None)
        if not isinstance(provider, ElectronNucleusCusp):
            raise ValueError(
                "analytic cusp evaluation requires one explicitly bound ElectronNucleusCusp provider"
            )
        if not callable(getattr(wavefunction, "factorized_local_energy_input", None)):
            raise ValueError("analytic cusp evaluation requires factorized_local_energy_input capability")
        same_geometry, _ = potential.atoms.compare(provider.atoms, atol=0.0, rtol=0.0)
        if not same_geometry:
            raise ValueError("ElectronNucleusPotential and analytic cusp provider must share matching atoms")
        if potential.atoms.spatial_dim != 3 or provider.atoms.spatial_dim != 3:
            raise ValueError("analytic cusp evaluation requires spatial dimension 3")
        if spatial_dim is not None and int(spatial_dim) != 3:
            raise ValueError("analytic cusp evaluation requires spatial dimension 3")
        if n_electrons is not None and int(n_electrons) < 1:
            raise ValueError("analytic cusp evaluation requires at least one electron")

    def validate_for_generator(self, terms, wavefunction, generator: object) -> None:
        """Validate static eligibility using a configured generator's sampler."""

        from tpen.physics.potential import ElectronNucleusPotential

        sampler = getattr(generator, "sampler", None)
        self.validate(
            terms,
            wavefunction,
            n_electrons=getattr(sampler, "n_electrons", None),
            spatial_dim=getattr(sampler, "spatial_dim", None),
        )
        sampler_atoms = getattr(sampler, "atomic_configuration", None)
        if sampler_atoms is not None:
            potential = next(
                term for term in normalize_hamiltonian_terms(terms).values()
                if isinstance(term, ElectronNucleusPotential)
            )
            same_geometry, _ = potential.atoms.compare(sampler_atoms, atol=0.0, rtol=0.0)
            if not same_geometry:
                raise ValueError(
                    "ElectronNucleusPotential and sampler must share matching atoms"
                )

    def evaluate(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: AnalyticCuspContext,
        *,
        return_terms: bool = False,
    ) -> torch.Tensor | LocalEnergyResult:
        """Evaluate with the batched analytic kernel."""

        return self._evaluate(terms, context, return_terms=return_terms, vectorized=True)

    def evaluate_reference(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: AnalyticCuspContext,
        *,
        return_terms: bool = False,
    ) -> torch.Tensor | LocalEnergyResult:
        """Evaluate with the independent loop-based oracle."""

        return self._evaluate(terms, context, return_terms=return_terms, vectorized=False)

    def _evaluate(
        self,
        terms: Mapping[Any, Any] | Sequence[Any],
        context: AnalyticCuspContext,
        *,
        return_terms: bool = False,
        vectorized: bool,
    ) -> torch.Tensor | LocalEnergyResult:
        if not isinstance(context, AnalyticCuspContext):
            raise TypeError(
                "AnalyticCuspEvaluator requires an AnalyticCuspContext, "
                f"got {type(context).__name__}"
            )
        from tpen.nn.cusp import ElectronNucleusCusp
        from tpen.physics.kinetic import KineticEnergy
        from tpen.physics.potential import ElectronNucleusPotential, _validate_batch_atoms_context

        normalized = normalize_hamiltonian_terms(terms)
        kinetic = [term for term in normalized.values() if isinstance(term, KineticEnergy)]
        potentials = [term for term in normalized.values() if isinstance(term, ElectronNucleusPotential)]
        if len(kinetic) != 1:
            raise ValueError("analytic cusp evaluation requires exactly one KineticEnergy term")
        if len(potentials) != 1:
            raise ValueError("analytic cusp evaluation requires exactly one ElectronNucleusPotential term")
        potential = potentials[0]
        if potential.eps != 0:
            raise ValueError("analytic cusp evaluation requires ElectronNucleusPotential.eps == 0")

        wavefunction = context.wavefunction
        provider = getattr(wavefunction, "analytic_cusp_provider", None)
        if not isinstance(provider, ElectronNucleusCusp):
            raise ValueError("analytic cusp evaluation requires one explicitly bound ElectronNucleusCusp provider")
        factorized = getattr(wavefunction, "factorized_local_energy_input", None)
        if not callable(factorized):
            raise ValueError("analytic cusp evaluation requires factorized_local_energy_input capability")
        same_geometry, _ = potential.atoms.compare(provider.atoms, atol=0.0, rtol=0.0)
        if not same_geometry:
            raise ValueError("ElectronNucleusPotential and analytic cusp provider must share matching atoms")

        flat = context.batch.flatten_samples()
        if flat.spatial_dim != 3:
            raise ValueError("analytic cusp evaluation requires spatial dimension 3")
        _validate_batch_atoms_context(potential.atoms, flat, term_name=type(potential).__name__)
        positions = flat.positions.detach().clone().requires_grad_(True)
        probe = ElectronBatch(
            positions=positions,
            system=flat.system,
            nuclear_positions=flat.nuclear_positions,
            nuclear_charges=flat.nuclear_charges,
            atomic_configuration=flat.atomic_configuration,
            spins=flat.spins,
            aux=dict(flat.aux),
        )
        factorized_input = factorized(probe)
        from tpen.data.batch import FactorizedLocalEnergyInput
        if not isinstance(factorized_input, FactorizedLocalEnergyInput):
            raise TypeError("factorized_local_energy_input must return FactorizedLocalEnergyInput")
        factorized_input.validate(batch_size=flat.batch_size)
        evaluation = factorized_input.electron_nucleus_cusp_evaluation
        if evaluation.n_electrons != flat.n_electrons or evaluation.n_nuclei != provider.atoms.n_nuclei:
            raise ValueError("analytic cusp evaluation geometry does not match the electron batch")
        if evaluation.displacement.shape[-1] != 3:
            raise ValueError("analytic cusp evaluation requires spatial dimension 3")
        atoms_on_device = provider.atoms.to(device=positions.device, dtype=positions.dtype)
        expected_displacement = positions.unsqueeze(2) - atoms_on_device.positions.view(1, 1, -1, 3)
        if not torch.equal(evaluation.displacement, expected_displacement):
            raise ValueError("analytic cusp evaluation displacement does not match the bound atoms")
        if not torch.equal(evaluation.distance, expected_displacement.norm(dim=-1)):
            raise ValueError("analytic cusp evaluation distance does not match its displacement")
        if torch.any(evaluation.distance <= 0):
            raise ValueError("analytic cusp evaluation does not support electron-nucleus coalescence")
        if not torch.equal(evaluation.nuclear_charges, provider.atoms.charges.to(evaluation.displacement)):
            raise ValueError("analytic cusp evaluation provider charges do not match its atoms")
        expected_slope = -evaluation.nuclear_charges
        if not torch.equal(evaluation.origin_radial_slope, expected_slope):
            raise ValueError("analytic cusp evaluation provider origin slopes do not match nuclear charges")

        regular = factorized_input.regular_wavefunction_output
        if regular.phase is not None:
            raise ValueError("analytic cusp evaluation requires a real wavefunction")
        if not torch.isfinite(regular.logabs).all() or not torch.isfinite(regular.sign).all() or torch.any(regular.sign == 0):
            raise ValueError("analytic cusp evaluation requires an off-node real wavefunction")
        # The zero quadratic keeps even constant or affine regular factors in
        # a second-derivative graph, without changing their value or gradient.
        differentiable_logabs = regular.logabs + positions.square().sum(dim=(1, 2)) * 0.0
        gradient = torch.autograd.grad(differentiable_logabs.sum(), positions, create_graph=True)[0]
        if gradient.shape != positions.shape:
            raise ValueError("regular wavefunction gradient must match electron positions")

        if vectorized:
            n_coordinates = flat.n_electrons * 3
            if n_coordinates:
                basis = torch.eye(n_coordinates, device=positions.device, dtype=positions.dtype)
                basis = basis.reshape(n_coordinates, 1, flat.n_electrons, 3)
                basis = basis.expand(-1, flat.batch_size, -1, -1)
                hessian = torch.autograd.grad(
                    gradient,
                    positions,
                    grad_outputs=basis,
                    is_grads_batched=True,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                electron_indices = torch.arange(flat.n_electrons, device=positions.device).repeat_interleave(3)
                coordinate_indices = torch.arange(3, device=positions.device).repeat(flat.n_electrons)
                diagonal = hessian[torch.arange(n_coordinates, device=positions.device), :, electron_indices, coordinate_indices]
                laplacian = diagonal.transpose(0, 1).reshape(flat.batch_size, flat.n_electrons, 3).sum(dim=-1)
            else:
                laplacian = torch.zeros(
                    (flat.batch_size, flat.n_electrons), device=positions.device, dtype=positions.dtype
                )
            cusp_gradient = (
                evaluation.radial_first_derivative.unsqueeze(-1)
                * evaluation.displacement
                / evaluation.distance.unsqueeze(-1)
            ).sum(dim=2)
            total_gradient = gradient + cusp_gradient
            fused = -0.5 * (
                laplacian.sum(dim=-1)
                + total_gradient.square().sum(dim=-1).sum(dim=-1)
            )
            fused = fused + evaluation.local_energy_pair().sum(dim=(1, 2))
        else:
            # Independent executable oracle: retain explicit sample/electron/
            # coordinate/nucleus loops so the fast kernel has a real reference.
            fused = torch.zeros(flat.batch_size, device=positions.device, dtype=positions.dtype)
            for sample in range(flat.batch_size):
                for electron in range(flat.n_electrons):
                    laplacian = torch.zeros((), device=positions.device, dtype=positions.dtype)
                    for coordinate in range(3):
                        second = torch.autograd.grad(
                            gradient[:, electron, coordinate].sum(),
                            positions,
                            create_graph=True,
                            retain_graph=True,
                        )[0]
                        laplacian = laplacian + second[sample, electron, coordinate]
                    cusp_gradient = torch.zeros(3, device=positions.device, dtype=positions.dtype)
                    for nucleus in range(evaluation.n_nuclei):
                        cusp_gradient = cusp_gradient + (
                            evaluation.radial_first_derivative[sample, electron, nucleus]
                            * evaluation.displacement[sample, electron, nucleus]
                            / evaluation.distance[sample, electron, nucleus]
                        )
                    total_gradient = gradient[sample, electron] + cusp_gradient
                    fused[sample] = fused[sample] - 0.5 * (laplacian + total_gradient.square().sum())
                fused[sample] = fused[sample] + evaluation.local_energy_pair()[sample].sum()

        if fused.shape != (flat.batch_size,):
            raise ValueError(
                "analytic cusp fused energy must have shape "
                f"{(flat.batch_size,)}, got {tuple(fused.shape)}"
            )

        full_output = WavefunctionOutput(
            logabs=regular.logabs + evaluation.pair_value.sum(dim=(1, 2)),
            sign=regular.sign,
            phase=regular.phase,
            aux=dict(regular.aux),
        )
        return self._aggregate(
            normalized,
            context.batch,
            wavefunction,
            fused,
            full_output,
            return_terms=return_terms,
        )

    def _aggregate(self, normalized, batch, wavefunction, fused, full_output, *, return_terms):
        flat = batch.flatten_samples()
        decomposition = {}
        total = None
        wavefunction_output = full_output
        per_electron = None
        provenance: dict[str, tuple[str, ...]] = {}
        fused_sources: list[str] = []
        inserted = False
        from tpen.physics.kinetic import KineticEnergy
        from tpen.physics.potential import ElectronNucleusPotential
        for name, term in normalized.items():
            if isinstance(term, (KineticEnergy, ElectronNucleusPotential)):
                fused_sources.append(name)
                if not inserted:
                    decomposition[self.fused_term_name] = fused
                    total = fused if total is None else total + fused
                    inserted = True
                continue
            result = _validate_local_energy_result(
                name,
                term.local_energy(wavefunction, batch),
                batch_size=flat.batch_size,
                n_electrons=flat.n_electrons,
            )
            decomposition[name] = result.total
            provenance[name] = result.term_provenance.get(name, (name,))
            total = result.total if total is None else total + result.total
            if result.wavefunction_output is not None:
                raise ValueError("analytic cusp evaluation cannot combine another wavefunction output")
            if result.per_electron_kinetic is not None:
                if per_electron is not None:
                    raise ValueError("local-energy evaluation produced more than one per-electron kinetic attribution")
                per_electron = result.per_electron_kinetic
        if total is None:
            total = fused
        if inserted:
            provenance[self.fused_term_name] = tuple(fused_sources)
        if return_terms:
            return LocalEnergyResult(
                total=total,
                terms=decomposition,
                wavefunction_output=wavefunction_output,
                per_electron_kinetic=per_electron,
                term_provenance=provenance,
                evaluator_id=self.evaluator_id,
            )
        return total


_NAIVE_EVALUATOR = NaiveLocalEnergyEvaluator()


def local_energy(
    terms: Mapping[Any, Any] | Sequence[Any],
    wavefunction,
    batch: ElectronBatch,
    *,
    return_terms: bool = False,
) -> torch.Tensor | LocalEnergyResult:
    """Evaluate the local energy of a collection of Hamiltonian terms.

    Parameters
    ----------
    terms : Mapping or Sequence of HamiltonianTerm
        Hamiltonian contributions to sum. A ``dict[str, HamiltonianTerm]`` names
        terms by its non-empty string keys; a sequence falls back to snake-case
        class names (see `normalize_hamiltonian_terms`).
    wavefunction : callable
        Wavefunction model or exact reference returning ``WavefunctionOutput``.
    batch : ElectronBatch
        Electron configuration batch.
    return_terms : bool, optional
        If ``True``, return a ``LocalEnergyResult`` whose ``terms`` decomposition
        is keyed by the resolved (unique) term names; otherwise return the summed
        tensor directly.

    Returns
    -------
    torch.Tensor or LocalEnergyResult
        Summed local energy with shape ``[batch]``, or a decomposed result when
        ``return_terms=True``.
    """

    # Explicit delegate: this entry point stays the single default path used
    # by training and evaluation; the naive evaluator owns the aggregation.
    context = NaiveLocalEnergyContext(wavefunction=wavefunction, batch=batch)
    return _NAIVE_EVALUATOR.evaluate(terms, context, return_terms=return_terms)


def _validate_local_energy_result(
    name: str,
    result: object,
    *,
    batch_size: int,
    n_electrons: int,
) -> LocalEnergyResult:
    """Validate the object returned by one Hamiltonian term."""

    if not isinstance(result, LocalEnergyResult):
        raise TypeError(
            f"hamiltonian term {name!r} must return LocalEnergyResult, got {type(result).__name__}"
        )
    if not isinstance(result.total, torch.Tensor):
        raise TypeError(f"hamiltonian term {name!r} total must be a torch.Tensor")
    expected_shape = (batch_size,)
    if tuple(result.total.shape) != expected_shape:
        raise ValueError(
            f"hamiltonian term {name!r} total must have shape {expected_shape}, "
            f"got {tuple(result.total.shape)}"
        )
    if not isinstance(result.terms, Mapping):
        raise TypeError(f"hamiltonian term {name!r} terms must be a mapping")
    for term_name, value in result.terms.items():
        if not isinstance(term_name, str) or not term_name.strip():
            raise ValueError(f"hamiltonian term {name!r} returned an empty decomposition name")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"hamiltonian term {name!r} decomposition {term_name!r} must be a torch.Tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"hamiltonian term {name!r} decomposition {term_name!r} must have shape "
                f"{expected_shape}, got {tuple(value.shape)}"
            )
    if result.term_provenance:
        if tuple(result.term_provenance) != tuple(result.terms):
            raise ValueError(f"hamiltonian term {name!r} provenance must cover terms in order")
        for decomposition_name, sources in result.term_provenance.items():
            if not sources or any(not source.strip() for source in sources):
                raise ValueError(
                    f"hamiltonian term {name!r} decomposition {decomposition_name!r} "
                    "has invalid provenance"
                )
    if result.wavefunction_output is not None:
        if not isinstance(result.wavefunction_output, WavefunctionOutput):
            raise TypeError(
                f"hamiltonian term {name!r} wavefunction_output must be a WavefunctionOutput"
            )
        result.wavefunction_output.validate(batch_size=batch_size)
    if result.per_electron_kinetic is not None:
        attribution = result.per_electron_kinetic
        if not isinstance(attribution, torch.Tensor):
            raise TypeError(
                f"hamiltonian term {name!r} per_electron_kinetic must be a torch.Tensor"
            )
        expected_attribution_shape = (batch_size, n_electrons)
        if tuple(attribution.shape) != expected_attribution_shape:
            raise ValueError(
                f"hamiltonian term {name!r} per_electron_kinetic must have shape "
                f"{expected_attribution_shape}, got {tuple(attribution.shape)}"
            )
        if attribution.device != result.total.device or attribution.dtype != result.total.dtype:
            raise ValueError(
                f"hamiltonian term {name!r} per_electron_kinetic must match total "
                "dtype/device"
            )
    return result


def summarize_local_energy(
    result: LocalEnergyResult | torch.Tensor,
) -> dict[str, Any]:
    """Summarize a sampled local energy into scalar logging metrics.

    Handles all-finite, partially-nonfinite, all-nonfinite, and empty inputs,
    and per-term decompositions. Returned keys follow the metric naming
    convention: callers provide the logging namespace, while this helper emits
    flat leaf keys such as ``energy``, ``local_energy_n_finite``, and
    ``energy_term_{name}``. This summary is reference-free; benchmark
    comparison belongs to evaluation diagnostics.

    Parameters
    ----------
    result : LocalEnergyResult or torch.Tensor
        Per-sample local energy, optionally with a per-term decomposition.

    Returns
    -------
    dict
        Scalar metrics. When no finite samples exist, ``energy`` and
        ``energy_variance`` are ``nan`` and ``energy_stderr`` is ``inf``.

    Notes
    -----
    ``energy_stderr`` and every ``{prefix}_stderr`` here is an **IID-only**
    standard error, ``sigma / sqrt(N)``, identical in meaning to the metric of
    the same name from `tpen.training.vmc.compute_vmc_objective`. It ignores serial
    correlation between MCMC walkers and so understates the true uncertainty by
    roughly ``sqrt(tau_int)``.

    The correlation-aware quantity is the MCSE from
    :func:`tpen.statistics.produce_trajectory_statistics`, which needs a
    ``[draw, walker]`` trajectory rather than the flat sample tensor available
    here. Do not reinterpret this metric as an MCSE.
    """

    if isinstance(result, LocalEnergyResult):
        eloc, terms = result.total, result.terms
    else:
        eloc, terms = result, {}

    metrics = _summarize_values("", eloc)
    for name, value in terms.items():
        metrics.update(_summarize_values(f"energy_term_{name}", value))
    return metrics


def _summarize_values(prefix: str, values: torch.Tensor) -> dict[str, Any]:
    """Return canonical finite-aware energy metrics for one value tensor.

    ``{prefix}_stderr`` is an **IID-only** standard error; see the Notes on
    `summarize_local_energy`. It is never an MCSE.
    """

    n_total = int(values.numel())
    finite_mask = torch.isfinite(values)
    n_finite = int(finite_mask.sum().item())
    if n_finite > 0:
        finite = values[finite_mask]
        energy = float(finite.mean().item())
        variance = float(finite.var(unbiased=False).item()) if n_finite > 1 else 0.0
        std = math.sqrt(variance)
        stderr = std / math.sqrt(n_finite)
    else:
        energy = float("nan")
        variance = float("nan")
        std = float("nan")
        stderr = float("inf")

    if prefix:
        return {
            prefix: energy,
            f"{prefix}_variance": variance,
            f"{prefix}_std": std,
            f"{prefix}_stderr": stderr,
            f"{prefix}_n_finite": n_finite,
            f"{prefix}_n_total": n_total,
            f"{prefix}_finite_fraction": float(n_finite / n_total) if n_total else 0.0,
            f"{prefix}_nonfinite_count": n_total - n_finite,
        }

    return {
        "energy": energy,
        "energy_variance": variance,
        "energy_std": std,
        "energy_stderr": stderr,
        "local_energy_n_finite": n_finite,
        "local_energy_n_total": n_total,
        "local_energy_finite_fraction": float(n_finite / n_total) if n_total else 0.0,
        "local_energy_nonfinite_count": n_total - n_finite,
    }


__all__ = [
    "HamiltonianTerm",
    "LocalEnergyEvaluator",
    "LocalEnergyResult",
    "NaiveLocalEnergyContext",
    "NaiveLocalEnergyEvaluator",
    "AnalyticCuspContext",
    "AnalyticCuspEvaluator",
    "local_energy",
    "normalize_hamiltonian_terms",
    "summarize_local_energy",
]
