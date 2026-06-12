"""Evaluation runner target."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from spenn.artifacts import RunContext, RunResult
from spenn.diagnostics import EvaluationContext, evaluate_diagnostics, validate_diagnostics
from spenn.dependencies import require_torch
from spenn.physics.hamiltonian import LocalEnergyResult, local_energy, normalize_hamiltonian_terms

from .base import Runner, _assert_eager_initialized, _is_torch_module, _place_module_for_runtime

torch = require_torch(feature="evaluation runner")


class Evaluate(Runner):
    """Generic sampled diagnostic evaluation runner.

    `Evaluate` owns sampling and shared evaluation work for one configured
    model/system. Diagnostics consume an `EvaluationContext`; they do not
    resample, log, emit callbacks, or write artifacts.

    Sampler contract assumed by this runner::

        walkers, sampler_stats = sampler.collect_samples(model, device=runtime_device)
        batch = walkers.make_batch()

    Parameters
    ----------
    model : callable
        Wavefunction model returning ``WavefunctionOutput``.
    sampler : object
        Sampler exposing ``collect_samples(model, device=...) -> (walkers, stats)``.
    hamiltonian_terms : sequence or mapping
        Hamiltonian terms summed by `local_energy`. A
        ``dict[str, HamiltonianTerm]`` uses its non-empty string keys as the
        public term names for decomposition and metrics; a sequence derives
        unique names from term class names.
    diagnostics : sequence of Diagnostic, optional
        Diagnostics to evaluate from the shared `EvaluationContext`. At least
        one diagnostic is required; configure `EnergyEvaluation` for energy
        summaries.
    return_terms : bool, optional
        Whether to request per-term local-energy components from `local_energy`.
    """

    def __init__(
        self,
        model,
        sampler,
        hamiltonian_terms,
        diagnostics: Sequence[object] | None = None,
        return_terms: bool = False,
    ) -> None:
        self.model = model
        self.sampler = sampler
        # Keep the configured form (sequence or ``dict[str, term]``);
        # ``local_energy`` normalizes it (see ``normalize_hamiltonian_terms``).
        self.hamiltonian_terms = hamiltonian_terms
        if diagnostics is None:
            raise ValueError(
                "Evaluate requires at least one diagnostic. Configure EnergyEvaluation to report energy metrics."
            )
        self.diagnostics = validate_diagnostics(diagnostics)
        self.return_terms = bool(return_terms)

    def run(self, context: RunContext) -> RunResult:
        """Sample configurations, evaluate local energy, and log metrics."""

        self.emit("run_start", context)

        if _is_torch_module(self.model):
            _place_module_for_runtime(self.model, context)
            self.model.eval()
            _assert_eager_initialized(self.model)

        self.emit("evaluate_start", context)

        # No torch.no_grad: local-energy evaluation needs position derivatives.
        walkers, sampler_stats = self.sampler.collect_samples(
            self.model,
            device=context.metadata.device,
        )
        batch = walkers.make_batch()

        self.emit("samples_collected", context, payload={"sampler_stats": dict(sampler_stats)})

        normalized_terms = normalize_hamiltonian_terms(self.hamiltonian_terms)
        energy_result = local_energy(normalized_terms, self.model, batch, return_terms=self.return_terms)
        total_energy, term_energies = _split_local_energy_result(energy_result)

        # PR6 keeps `wavefunction_output` in the shared context. Local-energy
        # terms may already evaluate the model internally; a future local-energy
        # API can avoid this extra no-grad forward when diagnostics do not need it.
        with torch.no_grad():
            wavefunction_output = self.model(batch)
        evaluation = EvaluationContext(
            model=self.model,
            batch=batch,
            wavefunction_output=wavefunction_output,
            local_energy=total_energy,
            local_energy_terms=term_energies,
            sampler_stats=dict(sampler_stats),
            hamiltonian_terms=normalized_terms,
        )
        metrics = evaluate_diagnostics(
            self.diagnostics,
            evaluation,
            emit=lambda name, *, payload=None: self.emit(name, context, payload=payload),
        )

        context.log(metrics, step=0, namespace="eval")
        if sampler_stats:
            context.log(dict(sampler_stats), step=0, namespace="eval/sampler")

        self.emit("evaluate_end", context, payload={"metrics": metrics})
        self.emit("run_end", context)
        return RunResult(status="completed")



def _split_local_energy_result(
    result: LocalEnergyResult | torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor] | None]:
    """Return ``(total, terms_or_none)`` from a local-energy result."""

    if isinstance(result, LocalEnergyResult):
        return result.total, result.terms
    return result, None



__all__ = ["Evaluate"]
