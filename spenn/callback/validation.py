"""Train-lifecycle validation callback for model/protocol selection."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

from .base import Callback, Event

# Namespaces owned by other phases; validation must not log into them.
_RESERVED_NAMESPACES = ("train", "eval", "checks")

# Exact-reference comparison belongs to final evaluation (eval/*) only.
_FORBIDDEN_METRICS = ("energy_error", "energy_abs_error", "reference_energy")


class Validation(Callback):
    """Run an independent validation sampler on the trained model.

    Listens to ``train_end`` (and optionally ``step_end`` with
    ``every_n_steps``), draws fresh samples from a validation sampler that is
    distinct from the training sampler, evaluates the configured diagnostics,
    and logs metrics under ``<namespace>/*``, ``<namespace>/sampler/*``, and
    ``<namespace>/perf/*``.

    Validation estimates energy and uncertainty for model/protocol/checkpoint
    selection. It must not compare against an exact reference energy, select
    hyperparameters, or mutate model/optimizer/trainer state; selection is
    owned by study scripts reading run outputs, and exact-reference reporting
    is owned by final evaluation.

    Parameters
    ----------
    triggers : iterable of str
        Event names that trigger validation (normally ``["train_end"]``).
    sampler : object
        Fresh validation sampler exposing
        ``collect_samples(model, device=...) -> (walkers, stats)``. Must be a
        different object from the training sampler; chain state is never
        shared with training.
    hamiltonian_terms : sequence or mapping
        Hamiltonian terms summed by `local_energy`, in the same configured
        form accepted by the runners.
    diagnostics : sequence of Diagnostic
        Diagnostics evaluated on the validation samples. Diagnostics carrying
        a non-``None`` ``reference_energy`` are rejected at construction.
    namespace : str, optional
        Root metric namespace, ``"validation"`` by default. Reserved phase
        namespaces (``train``, ``eval``, ``checks``) are rejected.
    return_terms : bool, optional
        Whether to request per-term local-energy components from
        `local_energy`.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        *,
        sampler,
        hamiltonian_terms,
        diagnostics: Sequence[object] | None = None,
        namespace: str = "validation",
        return_terms: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        from spenn.diagnostics import validate_diagnostics

        namespace = str(namespace).strip("/")
        if not namespace:
            raise ValueError("Validation namespace must be a non-empty string")
        root = namespace.split("/", 1)[0]
        if root in _RESERVED_NAMESPACES:
            raise ValueError(
                f"Validation namespace {namespace!r} collides with the reserved {root!r} phase namespace"
            )
        self.namespace = namespace
        self.sampler = sampler
        self.hamiltonian_terms = hamiltonian_terms
        self.return_terms = bool(return_terms)
        if diagnostics is None:
            raise ValueError(
                "Validation requires at least one diagnostic. Configure EnergyEvaluation "
                "without reference_energy to report validation energy metrics."
            )
        self.diagnostics = validate_diagnostics(diagnostics)
        for diagnostic in self.diagnostics:
            if getattr(diagnostic, "reference_energy", None) is not None:
                raise ValueError(
                    f"validation diagnostic {diagnostic.name!r} sets reference_energy; exact-reference "
                    "comparison belongs to final evaluation (eval/*), never to validation"
                )

    def should_run(self, event: Event) -> bool:
        """Let ``train_end`` bypass the periodic step filter.

        The ``every_n_steps`` cadence only applies to periodic ``step_end``
        validation. ``train_end`` always validates when triggered, even when
        the final step does not land on the cadence.
        """

        if event.name == "train_end":
            if event.name not in self.triggers:
                return False
            if self.max_calls is not None and self.num_calls >= self.max_calls:
                return False
            return self._draw_probability()
        return super().should_run(event)

    def on_train_end(self, event: Event) -> None:
        """Validate the final model with the independent validation sampler."""

        self._run_validation(event)

    def on_step_end(self, event: Event) -> None:
        """Optional periodic validation, gated by ``every_n_steps``."""

        self._run_validation(event)

    def _run_validation(self, event: Event) -> None:
        from spenn.dependencies import require_torch
        from spenn.diagnostics import EvaluationContext, evaluate_diagnostics
        from spenn.physics.hamiltonian import LocalEnergyResult, local_energy, normalize_hamiltonian_terms

        torch = require_torch(feature="Validation callback")
        start = time.perf_counter()

        model = event.payload.get("model")
        if model is None:
            model = getattr(event.state, "model", None)
        if model is None:
            raise ValueError(
                f"Validation requires the {event.name!r} event to provide the trained model "
                "in its payload or state"
            )

        training_sampler = getattr(event.state, "sampler", None)
        if training_sampler is not None and training_sampler is self.sampler:
            raise ValueError(
                "Validation must use an independent validation sampler instance; it received "
                "the training sampler object itself"
            )

        step = event.step
        if step is None:
            value = getattr(event.state, "step", None)
            step = None if value is None else int(value)

        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        try:
            # No torch.no_grad here: local-energy evaluation needs position
            # derivatives. Parameters are left untouched; nothing calls
            # backward() or an optimizer.
            walkers, sampler_stats = self.sampler.collect_samples(
                model,
                device=event.context.metadata.device,
            )
            batch = walkers.make_batch()

            normalized_terms = normalize_hamiltonian_terms(self.hamiltonian_terms)
            energy_result = local_energy(normalized_terms, model, batch, return_terms=self.return_terms)
            if isinstance(energy_result, LocalEnergyResult):
                total_energy, term_energies = energy_result.total, energy_result.terms
            else:
                total_energy, term_energies = energy_result, None

            with torch.no_grad():
                wavefunction_output = model(batch)

            evaluation = EvaluationContext(
                model=model,
                batch=batch,
                wavefunction_output=wavefunction_output,
                local_energy=total_energy,
                local_energy_terms=term_energies,
                sampler_stats=dict(sampler_stats),
                hamiltonian_terms=normalized_terms,
            )
            metrics = evaluate_diagnostics(self.diagnostics, evaluation, step=0 if step is None else step)
        finally:
            if was_training and hasattr(model, "train"):
                model.train()

        forbidden = sorted(set(_FORBIDDEN_METRICS).intersection(metrics))
        if forbidden:
            raise ValueError(
                f"validation diagnostics produced exact-reference metrics {forbidden}; these are "
                "only allowed in final evaluation (eval/*)"
            )

        event.context.log(metrics, step=step, namespace=self.namespace)
        if sampler_stats:
            event.context.log(dict(sampler_stats), step=step, namespace=f"{self.namespace}/sampler")
        event.context.log(
            {"wall_time_sec": time.perf_counter() - start},
            step=step,
            namespace=f"{self.namespace}/perf",
        )


__all__ = ["Validation"]
