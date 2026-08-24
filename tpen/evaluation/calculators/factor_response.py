"""Common-configuration factor-response calculator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch

from tpen.evaluation.bundle import EvaluationBundle, FactorResponseArmValues, FactorResponseValues
from tpen.evaluation.calculators.local_energy import evaluate_local_energy_in_chunks
from tpen.evaluation.factor_response import FactorParameterScale, helium_factor_parameter_scale
from tpen.evaluation.protocols import EvaluationContext
from tpen.physics.hamiltonian import HamiltonianTerm, LocalEnergyResult, normalize_hamiltonian_terms


class FactorResponseCalculator:
    """Evaluate declared factor arms on one unchanged configuration batch."""

    name = "factor_response"

    def __init__(
        self,
        *,
        hamiltonian_terms: Sequence[HamiltonianTerm] | Mapping[str, HamiltonianTerm],
        arms: Sequence[Mapping[str, object] | FactorParameterScale],
        baseline_label: str = "baseline",
        chunk_size: int | None = None,
    ) -> None:
        self.hamiltonian_terms = normalize_hamiltonian_terms(hamiltonian_terms)
        self.arms = tuple(
            arm if isinstance(arm, FactorParameterScale) else FactorParameterScale.from_mapping(arm)
            for arm in arms
        )
        labels = [arm.label for arm in self.arms]
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("factor-response arm labels must be non-empty and unique")
        self.baseline_label = str(baseline_label)
        if labels.count(self.baseline_label) != 1:
            raise ValueError("factor response requires exactly one named baseline arm")
        self.chunk_size = None if chunk_size is None else int(chunk_size)

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Return row-aligned local-energy and signed-log values for every arm."""

        del context
        batch = bundle.generated.batch.flatten_samples()
        values: list[FactorResponseArmValues] = []
        for arm in self.arms:
            with helium_factor_parameter_scale(model, arm) as realized:
                result = evaluate_local_energy_in_chunks(
                    self.hamiltonian_terms,
                    model,
                    batch,
                    return_terms=True,
                    chunk_size=self.chunk_size,
                )
            if not isinstance(result, LocalEnergyResult) or result.wavefunction_output is None:
                raise TypeError(
                    "factor response requires LocalEnergyResult with wavefunction primitives"
                )
            values.append(
                FactorResponseArmValues(
                    label=arm.label,
                    parameter_scales={
                        "b_ee": arm.b_ee,
                        "c_electron_nucleus": arm.c_electron_nucleus,
                        "d_electron_nucleus": arm.d_electron_nucleus,
                    },
                    realized_parameters=dict(realized),
                    local_energy=result.total.detach(),
                    logabs=result.wavefunction_output.logabs.detach().reshape(-1),
                    sign=result.wavefunction_output.sign.detach().reshape(-1),
                    term_energies={
                        name: term.detach().reshape(-1)
                        for name, term in result.terms.items()
                    },
                )
            )
        return replace(
            bundle,
            factor_response=FactorResponseValues(
                comparison_kind="common_configuration",
                baseline_label=self.baseline_label,
                arms=tuple(values),
                model_state_restored=True,
            ),
        )


class FactorArmCalculator:
    """Run one delegated calculator under a declared temporary factor arm.

    Re-equilibrated comparisons need both sampling and every subsequent model
    evaluation to see the same factor values. The evaluator restores component
    boundaries between generation and calculation, so this wrapper reapplies
    the arm for exactly one calculation and verifies restoration on exit.
    """

    name = "factor_arm"

    def __init__(
        self,
        *,
        calculator: object,
        arm: Mapping[str, object] | FactorParameterScale,
    ) -> None:
        calculate = getattr(calculator, "calculate", None)
        if not callable(calculate):
            raise TypeError("FactorArmCalculator requires a calculator delegate")
        self.calculator = calculator
        self.arm = (
            arm
            if isinstance(arm, FactorParameterScale)
            else FactorParameterScale.from_mapping(arm)
        )

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Delegate after checking the generated configurations use this arm."""

        metadata = bundle.generated.metadata
        if metadata.get("comparison_kind") != "re_equilibrated":
            raise ValueError(
                "FactorArmCalculator requires re-equilibrated generated configurations"
            )
        if metadata.get("factor_arm") != self.arm.label:
            raise ValueError(
                "factor calculator arm does not match generated configurations: "
                f"{self.arm.label!r} != {metadata.get('factor_arm')!r}"
            )
        with helium_factor_parameter_scale(model, self.arm):
            return self.calculator.calculate(
                model=model,
                bundle=bundle,
                context=context,
            )


__all__ = ["FactorArmCalculator", "FactorResponseCalculator"]
