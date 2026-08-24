"""Factor-overridden generators for separately re-equilibrated comparisons."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from tpen.evaluation.bundle import GeneratedConfigurations
from tpen.evaluation.factor_response import FactorParameterScale, helium_factor_parameter_scale
from tpen.evaluation.protocols import EvaluationContext


class ReequilibratedFactorGenerator:
    """Run a delegated generator under one temporary factor arm."""

    name = "reequilibrated_factor"

    def __init__(
        self,
        *,
        generator: object,
        arm: Mapping[str, object] | FactorParameterScale,
    ) -> None:
        generate = getattr(generator, "generate", None)
        if not callable(generate):
            raise TypeError("ReequilibratedFactorGenerator requires a generator delegate")
        self.generator = generator
        self.arm = arm if isinstance(arm, FactorParameterScale) else FactorParameterScale.from_mapping(arm)

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Generate fresh chains while restoring the checkpoint model afterwards."""

        if model is None:
            raise TypeError("ReequilibratedFactorGenerator requires a model")
        with helium_factor_parameter_scale(model, self.arm) as realized:
            generated = self.generator.generate(model=model, context=context)
        return GeneratedConfigurations(
            batch=generated.batch,
            metadata={
                **generated.metadata,
                "comparison_kind": "re_equilibrated",
                "factor_arm": self.arm.label,
                "factor_b_ee_scale": self.arm.b_ee,
                "factor_c_electron_nucleus_scale": self.arm.c_electron_nucleus,
                "factor_d_electron_nucleus_scale": self.arm.d_electron_nucleus,
                **{f"factor_realized_{name}": value for name, value in realized.items()},
            },
            trajectory_records=generated.trajectory_records,
        )


__all__ = ["ReequilibratedFactorGenerator"]
