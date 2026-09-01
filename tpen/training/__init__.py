"""Training-loop namespace."""

from tpen.training.optim import make_optimizer
from tpen.training.state import TrainerState, TrainingTiming
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ScoreUpdateInput,
    VMCStepData,
    VMCUpdateMethod,
    VMCUpdateResult,
)
from tpen.training.vmc import (
    VMCObjectiveResult,
    compute_vmc_objective,
    hamiltonian_term_metric_prefix,
    summarize_local_energy_terms,
    summarize_logabs,
)

__all__ = [
    "TrainerState",
    "TrainingTiming",
    "AutogradUpdateInput",
    "LegacyAutogradUpdate",
    "ScoreUpdateInput",
    "VMCStepData",
    "VMCUpdateMethod",
    "VMCUpdateResult",
    "VMCObjectiveResult",
    "VMCTrainer",
    "compute_vmc_objective",
    "hamiltonian_term_metric_prefix",
    "make_optimizer",
    "summarize_local_energy_terms",
    "summarize_logabs",
]
