"""Training-loop namespace."""

from tpen.training.optim import make_optimizer
from tpen.training.state import TrainerState, TrainingTiming
from tpen.training.statistics import (
    IdentityStatisticsReducer,
    StatisticsReducer,
    StatisticsSums,
    center_statistics,
)
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ModelParameterBinding,
    ScoreUpdateInput,
    VMCStepData,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
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
    "ModelParameterBinding",
    "ScoreUpdateInput",
    "VMCStepData",
    "VMCUpdateMethod",
    "VMCUpdateResult",
    "VMCUpdateState",
    "IdentityStatisticsReducer",
    "StatisticsReducer",
    "StatisticsSums",
    "center_statistics",
    "VMCObjectiveResult",
    "VMCTrainer",
    "compute_vmc_objective",
    "hamiltonian_term_metric_prefix",
    "make_optimizer",
    "summarize_local_energy_terms",
    "summarize_logabs",
]
