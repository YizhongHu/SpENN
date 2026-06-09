"""Training-loop namespace."""

from spenn.training.optim import make_optimizer
from spenn.training.state import TrainerState
from spenn.training.trainer import VMCTrainer
from spenn.training.vmc import (
    VMCObjectiveResult,
    compute_vmc_objective,
    hamiltonian_term_metric_prefix,
    summarize_local_energy_terms,
    summarize_logabs,
)

__all__ = [
    "TrainerState",
    "VMCObjectiveResult",
    "VMCTrainer",
    "compute_vmc_objective",
    "hamiltonian_term_metric_prefix",
    "make_optimizer",
    "summarize_local_energy_terms",
    "summarize_logabs",
]
