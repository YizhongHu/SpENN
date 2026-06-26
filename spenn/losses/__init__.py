"""Loss function namespace for VMC objectives."""

from spenn.losses.energy import mean_energy
from spenn.losses.variance import variance
from spenn.losses.vmc import VMCLoss

__all__ = ["VMCLoss", "mean_energy", "variance"]
