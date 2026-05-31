"""VMC loss."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch, WavefunctionOutput


def _extract_logabs(output: WavefunctionOutput | torch.Tensor) -> torch.Tensor:
    if isinstance(output, WavefunctionOutput):
        return output.logabs
    return output


class VMCLoss(nn.Module):
    """Compute the score-function VMC energy objective.

    Parameters
    ----------
    center_energy : bool, optional
        Whether to subtract the sample mean local energy before forming the
        score-function objective. Centering leaves the expected gradient
        unchanged and reduces variance.
    scale_factor : float, optional
        Multiplicative factor on the score-function objective. The default
        value ``2`` corresponds to gradients of an expectation under
        ``|psi|^2``.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, center_energy: bool = True, scale_factor: float = 2.0, **_: object) -> None:
        super().__init__()
        self.center_energy = center_energy
        self.scale_factor = float(scale_factor)

    def forward(self, model, hamiltonian, batch: ElectronBatch):
        """Evaluate the VMC objective and diagnostics for a batch.

        Parameters
        ----------
        model : callable
            Wavefunction model evaluated by `hamiltonian`.
        hamiltonian : object
            Hamiltonian object with a ``local_energy`` method.
        batch : ElectronBatch
            Electron configurations used for the estimate.

        Returns
        -------
        tuple
            Pair ``(loss, metrics)`` where ``loss`` is a differentiable
            score-function objective and ``metrics`` contains detached energy
            and variance tensors.
        """

        batch = batch.flatten_samples()
        model_output = model(batch)
        logabs = _extract_logabs(model_output)
        if logabs.shape != (batch.batch_size,):
            raise ValueError(f"Model logabs must have shape [{batch.batch_size}], got {tuple(logabs.shape)}")
        local_energy = hamiltonian.local_energy(model, batch)
        if local_energy.shape != (batch.batch_size,):
            raise ValueError(
                f"Hamiltonian local energy must have shape [{batch.batch_size}], got {tuple(local_energy.shape)}"
            )
        detached_energy = local_energy.detach()
        weight = detached_energy - detached_energy.mean() if self.center_energy else detached_energy
        loss = self.scale_factor * (weight * logabs).mean()
        metrics = {
            "energy": detached_energy.mean(),
            "variance": detached_energy.var(unbiased=False),
            "objective": loss.detach(),
        }
        return loss, metrics
