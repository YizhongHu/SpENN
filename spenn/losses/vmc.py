"""VMC loss."""

from __future__ import annotations

from torch import nn

from spenn.data.batch import ElectronBatch


class VMCLoss(nn.Module):
    """Compute the VMC mean-energy objective."""

    def forward(self, model, hamiltonian, batch: ElectronBatch):
        """Evaluate loss and diagnostics for a batch.

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
            Pair ``(loss, metrics)`` where ``loss`` is the mean local energy
            and ``metrics`` contains detached energy and variance tensors.
        """

        local_energy = hamiltonian.local_energy(model, batch)
        loss = local_energy.mean()
        metrics = {
            "energy": local_energy.mean().detach(),
            "variance": local_energy.var(unbiased=False).detach(),
        }
        return loss, metrics
