"""VMC trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spenn.data_structures.batch import ElectronBatch
from spenn.physics.systems import ElectronicSystem
from spenn.training.metrics import gradient_norm, parameter_norm


@dataclass
class TrainerConfig:
    """Store minimal trainer loop settings.

    Parameters
    ----------
    max_steps : int, optional
        Default number of optimization steps for `VMCTrainer.fit`.
    log_every : int, optional
        Step interval intended for logging.
    checkpoint_every : int, optional
        Step interval intended for checkpointing.
    """

    max_steps: int = 1000
    log_every: int = 10
    checkpoint_every: int = 100


class VMCTrainer:
    """Coordinate VMC sampling, loss evaluation, and optimization.

    Parameters
    ----------
    model : torch.nn.Module
        Wavefunction model to optimize.
    sampler : object
        Sampler with ``initialize`` and ``sample`` methods.
    hamiltonian : object
        Hamiltonian used by `loss` to compute local energies.
    loss : callable
        Loss callable returning ``(loss, metrics)``.
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters.
    scheduler : object or None, optional
        Optional scheduler with a ``step`` method.
    logger : object or None, optional
        Optional logger with a ``log`` method.
    cfg : object or None, optional
        Legacy config-like object. Values from `cfg` are used when present.
    max_steps, log_every, checkpoint_every : int or None, optional
        Training-loop settings used to build `TrainerConfig`.
    system : ElectronicSystem or None, optional
        System passed to the sampler when walkers are initialized.
    walkers : object or None, optional
        Initial walker state. If ``None``, the sampler initializes walkers.
    device : torch.device, str, or None, optional
        Device used for default walker initialization.
    **_ : object
        Ignored keyword arguments accepted for config compatibility.
    """

    def __init__(
        self,
        model,
        sampler,
        hamiltonian,
        loss,
        optimizer,
        scheduler=None,
        logger=None,
        cfg=None,
        max_steps: int | None = None,
        log_every: int | None = None,
        checkpoint_every: int | None = None,
        system: ElectronicSystem | None = None,
        walkers=None,
        device=None,
        **_: object,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.hamiltonian = hamiltonian
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        if cfg is not None:
            max_steps = getattr(cfg, "max_steps", max_steps)
            log_every = getattr(cfg, "log_every", log_every)
            checkpoint_every = getattr(cfg, "checkpoint_every", checkpoint_every)
        self.cfg = TrainerConfig(
            max_steps=TrainerConfig.max_steps if max_steps is None else max_steps,
            log_every=TrainerConfig.log_every if log_every is None else log_every,
            checkpoint_every=TrainerConfig.checkpoint_every if checkpoint_every is None else checkpoint_every,
        )
        self.system = system or getattr(sampler, "system", None)
        try:
            default_device = next(model.parameters()).device
        except StopIteration:
            default_device = torch.device("cpu")
        self.device = device or default_device
        self.walkers = walkers or sampler.initialize(system=self.system, device=self.device)
        self.global_step = 0

    def _log(self, metrics: dict) -> None:
        if self.logger is not None:
            self.logger.log(metrics)

    def train_step(self) -> dict:
        """Run one sampler and optimizer step.

        Returns
        -------
        dict
            Metrics from the loss with additional loss, acceptance-rate,
            gradient-norm, and parameter-norm entries.
        """

        self.model.train()
        self.walkers = self.sampler.sample(self.model, self.walkers, getattr(self.sampler, "steps_per_iter", 1))
        batch = ElectronBatch(
            positions=self.walkers.positions,
            spins=self.walkers.spins,
            system=self.walkers.aux.get("system"),
            aux=dict(self.walkers.aux),
        )
        loss, metrics = self.loss(self.model, self.hamiltonian, batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        metrics = dict(metrics)
        metrics["loss"] = loss.detach()
        metrics["acceptance_rate"] = torch.tensor(getattr(self.sampler, "acceptance_rate", 0.0))
        metrics["grad_norm"] = gradient_norm(self.model).detach()
        metrics["param_norm"] = parameter_norm(self.model).detach()
        self._log(metrics)
        self.global_step += 1
        return metrics

    def fit(self, max_steps: int | None = None) -> list[dict]:
        """Run the training loop for a fixed number of steps.

        Parameters
        ----------
        max_steps : int or None, optional
            Number of steps to run. If ``None``, `self.cfg.max_steps` is used.

        Returns
        -------
        list of dict
            Per-step metric dictionaries.
        """

        history = []
        total_steps = self.cfg.max_steps if max_steps is None else max_steps
        for _ in range(total_steps):
            history.append(self.train_step())
        return history
