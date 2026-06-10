"""Batched Metropolis sampler."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from spenn.data.batch import Walkers, WavefunctionOutput
from spenn.sampling.moves import GaussianMove


class MetropolisSampler(nn.Module):
    """Batched, stateful Metropolis-Hastings sampler.

    The sampler owns a persistent Markov chain: it holds the current walkers,
    burns in once, and advances the existing chain on each `collect_samples`
    call unless a reset is requested. It also owns all Markov-chain randomness
    through a sampler-local `torch.Generator` (initial walker positions,
    proposal noise, one-electron index selection, and accept/reject uniforms).
    Sampler code never mutates global Torch RNG state, and the runner/trainer
    must not seed on the sampler's behalf. Walker state and generator state are
    checkpointed together by `state_dict`/`load_state_dict`.

    Parameters
    ----------
    name : str, optional
        Human-readable sampler name.
    move : torch.nn.Module or None, optional
        Proposal kernel exposing ``propose(walkers, *, generator)`` and
        returning proposed positions plus a proposal log-ratio. The move
        consumes the sampler's generator; it does not own an RNG.
    n_walkers : int, optional
        Default number of walkers to initialize.
    burn_in : int, optional
        Number of equilibration steps run once per chain by `collect_samples`.
    n_steps : int, optional
        Default number of MCMC steps per sampling call.
    proposal_scale : float, optional
        Gaussian proposal scale used when `move` is ``None``.
    seed : int or None, optional
        Seed for the sampler-local generator. Controls only Markov-chain
        randomness, not model parameter initialization.
    n_electrons : int, optional
        Number of electrons per walker.
    spatial_dim : int, optional
        Spatial dimension of each electron coordinate.
    n_up, n_down : int or None, optional
        Spin partition. When both are given, walkers are initialized with the
        corresponding ``+1``/``-1`` spin labels.
    initial_scale : float, optional
        Standard deviation of normally initialized walker positions.
    dtype : torch.dtype or str, optional
        Floating-point dtype for initialized walkers.
    """

    def __init__(
        self,
        name: str = "metropolis",
        move: nn.Module | None = None,
        n_walkers: int = 1024,
        burn_in: int = 100,
        n_steps: int = 10,
        proposal_scale: float = 0.05,
        seed: int | None = None,
        n_electrons: int = 2,
        spatial_dim: int = 3,
        n_up: int | None = None,
        n_down: int | None = None,
        initial_scale: float = 1.0,
        dtype: torch.dtype | str = torch.float64,
    ) -> None:
        super().__init__()
        self.name = name
        self.move = move or GaussianMove(step_size=proposal_scale)
        self.n_walkers = n_walkers
        self.burn_in = burn_in
        self.n_steps = n_steps
        self.proposal_scale = proposal_scale
        self.seed = seed
        self.n_electrons = n_electrons
        self.spatial_dim = spatial_dim
        self.n_up = n_up
        self.n_down = n_down
        self.initial_scale = initial_scale
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.acceptance_rate = 0.0
        self.last_metrics: dict[str, float] = {}

        # Sampler-owned RNG and persistent Markov-chain state.
        self._generator_device = torch.device("cpu")
        self._generator = torch.Generator(device=self._generator_device)
        if self.seed is not None:
            self._generator.manual_seed(int(self.seed))
        self._walkers: Walkers | None = None
        self._has_burned_in = False

    @property
    def walkers(self) -> Walkers | None:
        """Return the current persistent walker state (``None`` before reset)."""

        return self._walkers

    @property
    def has_burned_in(self) -> bool:
        """Return whether the current chain has completed burn-in."""

        return self._has_burned_in

    def initialize(self, n_walkers: int | None = None, device=None) -> Walkers:
        """Initialize normally distributed walkers using the sampler generator.

        Parameters
        ----------
        n_walkers : int or None, optional
            Number of walkers to initialize. If ``None``, `self.n_walkers` is
            used.
        device : torch.device, str, or None, optional
            Optional device assertion. Must match the sampler generator device;
            use `reset` to move the chain to a new device.

        Returns
        -------
        Walkers
            Walker state with positions shaped ``[n_walkers, n_electrons,
            spatial_dim]``.
        """

        self._require_device(device)
        n_walkers = n_walkers or self.n_walkers
        positions = self.initial_scale * torch.randn(
            n_walkers,
            self.n_electrons,
            self.spatial_dim,
            device=self._generator_device,
            dtype=self.dtype,
            generator=self._generator,
        )
        spins = _default_spins(
            n_up=self.n_up,
            n_down=self.n_down,
            n_electrons=self.n_electrons,
            n_walkers=n_walkers,
            device=self._generator_device,
            dtype=self.dtype,
        )
        return Walkers(positions=positions, spins=spins)

    def reset(self, n_walkers: int | None = None, device=None) -> Walkers:
        """Re-seed the generator and start a fresh, un-burned-in chain.

        Parameters
        ----------
        n_walkers : int or None, optional
            Number of walkers to initialize.
        device : torch.device, str, or None, optional
            Device for the new chain and generator. Defaults to the current
            generator device.

        Returns
        -------
        Walkers
            The freshly initialized walker state.
        """

        target_device = torch.device(device) if device is not None else self._generator_device
        self._generator_device = target_device
        self._generator = torch.Generator(device=target_device)
        if self.seed is not None:
            self._generator.manual_seed(int(self.seed))
        self._walkers = self.initialize(n_walkers=n_walkers)
        self._has_burned_in = False
        return self._walkers

    def _require_device(self, device) -> None:
        if device is not None and torch.device(device) != self._generator_device:
            raise ValueError(
                f"sampler generator is on {self._generator_device}; cannot operate on "
                f"{torch.device(device)}. Call reset(device=...) to move the chain."
            )

    def _evaluate(self, model, walkers: Walkers) -> tuple[torch.Tensor, torch.Tensor]:
        batch = walkers.make_batch()
        with torch.no_grad():
            output = model(batch)
        if not isinstance(output, WavefunctionOutput):
            raise TypeError(f"Wavefunction model must return WavefunctionOutput, got {type(output)!r}")
        logabs = output.logabs
        sign = output.sign
        if logabs.shape != (walkers.batch_size,):
            raise ValueError(f"Model logabs must have shape [{walkers.batch_size}], got {tuple(logabs.shape)}")
        if sign.shape != (walkers.batch_size,):
            raise ValueError(f"Model sign must have shape [{walkers.batch_size}], got {tuple(sign.shape)}")
        return logabs, sign

    def _propose(self, model, walkers: Walkers) -> tuple[torch.Tensor, torch.Tensor]:
        del model
        if not hasattr(self.move, "propose"):
            raise TypeError("MetropolisSampler move must expose propose(walkers)")
        proposals, log_q_ratio = self.move.propose(walkers, generator=self._generator)
        if proposals.shape != walkers.positions.shape:
            raise ValueError(f"Proposal positions must have shape {tuple(walkers.positions.shape)}, got {tuple(proposals.shape)}")
        if log_q_ratio.shape != (walkers.batch_size,):
            raise ValueError(f"Proposal log-ratio must have shape [{walkers.batch_size}], got {tuple(log_q_ratio.shape)}")
        return proposals, log_q_ratio

    def step(self, model, walkers: Walkers) -> Walkers:
        """Run one Metropolis-Hastings step.

        Parameters
        ----------
        model : callable
            Wavefunction model returning `WavefunctionOutput`.
        walkers : Walkers
            Current walker state.

        Returns
        -------
        Walkers
            Updated walker state with cached wavefunction values and sampler
            diagnostics in ``aux``.
        """

        self._require_device(walkers.device)
        current_logabs = walkers.logabs
        current_sign = walkers.sign
        if current_logabs is None or current_sign is None:
            current_logabs, current_sign = self._evaluate(model, walkers)
        proposals, log_q_ratio = self._propose(model, walkers)
        proposal_walkers = Walkers(positions=proposals, spins=walkers.spins, aux=dict(walkers.aux))
        proposed_logabs, proposed_sign = self._evaluate(model, proposal_walkers)
        log_accept_ratio = torch.nan_to_num(2.0 * (proposed_logabs - current_logabs) + log_q_ratio, nan=-torch.inf)
        log_accept = torch.clamp(log_accept_ratio, max=0.0)
        uniforms = torch.rand(
            log_accept.shape,
            device=log_accept.device,
            dtype=log_accept.dtype,
            generator=self._generator,
        )
        accepted = torch.log(uniforms.clamp_min(1e-12)) < log_accept
        accepted_mask = accepted.view(-1, 1, 1)
        positions = torch.where(accepted_mask, proposals, walkers.positions)
        logabs = torch.where(accepted, proposed_logabs, current_logabs)
        sign = torch.where(accepted, proposed_sign, current_sign)
        self.acceptance_rate = accepted.to(dtype=torch.float32).mean().item()
        self.last_metrics = {
            "acceptance_rate": self.acceptance_rate,
            "mean_logabs": float(logabs.detach().mean().item()),
        }
        if hasattr(self.move, "step_size"):
            self.last_metrics["proposal_scale"] = float(self.move.step_size)
        return Walkers(
            positions=positions.detach(),
            logabs=logabs.detach(),
            sign=sign.detach(),
            spins=None if walkers.spins is None else walkers.spins.detach(),
            aux={
                **walkers.aux,
                "accepted": accepted.detach(),
                "log_accept_ratio": log_accept_ratio.detach(),
                "acceptance_rate": self.acceptance_rate,
            },
        )

    def sample(self, model, walkers: Walkers, n_steps: int | None = None) -> Walkers:
        """Run multiple Metropolis-Hastings steps.

        Parameters
        ----------
        model : callable
            Wavefunction model returning `WavefunctionOutput`.
        walkers : Walkers
            Current walker state.
        n_steps : int or None, optional
            Number of MCMC steps. If ``None``, `self.n_steps` is used.

        Returns
        -------
        Walkers
            Walker state after sampling. ``self.acceptance_rate`` is the mean
            acceptance rate over all steps in this call.
        """

        total_steps = self.n_steps if n_steps is None else n_steps
        if total_steps < 0:
            raise ValueError("n_steps must be non-negative")
        acceptance_sum = 0.0
        for _ in range(total_steps):
            walkers = self.step(model, walkers)
            acceptance_sum += float(walkers.aux["acceptance_rate"])
        if total_steps:
            self.acceptance_rate = acceptance_sum / total_steps
            self.last_metrics["acceptance_rate"] = self.acceptance_rate
        walkers.aux["sample_acceptance_rate"] = self.acceptance_rate
        return walkers

    def collect_samples(
        self,
        model,
        *,
        reset: bool = False,
        device=None,
    ) -> tuple[Walkers, dict[str, float]]:
        """Advance the persistent chain and draw production samples.

        On the first call (or when ``reset=True``) the chain is initialized and
        burned in once; subsequent calls advance the existing walkers without
        re-burning. The sampler owns its walkers and RNG across calls.

        Parameters
        ----------
        model : callable
            Wavefunction model returning `WavefunctionOutput`.
        reset : bool, optional
            Force a fresh, re-seeded, un-burned-in chain.
        device : torch.device, str, or None, optional
            Target device for the chain. On a fresh chain this selects the
            device; on an existing chain a mismatching device raises.

        Returns
        -------
        tuple
            Pair ``(walkers, stats)`` where ``walkers`` holds the final samples
            and ``stats`` reports sampler diagnostics for logging.
        """

        if reset or self._walkers is None:
            self.reset(device=device)
        else:
            self._require_device(device)
        if not self._has_burned_in and self.burn_in:
            self._walkers = self.sample(model, self._walkers, self.burn_in)
            self._has_burned_in = True
        self._walkers = self.sample(model, self._walkers, self.n_steps)
        stats = {
            "acceptance_rate": float(self.acceptance_rate),
            "n_walkers": int(self._walkers.batch_size),
            "burn_in": int(self.burn_in),
            "n_steps": int(self.n_steps),
        }
        return self._walkers, stats

    def mcmc_state_dict(self) -> dict[str, Any]:
        """Return checkpointable Markov-chain and RNG state.

        This is intentionally separate from `torch.nn.Module.state_dict`, which
        keeps its normal module-parameter semantics. MCMC state (walkers,
        burn-in flag, running acceptance, and generator state) is persisted here
        instead so checkpointing does not abuse the standard module API.
        """

        return {
            "walkers": self._walkers,
            "has_burned_in": self._has_burned_in,
            "acceptance_rate": float(self.acceptance_rate),
            "generator_state": self._generator.get_state(),
            "generator_device": str(self._generator_device),
        }

    def load_mcmc_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore Markov-chain and RNG state from `mcmc_state_dict`.

        Recreates the generator on the checkpointed device and restores its
        state, so a resumed run continues the same Markov chain.
        """

        self._generator_device = torch.device(state["generator_device"])
        self._generator = torch.Generator(device=self._generator_device)
        self._generator.set_state(state["generator_state"])
        self._walkers = state["walkers"]
        self._has_burned_in = bool(state["has_burned_in"])
        self.acceptance_rate = float(state.get("acceptance_rate", 0.0))


def _default_spins(
    *,
    n_up: int | None,
    n_down: int | None,
    n_electrons: int,
    n_walkers: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return repeated spin labels from a spin partition.

    Parameters
    ----------
    n_up, n_down : int or None
        Spin partition. If either is ``None``, no spin labels are produced.
    n_electrons : int
        Number of electrons; must equal ``n_up + n_down`` when both are given.
    n_walkers : int
        Number of walkers.
    device : torch.device, str, or None
        Target device for the spin tensor.
    dtype : torch.dtype
        Target dtype for the spin tensor.

    Returns
    -------
    torch.Tensor or None
        Spin labels with shape ``[n_walkers, n_electrons]`` when a partition is
        available, otherwise ``None``.
    """

    if n_up is None or n_down is None:
        return None
    spins = torch.tensor([1.0] * n_up + [-1.0] * n_down, device=device, dtype=dtype)
    if spins.numel() != n_electrons:
        raise ValueError("Spin partition must match n_electrons")
    return spins.unsqueeze(0).expand(n_walkers, -1).clone()
