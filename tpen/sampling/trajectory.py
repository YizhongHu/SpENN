"""Fixed-model trajectory collection with an explicit draw axis.

`MetropolisSampler.collect_samples` returns only the final state of its
persistent walkers, so nothing in the sampling package retains a time series.
This module adds the missing draw axis by calling the sampler's public
production entry point repeatedly and evaluating one observable per draw,
producing a `[draw, walker]` trajectory that the statistics package can
estimate an autocorrelation time from.

"Fixed model" is enforced, not assumed. Autocorrelation over training updates is
meaningless -- the target distribution is moving, so a lag-k correlation mixes
the chain's memory with the model's -- and a trajectory silently collected while
parameters drifted would produce a confident number for a quantity that does not
exist. The collector therefore evaluates under `torch.no_grad()` in eval mode
and, by default, verifies that no parameter moved.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from tpen.data.batch.walkers import Walkers
from tpen.sampling.stats import SamplerStats
from tpen.statistics.trajectory import ObservableTrajectory

__all__ = ["ModelDriftError", "collect_observable_trajectory", "parameter_fingerprint"]


class ModelDriftError(RuntimeError):
    """Raised when model parameters changed during a fixed-model collection."""


def parameter_fingerprint(model: torch.nn.Module) -> str:
    """Return a sha256 over a model's parameter and buffer values.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose state should be fingerprinted.

    Returns
    -------
    str
        Hex digest covering every parameter and buffer, in sorted name order,
        including each tensor's name, shape and raw bytes.
    """

    digest = hashlib.sha256()
    named = list(model.named_parameters()) + list(model.named_buffers())
    for name, tensor in sorted(named, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        values = tensor.detach().cpu().contiguous().to(torch.float64).numpy()
        digest.update(values.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


@contextmanager
def _frozen_eval(model: torch.nn.Module) -> Iterator[None]:
    """Run a block with `model` in eval mode, restoring the prior mode after."""

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def collect_observable_trajectory(
    sampler: object,
    model: torch.nn.Module,
    observable: Callable[[Walkers], torch.Tensor],
    *,
    observable_name: str,
    n_draws: int,
    discard_draws: int = 0,
    device: torch.device | str | None = None,
    reset: bool = False,
    verify_model_unchanged: bool = True,
) -> tuple[ObservableTrajectory, SamplerStats]:
    """Collect a `[draw, walker]` observable trajectory from a fixed model.

    Each draw advances the sampler's persistent chain by its configured
    ``n_steps`` and evaluates `observable` on the resulting walkers. Walker
    columns are kept separate for the whole collection: they are independent
    chains and the statistics layer must never see them concatenated.

    Parameters
    ----------
    sampler : object
        Sampler exposing ``collect_samples(model, *, reset=..., device=...)``
        returning ``(Walkers, SamplerStats)``.
    model : torch.nn.Module
        Wavefunction model, held fixed for the whole collection.
    observable : callable
        Maps one `Walkers` state to a ``[n_walkers]`` tensor of per-walker
        values. Autocorrelation is observable-specific, so exactly one
        observable is collected per call.
    observable_name : str
        Name recorded on the trajectory, for example ``"local_energy"``.
    n_draws : int
        Number of draws to retain per walker.
    discard_draws : int, optional
        Draws to collect and discard before retention begins, on top of the
        sampler's own burn-in. Recorded on the trajectory as ``burn_in_draws``.
    device : torch.device, str, or None, optional
        Forwarded to the sampler.
    reset : bool, optional
        Force a fresh, re-seeded, re-burned-in chain before the first draw.
    verify_model_unchanged : bool, optional
        Fingerprint parameters before and after collection and raise if they
        differ. Defaults to ``True``.

    Returns
    -------
    trajectory : ObservableTrajectory
        Retained samples with explicit ``[draw, walker]`` axes. ``draw_stride``
        is the sampler's own ``n_steps`` between draws, read from the typed
        `SamplerStats` record rather than probed off the sampler object.
    stats : SamplerStats
        The typed record from the final draw.

    Raises
    ------
    TypeError
        If `sampler` does not expose ``collect_samples``, or `observable` does
        not return a tensor.
    ValueError
        If `n_draws` is below one, `discard_draws` is negative, or the
        observable's shape does not match the walker count.
    ModelDriftError
        If `verify_model_unchanged` is set and any parameter moved.
    """

    collect = getattr(sampler, "collect_samples", None)
    if not callable(collect):
        raise TypeError("sampler must expose collect_samples(model, *, reset=..., device=...)")
    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}")
    if discard_draws < 0:
        raise ValueError(f"discard_draws must be non-negative, got {discard_draws}")

    fingerprint_before = parameter_fingerprint(model) if verify_model_unchanged else None

    columns: list[torch.Tensor] = []
    stats: SamplerStats | None = None
    with _frozen_eval(model):
        for draw_index in range(discard_draws + n_draws):
            # `reset` applies to the first call only: re-seeding mid-collection
            # would restart the chain and destroy the draw axis being built.
            walkers, stats = collect(
                model,
                reset=reset and draw_index == 0,
                device=device,
            )
            values = observable(walkers)
            if not isinstance(values, torch.Tensor):
                raise TypeError(
                    f"observable must return a torch.Tensor, got {type(values).__name__}"
                )
            values = values.detach().reshape(-1)
            if values.numel() != walkers.batch_size:
                raise ValueError(
                    f"observable returned {values.numel()} values for {walkers.batch_size} "
                    "walkers; one value per walker is required to keep chains separate"
                )
            if draw_index >= discard_draws:
                columns.append(values.to(torch.float64).cpu())

    if verify_model_unchanged:
        fingerprint_after = parameter_fingerprint(model)
        if fingerprint_after != fingerprint_before:
            raise ModelDriftError(
                "model parameters changed during trajectory collection; "
                "autocorrelation over a moving target distribution is undefined. "
                f"before={fingerprint_before} after={fingerprint_after}"
            )

    # The loop above runs at least once because n_draws >= 1 was validated.
    if stats is None:
        raise TypeError("sampler.collect_samples did not return a SamplerStats record")
    if stats.n_steps < 1:
        raise ValueError(
            f"sampler advances {stats.n_steps} steps per draw; consecutive draws would be "
            "identical and the draw axis would carry no information"
        )
    return (
        ObservableTrajectory(
            observable=observable_name,
            values=torch.stack(columns, dim=0),
            draw_stride=stats.n_steps,
            burn_in_draws=discard_draws,
        ),
        stats,
    )
