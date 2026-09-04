"""Native DDP VMC score-function step and reduction instrumentation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch

from tests.spikes.native_ddp.model_access import ModelAccess
from tests.spikes.native_ddp.runtime import DistributedRuntime
from tests.spikes.native_ddp.statistics import (
    FiniteStatistics,
    centered_terms,
    reduce_statistics,
)


@dataclass
class GradientReductionCounter:
    """Count only DDP reducer buckets, not ordinary diagnostic collectives."""

    world_size: int
    count: int = 0

    def communication_hook(self, _state, bucket):
        """All-reduce one bucket and perform the DDP average explicitly."""

        self.count += 1
        buffer = bucket.buffer()
        torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.SUM)
        buffer.div_(self.world_size)
        future = torch.futures.Future()
        future.set_result(buffer)
        return future


@dataclass(frozen=True)
class StepObservation:
    """Scalar and tensor evidence emitted by one native update."""

    stats: FiniteStatistics
    global_loss: float
    local_surrogate_loss: float
    scale_factor: float
    local_gradients: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor]
    gradient_reductions: int
    coordinate_gradient: torch.Tensor
    logabs: torch.Tensor


def install_gradient_counter(
    access: ModelAccess, runtime: DistributedRuntime
) -> GradientReductionCounter:
    """Install one reducer hook on the DDP wrapper."""

    counter = GradientReductionCounter(world_size=runtime.world_size)
    access.ddp_model.register_comm_hook(counter, counter.communication_hook)
    return counter


def prepare_statistics(
    runtime: DistributedRuntime, energy: torch.Tensor
) -> FiniteStatistics:
    """Run the rank-statistics collective before any backward call."""

    return reduce_statistics(runtime, energy)


def make_local_surrogate(
    logabs: torch.Tensor,
    energy: torch.Tensor,
    stats: FiniteStatistics,
    *,
    world_size: int,
) -> tuple[torch.Tensor, float]:
    """Build the W-scaled local surrogate and return its scale."""

    if stats.finite_count == 0:
        # The empty shard remains parameter-connected, but global M==0 is
        # refused by the worker before this function is used for backward.
        return (logabs * 0.0).sum(), 0.0
    scale = 2.0 * world_size / stats.finite_count
    return scale * centered_terms(logabs, energy, stats).sum(), scale


def run_score_function_step(
    access: ModelAccess,
    runtime: DistributedRuntime,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    energy: torch.Tensor,
    stats: FiniteStatistics,
    counter: GradientReductionCounter,
    *,
    before_backward: Callable[[], None] | None = None,
) -> StepObservation:
    """Run exactly one DDP-backed score update after coordinate work is done."""

    _, coordinate_gradient = access.coordinate_forward(features)
    raw_logabs = access.raw_model(features)
    diagnostic_surrogate, scale = make_local_surrogate(
        raw_logabs, energy, stats, world_size=runtime.world_size
    )
    local_gradients = {
        name: gradient.detach().clone()
        for (name, _parameter), gradient in zip(
            access.raw_model.named_parameters(),
            torch.autograd.grad(
                diagnostic_surrogate,
                tuple(access.raw_model.parameters()),
                allow_unused=True,
            ),
            strict=True,
        )
        if gradient is not None
    }

    optimizer.zero_grad(set_to_none=True)
    logabs = access.score_forward(features)
    local_surrogate, scale = make_local_surrogate(
        logabs, energy, stats, world_size=runtime.world_size
    )
    if before_backward is not None:
        before_backward()
    local_surrogate.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in access.raw_model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()

    local_term = float(centered_terms(logabs.detach(), energy, stats).sum().item())
    global_term = sum(float(value) for value in runtime.all_gather_objects(local_term))
    global_loss = 2.0 * global_term / stats.finite_count if stats.finite_count else 0.0
    return StepObservation(
        stats=stats,
        global_loss=global_loss,
        local_surrogate_loss=float(local_surrogate.detach().item()),
        scale_factor=scale,
        local_gradients=local_gradients,
        gradients=gradients,
        gradient_reductions=counter.count,
        coordinate_gradient=coordinate_gradient.detach(),
        logabs=logabs.detach(),
    )


def run_closure_step(
    access: ModelAccess,
    runtime: DistributedRuntime,
    optimizer: torch.optim.LBFGS,
    features: torch.Tensor,
    energy: torch.Tensor,
    stats: FiniteStatistics,
    counter: GradientReductionCounter,
    *,
    maximum_inner_iterates: int,
) -> tuple[StepObservation, int, int, int]:
    """Run a re-evaluating closure with all but its final backward suppressed."""

    if maximum_inner_iterates < 2:
        raise ValueError("closure test requires at least two inner iterates")
    closure_calls = 0
    synchronized_calls = 0
    final_gradient_call = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls, synchronized_calls, final_gradient_call
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        synchronize = closure_calls == maximum_inner_iterates
        context = nullcontext() if synchronize else access.ddp_model.no_sync()
        with context:
            logabs = access.score_forward(features)
            local_surrogate, _ = make_local_surrogate(
                logabs, energy, stats, world_size=runtime.world_size
            )
            local_surrogate.backward()
        if synchronize:
            synchronized_calls += 1
            final_gradient_call = closure_calls
        return local_surrogate

    optimizer.step(closure)
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in access.raw_model.named_parameters()
        if parameter.grad is not None
    }
    raw_logabs, coordinate_gradient = access.coordinate_forward(features)
    local_surrogate, scale = make_local_surrogate(
        raw_logabs, energy, stats, world_size=runtime.world_size
    )
    local_term = float(centered_terms(raw_logabs.detach(), energy, stats).sum().item())
    global_term = sum(float(value) for value in runtime.all_gather_objects(local_term))
    observation = StepObservation(
        stats=stats,
        global_loss=2.0 * global_term / stats.finite_count,
        local_surrogate_loss=float(local_surrogate.detach().item()),
        scale_factor=scale,
        local_gradients={},
        gradients=gradients,
        gradient_reductions=counter.count,
        coordinate_gradient=coordinate_gradient.detach(),
        logabs=raw_logabs.detach(),
    )
    return observation, closure_calls, synchronized_calls, final_gradient_call


__all__ = [
    "GradientReductionCounter",
    "StepObservation",
    "install_gradient_counter",
    "make_local_surrogate",
    "prepare_statistics",
    "run_closure_step",
    "run_score_function_step",
]
