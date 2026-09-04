"""DF1-compatible native DDP/DCP worker entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tests.helpers.ddp_fault_injection import FaultKind, FaultPhase, FaultPlan, read_fault_plan
from tests.spikes.native_ddp.checkpoint import (
    CheckpointCorrupt,
    CheckpointPayloadStore,
    CheckpointTopologyMismatch,
    _jsonable,
)
from tests.spikes.native_ddp.fixtures import scientific_fixture
from tests.spikes.native_ddp.model_access import ModelAccess, SemanticWavefunction
from tests.spikes.native_ddp.runtime import DistributedRuntime
from tests.spikes.native_ddp.sampler import RankLocalSampler
from tests.spikes.native_ddp.seed import (
    SeedPartition,
    capture_global_rng_state,
    restore_global_rng_state,
    seed_global_rngs,
)
from tests.spikes.native_ddp.vmc_step import (
    install_gradient_counter,
    prepare_statistics,
    run_closure_step,
    run_score_function_step,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # DF1's fixed worker CLI is intentionally accepted unchanged.
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--rendezvous-file", type=str, required=True)
    parser.add_argument("--receipt-path", type=str, required=True)
    parser.add_argument("--state-path", type=str, required=True)
    parser.add_argument("--complete-marker-path", type=str, required=True)
    parser.add_argument("--pg-timeout", type=float, required=True)
    parser.add_argument("--fault-plan-path", type=str, default=None)
    parser.add_argument("--experiment", choices=("scientific", "resume"), default="scientific")
    parser.add_argument("--fixture", choices=("regular", "m2", "all_invalid"), default="regular")
    parser.add_argument("--optimizer", choices=("sgd", "adam", "closure"), default="sgd")
    parser.add_argument("--mcmc-steps", type=int, default=1)
    parser.add_argument("--kinetic-forwards", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--closure-inner-iterates", type=int, default=3)
    parser.add_argument("--checkpoint-root", type=str, default=None)
    parser.add_argument("--checkpoint-generation", type=int, default=None)
    parser.add_argument("--resume-generation", type=int, default=None)
    parser.add_argument("--checkpoint-failure-rank", type=int, default=None)
    parser.add_argument("--checkpoint-delay-rank", type=int, default=None)
    parser.add_argument("--checkpoint-delay-seconds", type=float, default=0.0)
    return parser


def _fault_matches(plan: FaultPlan | None, rank: int, phase: FaultPhase) -> bool:
    return plan is not None and plan.target_rank == rank and plan.phase == phase


def _apply_fault(
    plan: FaultPlan | None,
    *,
    rank: int,
    phase: FaultPhase,
    state_path: Path,
) -> None:
    """Apply only the unchanged DF1 fault taxonomy at a native hook point."""

    if not _fault_matches(plan, rank, phase):
        return
    assert plan is not None
    print(
        f"ddp harness injected fault: rank {rank} phase {phase.name} kind {plan.kind.name}",
        file=sys.stderr,
        flush=True,
    )
    failure = {
        "status": "fault_applied",
        "rank": rank,
        "fault_kind": plan.kind.name,
        "fault_phase": phase.name,
        "failure_evidence": "native worker matched the configured rank and phase",
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(failure, sort_keys=True))
    if plan.kind == FaultKind.RAISE_BEFORE_BACKWARD:
        raise RuntimeError(f"ddp harness injected fault: rank {rank} phase {phase.name}")
    if plan.kind in (FaultKind.SKIP_COLLECTIVE, FaultKind.MISMATCH_COLLECTIVE, FaultKind.MISMATCH_SHAPE):
        os._exit(2)
    if plan.kind in (FaultKind.STALL_BEFORE_COLLECTIVE,):
        time.sleep(plan.delay_seconds)


def _optimizer(model: SemanticWavefunction, name: str) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=0.01)
    if name == "closure":
        return torch.optim.LBFGS(
            model.parameters(),
            lr=0.25,
            max_iter=3,
            history_size=5,
            tolerance_grad=0.0,
            tolerance_change=0.0,
        )
    raise ValueError(f"unsupported optimizer {name!r}")


def _parameters(model: SemanticWavefunction) -> dict[str, list[float]]:
    return {name: parameter.detach().cpu().tolist() for name, parameter in model.named_parameters()}


def _tensors(values: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {name: _jsonable(value) for name, value in values.items()}


def _common_observability(
    *,
    rank: int,
    world_size: int,
    access: ModelAccess,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "world_size": world_size,
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "access": {
            "coordinate_forward_owner": "raw_model",
            "score_forward_owner": "ddp_model",
            "used_module_attribute": False,
        },
        "canonical_model_keys": list(access.raw_model.state_dict().keys()),
        "ddp_forward_calls": access.forward_counts["ddp"],
        "optimizer_type": type(optimizer).__name__,
        "torch_version": torch.__version__,
        "api_inventory": {
            "stable": [
                "torch.nn.parallel.DistributedDataParallel",
                "torch.distributed.checkpoint.FileSystemReader",
                "torch.distributed.checkpoint.FileSystemWriter",
            ],
            "experimental": [
                "torch.distributed.checkpoint.state_dict.get_state_dict",
                "torch.distributed.checkpoint.state_dict.set_state_dict",
            ],
            "spike_prototype": [
                "tests.spikes.native_ddp.DistributedRuntime",
                "tests.spikes.native_ddp.CheckpointPayloadStore",
                "run_gloo_subprocess_group(worker_module=..., worker_extra_args=...)",
            ],
        },
        "accelerator_execution": False,
        "accelerator_path_inspected": "CUDA -> NCCL; ROCm -> RCCL; XPU -> XCCL",
    }


def _write_receipt(args: argparse.Namespace, phase_sequence: list[str], fault_kind: str) -> None:
    payload = {
        "rank": args.rank,
        "world_size": args.world_size,
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "phase_sequence": phase_sequence,
        "collective_result": None,
        "fault_kind": fault_kind,
    }
    Path(args.receipt_path).write_text(json.dumps(payload, sort_keys=True))


def _failure_state(args: argparse.Namespace, status: str, error: BaseException) -> None:
    path = Path(args.state_path)
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if state.get("status") not in {"success", "checkpoint_pending"}:
            return
        state.update(
            {
                "status": status,
                "failure_exception": f"{type(error).__name__}: {error}",
                "failure_evidence": "native worker failed after provisional state emission",
            }
        )
        path.write_text(json.dumps(state, sort_keys=True))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "rank": args.rank,
                "world_size": args.world_size,
                "failure_exception": f"{type(error).__name__}: {error}",
                "failure_evidence": "native worker preserved its exception classification",
            },
            sort_keys=True,
        )
    )


def _coordinates_and_energy(
    access: ModelAccess, sampler: RankLocalSampler
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert sampler.walkers is not None
    coordinates = sampler.walkers
    _, coordinate_gradient = access.coordinate_forward(coordinates)
    energy = coordinates.square().sum(dim=1) + 0.5 * coordinate_gradient.square().sum(dim=1)
    return coordinates, energy.detach(), coordinate_gradient.detach()


def _run(args: argparse.Namespace) -> int:
    plan = read_fault_plan(Path(args.fault_plan_path)) if args.fault_plan_path else None
    seed_global_rngs(12_345 + args.rank)
    runtime = DistributedRuntime.initialize(
        rank=args.rank,
        world_size=args.world_size,
        rendezvous_file=Path(args.rendezvous_file),
        process_group_timeout_seconds=args.pg_timeout,
    )
    model = SemanticWavefunction()
    access = ModelAccess.create(model)
    optimizer = _optimizer(model, args.optimizer)
    counter = install_gradient_counter(access, runtime)
    store = (
        None
        if args.checkpoint_root is None
        else CheckpointPayloadStore(root=Path(args.checkpoint_root), runtime=runtime)
    )
    sampler = RankLocalSampler(SeedPartition(base_seed=50_000, rank=args.rank, world_size=args.world_size))
    completed_updates = 0
    phase_sequence: list[str] = []

    if args.experiment == "resume" and args.resume_generation is not None:
        if store is None:
            raise ValueError("resume requires --checkpoint-root")
        resume_parameters_before = _parameters(model)
        resume_optimizer_before = _jsonable(optimizer.state_dict())
        try:
            loaded = store.load(model, optimizer, generation=args.resume_generation)
        except CheckpointTopologyMismatch as exc:
            state = _common_observability(
                rank=args.rank, world_size=args.world_size, access=access, optimizer=optimizer
            )
            state.update(
                {
                    "status": "topology_refused",
                    "counter_before": 0,
                    "counter_after": 0,
                    "parameters_before": resume_parameters_before,
                    "parameters_after": _parameters(model),
                    "optimizer_state_before": resume_optimizer_before,
                    "optimizer_state_after": _jsonable(optimizer.state_dict()),
                    "failure_exception": f"{type(exc).__name__}: {exc}",
                }
            )
            Path(args.state_path).write_text(json.dumps(state, sort_keys=True))
            runtime.close()
            _write_receipt(args, phase_sequence, plan.kind.name if plan else FaultKind.NONE.name)
            return 0
        except CheckpointCorrupt as exc:
            _failure_state(args, "resume_failed", exc)
            runtime.close()
            _write_receipt(args, phase_sequence, plan.kind.name if plan else FaultKind.NONE.name)
            return 3
        sidecar = loaded["sidecar"]
        sampler.load_state_dict(sidecar["sampler_state"])
        restore_global_rng_state(sidecar["rng_state"])
        completed_updates = int(sidecar["completed_updates"])

    parameters_before = _parameters(model)
    optimizer_before = _jsonable(optimizer.state_dict())
    last_observation = None
    last_coordinates = None
    last_energy = None
    last_coordinate_gradient = None
    closure_calls = 0
    synchronized_closure_calls = 0
    final_gradient_call = 0
    kinetic_raw_model_calls = 0

    for iteration in range(completed_updates, args.iterations):
        reductions_before_sampling = counter.count
        if args.experiment == "scientific":
            kinetic_raw_model_calls = 0
            features, energy = scientific_fixture(args.world_size, args.rank, kind=args.fixture)
            sampler.advance(model, args.mcmc_steps)
            for _ in range(args.kinetic_forwards):
                _, kinetic_gradient = access.coordinate_forward(features)
                last_coordinate_gradient = kinetic_gradient.detach()
                kinetic_raw_model_calls += 1
            last_coordinates = features
            last_energy = energy
        else:
            sampler.advance(model, args.mcmc_steps)
            last_coordinates, last_energy, last_coordinate_gradient = _coordinates_and_energy(access, sampler)

        phase_sequence.append(FaultPhase.BEFORE_COLLECTIVE.name)
        _apply_fault(plan, rank=args.rank, phase=FaultPhase.BEFORE_COLLECTIVE, state_path=Path(args.state_path))
        stats = prepare_statistics(runtime, last_energy)
        phase_sequence.append(FaultPhase.AFTER_COLLECTIVE.name)

        if stats.finite_count == 0:
            state = _common_observability(
                rank=args.rank, world_size=args.world_size, access=access, optimizer=optimizer
            )
            state.update(
                {
                    "status": "refused",
                    "reason": "global finite-energy count is zero",
                    "backward_called": False,
                    "optimizer_step": False,
                    "counter_before": completed_updates,
                    "counter_after": completed_updates,
                    "parameters_before": parameters_before,
                    "parameters_after": _parameters(model),
                    "optimizer_state_before": optimizer_before,
                    "optimizer_state_after": _jsonable(optimizer.state_dict()),
                    "ddp_gradient_reductions": counter.count,
                    "ddp_gradient_reductions_per_update": 0,
                    "phase_sequence": phase_sequence,
                    "global_statistics": stats.as_dict(),
                }
            )
            Path(args.state_path).write_text(json.dumps(state, sort_keys=True))
            runtime.barrier()
            runtime.close()
            _write_receipt(args, phase_sequence, plan.kind.name if plan else FaultKind.NONE.name)
            return 0

        phase_sequence.append(FaultPhase.BEFORE_OPTIMIZER_STEP.name)
        _apply_fault(
            plan, rank=args.rank, phase=FaultPhase.BEFORE_OPTIMIZER_STEP, state_path=Path(args.state_path)
        )
        reductions_before = counter.count
        if args.optimizer == "closure":
            last_observation, closure_calls, synchronized_closure_calls, final_gradient_call = run_closure_step(
                access,
                runtime,
                optimizer,
                last_coordinates,
                last_energy,
                stats,
                counter,
                maximum_inner_iterates=args.closure_inner_iterates,
            )
        else:
            last_observation = run_score_function_step(
                access,
                runtime,
                optimizer,
                last_coordinates,
                last_energy,
                stats,
                counter,
            )
        phase_sequence.append(FaultPhase.AFTER_OPTIMIZER_STEP.name)
        completed_updates += 1

    assert last_observation is not None
    assert last_coordinates is not None and last_energy is not None
    state = _common_observability(
        rank=args.rank, world_size=args.world_size, access=access, optimizer=optimizer
    )
    state.update(
        {
            "status": "success",
            "backward_called": True,
            "optimizer_step": True,
            "loss_requires_grad": True,
            "phase_sequence": phase_sequence,
            "coordinates": _jsonable(last_coordinates),
            "logabs": _jsonable(last_observation.logabs),
            "energy": _jsonable(last_energy),
            "coordinate_gradient": _jsonable(last_observation.coordinate_gradient),
            "global_statistics": last_observation.stats.as_dict(),
            "global_metrics": {
                "energy": last_observation.stats.mean,
                "energy_variance": last_observation.stats.variance,
                "energy_std": last_observation.stats.variance**0.5,
                "energy_stderr": (
                    last_observation.stats.variance / last_observation.stats.finite_count
                )
                ** 0.5,
                "local_energy_n_finite": last_observation.stats.finite_count,
                "local_energy_n_total": last_observation.stats.total_count,
                "local_energy_nonfinite_count": (
                    last_observation.stats.total_count - last_observation.stats.finite_count
                ),
            },
            "global_loss": last_observation.global_loss,
            "local_surrogate_loss": last_observation.local_surrogate_loss,
            "ddp_backward_scale": last_observation.scale_factor,
            "local_gradients": _tensors(last_observation.local_gradients),
            "gradients": _tensors(last_observation.gradients),
            "ddp_gradient_reductions": last_observation.gradient_reductions,
            "ddp_gradient_reductions_per_update": last_observation.gradient_reductions - reductions_before,
            "sampling_gradient_reductions": reductions_before - reductions_before_sampling,
            "coordinate_forward_count": access.forward_counts["raw"],
            "sampling_raw_model_calls": sampler.coordinate_forward_count,
            "kinetic_raw_model_calls": kinetic_raw_model_calls,
            "closure_calls": closure_calls,
            "synchronized_closure_calls": synchronized_closure_calls,
            "final_gradient_call": final_gradient_call,
            "counter_before": completed_updates - 1,
            "counter_after": completed_updates,
            "parameters_before": parameters_before,
            "parameters_after": _parameters(model),
            "optimizer_state_before": optimizer_before,
            "optimizer_state_after": _jsonable(optimizer.state_dict()),
            "sampler_state": _jsonable(sampler.state_dict()),
            "rng_state": _jsonable(capture_global_rng_state()),
        }
    )
    state_path = Path(args.state_path)
    if store is not None and args.checkpoint_generation is not None:
        state["status"] = "checkpoint_pending"
        state["checkpoint_status"] = "pending"
    state_path.write_text(json.dumps(state, sort_keys=True))

    if store is not None and args.checkpoint_generation is not None:
        phase_sequence.append(FaultPhase.BEFORE_STATE_WRITE.name)
        phase_sequence.append(FaultPhase.AFTER_STATE_WRITE.name)
        phase_sequence.append(FaultPhase.BEFORE_PUBLICATION.name)
        store.save(
            model,
            optimizer,
            generation=args.checkpoint_generation,
            sampler_state=sampler.state_dict(),
            rng_state=capture_global_rng_state(),
            completed_updates=completed_updates,
            failure_rank=args.checkpoint_failure_rank,
            delay_rank=args.checkpoint_delay_rank,
            delay_seconds=args.checkpoint_delay_seconds,
        )
        state["status"] = "success"
        state["checkpoint_status"] = "published"
        state_path.write_text(json.dumps(state, sort_keys=True))
        phase_sequence.append(FaultPhase.AFTER_PUBLICATION.name)

    runtime.barrier()
    if args.rank == 0:
        Path(args.complete_marker_path).write_text("COMPLETE")
    runtime.close()
    _write_receipt(args, phase_sequence, plan.kind.name if plan else FaultKind.NONE.name)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except BaseException as exc:
        _failure_state(args, "worker_failed", exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
