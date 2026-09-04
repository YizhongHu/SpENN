"""DS-N0: native c10d/DDP/DCP mechanics through the shared subprocess harness.

Coverage is intentionally CPU/Gloo-only and every test is capability-gated.
This module is an evidence instrument, not production authorization.  Its
launch scope is the file-store subprocess harness; launcher fidelity for the
real ``torchrun`` binary is deferred to production.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tests.helpers.ddp_capability import missing_capability_reason, probe_gloo_capability
from tests.helpers.ddp_fault_injection import FaultKind, FaultPhase, FaultPlan
from tests.helpers.ddp_subprocess_harness import HarnessBounds, RankReceipt, run_gloo_subprocess_group
from tests.helpers.vmc_scientific_oracle import (
    loss_tolerance_envelope,
    naive_per_shard_clip_then_sum,
    oracle_global_clip,
    oracle_vmc_objective,
)
from tests.spikes.native_ddp.statistics import local_centered_objective


_CAPABILITY = probe_gloo_capability()
_DEFAULT_BOUNDS = HarnessBounds(process_group_timeout=6.0, watchdog_timeout=20.0)
_FAULT_BOUNDS = HarnessBounds(process_group_timeout=2.0, watchdog_timeout=12.0)


@pytest.fixture(autouse=True)
def _require_gloo_subprocess_capability() -> None:
    """Make every test's capability decision explicit and attributable."""

    capability = _CAPABILITY
    if not capability.gloo_available:
        pytest.skip(missing_capability_reason(capability, "gloo_available"))
    if not capability.subprocess_spawn_available:
        pytest.skip(missing_capability_reason(capability, "subprocess_spawn_available"))


def _run_native(
    tmp_path: Path,
    *,
    world_size: int,
    extra_args: tuple[str, ...] = (),
    fault_plan: FaultPlan | None = None,
    bounds: HarnessBounds = _DEFAULT_BOUNDS,
):
    return run_gloo_subprocess_group(
        world_size,
        fault_plan,
        bounds,
        tmp_path,
        worker_module="tests.spikes.native_ddp.worker",
        worker_extra_args=extra_args,
    )


def _state(result, rank: int) -> dict:
    return json.loads((Path(result.invocation_dir) / f"state_{rank}.json").read_text())


def _states(result) -> list[dict]:
    return [_state(result, rank) for rank in range(len(result.receipts))]


def _encoded_tensor(value: dict) -> torch.Tensor:
    dtype = {
        "torch.float64": torch.float64,
        "torch.float32": torch.float32,
        "torch.uint8": torch.uint8,
        "torch.int64": torch.int64,
    }[value["dtype"]]
    return torch.tensor(value["__tensor__"], dtype=dtype)


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__tensor__" in value:
        return _encoded_tensor(value)
    if "__tuple__" in value:
        return tuple(_decode(item) for item in value["__tuple__"])
    return {
        int(key) if isinstance(key, str) and key.isdigit() else key: _decode(item)
        for key, item in value.items()
    }


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, terms: torch.Tensor) -> None:
    atol = loss_tolerance_envelope(terms)
    assert torch.allclose(actual, expected, atol=atol, rtol=0.0), (
        f"max abs diff {(actual - expected).abs().max().item()} exceeds atol={atol}"
    )


def _oracle_parameter_gradients(states: list[dict]) -> tuple[dict[str, torch.Tensor], object]:
    logabs = [_encoded_tensor(state["logabs"]) for state in states]
    energy = [_encoded_tensor(state["energy"]) for state in states]
    oracle = oracle_vmc_objective(logabs, energy)
    gradients = [item for item in oracle.per_sample_gradients]
    features = [_encoded_tensor(state["coordinates"]) for state in states]
    return (
        {
            "weight": sum(
                (sample_gradient * feature.sum(dim=1)).sum()
                for sample_gradient, feature in zip(gradients, features, strict=True)
            ).reshape(1),
            "bias": sum(sample_gradient.sum() for sample_gradient in gradients).reshape(1),
        },
        oracle,
    )


def _assert_global_metrics(state: dict, oracle, terms: torch.Tensor) -> None:
    for key in ("energy", "energy_variance", "energy_std", "energy_stderr"):
        assert state["global_metrics"][key] == pytest.approx(
            oracle.metrics[key], abs=loss_tolerance_envelope(terms)
        )
    for key in (
        "local_energy_n_finite",
        "local_energy_n_total",
        "local_energy_nonfinite_count",
    ):
        assert state["global_metrics"][key] == oracle.metrics[key]


def _assert_scientific_result(result, *, expected_raw_counts: tuple[int, ...] | None = None) -> None:
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert result.publication_observed is True
    assert result.exit_codes == (0,) * len(result.receipts)
    for receipt in result.receipts:
        assert isinstance(receipt, RankReceipt)
    states = _states(result)
    assert all(state["status"] == "success" for state in states)
    if expected_raw_counts is not None:
        assert tuple(state["global_statistics"]["total_count"] for state in states) == expected_raw_counts


def test_native_ddp_world_size_two_matches_concatenated_oracle_and_average(tmp_path: Path) -> None:
    result = _run_native(
        tmp_path,
        world_size=2,
        extra_args=("--experiment", "scientific", "--fixture", "regular", "--optimizer", "sgd"),
    )
    _assert_scientific_result(result)
    states = _states(result)
    expected_gradients, oracle = _oracle_parameter_gradients(states)
    finite_terms = torch.cat(
        [_encoded_tensor(state["energy"])[torch.isfinite(_encoded_tensor(state["energy"]))] for state in states]
    )
    for state in states:
        _assert_close(_encoded_tensor(state["gradients"]["weight"]), expected_gradients["weight"], finite_terms)
        _assert_close(_encoded_tensor(state["gradients"]["bias"]), expected_gradients["bias"], finite_terms)
        assert state["ddp_backward_scale"] == pytest.approx(4.0 / oracle.metrics["local_energy_n_finite"])
        assert state["global_loss"] == pytest.approx(oracle.loss.item(), abs=loss_tolerance_envelope(finite_terms))
        _assert_global_metrics(state, oracle, finite_terms)
        _assert_close(
            torch.tensor(state["parameters_after"]["weight"], dtype=torch.float64),
            torch.tensor([0.25], dtype=torch.float64) - 0.05 * expected_gradients["weight"],
            finite_terms,
        )
        _assert_close(
            torch.tensor(state["parameters_after"]["bias"], dtype=torch.float64),
            torch.tensor([-0.10], dtype=torch.float64) - 0.05 * expected_gradients["bias"],
            finite_terms,
        )


def test_native_ddp_world_size_three_matches_m2_uneven_oracle_and_empty_shard(tmp_path: Path) -> None:
    result = _run_native(
        tmp_path,
        world_size=3,
        extra_args=("--experiment", "scientific", "--fixture", "m2", "--optimizer", "sgd"),
    )
    _assert_scientific_result(result, expected_raw_counts=(15, 15, 15))
    states = _states(result)
    assert [state["global_statistics"]["total_count"] for state in states] == [15, 15, 15]
    assert [state["global_statistics"]["finite_count"] for state in states] == [8, 8, 8]
    expected_gradients, oracle = _oracle_parameter_gradients(states)
    finite_terms = torch.cat(
        [_encoded_tensor(state["energy"])[torch.isfinite(_encoded_tensor(state["energy"]))] for state in states]
    )
    for state in states:
        _assert_close(_encoded_tensor(state["gradients"]["weight"]), expected_gradients["weight"], finite_terms)
        _assert_close(_encoded_tensor(state["gradients"]["bias"]), expected_gradients["bias"], finite_terms)
        assert state["global_loss"] == pytest.approx(oracle.loss.item(), abs=loss_tolerance_envelope(finite_terms))
        _assert_global_metrics(state, oracle, finite_terms)
        assert state["ddp_backward_scale"] == pytest.approx(6.0 / oracle.metrics["local_energy_n_finite"])
        _assert_close(
            torch.tensor(state["parameters_after"]["weight"], dtype=torch.float64),
            torch.tensor([0.25], dtype=torch.float64) - 0.05 * expected_gradients["weight"],
            finite_terms,
        )
        _assert_close(
            torch.tensor(state["parameters_after"]["bias"], dtype=torch.float64),
            torch.tensor([-0.10], dtype=torch.float64) - 0.05 * expected_gradients["bias"],
            finite_terms,
        )
    assert _encoded_tensor(states[1]["local_gradients"]["weight"]).item() == 0.0
    assert _encoded_tensor(states[1]["local_gradients"]["bias"]).item() == 0.0
    assert states[1]["loss_requires_grad"] is True
    assert all(state["ddp_gradient_reductions_per_update"] == 1 for state in states)


def test_native_negative_controls_fail_local_centering_and_per_shard_clipping(tmp_path: Path) -> None:
    del tmp_path
    energies = [
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([5.0], dtype=torch.float64),
    ]
    logabs = [
        torch.tensor([0.3], dtype=torch.float64, requires_grad=True),
        torch.tensor([0.7], dtype=torch.float64, requires_grad=True),
    ]
    oracle = oracle_vmc_objective(logabs, energies)
    local = local_centered_objective(logabs, energies)
    assert local.item() == 0.0
    assert oracle.loss.item() != local.item()

    shard_a = torch.tensor([10.0, 10.0], dtype=torch.float64)
    shard_b = torch.tensor([-10.05, -10.05], dtype=torch.float64)
    global_clipped = oracle_global_clip([shard_a, shard_b], 1.0)
    per_shard = naive_per_shard_clip_then_sum([shard_a, shard_b], 1.0)
    assert not torch.allclose(global_clipped, per_shard)


def test_native_global_zero_valid_energy_refuses_before_backward_and_optimizer_mutation(
    tmp_path: Path,
) -> None:
    result = _run_native(
        tmp_path,
        world_size=2,
        extra_args=("--experiment", "scientific", "--fixture", "all_invalid"),
    )
    assert result.all_reaped is True
    assert result.publication_observed is False
    assert result.exit_codes == (0, 0)
    for state in _states(result):
        assert state["status"] == "refused"
        assert state["backward_called"] is False
        assert state["optimizer_step"] is False
        assert state["counter_before"] == state["counter_after"] == 0
        _assert_nested_equal(
            _decode(state["optimizer_state_before"]), _decode(state["optimizer_state_after"])
        )


def test_native_same_topology_dcp_resume_matches_continuous_model_optimizer_sampler_and_rng(
    tmp_path: Path,
) -> None:
    continuous_root = tmp_path / "continuous"
    split_root = tmp_path / "split"
    continuous = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "resume", "--iterations", "4", "--mcmc-steps", "2",
            "--checkpoint-root", str(continuous_root), "--checkpoint-generation", "4",
        ),
    )
    first = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "resume", "--iterations", "2", "--mcmc-steps", "2",
            "--checkpoint-root", str(split_root), "--checkpoint-generation", "2",
        ),
    )
    resumed = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "resume", "--iterations", "4", "--mcmc-steps", "2",
            "--checkpoint-root", str(split_root), "--resume-generation", "2",
        ),
    )
    for run in (continuous, first, resumed):
        _assert_scientific_result(run)
    first_states = _states(first)
    for continuous_state, resumed_state in zip(_states(continuous), _states(resumed), strict=True):
        first_state = first_states[resumed_state["rank"]]
        assert first_state["parameters_after"] != {"weight": [0.25], "bias": [-0.1]}
        assert first_state["optimizer_state_after"]["state"]
        assert first_state["parameters_after"] == resumed_state["parameters_before"]
        assert first_state["optimizer_state_after"] == resumed_state["optimizer_state_before"]
        for key in (
            "parameters_after", "optimizer_state_after", "counter_after", "sampler_state", "rng_state",
        ):
            assert continuous_state[key] == resumed_state[key]
        assert all(not key.startswith("module.") for key in continuous_state["canonical_model_keys"])
    latest = json.loads((split_root / "latest.json").read_text())
    manifest = json.loads((split_root / latest["path"] / "manifest.json").read_text())
    assert manifest["publisher_rank"] == 0
    assert any(item["relative_path"].startswith("dcp/") for item in manifest["files"])
    assert all(item["relative_path"].startswith(("dcp/", "sidecars/")) for item in manifest["files"])


def test_native_topology_change_is_refused_before_any_resume_mutation(tmp_path: Path) -> None:
    root = tmp_path / "topology"
    saved = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "resume", "--iterations", "1", "--checkpoint-root", str(root),
            "--checkpoint-generation", "1",
        ),
    )
    _assert_scientific_result(saved)
    changed = _run_native(
        tmp_path,
        world_size=3,
        extra_args=(
            "--experiment", "resume", "--iterations", "1", "--checkpoint-root", str(root),
            "--resume-generation", "1",
        ),
    )
    assert changed.all_reaped is True
    assert changed.publication_observed is False
    assert changed.exit_codes == (0, 0, 0)
    for state in _states(changed):
        assert state["status"] == "topology_refused"
        assert state["mutation_started"] is False
        assert state["counter_before"] == state["counter_after"] == 0
        _assert_nested_equal(
            _decode(state["optimizer_state_before"]), _decode(state["optimizer_state_after"])
        )


def test_native_perturbed_rank_sampler_sidecar_fails_resume(tmp_path: Path) -> None:
    root = tmp_path / "perturbed"
    saved = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "resume", "--iterations", "1", "--checkpoint-root", str(root),
            "--checkpoint-generation", "1",
        ),
    )
    _assert_scientific_result(saved)
    latest = json.loads((root / "latest.json").read_text())
    sidecar = root / latest["path"] / "sidecars" / "rank-00001.json"
    payload = json.loads(sidecar.read_text())
    payload["sampler_state"]["walkers"]["__tensor__"][0][0] += 1.0
    sidecar.write_text(json.dumps(payload, sort_keys=True))

    failed = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "resume", "--iterations", "2", "--checkpoint-root", str(root),
            "--resume-generation", "1",
        ),
    )
    assert failed.all_reaped is True
    assert failed.publication_observed is False
    assert all(code is not None and code != 0 for code in failed.exit_codes)
    assert all(state["status"] == "resume_failed" for state in _states(failed))


def test_native_raise_before_backward_preserves_culprit_and_nonpublication(tmp_path: Path) -> None:
    plan = FaultPlan(target_rank=1, kind=FaultKind.RAISE_BEFORE_BACKWARD, phase=FaultPhase.BEFORE_OPTIMIZER_STEP)
    result = _run_native(tmp_path, world_size=2, fault_plan=plan, bounds=_FAULT_BOUNDS)
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert result.culprit_rank == 1
    assert result.publication_observed is False
    assert Path(result.invocation_dir, "state_1.json").exists()
    assert Path(result.invocation_dir, "state_0.json").exists()


def test_native_skip_collective_preserves_culprit_and_nonpublication(tmp_path: Path) -> None:
    plan = FaultPlan(target_rank=1, kind=FaultKind.SKIP_COLLECTIVE, phase=FaultPhase.BEFORE_COLLECTIVE)
    result = _run_native(tmp_path, world_size=2, fault_plan=plan, bounds=_FAULT_BOUNDS)
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert result.culprit_rank == 1
    assert result.publication_observed is False
    assert Path(result.invocation_dir, "state_1.json").exists()


def test_native_stall_before_collective_is_bounded_and_nonpublishing(tmp_path: Path) -> None:
    plan = FaultPlan(
        target_rank=1,
        kind=FaultKind.STALL_BEFORE_COLLECTIVE,
        phase=FaultPhase.BEFORE_COLLECTIVE,
        delay_seconds=6.0,
    )
    result = _run_native(tmp_path, world_size=2, fault_plan=plan, bounds=_FAULT_BOUNDS)
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert result.culprit_rank == 1
    assert result.publication_observed is False
    assert Path(result.invocation_dir, "state_1.json").exists()


def test_native_failed_shard_writer_keeps_previous_generation_selectable(tmp_path: Path) -> None:
    root = tmp_path / "failed-writer"
    first = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "resume", "--iterations", "1", "--checkpoint-root", str(root),
            "--checkpoint-generation", "1",
        ),
    )
    _assert_scientific_result(first)
    previous_latest = json.loads((root / "latest.json").read_text())
    second = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "resume", "--iterations", "2", "--checkpoint-root", str(root),
            "--checkpoint-generation", "2", "--checkpoint-failure-rank", "1",
        ),
        bounds=_FAULT_BOUNDS,
    )
    assert second.watchdog_fired is False
    assert second.all_reaped is True
    assert second.publication_observed is False
    assert json.loads((root / "latest.json").read_text()) == previous_latest
    assert not (root / "generations" / "gen-000002" / "COMPLETE").exists()
    assert (root / "generations" / "gen-000001" / "COMPLETE").exists()
    assert (root / "generations" / "gen-000001" / "sidecars" / "rank-00000.json").exists()
    assert (root / "generations" / "gen-000001" / "sidecars" / "rank-00001.json").exists()


def test_native_delayed_shard_writer_keeps_previous_generation_selectable(tmp_path: Path) -> None:
    root = tmp_path / "delayed-writer"
    first = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "resume", "--iterations", "1", "--checkpoint-root", str(root),
            "--checkpoint-generation", "1",
        ),
    )
    _assert_scientific_result(first)
    previous_latest = json.loads((root / "latest.json").read_text())
    second = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "resume", "--iterations", "2", "--checkpoint-root", str(root),
            "--checkpoint-generation", "2", "--checkpoint-delay-rank", "1", "--checkpoint-delay-seconds", "6",
        ),
        bounds=_FAULT_BOUNDS,
    )
    assert second.watchdog_fired is False
    assert second.all_reaped is True
    assert second.publication_observed is False
    assert json.loads((root / "latest.json").read_text()) == previous_latest
    assert not (root / "generations" / "gen-000002" / "COMPLETE").exists()


def test_native_ddp_gradient_reduction_count_is_constant_across_mcmc_and_kinetic_work(tmp_path: Path) -> None:
    short = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "scientific", "--fixture", "regular", "--mcmc-steps", "1",
            "--kinetic-forwards", "1",
        ),
    )
    long = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "scientific", "--fixture", "regular", "--mcmc-steps", "5",
            "--kinetic-forwards", "5",
        ),
    )
    _assert_scientific_result(short)
    _assert_scientific_result(long)
    for short_state, long_state in zip(_states(short), _states(long), strict=True):
        assert short_state["ddp_gradient_reductions_per_update"] == 1
        assert long_state["ddp_gradient_reductions_per_update"] == 1
        assert short_state["sampling_gradient_reductions"] == long_state["sampling_gradient_reductions"] == 0
        assert short_state["sampling_collectives"] == long_state["sampling_collectives"] == 0
        assert long_state["coordinate_forward_count"] > short_state["coordinate_forward_count"]


def test_native_raw_model_boundary_matches_coordinate_gradient_without_ddp_wrapper(tmp_path: Path) -> None:
    result = _run_native(
        tmp_path,
        world_size=2,
        extra_args=("--experiment", "scientific", "--fixture", "regular", "--kinetic-forwards", "3"),
    )
    _assert_scientific_result(result)
    for state in _states(result):
        coordinates = _encoded_tensor(state["coordinates"])
        expected_coordinate_gradient = torch.full_like(coordinates, 0.25)
        _assert_close(_encoded_tensor(state["coordinate_gradient"]), expected_coordinate_gradient, coordinates.abs())
        assert state["access"] == {
            "coordinate_forward_owner": "raw_model",
            "score_forward_owner": "ddp_model",
            "used_module_attribute": False,
        }
        assert state["ddp_forward_calls"] == 1
        assert all(not key.startswith("module.") for key in state["canonical_model_keys"])


def test_native_sgd_momentum_and_adam_optimizer_states_match_independent_gradients(tmp_path: Path) -> None:
    for optimizer_name in ("sgd", "adam"):
        result = _run_native(
            tmp_path,
            world_size=2,
            extra_args=("--experiment", "scientific", "--fixture", "regular", "--optimizer", optimizer_name),
        )
        _assert_scientific_result(result)
        states = _states(result)
        expected_gradients, _ = _oracle_parameter_gradients(states)
        expected_weight = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float64))
        expected_bias = torch.nn.Parameter(torch.tensor([-0.10], dtype=torch.float64))
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD((expected_weight, expected_bias), lr=0.05, momentum=0.9)
        else:
            optimizer = torch.optim.Adam((expected_weight, expected_bias), lr=0.01)
        expected_weight.grad = expected_gradients["weight"].clone()
        expected_bias.grad = expected_gradients["bias"].clone()
        optimizer.step()
        for state in states:
            _assert_nested_equal(_decode(state["optimizer_state_after"]), optimizer.state_dict())
            _assert_close(
                torch.tensor(state["parameters_after"]["weight"], dtype=torch.float64),
                expected_weight.detach(),
                torch.tensor([1.0], dtype=torch.float64),
            )
            _assert_close(
                torch.tensor(state["parameters_after"]["bias"], dtype=torch.float64),
                expected_bias.detach(),
                torch.tensor([1.0], dtype=torch.float64),
            )


def test_native_closure_optimizer_suppresses_inner_ddp_reductions_and_counts_final_gradient(tmp_path: Path) -> None:
    result = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "scientific", "--fixture", "regular", "--optimizer", "closure",
            "--closure-inner-iterates", "3",
        ),
    )
    _assert_scientific_result(result)
    for state in _states(result):
        assert state["closure_calls"] == 3
        assert state["synchronized_closure_calls"] == 1
        assert state["final_gradient_call"] == state["closure_calls"]
        assert state["ddp_gradient_reductions_per_update"] == 1


def test_native_api_inventory_records_torch_version_and_inspection_only_accelerator_path(tmp_path: Path) -> None:
    result = _run_native(tmp_path, world_size=2)
    _assert_scientific_result(result)
    state = _state(result, 0)
    assert isinstance(state["torch_version"], str)
    assert state["torch_version"]
    assert "torch.distributed.checkpoint.state_dict.get_state_dict" in state["api_inventory"]["stable"]
    assert "tests.spikes.native_ddp.CheckpointPayloadStore" in state["api_inventory"]["spike_prototype"]
    assert state["accelerator_execution"] is False
    assert state["accelerator_path_inspected"] == "CUDA -> NCCL; ROCm -> RCCL; XPU -> XCCL"


def test_native_rank_state_artifact_complements_shared_receipt_observability(tmp_path: Path) -> None:
    result = _run_native(tmp_path, world_size=2)
    _assert_scientific_result(result)
    required_state_keys = {
        "rank", "world_size", "hostname", "pid", "phase_sequence", "status", "global_statistics",
        "global_loss", "parameters_after", "optimizer_state_after", "sampler_state", "rng_state",
    }
    for receipt, state in zip(result.receipts, _states(result), strict=True):
        assert isinstance(receipt, RankReceipt)
        assert required_state_keys <= state.keys()
        assert state["rank"] == receipt.rank
        assert state["world_size"] == receipt.world_size
        assert state["pid"] == receipt.pid
        assert state["hostname"] == receipt.hostname
        assert state["phase_sequence"] == receipt.phase_sequence
        assert (Path(result.invocation_dir) / f"state_{state['rank']}.json").name == f"state_{receipt.rank}.json"
