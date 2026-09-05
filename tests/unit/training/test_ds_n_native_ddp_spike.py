"""DS-N0: native c10d/DDP/DCP mechanics through the shared subprocess harness.

Coverage is intentionally CPU/Gloo-only and every test is capability-gated.
This module is an evidence instrument, not production authorization.  Its
launch scope is the file-store subprocess harness; launcher fidelity for the
real ``torchrun`` binary is deferred to production.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from unittest.mock import Mock

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
from tests.spikes.native_ddp import checkpoint as checkpoint_module
from tests.spikes.native_ddp.checkpoint import CheckpointCorrupt, CheckpointPayloadStore, CheckpointTopologyMismatch
from tests.spikes.native_ddp.model_access import SemanticWavefunction
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


def _assert_nested_close(left, right, *, atol: float) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        if left.dtype.is_floating_point:
            assert torch.allclose(left, right, atol=atol, rtol=0.0)
        else:
            assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_close(left[key], right[key], atol=atol)
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_close(left_value, right_value, atol=atol)
    elif isinstance(left, float):
        assert left == pytest.approx(right, abs=atol)
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


def _assert_no_published_generation(root: Path, generation: int) -> None:
    final = root / "generations" / f"gen-{generation:06d}"
    assert not (final / "COMPLETE").exists()
    assert not final.exists()
    assert not (root / "latest.json").exists()


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
    expected_finite_masks = [
        [True, False, True, True, False],
        [False, False, False],
        [True, True, False, True, True, True, False],
    ]
    expected_finite_values = [
        [1.0, 2.0, -1.0],
        [],
        [3.0, -2.0, 1.0, 0.5, 2.5],
    ]
    for state, expected_mask, expected_values in zip(
        states, expected_finite_masks, expected_finite_values, strict=True
    ):
        encoded_energy = state["energy"]["__tensor__"]
        assert len(encoded_energy) == len(expected_mask)
        observed_mask = [
            math.isfinite(float(value)) for value in encoded_energy
        ]
        assert observed_mask == expected_mask
        assert [
            float(value) for value, is_finite in zip(encoded_energy, observed_mask, strict=True)
            if is_finite
        ] == expected_values
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
        assert state["ddp_forward_calls"] == 0
        assert state["ddp_gradient_reductions"] == 0
        assert state["parameter_gradient_events"] == 0
        assert state["counter_before"] == state["counter_after"] == 0
        assert state["parameters_before"] == state["parameters_after"]
        _assert_nested_equal(
            _decode(state["optimizer_state_before"]), _decode(state["optimizer_state_after"])
        )
        assert "gradients" not in state


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
        assert state["counter_before"] == state["counter_after"] == 0
        assert state["ddp_forward_calls"] == 0
        assert state["parameters_before"] == state["parameters_after"]
        _assert_nested_equal(
            _decode(state["optimizer_state_before"]), _decode(state["optimizer_state_after"])
        )


def test_native_topology_load_boundary_never_enters_dcp_or_mutates_state(
    tmp_path: Path, monkeypatch
) -> None:
    model = SemanticWavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    runtime = Mock(rank=1, world_size=3)
    runtime.broadcast_object.return_value = {
        "world_size": 2,
        "path": "generations/gen-000001",
        "files": [],
    }
    store = CheckpointPayloadStore(root=tmp_path, runtime=runtime)
    dcp_load = Mock(side_effect=AssertionError("dcp.load must not be entered"))
    state_apply = Mock(side_effect=AssertionError("set_state_dict must not be entered"))
    monkeypatch.setattr(checkpoint_module.dcp, "load", dcp_load)
    monkeypatch.setattr(checkpoint_module, "set_state_dict", state_apply)

    with pytest.raises(CheckpointTopologyMismatch):
        store.load(model, optimizer, generation=1)

    assert dcp_load.call_count == 0
    assert state_apply.call_count == 0
    for name, value in model.state_dict().items():
        assert torch.equal(value, model_before[name])
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)


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
    root = tmp_path / "raise-checkpoint"
    result = _run_native(
        tmp_path,
        world_size=2,
        fault_plan=plan,
        bounds=_FAULT_BOUNDS,
        extra_args=("--checkpoint-root", str(root), "--checkpoint-generation", "1"),
    )
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert all(code is not None for code in result.exit_codes)
    assert any(code != 0 for code in result.exit_codes)
    assert result.culprit_rank == 1
    assert result.publication_observed is False
    assert Path(result.invocation_dir, "state_1.json").exists()
    assert Path(result.invocation_dir, "state_0.json").exists()
    _assert_no_published_generation(root, 1)


def test_native_skip_collective_preserves_culprit_and_nonpublication(tmp_path: Path) -> None:
    plan = FaultPlan(target_rank=1, kind=FaultKind.SKIP_COLLECTIVE, phase=FaultPhase.BEFORE_COLLECTIVE)
    root = tmp_path / "skip-checkpoint"
    result = _run_native(
        tmp_path,
        world_size=2,
        fault_plan=plan,
        bounds=_FAULT_BOUNDS,
        extra_args=("--checkpoint-root", str(root), "--checkpoint-generation", "1"),
    )
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert all(code is not None for code in result.exit_codes)
    assert any(code != 0 for code in result.exit_codes)
    assert result.culprit_rank == 1
    assert result.publication_observed is False
    assert Path(result.invocation_dir, "state_1.json").exists()
    _assert_no_published_generation(root, 1)


def test_native_stall_before_collective_is_bounded_and_nonpublishing(tmp_path: Path) -> None:
    plan = FaultPlan(
        target_rank=1,
        kind=FaultKind.STALL_BEFORE_COLLECTIVE,
        phase=FaultPhase.BEFORE_COLLECTIVE,
        delay_seconds=6.0,
    )
    root = tmp_path / "stall-checkpoint"
    result = _run_native(
        tmp_path,
        world_size=2,
        fault_plan=plan,
        bounds=_FAULT_BOUNDS,
        extra_args=("--checkpoint-root", str(root), "--checkpoint-generation", "1"),
    )
    assert result.watchdog_fired is False
    assert result.all_reaped is True
    assert all(code is not None for code in result.exit_codes)
    assert any(code != 0 for code in result.exit_codes)
    assert result.culprit_rank == 1
    assert result.publication_observed is False
    assert Path(result.invocation_dir, "state_1.json").exists()
    _assert_no_published_generation(root, 1)


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
    assert all(state["status"] != "success" for state in _states(second))
    assert second.publication_observed is False
    assert json.loads((root / "latest.json").read_text()) == previous_latest
    assert not (root / "generations" / "gen-000002" / "COMPLETE").exists()
    assert (root / "generations" / "gen-000001" / "COMPLETE").exists()
    assert (root / "generations" / "gen-000001" / "sidecars" / "rank-00000.json").exists()
    assert (root / "generations" / "gen-000001" / "sidecars" / "rank-00001.json").exists()


def test_native_save_digest_change_blocks_coordinator_publication(tmp_path: Path, monkeypatch) -> None:
    runtime = Mock(rank=0, world_size=2)
    runtime.barrier.return_value = None
    model = SemanticWavefunction()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def gather_after_local_digest(local_digest):
        stage = tmp_path / "staging" / "gen-000001"
        rank_one = stage / "sidecars" / "rank-00001.json"
        rank_one.parent.mkdir(parents=True, exist_ok=True)
        rank_one.write_text(json.dumps({"rank": 1, "closed": True}, sort_keys=True))
        expected_rank_one = checkpoint_module._digest(rank_one, root=stage).as_dict()
        rank_one.write_text(json.dumps({"rank": 1, "closed": False}, sort_keys=True))
        return [local_digest, expected_rank_one]

    runtime.all_gather_objects.side_effect = gather_after_local_digest
    monkeypatch.setattr(checkpoint_module, "get_state_dict", lambda *_args: ({}, {}))
    monkeypatch.setattr(checkpoint_module.dcp, "save", lambda *_args, **_kwargs: None)

    with pytest.raises(CheckpointCorrupt, match="digest changed"):
        CheckpointPayloadStore(root=tmp_path, runtime=runtime).save(
            model,
            optimizer,
            generation=1,
            sampler_state={},
            rng_state={},
            completed_updates=1,
        )

    _assert_no_published_generation(tmp_path, 1)


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
        assert short_state["sampling_gradient_reductions"] == 0
        assert long_state["sampling_gradient_reductions"] == 0
        assert short_state["sampling_raw_model_calls"] == 2
        assert long_state["sampling_raw_model_calls"] == 6
        assert short_state["kinetic_raw_model_calls"] == 1
        assert long_state["kinetic_raw_model_calls"] == 5
        assert short_state["coordinate_forward_count"] == 6
        assert long_state["coordinate_forward_count"] == 14


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


def test_native_closure_optimizer_matches_global_reference_across_ranks(tmp_path: Path) -> None:
    result = _run_native(
        tmp_path,
        world_size=2,
        extra_args=(
            "--experiment", "scientific", "--fixture", "regular", "--optimizer", "closure",
            "--closure-inner-iterates", "3",
        ),
    )
    _assert_scientific_result(result)
    states = _states(result)
    assert states[0]["parameters_after"] == states[1]["parameters_after"]
    _assert_nested_equal(
        _decode(states[0]["optimizer_state_after"]),
        _decode(states[1]["optimizer_state_after"]),
    )
    for state in states:
        assert state["synchronized_closure_calls"] == state["closure_calls"]
        assert state["final_gradient_call"] == state["closure_calls"]
        assert state["ddp_gradient_reductions_per_update"] == state["closure_calls"]

    features = torch.tensor(
        [
            [0.2, 0.1], [-0.5, 0.3], [0.7, -0.2], [0.4, 0.9],
            [-0.3, 0.8], [0.4, -0.1], [1.1, 0.2],
        ],
        dtype=torch.float64,
    )
    energy = torch.tensor(
        [1.0, 2.0, 0.5, float("nan"), 3.0, -1.0, 2.0],
        dtype=torch.float64,
    )
    reference_model = SemanticWavefunction()
    reference_optimizer = torch.optim.LBFGS(
        reference_model.parameters(),
        lr=0.25,
        max_iter=3,
        history_size=5,
        tolerance_grad=0.0,
        tolerance_change=0.0,
    )

    def reference_closure() -> torch.Tensor:
        reference_optimizer.zero_grad(set_to_none=True)
        logabs = reference_model(features)
        objective = oracle_vmc_objective([logabs], [energy])
        objective.loss.backward()
        return objective.loss

    reference_optimizer.step(reference_closure)
    optimizer_atol = 10.0 * loss_tolerance_envelope(energy[torch.isfinite(energy)].abs())
    for state in states:
        assert state["parameters_after"]["weight"] == pytest.approx(
            reference_model.weight.detach().tolist(), abs=optimizer_atol
        )
        assert state["parameters_after"]["bias"] == pytest.approx(
            reference_model.bias.detach().tolist(), abs=optimizer_atol
        )
        _assert_nested_close(
            _decode(state["optimizer_state_after"]),
            reference_optimizer.state_dict(),
            atol=optimizer_atol,
        )


def test_native_api_inventory_separates_experimental_dcp_helpers(tmp_path: Path) -> None:
    """Record consumed API buckets and version-bound reach references."""

    result = _run_native(tmp_path, world_size=2)
    _assert_scientific_result(result)
    state = _state(result, 0)
    assert isinstance(state["torch_version"], str)
    assert state["torch_version"]
    inventory = state["api_inventory"]
    stable = set(inventory["stable"])
    experimental = set(state["api_inventory"]["experimental"])
    prototype = set(state["api_inventory"]["spike_prototype"])
    assert stable & experimental == set()
    assert stable & prototype == set()
    assert experimental & prototype == set()
    consumed_stable = {
        "torch.nn.parallel.DistributedDataParallel",
        "torch.nn.parallel.DistributedDataParallel.register_comm_hook",
        "torch.distributed.init_process_group",
        "torch.distributed.barrier",
        "torch.distributed.all_gather_object",
        "torch.distributed.broadcast_object_list",
        "torch.distributed.all_reduce",
        "torch.distributed.ReduceOp.SUM",
        "torch.distributed.is_initialized",
        "torch.distributed.destroy_process_group",
        "torch.distributed.checkpoint.save",
        "torch.distributed.checkpoint.load",
        "torch.distributed.checkpoint.FileSystemReader",
        "torch.distributed.checkpoint.FileSystemWriter",
    }
    for api in consumed_stable:
        buckets = [name for name, entries in inventory.items() if api in entries]
        assert buckets == ["stable"], f"{api} must be in exactly one stable bucket; observed={buckets}"
    assert "torch.distributed.checkpoint.state_dict.get_state_dict" in experimental
    assert "torch.distributed.checkpoint.state_dict.set_state_dict" in experimental
    assert "tests.spikes.native_ddp.CheckpointPayloadStore" in prototype
    assert state["accelerator_execution"] is False
    assert state["api_references"] == {
        "dcp": "https://docs.pytorch.org/docs/2.12/distributed.checkpoint.html",
        "c10d_and_ddp": "https://docs.pytorch.org/docs/2.12/distributed.html",
        "backend_selection": "https://docs.pytorch.org/docs/2.12/distributed.html#which-backend-to-use",
        "rocm_backends": "https://docs.pytorch.org/docs/2.12/notes/hip.html#torch-distributed-backends",
    }
    assert state["accelerator_inspection"] == {
        "cuda": {
            "backend": "nccl",
            "communication_library": "NCCL",
            "build_scope": "CUDA build only; not executed",
            "reference": "https://docs.pytorch.org/docs/2.12/distributed.html#which-backend-to-use",
        },
        "rocm": {
            "backend": "nccl",
            "communication_library": "RCCL",
            "build_scope": "ROCm build only; not executed",
            "reference": "https://docs.pytorch.org/docs/2.12/notes/hip.html#torch-distributed-backends",
        },
        "xpu": {
            "backend": "xccl",
            "communication_library": "XCCL",
            "build_scope": "XPU build only; not executed",
            "reference": "https://docs.pytorch.org/docs/2.12/distributed.html#which-backend-to-use",
        },
    }


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
