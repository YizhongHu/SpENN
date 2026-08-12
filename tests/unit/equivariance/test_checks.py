"""Tests for FullModelEquivarianceChecker and TraceEquivarianceChecker.

Toy typed models live here (pytest-only), not in tpen.testing. The checkers
call the normal model ``forward`` and act on semantic typed values.
"""

from __future__ import annotations

import torch
from torch import nn

from tpen.data.real import Feature, zero_block
from tpen.equivariance import EquivariantMap
from tpen.equivariance.checks import FullModelEquivarianceChecker, TraceEquivarianceChecker


def _feature() -> Feature:
    # Last axis is the particle index (3 particles); channels = 2.
    return Feature(
        [
            zero_block(dtype=torch.float64),
            torch.arange(1 * 2 * 3, dtype=torch.float64).reshape(1, 2, 3),
            torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3),
        ]
    )


class _State:
    def __init__(self, model, batch, step: int = 1) -> None:
        self.model = model
        self.batch = batch
        self.step = step


# --- plain nn.Module toys: prove the checker uses normal forward (no forward_impl) ---


class IdentityModule(nn.Module):
    def forward(self, x: Feature) -> Feature:
        return x.clone()


class LabelBiasModule(nn.Module):
    def forward(self, x: Feature) -> Feature:
        n = x.blocks[1].shape[-1]
        bias = torch.arange(n, dtype=x.blocks[1].dtype).reshape(1, 1, n)
        return Feature([x.blocks[0].clone(), x.blocks[1] + bias, x.blocks[2].clone()])


# --- EquivariantMap toys whose forward records a trace ---


class TracedIdentity(EquivariantMap):
    def forward_impl(self, x: Feature) -> Feature:
        return x.clone()


class TracedLabelBias(EquivariantMap):
    def forward_impl(self, x: Feature) -> Feature:
        n = x.blocks[1].shape[-1]
        bias = torch.arange(n, dtype=x.blocks[1].dtype).reshape(1, 1, n)
        return Feature([x.blocks[0].clone(), x.blocks[1] + bias, x.blocks[2].clone()])


class KeyVaryingMap(EquivariantMap):
    def forward_impl(self, x: Feature) -> Feature:
        # Trace a semantic value under a key that depends on particle order, so
        # the recorded *key set* differs between x and sigma.x.
        peak = int(x.blocks[1][0, 0].argmax().item())
        self.trace(f"peak_{peak}", x.clone())
        return x.clone()


class TraceModel(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Feature) -> Feature:
        return self.layer(x)


# --- FullModelEquivarianceChecker ---


def test_full_model_passes_on_equivariant_module_via_normal_forward() -> None:
    # IdentityModule defines only forward (no forward_impl); passing proves the
    # checker uses the normal forward path.
    result = FullModelEquivarianceChecker(permutation_fraction=1.0, max_permutations=8, seed=0).run(
        _State(IdentityModule(), _feature())
    )

    assert result.passed is True
    assert result.metrics["n_permutations_tested"] == 5  # all of 3! - 1
    assert result.metrics["n_failed_permutations"] == 0
    assert result.n_comparisons == 5  # one comparison per permutation
    assert result.artifact is None


def test_full_model_fails_on_non_equivariant_module() -> None:
    result = FullModelEquivarianceChecker(permutation_fraction=1.0, max_permutations=8, seed=0).run(
        _State(LabelBiasModule(), _feature())
    )

    assert result.passed is False
    assert result.metrics["n_failed_permutations"] > 0
    assert result.metrics["worst_permutation"] != ""
    assert result.artifact is not None
    assert result.artifact["failed_permutations"]


def test_full_model_trivial_pass_without_particle_count() -> None:
    result = FullModelEquivarianceChecker().run(_State(IdentityModule(), batch=None))

    assert result.passed is True
    assert result.metrics["n_permutations_tested"] == 0
    # The pass is real -- there was nothing to permute -- and the count says so
    # rather than leaving `passed` to imply a measurement happened.
    assert result.n_comparisons == 0


def test_full_model_trivial_pass_for_zero_particles() -> None:
    batch = type("ZeroElectronBatch", (), {"n_electrons": 0})()

    result = FullModelEquivarianceChecker().run(_State(IdentityModule(), batch=batch))

    assert result.passed is True
    assert result.metrics["n_particles"] == 0
    assert result.metrics["n_available_permutations"] == 0
    assert result.metrics["n_permutations_tested"] == 0
    assert result.n_comparisons == 0


# --- TraceEquivarianceChecker ---


def test_trace_passes_when_traced_values_transform_correctly() -> None:
    model = TraceModel(TracedIdentity())
    result = TraceEquivarianceChecker(permutation_fraction=1.0, max_permutations=4, seed=0).run(
        _State(model, _feature())
    )

    assert result.passed is True
    assert result.metrics["n_trace_entries"] == 1  # layer/output
    assert result.metrics["n_failed_entries"] == 0
    # One comparison per shared trace key per permutation; compare_output is off.
    assert result.metrics["n_permutations_tested"] == 4
    assert result.n_comparisons == 4


def test_trace_fails_with_worst_key_on_violation() -> None:
    model = TraceModel(TracedLabelBias())
    result = TraceEquivarianceChecker(permutation_fraction=1.0, max_permutations=4, seed=0).run(
        _State(model, _feature())
    )

    assert result.passed is False
    assert result.metrics["worst_key"] == "layer/output"
    assert result.metrics["max_abs_error"] != 0.0
    assert result.artifact is not None


def test_trace_reports_missing_and_extra_keys() -> None:
    model = TraceModel(KeyVaryingMap())
    result = TraceEquivarianceChecker(permutation_fraction=1.0, max_permutations=4, seed=0).run(
        _State(model, _feature())
    )

    assert result.passed is False
    assert result.metrics["n_missing_keys"] > 0
    assert result.metrics["n_extra_keys"] > 0
    assert result.artifact["missing_keys"] or result.artifact["extra_keys"]


def test_trace_pass_over_an_empty_trace_reports_zero_comparisons() -> None:
    # The case `n_permutations_tested` cannot express. `IdentityModule` is a
    # plain nn.Module, so it records nothing; the key sets are both empty, so no
    # key is missing, no key is extra, no key fails, and the checker reports
    # `passed` having compared literally nothing across four permutations.
    # `n_comparisons` is the only published number that contradicts that.
    result = TraceEquivarianceChecker(permutation_fraction=1.0, max_permutations=4, seed=0).run(
        _State(TraceModel(IdentityModule()), _feature())
    )

    assert result.passed is True
    assert result.metrics["n_permutations_tested"] == 4
    assert result.metrics["n_trace_entries"] == 0
    assert result.n_comparisons == 0


def test_trace_counts_the_output_comparison_when_compare_output_is_on() -> None:
    # compare_output adds one comparison per permutation on top of the per-key
    # ones, so the count tracks comparisons performed rather than permutations.
    result = TraceEquivarianceChecker(
        permutation_fraction=1.0, max_permutations=4, seed=0, compare_output=True
    ).run(_State(TraceModel(TracedIdentity()), _feature()))

    assert result.passed is True
    assert result.metrics["n_trace_entries"] == 1
    assert result.n_comparisons == 8  # 4 permutations x (1 trace key + 1 output)


def test_trace_dump_on_failure_false_returns_no_artifact() -> None:
    model = TraceModel(TracedLabelBias())
    result = TraceEquivarianceChecker(
        permutation_fraction=1.0, max_permutations=4, seed=0, dump_on_failure=False
    ).run(_State(model, _feature()))

    assert result.passed is False
    assert result.artifact is None
