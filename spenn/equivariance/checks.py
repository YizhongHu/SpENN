"""Runtime equivariance checkers driven by the RuntimeEquivariance callback.

Checkers own *how* to check (which permutations, what to compare); the callback
owns *when* to run and where to log/persist. Both checkers here call the normal
model ``forward`` (never ``forward_impl``) and act on semantic, typed values via
their own ``permute`` contracts rather than generic tensor-tree inference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from spenn.data.equivariant_state import infer_particle_count
from spenn.data.permutation import (
    apply_particle_permutation,
    count_nonidentity_permutations,
    select_nonidentity_permutations,
)


@dataclass
class EquivarianceCheckResult:
    """Outcome of one runtime equivariance check.

    Parameters
    ----------
    passed : bool
        Whether the check passed.
    metrics : dict
        Scalar metrics for logging (JSON/CSV-safe).
    failures : list of str, optional
        Human-readable failure messages.
    artifact : Mapping or None, optional
        JSON-safe failure metadata for the callback to persist. ``None`` when
        there is nothing to write.
    """

    passed: bool
    metrics: dict[str, float | int | bool | str]
    failures: list[str] = field(default_factory=list)
    artifact: Mapping[str, Any] | None = None


@runtime_checkable
class RuntimeEquivarianceChecker(Protocol):
    """Protocol for runtime equivariance checkers."""

    def run(self, state: Any) -> EquivarianceCheckResult:
        """Inspect a `TrainerState` and return an `EquivarianceCheckResult`."""

        ...


class FullModelEquivarianceChecker:
    """Check final-output equivariance: ``F(sigma . x) ~= sigma . F(x)``.

    Calls the normal model ``forward`` and permutes the batch and output through
    their semantic ``permute`` contracts.

    Parameters
    ----------
    permutation_fraction : float, optional
        Fraction of non-identity permutations to target, in ``[0, 1]``.
    max_permutations : int, optional
        Hard cap on the number of permutations tested.
    seed : int or None, optional
        Seed controlling which permutations are selected (mixed with the step).
    atol, rtol : float, optional
        Comparison tolerances.
    """

    def __init__(
        self,
        *,
        permutation_fraction: float = 1.0,
        max_permutations: int = 8,
        seed: int | None = None,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> None:
        self.permutation_fraction = float(permutation_fraction)
        self.max_permutations = int(max_permutations)
        self.seed = seed
        self.atol = float(atol)
        self.rtol = float(rtol)

    def run(self, state: Any) -> EquivarianceCheckResult:
        """Run the full-model equivariance check against ``state``."""

        model = getattr(state, "model", None)
        batch = getattr(state, "batch", None)
        step = getattr(state, "step", None)
        n_particles = infer_particle_count(batch)
        metrics = self._base_metrics(n_particles)

        if model is None or batch is None or n_particles is None or n_particles < 2:
            metrics["n_permutations_tested"] = 0
            metrics["n_failed_permutations"] = 0
            metrics["max_abs_error"] = 0.0
            metrics["worst_permutation"] = ""
            metrics["passed"] = True
            return EquivarianceCheckResult(passed=True, metrics=metrics)

        permutations = select_nonidentity_permutations(
            n_particles=n_particles,
            fraction=self.permutation_fraction,
            max_count=self.max_permutations,
            seed=self.seed,
            step=step,
        )
        failed: list[list[int]] = []
        max_abs_error = 0.0
        worst: list[int] | None = None
        with torch.no_grad():
            output = model(batch)
            for permutation in permutations:
                permuted_batch = apply_particle_permutation(batch, permutation)
                lhs = model(permuted_batch)
                rhs = apply_particle_permutation(output, permutation)
                close, error = lhs.compare(rhs, atol=self.atol, rtol=self.rtol)
                if error > max_abs_error:
                    max_abs_error = error
                    worst = list(permutation.image)
                if not close:
                    failed.append(list(permutation.image))

        passed = not failed
        reported_error = max_abs_error if math.isfinite(max_abs_error) else str(max_abs_error)
        metrics["n_permutations_tested"] = len(permutations)
        metrics["n_failed_permutations"] = len(failed)
        metrics["max_abs_error"] = reported_error
        metrics["worst_permutation"] = str(worst) if worst is not None else ""
        metrics["passed"] = passed

        failures: list[str] = []
        artifact: dict[str, Any] | None = None
        if not passed:
            failures = [
                f"{len(failed)}/{len(permutations)} permutations failed; "
                f"worst {worst} with max_abs_error={max_abs_error}"
            ]
            artifact = {
                "checker_class": type(self).__name__,
                "step": step,
                "n_particles": int(n_particles),
                "permutations_tested": [list(p.image) for p in permutations],
                "failed_permutations": failed,
                "worst_permutation": worst,
                "max_abs_error": reported_error,
                "atol": self.atol,
                "rtol": self.rtol,
                "failures": failures,
            }
        return EquivarianceCheckResult(passed=passed, metrics=metrics, failures=failures, artifact=artifact)

    def _base_metrics(self, n_particles: int | None) -> dict[str, float | int | bool | str]:
        n_available = count_nonidentity_permutations(n_particles) if n_particles is not None else 0
        return {
            "n_particles": int(n_particles) if n_particles is not None else 0,
            "n_available_permutations": n_available,
            "permutation_fraction": self.permutation_fraction,
            "max_permutations": self.max_permutations,
            "atol": self.atol,
            "rtol": self.rtol,
        }


class TraceEquivarianceChecker:
    """Check semantic trace equivariance: ``trace_B[key] ~= sigma . trace_A[key]``.

    Captures two passive traces from the normal model ``forward`` (on ``x`` and
    on ``sigma . x``) and compares matching trace entries via each recorded
    value's own ``permute`` contract. No ``forward_impl``; no generic
    tensor-tree permutation.

    Parameters
    ----------
    permutation_fraction : float, optional
        Fraction of non-identity permutations to target.
    max_permutations : int, optional
        Hard cap on permutations tested.
    seed : int or None, optional
        Seed controlling permutation selection.
    comparison : {"stepwise", "full_trace", "both"}, optional
        ``stepwise`` compares entries key-by-key and reports the worst/failing
        key; ``full_trace`` adds whole-trace schema aggregation; ``both`` does
        both. All modes treat missing/extra keys as failures.
    compare_output : bool, optional
        Also compare the final model outputs like the full-model checker.
    dump_on_failure : bool, optional
        Whether to return a failure artifact.
    atol, rtol : float, optional
        Comparison tolerances.
    """

    def __init__(
        self,
        *,
        permutation_fraction: float = 1.0,
        max_permutations: int = 4,
        seed: int | None = None,
        comparison: str = "stepwise",
        compare_output: bool = False,
        dump_on_failure: bool = True,
        atol: float = 1.0e-6,
        rtol: float = 1.0e-6,
    ) -> None:
        if comparison not in ("stepwise", "full_trace", "both"):
            raise ValueError(f"comparison must be stepwise/full_trace/both, got {comparison!r}")
        self.permutation_fraction = float(permutation_fraction)
        self.max_permutations = int(max_permutations)
        self.seed = seed
        self.comparison = comparison
        self.compare_output = bool(compare_output)
        self.dump_on_failure = bool(dump_on_failure)
        self.atol = float(atol)
        self.rtol = float(rtol)

    def run(self, state: Any) -> EquivarianceCheckResult:
        """Run the trace-based equivariance check against ``state``."""

        from spenn.equivariance.trace import EquivarianceTrace

        model = getattr(state, "model", None)
        batch = getattr(state, "batch", None)
        step = getattr(state, "step", None)
        n_particles = infer_particle_count(batch)
        metrics = self._base_metrics(n_particles)

        if model is None or batch is None or n_particles is None or n_particles < 2:
            metrics.update(
                {
                    "n_permutations_tested": 0,
                    "n_trace_entries": 0,
                    "n_missing_keys": 0,
                    "n_extra_keys": 0,
                    "n_failed_entries": 0,
                    "worst_key": "",
                    "worst_permutation": "",
                    "max_abs_error": 0.0,
                    "passed": True,
                }
            )
            return EquivarianceCheckResult(passed=True, metrics=metrics)

        permutations = select_nonidentity_permutations(
            n_particles=n_particles,
            fraction=self.permutation_fraction,
            max_count=self.max_permutations,
            seed=self.seed,
            step=step,
        )

        missing_keys: set[str] = set()
        extra_keys: set[str] = set()
        failed_keys: set[str] = set()
        per_key_error: dict[str, float] = {}
        failed_permutations: list[list[int]] = []
        max_abs_error = 0.0
        worst_key: str | None = None
        worst_permutation: list[int] | None = None
        output_failed = False

        with torch.no_grad():
            with EquivarianceTrace.capture(model=model) as trace_a:
                output_a = model(batch)
            n_trace_entries = len(trace_a)
            keys_a = set(trace_a.keys())

            for permutation in permutations:
                permuted_batch = apply_particle_permutation(batch, permutation)
                with EquivarianceTrace.capture(model=model) as trace_b:
                    output_b = model(permuted_batch)
                keys_b = set(trace_b.keys())
                missing_keys |= keys_a - keys_b
                extra_keys |= keys_b - keys_a

                permutation_failed = bool((keys_a - keys_b) or (keys_b - keys_a))
                for key in keys_a & keys_b:
                    expected = apply_particle_permutation(trace_a[key].value, permutation)
                    actual = trace_b[key].value
                    close, error = actual.compare(expected, atol=self.atol, rtol=self.rtol)
                    per_key_error[key] = max(per_key_error.get(key, 0.0), error)
                    if error > max_abs_error:
                        max_abs_error = error
                        worst_key = key
                        worst_permutation = list(permutation.image)
                    if not close:
                        failed_keys.add(key)
                        permutation_failed = True

                if self.compare_output:
                    expected_output = apply_particle_permutation(output_a, permutation)
                    close, error = output_b.compare(expected_output, atol=self.atol, rtol=self.rtol)
                    if error > max_abs_error:
                        max_abs_error = error
                        worst_key = "output"
                        worst_permutation = list(permutation.image)
                    if not close:
                        output_failed = True
                        permutation_failed = True

                if permutation_failed:
                    failed_permutations.append(list(permutation.image))

        passed = not failed_keys and not missing_keys and not extra_keys and not output_failed
        reported_error = max_abs_error if math.isfinite(max_abs_error) else str(max_abs_error)
        metrics.update(
            {
                "n_permutations_tested": len(permutations),
                "n_trace_entries": n_trace_entries,
                "n_missing_keys": len(missing_keys),
                "n_extra_keys": len(extra_keys),
                "n_failed_entries": len(failed_keys),
                "worst_key": worst_key or "",
                "worst_permutation": str(worst_permutation) if worst_permutation is not None else "",
                "max_abs_error": reported_error,
                "passed": passed,
            }
        )

        failures: list[str] = []
        artifact: dict[str, Any] | None = None
        if not passed:
            if failed_keys:
                failures.append(f"failed trace keys: {sorted(failed_keys)}")
            if missing_keys:
                failures.append(f"missing trace keys: {sorted(missing_keys)}")
            if extra_keys:
                failures.append(f"extra trace keys: {sorted(extra_keys)}")
            if output_failed:
                failures.append("final output failed equivariance")
            if self.dump_on_failure:
                artifact = {
                    "checker_class": type(self).__name__,
                    "step": step,
                    "comparison": self.comparison,
                    "n_particles": int(n_particles),
                    "permutations_tested": [list(p.image) for p in permutations],
                    "failed_permutations": failed_permutations,
                    "missing_keys": sorted(missing_keys),
                    "extra_keys": sorted(extra_keys),
                    "failed_keys": sorted(failed_keys),
                    "worst_key": worst_key,
                    "worst_permutation": worst_permutation,
                    "max_abs_error": reported_error,
                    "atol": self.atol,
                    "rtol": self.rtol,
                    "failures": failures,
                    "trace_errors": [
                        {
                            "key": key,
                            "passed": key not in failed_keys,
                            "max_abs_error": (
                                per_key_error[key]
                                if math.isfinite(per_key_error[key])
                                else str(per_key_error[key])
                            ),
                        }
                        for key in sorted(per_key_error)
                    ],
                }
        return EquivarianceCheckResult(passed=passed, metrics=metrics, failures=failures, artifact=artifact)

    def _base_metrics(self, n_particles: int | None) -> dict[str, float | int | bool | str]:
        n_available = count_nonidentity_permutations(n_particles) if n_particles is not None else 0
        return {
            "n_particles": int(n_particles) if n_particles is not None else 0,
            "n_available_permutations": n_available,
            "permutation_fraction": self.permutation_fraction,
            "max_permutations": self.max_permutations,
            "comparison": self.comparison,
            "atol": self.atol,
            "rtol": self.rtol,
        }


__all__ = [
    "EquivarianceCheckResult",
    "FullModelEquivarianceChecker",
    "RuntimeEquivarianceChecker",
    "TraceEquivarianceChecker",
]
