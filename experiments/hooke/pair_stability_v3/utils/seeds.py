"""Named seed override helpers for staged studies."""

from __future__ import annotations

from typing import Any

DEFAULT_SEED_OVERRIDES = {
    "scan_train": {
        "run_parameters.seed": "scan_seed",
        "runtime.seed": "scan_seed",
        "model_initialization.seed": "scan_seed",
        "sampler.seed": "scan_seed",
    },
    "validation": {
        "run_parameters.seed": "scan_seed",
        "runtime.seed": "scan_seed",
        "model_initialization.seed": "scan_seed",
        "evaluation.seed": "scan_seed",
    },
    "final_train": {
        "run_parameters.seed": "final_train_model_seed",
        "runtime.seed": "final_train_model_seed",
        "model_initialization.seed": "final_train_model_seed",
        "sampler.seed": "final_train_sampler_seed",
    },
    "final_eval": {
        "run_parameters.seed": "final_eval_seed",
        "runtime.seed": "final_eval_seed",
        "model_initialization.seed": "final_eval_seed",
        "evaluation.seed": "final_eval_seed",
    },
}
DEFAULT_FINAL_SEED_SEQUENCES = {
    "final_train_sampler_seed": {"start": 101, "step": 1},
    "final_train_model_seed": {"start": 1001, "step": 1},
    "final_eval_seed": {"start": 10001, "step": 1},
}


def seed_override_policy(configured: Any | None = None) -> dict[str, dict[str, str]]:
    """Return normalized stage -> override path -> named seed mapping."""

    source = DEFAULT_SEED_OVERRIDES if configured is None else configured
    if not isinstance(source, dict):
        raise ValueError("seed_overrides must be a mapping")
    policy: dict[str, dict[str, str]] = {}
    for stage, overrides in source.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"seed_overrides.{stage} must be a mapping")
        policy[str(stage)] = {str(path): str(seed_name) for path, seed_name in overrides.items()}
    return policy


def seed_override_values(
    policy: dict[str, dict[str, str]] | None,
    stage: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Resolve configured seed overrides for ``stage`` from named seed values."""

    resolved_policy = seed_override_policy(policy)
    overrides = resolved_policy.get(stage, {})
    resolved = {}
    for path, seed_name in overrides.items():
        if seed_name not in values:
            raise KeyError(f"seed policy for {stage!r} references missing seed {seed_name!r}")
        resolved[path] = values[seed_name]
    return resolved


def final_seed_sequences(configured: Any | None = None) -> dict[str, dict[str, int]]:
    """Return normalized final seed sequence specs."""

    source = DEFAULT_FINAL_SEED_SEQUENCES if configured is None else configured
    if not isinstance(source, dict):
        raise ValueError("final_seed_sequences must be a mapping")
    sequences: dict[str, dict[str, int]] = {}
    for name, spec in source.items():
        if not isinstance(spec, dict):
            raise ValueError(f"final_seed_sequences.{name} must be a mapping")
        sequences[str(name)] = {
            "start": int(spec.get("start", 0)),
            "step": int(spec.get("step", 1)),
        }
    return sequences


def final_seed_values(
    sequences: dict[str, dict[str, int]] | None,
    replicate_index: int,
) -> dict[str, int]:
    """Return named final seeds for one replicate index."""

    resolved_sequences = final_seed_sequences(sequences)
    return {
        name: int(spec["start"]) + int(replicate_index) * int(spec["step"])
        for name, spec in resolved_sequences.items()
    }
