"""Plan Hooke pair final-train and final-eval job manifests."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .study_manifest import (
        cartesian_phase_jobs,
        dotlist,
        eval_job_run_id,
        final_eval_plan_dir,
        final_train_plan_dir,
        final_train_report_dir,
        load_yaml,
        phase_base_config,
        phase_config,
        phase_provenance_overrides,
        phase_run_root,
        save_yaml,
        select_report_dir,
        selected_config_id,
        selected_hyperparameters,
        stage_run_id,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - direct script execution
    from study_manifest import (
        cartesian_phase_jobs,
        dotlist,
        eval_job_run_id,
        final_eval_plan_dir,
        final_train_plan_dir,
        final_train_report_dir,
        load_yaml,
        phase_base_config,
        phase_config,
        phase_provenance_overrides,
        phase_run_root,
        save_yaml,
        select_report_dir,
        selected_config_id,
        selected_hyperparameters,
        stage_run_id,
        write_jsonl,
    )


PHASE_OUTPUTS = {
    "final_train": ("final_train_manifest.yaml", "final_train_jobs.jsonl"),
    "smoke_eval": ("smoke_eval_manifest.yaml", "smoke_eval_jobs.jsonl"),
    "final_eval": ("final_eval_manifest.yaml", "final_eval_jobs.jsonl"),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run planner CLI."""

    args = _parse_args(argv)
    plan_final(
        manifest_path=args.manifest,
        selected_config_path=args.selected_config,
        phase=args.phase,
        output_dir=args.output_dir,
        final_train_runs_path=args.final_train_runs,
    )
    return 0


def plan_final(
    *,
    manifest_path: str | Path,
    selected_config_path: str | Path | None = None,
    phase: str,
    output_dir: str | Path | None = None,
    final_train_runs_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write concrete final job manifest artifacts."""

    if phase not in PHASE_OUTPUTS:
        raise ValueError(f"phase must be one of {sorted(PHASE_OUTPUTS)}, got {phase!r}")

    manifest = load_yaml(manifest_path)
    if selected_config_path is None:
        selected_config_path = Path(select_report_dir(manifest)) / "selected_config.yaml"
    selected = load_yaml(selected_config_path)
    if output_dir is not None:
        output = Path(output_dir)
    elif phase == "final_train":
        output = Path(final_train_plan_dir(manifest))
    else:
        output = Path(final_eval_plan_dir(manifest))
    output.mkdir(parents=True, exist_ok=True)

    if phase == "final_train":
        jobs = cartesian_phase_jobs(manifest, "final_train", selected=selected)
    else:
        if final_train_runs_path is None:
            final_train_runs_path = Path(final_train_report_dir(manifest)) / "final_train_runs.csv"
        jobs = _final_eval_jobs(
            manifest,
            selected,
            final_train_runs_path=final_train_runs_path,
            smoke=(phase == "smoke_eval"),
        )

    manifest_name, jobs_name = PHASE_OUTPUTS[phase]
    phase_manifest = {
        "phase": phase,
        "source_manifest": str(manifest_path),
        "selected_config": str(selected_config_path),
        "final_train_runs": None if final_train_runs_path is None else str(final_train_runs_path),
        "job_count": len(jobs),
        "jobs": jobs_name,
    }
    save_yaml(phase_manifest, output / manifest_name)
    write_jsonl(jobs, output / jobs_name)
    return {"manifest": phase_manifest, "jobs": jobs}


def _final_eval_jobs(
    manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
    *,
    final_train_runs_path: str | Path,
    smoke: bool,
) -> list[dict[str, Any]]:
    checkpoint_rows = sorted(_checkpoint_rows(final_train_runs_path), key=lambda row: _seed_sort_key(row["train_seed"]))
    eval_seeds = list(_select(manifest, "final_evaluation.eval_seeds") or [])
    if not eval_seeds:
        raise ValueError("manifest final_evaluation.eval_seeds must be non-empty")
    if smoke:
        checkpoint_rows = checkpoint_rows[:1]
        eval_seeds = eval_seeds[:1]
    if len(checkpoint_rows) != len(eval_seeds):
        raise ValueError(
            "final_eval requires one eval seed per completed final-train checkpoint row; "
            f"got {len(checkpoint_rows)} checkpoint rows and {len(eval_seeds)} eval seeds"
        )

    sampler = _select(manifest, "final_evaluation.sampler")
    sampler = sampler if isinstance(sampler, Mapping) else {}
    jobs: list[dict[str, Any]] = []
    for checkpoint_row, eval_seed in zip(checkpoint_rows, eval_seeds, strict=True):
        train_seed = checkpoint_row["train_seed"]
        checkpoint_path = checkpoint_row["checkpoint_path"]
        if smoke and not Path(str(checkpoint_path)).exists():
            raise FileNotFoundError(f"smoke_eval requires an existing checkpoint path: {checkpoint_path}")
        row_values = {
            "eval_seed": eval_seed,
            "train_seed": train_seed,
            "checkpoint_path": checkpoint_path,
            "sampler_n_walkers": sampler.get("n_walkers"),
            "sampler_burn_in": sampler.get("burn_in"),
            "sampler_n_steps": sampler.get("n_steps"),
            "sampler_proposal_scale": sampler.get("proposal_scale"),
        }
        if smoke:
            row_values.update(_smoke_sampler_values(manifest, row_values))
        jobs.append(_eval_job(manifest, selected, row_values, smoke=smoke))
    return jobs


def _eval_job(
    manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
    row_values: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    phase = "final_eval"
    block = phase_config(manifest, phase)
    overrides = _fixed_overrides(manifest, phase, block)
    overrides.update(_selection_overrides(block, selected))
    overrides.update(_row_overrides(block, row_values))
    config_id = selected_config_id(selected) or "selected"
    overrides["study.config_id"] = config_id
    run_root = phase_run_root(manifest, phase)
    kind = "smoke" if smoke else "full"
    run_id = stage_run_id(kind, config_id, eval_job_run_id(row_values["train_seed"], row_values["eval_seed"]))
    overrides["run.root"] = run_root
    overrides["run.run_id"] = run_id
    overrides["run.layout"] = "flat"
    if smoke:
        overlay = _select(block, "smoke.overlay")
        if isinstance(overlay, Mapping):
            overrides.update({str(key): value for key, value in overlay.items()})
    return {
        "phase": "smoke_eval" if smoke else "final_eval",
        "target_phase": phase,
        "base_config": phase_base_config(manifest, phase),
        "run_root": run_root,
        "run_id": run_id,
        "run_dir": f"{run_root}/{run_id}",
        "config_id": config_id,
        "train_seed": row_values["train_seed"],
        "eval_seed": row_values["eval_seed"],
        "checkpoint_path": row_values["checkpoint_path"],
        "overrides": dotlist(overrides),
    }


def _checkpoint_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checkpoints: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "").lower()
        checkpoint_path = row.get("checkpoint/latest_path") or row.get("checkpoint_path")
        if status and status != "completed":
            continue
        if not checkpoint_path:
            continue
        train_seed = row.get("runtime.seed") or row.get("training_seed") or row.get("train_seed")
        if train_seed in (None, ""):
            raise ValueError(f"final train row lacks runtime.seed/training_seed: {row}")
        checkpoints.append({"train_seed": _parse_scalar(train_seed), "checkpoint_path": checkpoint_path})
    if not checkpoints:
        raise ValueError("no completed final-train checkpoint rows found")
    return checkpoints


def _fixed_overrides(manifest: Mapping[str, Any], phase: str, block: Mapping[str, Any]) -> dict[str, Any]:
    fixed = _select(block, "overrides.fixed")
    values = {str(key): value for key, value in fixed.items()} if isinstance(fixed, Mapping) else {}
    values.update({key: value for key, value in phase_provenance_overrides(manifest, phase).items() if key not in values})
    return values


def _selection_overrides(block: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    mapping = _select(block, "overrides.from_selection")
    if not isinstance(mapping, Mapping):
        return {}
    dotted = selected_hyperparameters(selected)
    values: dict[str, Any] = {}
    for target, source in mapping.items():
        value = _select(selected, str(source))
        if value is None and str(target) in dotted:
            value = dotted[str(target)]
        if value is None:
            raise ValueError(f"selected_config.yaml does not provide {source!r}")
        values[str(target)] = value
    return values


def _row_overrides(block: Mapping[str, Any], row_values: Mapping[str, Any]) -> dict[str, Any]:
    mapping = _select(block, "overrides.from_rows")
    if not isinstance(mapping, Mapping):
        return {}
    values: dict[str, Any] = {}
    for target, source in mapping.items():
        if source not in row_values:
            raise ValueError(f"row value {source!r} required for override {target!r}")
        values[str(target)] = row_values[str(source)]
    return values


def _smoke_sampler_values(manifest: Mapping[str, Any], row_values: Mapping[str, Any]) -> dict[str, Any]:
    overlay = _select(manifest, "phases.final_eval.smoke.overlay")
    if not isinstance(overlay, Mapping):
        return {}
    result = dict(row_values)
    for key, value in overlay.items():
        if key == "sampler_params.n_walkers":
            result["sampler_n_walkers"] = value
        elif key == "sampler_params.burn_in":
            result["sampler_burn_in"] = value
        elif key == "sampler_params.n_steps":
            result["sampler_n_steps"] = value
        elif key == "sampler_params.proposal_scale":
            result["sampler_proposal_scale"] = value
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selected-config", type=Path)
    parser.add_argument("--phase", required=True, choices=sorted(PHASE_OUTPUTS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--final-train-runs", type=Path)
    return parser.parse_args(argv)


def _select(container: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = container
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _parse_scalar(value: Any) -> Any:
    text = str(value).strip()
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _seed_sort_key(value: Any) -> tuple[int, float | str]:
    parsed = _parse_scalar(value)
    if isinstance(parsed, bool):
        return (1, str(parsed))
    if isinstance(parsed, (int, float)):
        return (0, float(parsed))
    return (1, str(parsed))


__all__ = ["PHASE_OUTPUTS", "main", "plan_final"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
