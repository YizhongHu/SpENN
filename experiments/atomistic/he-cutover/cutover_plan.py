"""Build the attempt-free He-cutover train and evaluation plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.toolkit.dispatch import LogicalTaskSpec, StagePlanV2, logical_task_id_from_parts
from experiments.toolkit.resources import ResourceSpec
from experiments.toolkit.specs import CompletionSpec

import cutover_strata

SCHEMA = "he-cutover-smoke-grid/v1"
STUDY = "he-cutover"
GRID_KEYS = {"schema", "study", "train_config", "eval_config", "train", "eval", "facilities"}
TRAIN_KEYS = {"seed", "max_steps", "n_walkers"}
EVAL_KEYS = {"chains", "n_walkers", "n_draws", "burn_in", "discard_draws", "stride", "chunk_size", "task_names"}
FACILITY_KEYS = {"runtime", "partition", "stratum", "timeout_min", "cpus", "mem_gb", "gpus"}


class PlanError(ValueError):
    """The tracked grid is not exactly the declared smoke."""


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise PlanError(f"{label} keys mismatch: expected={sorted(keys)}, actual={sorted(value)}")


def load_grid(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the single two-facility smoke grid."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PlanError("grid must be a mapping")
    _exact(payload, GRID_KEYS, "grid")
    if payload["schema"] != SCHEMA or payload["study"] != STUDY:
        raise PlanError("grid schema/study identity changed")
    if payload["train_config"] != "experiments/atomistic/he-v1/configs/train.yaml" or payload["eval_config"] != "experiments/atomistic/he-v1/configs/eval.yaml":
        raise PlanError("He-v1 configs must be referenced verbatim")
    train, evaluation, facilities = payload["train"], payload["eval"], payload["facilities"]
    if not all(isinstance(item, Mapping) for item in (train, evaluation, facilities)):
        raise PlanError("train, eval, and facilities must be mappings")
    _exact(train, TRAIN_KEYS, "train")
    _exact(evaluation, EVAL_KEYS, "eval")
    if dict(train) != {"seed": 0, "max_steps": 25, "n_walkers": 16}:
        raise PlanError("train smoke must be seed 0, 25 steps, 16 walkers")
    expected_eval = {"chains": 2, "n_walkers": 16, "n_draws": 4, "burn_in": 4, "discard_draws": 0, "stride": 2, "chunk_size": 16, "task_names": ["mcmc_energy"]}
    if dict(evaluation) != expected_eval:
        raise PlanError("evaluation smoke coordinates changed")
    if set(facilities) != {"cannon", "polaris"}:
        raise PlanError("facilities must be exactly cannon and polaris")
    normalized = dict(payload)
    normalized["facilities"] = {}
    for name, raw in facilities.items():
        if not isinstance(raw, Mapping):
            raise PlanError(f"facility {name} must be a mapping")
        _exact(raw, FACILITY_KEYS, f"facility {name}")
        resolved = {key: (str(value) if key in {"runtime", "partition", "stratum"} else int(value)) for key, value in raw.items()}
        if resolved["gpus"] != 1:
            raise PlanError("each smoke row requires exactly one GPU")
        cutover_strata.validate_placement(facility=name, partition=resolved["partition"], stratum=resolved["stratum"], timeout_min=resolved["timeout_min"])
        normalized["facilities"][name] = resolved
    return normalized


def expand_rows(grid: Mapping[str, Any], *, facility: str, results_root: str | Path) -> tuple[dict[str, Any], ...]:
    """Return one train row and two same-allocation eval rows."""

    placement = dict(grid["facilities"][facility])
    root = Path(results_root).resolve()
    train_dir = root / "02_train" / "seed-000"
    checkpoint = train_dir / "checkpoints" / "step_000025"
    common = {"facility": facility, "runtime": placement.pop("runtime"), "resources": placement}
    train = {**common, "kind": "train", "stage": "02_train", "row_id": "seed-000", "config": grid["train_config"], "seed": 0, "max_steps": 25, "n_walkers": 16, "result_dir": str(train_dir), "checkpoint_dir": str(checkpoint)}
    rows = [train]
    for chain in range(2):
        row_id = f"seed-000-chain-{chain:02d}"
        rows.append({**common, "kind": "eval", "stage": "03_eval", "row_id": row_id, "config": grid["eval_config"], "seed": chain + 1, **dict(grid["eval"]), "record_capacity": 64, "result_dir": str(root / "03_eval" / row_id), "checkpoint_dir": str(checkpoint)})
    return tuple(rows)


def build_plans(grid: Mapping[str, Any], *, facility: str, results_root: str | Path, plan_id: str) -> tuple[StagePlanV2, StagePlanV2, dict[str, Any]]:
    """Build train/eval plans and the study manifest."""

    rows = expand_rows(grid, facility=facility, results_root=results_root)
    tasks: list[LogicalTaskSpec] = []
    for row in rows:
        result = Path(row["result_dir"])
        status = result / "status.json"
        checkpoint = row.get("checkpoint_dir")
        completion = CompletionSpec(policy="status_completed_with_checkpoint" if row["kind"] == "train" else "status_completed", status_path=str(status), checkpoint_path=str(Path(checkpoint) / "COMPLETE") if row["kind"] == "train" else None)
        script = "run_train_row.py" if row["kind"] == "train" else "run_eval_row.py"
        command = ("{python}", str(Path(__file__).resolve().parent / script), "--row-json", json.dumps(row, sort_keys=True), "--plan-attempt-id", plan_id)
        logical_id = logical_task_id_from_parts(stage=row["stage"], run_id=row["row_id"], plan_id=plan_id)
        dependencies = () if row["kind"] == "train" else (logical_task_id_from_parts(stage="02_train", run_id="seed-000", plan_id=plan_id),)
        resources = row["resources"]
        tasks.append(LogicalTaskSpec(logical_task_id=logical_id, stage=row["stage"], run_id=row["row_id"], command=command, result_dir=str(result), logs=(str(status),), params=dict(row), resources=ResourceSpec(profile="cuda", device="cuda", partition=resources["partition"], threads=resources["cpus"], mem_gb=resources["mem_gb"], gpus=resources["gpus"], timeout_min=resources["timeout_min"], metadata={"stratum": resources["stratum"]}), dependencies=dependencies, completion=completion, metadata={"facility": facility, "runtime": row["runtime"]}))
    train_plan = StagePlanV2(STUDY, "02_train", plan_id, str(results_root), (tasks[0],)).validate()
    eval_plan = StagePlanV2(STUDY, "03_eval", plan_id, str(results_root), tuple(tasks[1:])).validate()
    manifest = {"schema": "he-cutover-plan/v1", "study": STUDY, "facility": facility, "plan_id": plan_id, "results_root": str(Path(results_root).resolve()), "rows": list(rows), "stages": {"02_train": "02_train", "03_eval": "03_eval"}}
    return train_plan, eval_plan, manifest


def write_plans(output_dir: str | Path, train_plan: StagePlanV2, eval_plan: StagePlanV2, manifest: Mapping[str, Any]) -> Path:
    output = Path(output_dir)
    train_plan.write(output / "02_train")
    eval_plan.write(output / "03_eval")
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default=str(Path(__file__).with_name("smoke_grid.yaml")))
    parser.add_argument("--facility", choices=("cannon", "polaris"), required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--plan-attempt-id", required=True)
    args = parser.parse_args(argv)
    plans = build_plans(load_grid(args.grid), facility=args.facility, results_root=args.results_root, plan_id=args.plan_attempt_id)
    write_plans(Path(args.results_root) / "00_plan" / args.plan_attempt_id, *plans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
