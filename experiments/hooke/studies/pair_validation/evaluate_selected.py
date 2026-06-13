"""Generate final Hooke pair benchmark train/evaluate commands."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

FINAL_SUMMARY_FIELDS = (
    "run_dir",
    "config_id",
    "training_seed",
    "eval_seed",
    "eval/energy",
    "eval/energy_stderr",
    "eval/energy_variance",
    "eval/energy_error",
    "eval/energy_abs_error",
    "eval/sampler/acceptance_rate",
    "eval/sampler/radius_mean",
    "eval/sampler/radius_q99",
    "eval/sampler/electron_distance_q01",
    "runtime/wall_time_sec",
    "status",
    "git/sha",
    "wandb/run_id",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the final-evaluation orchestration CLI."""

    args = _parse_args(argv)
    plan = generate_final_evaluation(
        manifest_path=args.manifest,
        selected_config_path=args.selected_config,
        run_root=args.run_root,
        output_dir=args.output_dir,
        execute=args.execute,
        collect=args.collect,
        eval_config=args.eval_config,
    )
    if args.execute:
        for command in plan["commands"]:
            subprocess.run(command, shell=True, check=True)
    return 0


def generate_final_evaluation(
    *,
    manifest_path: str | Path,
    selected_config_path: str | Path,
    run_root: str | Path,
    output_dir: str | Path,
    execute: bool = False,
    collect: bool = False,
    eval_config: str | Path | None = None,
) -> dict[str, Any]:
    """Write final train/evaluate configs, command script, manifest, and inputs."""

    manifest_path = Path(manifest_path)
    selected_config_path = Path(selected_config_path)
    run_root = Path(run_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_yaml(manifest_path)
    selected = _load_yaml(selected_config_path)
    final_eval = _final_eval_block(manifest)
    validation_seeds = set(_grid_seeds(manifest))
    training_seeds = [int(seed) for seed in final_eval.get("training_seeds", [])]
    eval_seeds = [int(seed) for seed in final_eval.get("eval_seeds", [])]
    if not training_seeds:
        raise ValueError("manifest final_evaluation.training_seeds must be non-empty")
    if not eval_seeds:
        raise ValueError("manifest final_evaluation.eval_seeds must be non-empty")
    if validation_seeds.intersection(training_seeds) and not bool(final_eval.get("allow_validation_seed_reuse", False)):
        raise ValueError("final training seeds reuse validation seeds without allow_validation_seed_reuse=true")

    hyperparameters = _selected_hyperparameters(selected)
    config_id = str(_select(selected, "selected.config_id") or _select(selected, "selection.selected_config_id"))
    if not config_id or config_id == "None":
        raise ValueError("selected_config.yaml must include selected.config_id")

    train_template = Path(str(manifest.get("train_config")))
    eval_template = Path(str(eval_config or manifest.get("final_eval_config")))
    if not train_template:
        raise ValueError("manifest train_config is required")
    if not eval_template:
        raise ValueError("manifest final_eval_config is required")

    pairs = _seed_pairs(training_seeds, eval_seeds)
    train_config_dir = output_dir / "final_train_configs"
    eval_config_dir = output_dir / "final_eval_configs"
    train_config_dir.mkdir(parents=True, exist_ok=True)
    eval_config_dir.mkdir(parents=True, exist_ok=True)

    inputs: list[dict[str, Any]] = []
    commands: list[str] = []
    for training_seed, eval_seed in pairs:
        train_run_id = f"final_train_seed{training_seed}_{_slug(config_id)}"
        eval_run_id = f"final_eval_seed{training_seed}_eval{eval_seed}_{_slug(config_id)}"
        train_config_path = train_config_dir / f"final_train_config_seed{training_seed}.yaml"
        eval_config_path = eval_config_dir / f"final_eval_config_seed{training_seed}_eval{eval_seed}.yaml"
        checkpoint_path = _run_dir(train_template, run_root, train_run_id) / "checkpoints" / "latest.json"

        _write_train_config(
            template=train_template,
            path=train_config_path,
            manifest=manifest,
            config_id=config_id,
            hyperparameters=hyperparameters,
            run_root=run_root,
            run_id=train_run_id,
            training_seed=training_seed,
        )
        _write_eval_config(
            template=eval_template,
            path=eval_config_path,
            manifest=manifest,
            config_id=config_id,
            hyperparameters=hyperparameters,
            run_root=run_root,
            run_id=eval_run_id,
            training_seed=training_seed,
            eval_seed=eval_seed,
            checkpoint_path=checkpoint_path,
        )

        train_command = f"uv run python -u run.py --config {shlex.quote(str(train_config_path))}"
        eval_command = f"uv run python -u run.py --config {shlex.quote(str(eval_config_path))}"
        commands.extend([train_command, eval_command])
        inputs.append(
            {
                "config_id": config_id,
                "training_seed": training_seed,
                "eval_seed": eval_seed,
                "train_config": str(train_config_path),
                "eval_config": str(eval_config_path),
                "checkpoint_path": str(checkpoint_path),
                "train_run_dir": str(_run_dir(train_template, run_root, train_run_id)),
                "eval_run_dir": str(_run_dir(eval_template, run_root, eval_run_id)),
                "train_command": train_command,
                "eval_command": eval_command,
            }
        )

    command_script = output_dir / "final_eval_commands.sh"
    command_script.write_text(_command_script(commands), encoding="utf-8")
    _write_inputs_csv(inputs, output_dir / "final_eval_inputs.csv")
    final_manifest = _final_manifest(
        manifest=manifest,
        selected=selected,
        manifest_path=manifest_path,
        selected_config_path=selected_config_path,
        output_dir=output_dir,
        inputs=inputs,
        execute=execute,
    )
    OmegaConf.save(config=OmegaConf.create(final_manifest), f=output_dir / "final_eval_manifest.yaml", resolve=False)

    result = {"commands": commands, "inputs": inputs, "manifest": final_manifest}
    if collect:
        result["summary"] = collect_final_outputs(inputs=inputs, output_dir=output_dir)
    return result


def collect_final_outputs(*, inputs: Sequence[Mapping[str, Any]], output_dir: str | Path) -> list[dict[str, Any]]:
    """Collect existing final-evaluation run outputs into summary files."""

    output = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for item in inputs:
        run_dir = Path(str(item["eval_run_dir"]))
        row = {field: None for field in FINAL_SUMMARY_FIELDS}
        row.update(
            {
                "run_dir": str(run_dir),
                "config_id": item.get("config_id"),
                "training_seed": item.get("training_seed"),
                "eval_seed": item.get("eval_seed"),
            }
        )
        metrics = _read_metrics(run_dir)
        row.update({key: metrics.get(key) for key in FINAL_SUMMARY_FIELDS if key in metrics})
        status = _load_json_if_present(run_dir / "status.json")
        metadata = _load_json_if_present(run_dir / "metadata.json")
        run_start = _load_json_if_present(run_dir / "run_start.json")
        row["status"] = status.get("status") or metadata.get("status") or ("incomplete" if not run_dir.exists() else "unknown")
        row["git/sha"] = _select(run_start, "git.sha") or metadata.get("git_commit")
        row["wandb/run_id"] = _select(metadata, "wandb.run_id") or _select(metadata, "wandb_run_id")
        rows.append(row)

    _write_summary_csv(rows, output / "final_eval_runs.csv")
    _write_summary_csv(rows, output / "final_benchmark_summary.csv")
    with (output / "final_benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump([_jsonable(row) for row in rows], handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    (output / "final_benchmark_report.md").write_text(_benchmark_report(rows), encoding="utf-8")
    return rows


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selected-config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Default behavior; write plans without executing.")
    parser.add_argument("--execute", action="store_true", help="Execute generated commands sequentially.")
    parser.add_argument("--collect", action="store_true", help="Collect existing final eval outputs into summary files.")
    parser.add_argument("--eval-config", type=Path, default=None)
    return parser.parse_args(argv)


def _write_train_config(
    *,
    template: Path,
    path: Path,
    manifest: Mapping[str, Any],
    config_id: str,
    hyperparameters: Mapping[str, Any],
    run_root: Path,
    run_id: str,
    training_seed: int,
) -> None:
    cfg = OmegaConf.load(template)
    _apply_hyperparameters(cfg, hyperparameters)
    OmegaConf.update(cfg, "study.name", _select(manifest, "study.name"), merge=False, force_add=True)
    OmegaConf.update(cfg, "study.config_id", config_id, merge=False, force_add=True)
    OmegaConf.update(cfg, "runtime.seed", training_seed, merge=False, force_add=True)
    OmegaConf.update(cfg, "run.root", str(run_root), merge=False, force_add=True)
    OmegaConf.update(cfg, "run.run_id", run_id, merge=False, force_add=True)
    OmegaConf.update(cfg, "load.path", None, merge=False, force_add=True)
    OmegaConf.update(cfg, "load.mode", "none", merge=False, force_add=True)
    OmegaConf.save(config=cfg, f=path, resolve=False)


def _write_eval_config(
    *,
    template: Path,
    path: Path,
    manifest: Mapping[str, Any],
    config_id: str,
    hyperparameters: Mapping[str, Any],
    run_root: Path,
    run_id: str,
    training_seed: int,
    eval_seed: int,
    checkpoint_path: Path,
) -> None:
    cfg = OmegaConf.load(template)
    _apply_hyperparameters(cfg, hyperparameters)
    sampler = _final_eval_block(manifest).get("sampler") or {}
    for key, value in sampler.items():
        OmegaConf.update(cfg, f"sampler_params.{key}", value, merge=False, force_add=True)
    OmegaConf.update(cfg, "study.name", _select(manifest, "study.name"), merge=False, force_add=True)
    OmegaConf.update(cfg, "study.config_id", config_id, merge=False, force_add=True)
    OmegaConf.update(cfg, "runtime.seed", eval_seed, merge=False, force_add=True)
    OmegaConf.update(cfg, "evaluation.training_seed", training_seed, merge=False, force_add=True)
    OmegaConf.update(cfg, "run.root", str(run_root), merge=False, force_add=True)
    OmegaConf.update(cfg, "run.run_id", run_id, merge=False, force_add=True)
    OmegaConf.update(cfg, "load.path", str(checkpoint_path.resolve()), merge=False, force_add=True)
    OmegaConf.update(cfg, "load.mode", "model_only", merge=False, force_add=True)
    OmegaConf.update(cfg, "load.strict", True, merge=False, force_add=True)
    OmegaConf.update(cfg, "load.allow_protocol_mismatch", False, merge=False, force_add=True)
    OmegaConf.save(config=cfg, f=path, resolve=False)


def _final_manifest(
    *,
    manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
    manifest_path: Path,
    selected_config_path: Path,
    output_dir: Path,
    inputs: Sequence[Mapping[str, Any]],
    execute: bool,
) -> dict[str, Any]:
    final_eval = _final_eval_block(manifest)
    return {
        "selected_config_id": _select(selected, "selected.config_id"),
        "selected_hyperparameters": _selected_hyperparameters(selected),
        "selection_report_path": _select(selected, "study.selection_report") or str(output_dir / "selection_report.md"),
        "selected_config_path": str(selected_config_path),
        "source_validation_study_name": _select(manifest, "study.name"),
        "source_git_sha": _first_git_sha(selected),
        "manifest_path": str(manifest_path),
        "final_evaluation_config": str(manifest.get("final_eval_config")),
        "final_evaluation_sampler_settings": final_eval.get("sampler", {}),
        "final_training_seeds": final_eval.get("training_seeds", []),
        "final_evaluation_seeds": final_eval.get("eval_seeds", []),
        "exact_reference_source": "experiments/hooke/configs/benchmark/pair_final_eval.yaml references.exact_energy",
        "expected_model_config_hash_provided": True,
        "dry_run": not execute,
        "inputs": list(inputs),
    }


def _selected_hyperparameters(selected: Mapping[str, Any]) -> dict[str, Any]:
    values = _select(selected, "selected.hyperparameters")
    if not isinstance(values, Mapping):
        raise ValueError("selected_config.yaml must include selected.hyperparameters")
    return dict(values)


def _apply_hyperparameters(cfg: Any, hyperparameters: Mapping[str, Any]) -> None:
    for key, value in hyperparameters.items():
        OmegaConf.update(cfg, str(key), value, merge=False, force_add=True)


def _seed_pairs(training_seeds: Sequence[int], eval_seeds: Sequence[int]) -> list[tuple[int, int]]:
    if len(training_seeds) == len(eval_seeds):
        return list(zip(training_seeds, eval_seeds, strict=True))
    return [(training_seed, eval_seed) for training_seed in training_seeds for eval_seed in eval_seeds]


def _run_dir(config_path: Path, run_root: Path, run_id: str) -> Path:
    cfg = OmegaConf.load(config_path)
    experiment = str(OmegaConf.select(cfg, "experiment.name", default="experiment"))
    sector = str(OmegaConf.select(cfg, "experiment.sector", default="default"))
    root = run_root if run_root.is_absolute() else Path.cwd() / run_root
    return root / experiment / sector / run_id


def _command_script(commands: Sequence[str]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(commands)
    return "\n".join(lines) + "\n"


def _write_inputs_csv(inputs: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns = (
        "config_id",
        "training_seed",
        "eval_seed",
        "train_config",
        "eval_config",
        "checkpoint_path",
        "train_run_dir",
        "eval_run_dir",
        "train_command",
        "eval_command",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in inputs:
            writer.writerow({column: row.get(column) for column in columns})


def _write_summary_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in FINAL_SUMMARY_FIELDS})


def _benchmark_report(rows: Sequence[Mapping[str, Any]]) -> str:
    completed = [row for row in rows if row.get("status") == "completed"]
    lines = [
        "# Hooke Pair Final Benchmark Summary",
        "",
        f"Runs summarized: {len(rows)}",
        f"Completed runs: {len(completed)}",
        "",
        "Final evaluation may use exact-reference energy because validation selection is already frozen.",
        "",
    ]
    if completed:
        energies = [_as_float(row.get("eval/energy"), default=math.nan) for row in completed]
        finite = [value for value in energies if math.isfinite(value)]
        if finite:
            lines.append(f"Mean eval/energy: {sum(finite) / len(finite):.12g}")
    return "\n".join(lines) + "\n"


def _read_metrics(run_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for path in (run_dir / "metrics.csv", run_dir / "metrics.jsonl"):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    namespace = str(row.get("namespace") or "").strip("/")
                    key = str(row.get("key") or "").strip("/")
                    if namespace and key:
                        metrics[f"{namespace}/{key}"] = _parse_scalar(row.get("value"))
        else:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    namespace = str(record.get("namespace") or "").strip("/")
                    values = record.get("metrics") or {}
                    if isinstance(values, Mapping):
                        for key, value in values.items():
                            metrics[f"{namespace}/{key}"] = value
    return metrics


def _load_yaml(path: str | Path) -> dict[str, Any]:
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    return data if isinstance(data, dict) else {}


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _final_eval_block(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    block = manifest.get("final_evaluation")
    return block if isinstance(block, Mapping) else {}


def _grid_seeds(manifest: Mapping[str, Any]) -> list[int]:
    seed_key = str(manifest.get("seed_key") or "runtime.seed")
    grid = manifest.get("grid") if isinstance(manifest.get("grid"), Mapping) else {}
    return [int(seed) for seed in grid.get(seed_key, [])]


def _first_git_sha(selected: Mapping[str, Any]) -> str | None:
    runs = _select(selected, "selected.validation_runs")
    if isinstance(runs, Sequence):
        for run in runs:
            if isinstance(run, Mapping) and run.get("git_sha"):
                return str(run["git_sha"])
    return None


def _select(container: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = container
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _parse_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if lowered in {"-inf", "-infinity"}:
        return -math.inf
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _as_float(value: Any, *, default: float) -> float:
    parsed = _parse_scalar(value)
    if parsed is None or isinstance(parsed, bool):
        return default
    try:
        return float(parsed)
    except (TypeError, ValueError):
        return default


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() else "-" for char in text).strip("-")


__all__ = [
    "FINAL_SUMMARY_FIELDS",
    "collect_final_outputs",
    "generate_final_evaluation",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
