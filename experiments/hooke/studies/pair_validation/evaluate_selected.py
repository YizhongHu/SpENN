#!/usr/bin/env python
"""Generate (and optionally run) the final held-out benchmark for the winner.

Reads ``manifest.yaml`` and ``selected_config.yaml`` (from select.py) and
writes ``final_eval_commands.sh``, ``final_eval_manifest.yaml``, and
``final_eval_inputs.csv``. Dry-run is the default: nothing executes unless
``--execute`` is passed.

The final benchmark is two staged sets of standard ``run.py`` commands:

1. retrain the selected config once per fresh final training seed
   (``pair_train.yaml``), and
2. evaluate each trained checkpoint with the large final-evaluation sampler
   and its paired evaluation seed, via a generated, self-contained eval
   config derived from ``pair_final_eval.yaml`` (the Evaluate runner owns
   all physics diagnostics).

Checkpoint/model pairing contract (issue #45 addendum): each generated eval
config carries the *explicit* model spec of its training run — copied from
the training run's ``resolved_config.yaml`` when it exists, otherwise
resolved from the training config plus the selected overrides (the same
resolution the training command performs). Architecture is never inferred
from checkpoint keys. ``final_evaluation.checkpoint_loading`` in the
manifest picks the mode:

- ``structured_checkpoint`` (default): training runs write schema-v1
  checkpoints carrying ``model_config_hash``; the generated eval config pins
  ``evaluation.expected_model_config_hash`` so loading verifies the pairing.
- ``legacy_resolved_config_workaround``: for pre-schema checkpoints from
  already-completed training runs. The model spec is copied from the
  training ``resolved_config.yaml``, loading stays strict, but no hash is
  verified (legacy payloads carry none). Recorded as such in
  ``final_eval_manifest.yaml``; this mode must not become the long-term
  benchmark path.

``--collect`` summarizes existing final-eval run directories into
``final_benchmark_summary.csv/json`` and ``final_benchmark_report.md``.
Exact-reference error metrics (``eval/energy_error``) are allowed here —
and only here — because selection is already frozen.

Local run outputs are authoritative end to end; this script never reads W&B.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import median

import yaml
from omegaconf import OmegaConf

# Canonical config hashing is owned by the checkpoint module (torch-free).
from spenn.callback.checkpoint import model_config_hash

from collect import _read_json, _read_yaml, load_manifest, lookup_dotted, read_last_metrics

# Manifest final_evaluation.checkpoint_loading values (issue #45 addendum).
CHECKPOINT_LOADING_MODES = ("structured_checkpoint", "legacy_resolved_config_workaround")

# Columns of the final benchmark summary (issue-required field list).
_SUMMARY_COLUMNS = (
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


def load_selected_config(path: Path) -> dict:
    """Load select.py's frozen winner and minimally validate it."""

    with open(path, encoding="utf-8") as handle:
        selected = yaml.safe_load(handle)
    for required in ("study", "selected", "overrides"):
        if required not in selected:
            raise ValueError(f"selected config {path} is missing the {required!r} section")
    return selected


def final_evaluation_policy(manifest: dict) -> dict:
    """Return the manifest final_evaluation block, validated."""

    policy = manifest.get("final_evaluation")
    if not isinstance(policy, dict):
        raise ValueError("manifest is missing the final_evaluation section")
    training_seeds = list(policy.get("training_seeds", ()))
    eval_seeds = list(policy.get("eval_seeds", ()))
    if not training_seeds or not eval_seeds:
        raise ValueError("final_evaluation needs non-empty training_seeds and eval_seeds")
    if len(training_seeds) != len(eval_seeds):
        raise ValueError(
            "final_evaluation training_seeds and eval_seeds are paired index-wise "
            f"and must have equal length ({len(training_seeds)} != {len(eval_seeds)})"
        )

    # Fresh seeds only: the benchmark must not silently reuse the runs that
    # drove selection.
    validation_seeds = {str(seed) for seed in manifest["grid"][str(manifest["seed_key"])]}
    if not policy.get("allow_validation_seed_reuse", False):
        reused = [
            seed for seed in (*training_seeds, *eval_seeds) if str(seed) in validation_seeds
        ]
        if reused:
            raise ValueError(
                f"final seeds {reused} reuse validation seeds; set "
                "final_evaluation.allow_validation_seed_reuse in the manifest to permit this"
            )

    mode = checkpoint_loading_mode(policy)
    if mode not in CHECKPOINT_LOADING_MODES:
        raise ValueError(
            f"final_evaluation.checkpoint_loading must be one of {CHECKPOINT_LOADING_MODES}, "
            f"got {mode!r}"
        )
    return policy


def checkpoint_loading_mode(policy: dict) -> str:
    """Checkpoint pairing mode declared by the manifest (structured default)."""

    return str(policy.get("checkpoint_loading", "structured_checkpoint"))


def trained_model_spec(train_run_dir: Path) -> dict | None:
    """Explicit model spec from an existing training run, or None.

    The training run's ``resolved_config.yaml`` is the authoritative source of
    the trained architecture (issue #45 addendum workaround): copying it means
    the eval model is explicitly configured, never guessed from checkpoint keys.
    """

    resolved_path = train_run_dir / "resolved_config.yaml"
    if not resolved_path.is_file():
        return None
    return (_read_yaml(resolved_path) or {}).get("model")


def planned_model_spec(train_config: Path, overrides: list[str]) -> dict | None:
    """Model spec the planned training command will resolve to.

    Replays the training command's config resolution (OmegaConf load + dotlist
    overrides) and resolves only the ``model`` subtree, so the hash of this
    spec equals the ``model_config_hash`` a schema-v1 checkpoint will store.
    """

    cfg = OmegaConf.load(train_config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    node = OmegaConf.select(cfg, "model")
    if node is None:
        return None
    return OmegaConf.to_container(node, resolve=True)


def _experiment_subdir(config_path: Path) -> str:
    """Run-dir subpath ``<experiment.name>/<sector>`` declared by a config."""

    config = _read_yaml(config_path)
    name = lookup_dotted(config, "experiment.name")
    sector = lookup_dotted(config, "experiment.sector")
    if not name or not sector:
        raise ValueError(f"{config_path} does not declare experiment.name and experiment.sector")
    return f"{name}/{sector}"


def _selected_model_params(overrides: list[str]) -> dict[str, object]:
    """Selected ``model_params.*`` override values as a provenance mapping."""

    params: dict[str, object] = {}
    for override in overrides:
        key, _, value = override.partition("=")
        if key.startswith("model_params."):
            params[key.removeprefix("model_params.")] = yaml.safe_load(value)
    return params


def _baked_eval_config(
    eval_config: Path,
    entry: dict[str, object],
    *,
    study_name: str,
    model_spec: dict | None,
    model_params: dict[str, object],
    sampler: dict,
    mode: str,
    run_root: Path,
) -> dict:
    """One self-contained eval config derived from the base eval config.

    Bakes in the explicit model spec, checkpoint pairing, seeds, study
    identity, and sampler settings, so the eval run is fully reproducible from
    the generated file alone (no command-line overrides).
    """

    config = copy.deepcopy(_read_yaml(eval_config))
    config.setdefault("run", {})["root"] = str(run_root / study_name)
    config["run"]["run_id"] = entry["eval_run_id"]
    config.setdefault("study", {})["name"] = study_name
    config["study"]["config_id"] = entry["config_id"]
    config.setdefault("runtime", {})["seed"] = entry["eval_seed"]
    if model_spec is not None:
        config["model"] = copy.deepcopy(model_spec)
    # Keep the user-facing knob block consistent with the explicit model spec.
    config.setdefault("model_params", {}).update(model_params)
    config.setdefault("sampler_params", {}).update(sampler)

    evaluation = config.setdefault("evaluation", {})
    evaluation["checkpoint"] = entry["source_checkpoint_path"]
    evaluation["checkpoint_strict"] = True
    evaluation["allow_model_config_mismatch"] = False
    evaluation["expected_model_config_hash"] = (
        entry["source_model_config_hash"] if mode == "structured_checkpoint" else None
    )
    evaluation["training_seed"] = entry["training_seed"]

    # Checkpoint pairing provenance (issue #45 addendum); lands verbatim in the
    # eval run's resolved_config.yaml.
    config["checkpoint_loading"] = {
        "mode": mode,
        "source_resolved_config": entry["source_resolved_config"],
        "checkpoint_path": entry["source_checkpoint_path"],
        "strict": True,
        "model_config_hash_verified": mode == "structured_checkpoint",
    }
    return config


def build_plan(
    manifest: dict,
    selected: dict,
    *,
    run_root: Path,
    train_config: Path,
    eval_config: Path,
    output_dir: Path,
) -> list[dict[str, object]]:
    """Plan one train command + one generated eval config per training seed."""

    policy = final_evaluation_policy(manifest)
    mode = checkpoint_loading_mode(policy)
    study_name = str(policy.get("study_name") or f"{manifest['study']['name']}_final")
    config_id = str(selected["selected"]["config_id"])
    overrides = [str(override) for override in selected["overrides"]]
    model_params = _selected_model_params(overrides)
    sampler = dict(policy.get("sampler", {}))

    root = run_root / study_name
    train_subdir = _experiment_subdir(train_config)
    eval_subdir = _experiment_subdir(eval_config)

    plan: list[dict[str, object]] = []
    for training_seed, eval_seed in zip(policy["training_seeds"], policy["eval_seeds"]):
        train_run_id = f"final_train_seed{training_seed}"
        eval_run_id = f"final_eval_seed{training_seed}_eval{eval_seed}"
        train_run_dir = root / train_subdir / train_run_id
        eval_run_dir = root / eval_subdir / eval_run_id
        checkpoint = train_run_dir / "checkpoints" / "latest.pt"
        final_eval_config_path = output_dir / f"final_eval_config_seed{training_seed}_eval{eval_seed}.yaml"

        # The explicit model spec for this eval run: the training run's
        # resolved config when it already exists; otherwise (fresh benchmark
        # dry-run) the spec the planned training command will resolve to.
        model_spec = trained_model_spec(train_run_dir)
        source_resolved_config = (
            str(train_run_dir / "resolved_config.yaml") if model_spec is not None else None
        )
        if model_spec is None:
            if mode == "legacy_resolved_config_workaround":
                raise ValueError(
                    "legacy_resolved_config_workaround needs completed training runs; "
                    f"missing {train_run_dir / 'resolved_config.yaml'}"
                )
            model_spec = planned_model_spec(
                train_config, [f"runtime.seed={training_seed}", *overrides]
            )
        source_model_config_hash = None if model_spec is None else model_config_hash(model_spec)
        train_metadata = _read_json(train_run_dir / "metadata.json")

        train_command = [
            "uv", "run", "python", "run.py",
            "--config", str(train_config),
            f"run.root={root}",
            f"run.run_id={train_run_id}",
            f"study.name={study_name}",
            f"study.config_id={config_id}",
            f"runtime.seed={training_seed}",
            *overrides,
        ]
        # Everything eval-side is baked into the generated config.
        eval_command = ["uv", "run", "python", "run.py", "--config", str(final_eval_config_path)]

        entry: dict[str, object] = {
            "config_id": config_id,
            "training_seed": training_seed,
            "eval_seed": eval_seed,
            "train_run_id": train_run_id,
            "source_train_run_dir": str(train_run_dir),
            "source_resolved_config": source_resolved_config,
            "source_checkpoint_path": str(checkpoint),
            "source_model_config_hash": source_model_config_hash,
            "source_train_git_sha": train_metadata.get("git_commit") or None,
            "eval_run_id": eval_run_id,
            "eval_run_dir": str(eval_run_dir),
            "final_eval_config_path": str(final_eval_config_path),
            "train_command": train_command,
            "eval_command": eval_command,
        }
        entry["eval_config"] = _baked_eval_config(
            eval_config,
            entry,
            study_name=study_name,
            model_spec=model_spec,
            model_params=model_params,
            sampler=sampler,
            mode=mode,
            run_root=run_root,
        )
        plan.append(entry)
    return plan


def write_eval_configs(plan: list[dict[str, object]]) -> None:
    """Write each planned eval run's self-contained generated config."""

    for entry in plan:
        path = Path(entry["final_eval_config_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(entry["eval_config"], handle, sort_keys=False)


def _shell_line(command: list[str]) -> str:
    """One command as a readable (multi-line when long) shell invocation."""

    head, args = command[:6], command[6:]
    if not args:
        return " ".join(head)
    lines = [" ".join(head) + " \\"]
    lines += [f"  {arg} \\" for arg in args[:-1]]
    lines.append(f"  {args[-1]}")
    return "\n".join(lines)


def write_commands(plan: list[dict[str, object]], path: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "# Final benchmark commands generated by evaluate_selected.py.",
        "# Each eval run restores its train run's checkpoint, so a train command",
        "# must finish before its paired eval command starts. Run locally with",
        "#   bash final_eval_commands.sh",
        "# or submit each command through SLURM (sbatch --wrap or a job array),",
        "# keeping the train -> eval dependency per seed (e.g. --dependency=afterok).",
        "set -euo pipefail",
        "",
    ]
    for entry in plan:
        lines += [
            f"# --- training seed {entry['training_seed']} ---",
            _shell_line(entry["train_command"]),
            "",
            f"# eval seed {entry['eval_seed']} (checkpoint from seed {entry['training_seed']})",
            _shell_line(entry["eval_command"]),
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""


def write_final_manifest(
    manifest: dict,
    selected: dict,
    plan: list[dict[str, object]],
    *,
    eval_config: Path,
    selection_report: Path,
    path: Path,
) -> None:
    policy = final_evaluation_policy(manifest)
    mode = checkpoint_loading_mode(policy)
    payload = {
        "study": {
            "name": str(policy.get("study_name") or f"{manifest['study']['name']}_final"),
            "purpose": "final_benchmark",
            "source_validation_study": manifest["study"]["name"],
        },
        "selected": dict(selected["selected"]),
        "selection_report": str(selection_report),
        "source_git_sha": _git_sha(),
        "final_eval_config": str(eval_config),
        "final_eval_sampler": dict(policy.get("sampler", {})),
        "final_training_seeds": list(policy["training_seeds"]),
        "final_eval_seeds": list(policy["eval_seeds"]),
        # How eval models are paired with checkpoints (issue #45 addendum):
        # structured checkpoints verify model_config_hash; the legacy
        # resolved-config workaround copies the model spec but cannot verify.
        "checkpoint_loading": {
            "mode": mode,
            "strict": True,
            "model_config_hash_verified": mode == "structured_checkpoint",
        },
        # The exact reference enters only here, after selection froze.
        "exact_reference": {
            "source": "references.exact_energy in the final eval config (Taut 1993, omega=0.5 Hooke singlet)",
            "used_by": "eval/energy_error, eval/energy_abs_error, eval/reference_energy",
        },
        "runs": [
            {
                "training_seed": entry["training_seed"],
                "eval_seed": entry["eval_seed"],
                "source_train_run_dir": entry["source_train_run_dir"],
                "source_resolved_config": entry["source_resolved_config"],
                "source_checkpoint_path": entry["source_checkpoint_path"],
                "source_model_config_hash": entry["source_model_config_hash"],
                "source_train_git_sha": entry["source_train_git_sha"],
                "final_eval_config_path": entry["final_eval_config_path"],
                "eval_run_dir": entry["eval_run_dir"],
            }
            for entry in plan
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_inputs_csv(plan: list[dict[str, object]], path: Path) -> None:
    columns = [
        "config_id",
        "training_seed",
        "eval_seed",
        "train_run_id",
        "source_train_run_dir",
        "source_resolved_config",
        "source_checkpoint_path",
        "source_model_config_hash",
        "eval_run_id",
        "eval_run_dir",
        "final_eval_config_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for entry in plan:
            writer.writerow([entry[column] for column in columns])


def execute_plan(plan: list[dict[str, object]], output_dir: Path) -> int:
    """Run train then eval per seed, recording statuses in final_eval_runs.csv."""

    records: list[dict[str, object]] = []
    failed = False
    for entry in plan:
        for stage, command, run_dir in (
            ("train", entry["train_command"], entry["source_train_run_dir"]),
            ("eval", entry["eval_command"], entry["eval_run_dir"]),
        ):
            if failed:
                records.append(
                    {"stage": stage, "run_dir": run_dir, "returncode": "", "status": "skipped"}
                )
                continue
            print(f"[{stage}] {' '.join(command)}", flush=True)
            result = subprocess.run(command)
            status = "completed" if result.returncode == 0 else "failed"
            records.append(
                {
                    "stage": stage,
                    "run_dir": run_dir,
                    "returncode": result.returncode,
                    "status": status,
                }
            )
            if result.returncode != 0:
                failed = True  # an eval without its checkpoint cannot succeed

    runs_csv = output_dir / "final_eval_runs.csv"
    with open(runs_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "run_dir", "returncode", "status"])
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {runs_csv}")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Collection mode: summarize completed final-eval run directories
# ---------------------------------------------------------------------------


def collect_final_eval_run(run_dir: Path) -> dict[str, object] | None:
    """Normalize one final-eval run directory; None for non-eval runs."""

    resolved = _read_yaml(run_dir / "resolved_config.yaml")
    if lookup_dotted(resolved, "evaluation.checkpoint") is None:
        return None  # a training run (or unrelated run), not a final eval run

    metadata = _read_json(run_dir / "metadata.json")
    status_file = _read_json(run_dir / "status.json")
    metrics = read_last_metrics(run_dir / "metrics.jsonl")

    status = str(status_file.get("status", "")).lower()
    if status in ("failed", "exception", "error"):
        status = "failed"
    elif status == "completed" and "eval/energy" in metrics:
        status = "completed"
    else:
        status = "incomplete"

    row: dict[str, object] = {
        "run_dir": str(run_dir),
        "config_id": lookup_dotted(resolved, "study.config_id") or "",
        "training_seed": lookup_dotted(resolved, "evaluation.training_seed"),
        "eval_seed": lookup_dotted(resolved, "runtime.seed"),
        "status": status,
        "git/sha": metadata.get("git_commit") or "",
        "wandb/run_id": metadata.get("wandb_run_id") or "",
    }
    for column in _SUMMARY_COLUMNS:
        if column.startswith(("eval/", "runtime/")):
            row[column] = metrics.get(column)
    return row


def _finite_values(rows: list[dict[str, object]], column: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(column)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    csv_path = output_dir / "final_benchmark_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_SUMMARY_COLUMNS)
        for row in rows:
            writer.writerow(
                ["" if row.get(column) is None else row.get(column) for column in _SUMMARY_COLUMNS]
            )

    json_path = output_dir / "final_benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=str)

    completed = [row for row in rows if row.get("status") == "completed"]
    lines = [
        "# Final benchmark report",
        "",
        f"- runs: {len(rows)} total, {len(completed)} completed",
        "- inputs: local final-eval run directories only (W&B is visualization only)",
        "",
        "| training_seed | eval_seed | eval/energy | eval/energy_stderr | eval/energy_error | status |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (str(r.get("training_seed")), str(r.get("eval_seed")))):
        lines.append(
            "| "
            + " | ".join(
                str(row.get(column) if row.get(column) is not None else "-")
                for column in (
                    "training_seed",
                    "eval_seed",
                    "eval/energy",
                    "eval/energy_stderr",
                    "eval/energy_error",
                    "status",
                )
            )
            + " |"
        )
    energies = _finite_values(completed, "eval/energy")
    abs_errors = _finite_values(completed, "eval/energy_abs_error")
    lines.append("")
    if energies:
        lines.append(f"Median eval/energy over completed runs: {median(energies):.8g}")
    if abs_errors:
        lines.append(f"Median eval/energy_abs_error over completed runs: {median(abs_errors):.3g}")
    if not energies:
        lines.append("No completed final-eval runs found.")
    lines.append("")
    (output_dir / "final_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {csv_path}, {json_path}, {output_dir / 'final_benchmark_report.md'}")


def collect_summary(run_root: Path, manifest: dict, output_dir: Path) -> int:
    policy = final_evaluation_policy(manifest)
    study_name = str(policy.get("study_name") or f"{manifest['study']['name']}_final")
    root = run_root / study_name
    if not root.is_dir():
        print(f"no final-eval runs under {root}", file=sys.stderr)
        return 1
    run_dirs = sorted(path.parent for path in root.rglob("metadata.json"))
    rows = [row for run_dir in run_dirs if (row := collect_final_eval_run(run_dir)) is not None]
    if not rows:
        print(f"no final-eval run directories under {root}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(rows, output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    study_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=study_dir / "manifest.yaml", help="Study manifest path."
    )
    parser.add_argument(
        "--selected-config",
        type=Path,
        default=study_dir / "results" / "selected_config.yaml",
        help="Frozen winner written by select.py.",
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("outputs"), help="Root for final benchmark run dirs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=study_dir / "results",
        help="Directory for generated commands, manifest, inputs, and summaries.",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help="Final eval config (default: manifest final_evaluation.eval_config).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Write commands without executing (the default behavior).",
    )
    mode.add_argument(
        "--execute", action="store_true", help="Run the generated commands sequentially."
    )
    mode.add_argument(
        "--collect",
        action="store_true",
        help="Summarize existing final-eval runs instead of generating commands.",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    if args.collect:
        return collect_summary(args.run_root, manifest, args.output_dir)

    selected = load_selected_config(args.selected_config)
    policy = final_evaluation_policy(manifest)
    train_config = Path(selected.get("train_config") or manifest["train_config"])
    eval_config = args.eval_config or Path(policy["eval_config"])

    plan = build_plan(
        manifest,
        selected,
        run_root=args.run_root,
        train_config=train_config,
        eval_config=eval_config,
        output_dir=args.output_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_eval_configs(plan)
    commands_path = args.output_dir / "final_eval_commands.sh"
    write_commands(plan, commands_path)
    write_final_manifest(
        manifest,
        selected,
        plan,
        eval_config=eval_config,
        selection_report=args.selected_config.parent / "selection_report.md",
        path=args.output_dir / "final_eval_manifest.yaml",
    )
    write_inputs_csv(plan, args.output_dir / "final_eval_inputs.csv")
    print(
        f"planned {len(plan)} train+eval pairs -> {commands_path} "
        f"({'executing' if args.execute else 'dry-run; pass --execute to run'})"
    )

    if args.execute:
        return execute_plan(plan, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
