"""Run one frozen ``he-v1-diagnostic-v1`` row inside its Slurm allocation.

This driver restores an existing complete checkpoint only. It binds the row to
the checkpoint bytes, evaluator Git SHA, delivered GPU, selected task graph,
and resolved scientific controls before invoking the sanctioned configured-run
entrypoint. No training or resume path exists in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from omegaconf import DictConfig, OmegaConf

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import diagnostic_plan as plan_stage  # noqa: E402
import layout  # noqa: E402

EVALUATOR_ID = "he-v1-diagnostic-v1"
ALLOCATION_RECEIPT = "allocation_receipt.json"
CHECKPOINT_BINDING = "checkpoint_binding.json"

DeviceReader = Callable[[], str | None]
ConfigRunner = Callable[..., int]


class DiagnosticDriverError(RuntimeError):
    """A diagnostic row cannot execute with the supplied immutable inputs."""


def require_scheduler(environ: Mapping[str, str] | None = None) -> str:
    """Return the Slurm job id and reject login-node execution."""

    environ = os.environ if environ is None else environ
    job_id = str(environ.get("SLURM_JOB_ID") or "").strip()
    if not job_id:
        raise DiagnosticDriverError(
            "SLURM_JOB_ID is empty: diagnostic rows run only inside Slurm"
        )
    return job_id


def torch_device_name() -> str | None:
    """Return the delivered CUDA device name from inside the allocation."""

    try:
        import torch  # noqa: PLC0415 - allocation-only runtime dependency
    except ImportError:
        return None
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        return None
    return str(torch.cuda.get_device_name(0))


def verify_delivered_device(
    row: Mapping[str, Any],
    *,
    receipt_dir: str | Path,
    job_id: str,
    device_reader: DeviceReader = torch_device_name,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify the production A100 or smoke A100-MIG stratum and write a receipt."""

    environ = os.environ if environ is None else environ
    resources = row["resources"]
    requested = str(resources["stratum"])
    delivered = device_reader()
    text = "" if delivered is None else delivered.lower()
    mismatch: str | None = None
    if requested == "a100":
        if "a100-sxm4-80gb" not in text or "mig" in text:
            mismatch = "production diagnostic requires a full NVIDIA A100-SXM4-80GB"
    elif requested == "a100_mig":
        if "a100" not in text or ("mig" not in text and "3g.20gb" not in text):
            mismatch = "smoke diagnostic requires an NVIDIA A100 MIG slice"
    else:
        mismatch = f"unknown diagnostic GPU stratum {requested!r}"
    declared_partition = str(resources["partition"])
    delivered_partition = str(environ.get("SLURM_JOB_PARTITION") or "")
    if delivered_partition and delivered_partition != declared_partition:
        mismatch = (
            f"delivered partition {delivered_partition!r} does not match "
            f"planned {declared_partition!r}"
        )
    receipt = {
        "schema": "he-v1-diagnostic-allocation/v1",
        "row_id": row["row_id"],
        "job_id": job_id,
        "hostname": socket.gethostname(),
        "requested_partition": declared_partition,
        "delivered_partition": delivered_partition or None,
        "requested_stratum": requested,
        "requested_constraint": resources.get("constraint"),
        "delivered_device": delivered,
        "delivered_matches_requested": mismatch is None,
        "mismatch_reason": mismatch,
        "cuda_visible_devices": environ.get("CUDA_VISIBLE_DEVICES"),
        "recorded_at": datetime.now(ZoneInfo(plan_stage.STUDY_TIMEZONE)).isoformat(),
        "timezone": plan_stage.STUDY_TIMEZONE,
    }
    _write_new_json(Path(receipt_dir) / ALLOCATION_RECEIPT, receipt)
    if mismatch is not None:
        raise DiagnosticDriverError(mismatch)
    return receipt


def require_evaluation_checkout(manifest: Mapping[str, Any]) -> str:
    """Require the allocation checkout to be clean and at the planned full SHA."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=STUDY_DIR.parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise DiagnosticDriverError(
            f"cannot read evaluator Git SHA: {completed.stderr.strip()}"
        )
    actual = completed.stdout.strip()
    expected = str(manifest["evaluation_git_sha"])
    if actual != expected:
        raise DiagnosticDriverError(
            f"evaluator Git SHA mismatch: planned={expected}, actual={actual}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=STUDY_DIR.parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise DiagnosticDriverError(
            "evaluator checkout has tracked modifications; exact-SHA execution required"
        )
    return actual


def reconcile_checkpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash and validate the real-format checkpoint immediately before restore."""

    source = Path(str(row["checkpoint_source_dir"]))
    required = {
        "model": source / "model.pt",
        "manifest": source / "manifest.json",
        "complete": source / "COMPLETE",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise DiagnosticDriverError(
            f"checkpoint {row['checkpoint_label']} is incomplete; missing={missing}"
        )
    actual_hashes = {name: plan_stage.file_sha256(path) for name, path in required.items()}
    expected_hashes = {
        "model": str(row["checkpoint_model_sha256"]),
        "manifest": str(row["checkpoint_manifest_sha256"]),
        "complete": str(row["checkpoint_complete_sha256"]),
    }
    if actual_hashes != expected_hashes:
        raise DiagnosticDriverError(
            f"checkpoint {row['checkpoint_label']} content mismatch: "
            f"expected={expected_hashes}, actual={actual_hashes}"
        )
    checkpoint_manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    files = checkpoint_manifest.get("files")
    provenance = checkpoint_manifest.get("provenance")
    if (
        checkpoint_manifest.get("schema_version") != row["checkpoint_schema_version"]
        or checkpoint_manifest.get("kind") != row["checkpoint_kind"]
        or checkpoint_manifest.get("completed_updates") != row["checkpoint_step"]
    ):
        raise DiagnosticDriverError("checkpoint real-format identity changed after planning")
    if not isinstance(files, Mapping) or files.get("model") != "model.pt":
        raise DiagnosticDriverError("checkpoint manifest does not bind model.pt")
    if not isinstance(provenance, Mapping):
        raise DiagnosticDriverError("checkpoint manifest lacks provenance")
    if (
        provenance.get("git_sha") != row["checkpoint_source_git_sha"]
        or provenance.get("tpen_version") != row["checkpoint_source_tpen_version"]
    ):
        raise DiagnosticDriverError("checkpoint source provenance changed after planning")
    resolved_name = files.get("resolved_config")
    if not isinstance(resolved_name, str) or not (source / resolved_name).is_file():
        raise DiagnosticDriverError("checkpoint lacks its real-format resolved config")
    return {
        "schema": "he-v1-diagnostic-checkpoint-binding/v1",
        "checkpoint_label": row["checkpoint_label"],
        "checkpoint_step": row["checkpoint_step"],
        "checkpoint_dir": str(source),
        "checkpoint_model_file": str(required["model"]),
        "hashes": actual_hashes,
        "checkpoint_schema_version": checkpoint_manifest["schema_version"],
        "checkpoint_kind": checkpoint_manifest["kind"],
        "source_git_sha": provenance["git_sha"],
        "source_tpen_version": provenance["tpen_version"],
        "resolved_config_file": str(source / resolved_name),
    }


def build_diagnostic_config(
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    results_root: str | Path,
    plan_attempt_id: str,
) -> tuple[DictConfig, str]:
    """Merge the frozen overlay, select the exact row tasks, and inject identity."""

    repo_root = STUDY_DIR.parents[2]
    base_path = repo_root / str(manifest["base_eval_config"])
    overlay_path = repo_root / str(manifest["overlay_config"])
    if plan_stage.file_sha256(base_path) != manifest["base_eval_config_sha256"]:
        raise DiagnosticDriverError("base evaluation config changed after planning")
    if plan_stage.file_sha256(overlay_path) != manifest["overlay_config_sha256"]:
        raise DiagnosticDriverError("diagnostic overlay changed after planning")
    cfg = OmegaConf.merge(OmegaConf.load(base_path), OmegaConf.load(overlay_path))

    selected_tasks = []
    for name in row["task_names"]:
        diagnostic_task = OmegaConf.select(cfg, f"diagnostic_tasks.{name}")
        base_task = OmegaConf.select(cfg, f"evaluation_tasks.{name}")
        task = diagnostic_task if diagnostic_task is not None else base_task
        if task is None:
            raise DiagnosticDriverError(f"planned task {name!r} is not declared")
        selected_tasks.append(task)
    if len(selected_tasks) != len(row["task_names"]):
        raise DiagnosticDriverError("selected diagnostic task count mismatch")
    OmegaConf.update(cfg, "evaluator.tasks", selected_tasks, merge=False)

    callbacks = list(OmegaConf.select(cfg, "callbacks", default=[]))
    callbacks.extend(OmegaConf.select(cfg, "diagnostic_callbacks", default=[]))
    OmegaConf.update(cfg, "callbacks", callbacks, merge=False)

    factor_arm = row.get("factor_arm") or {
        "label": "baseline",
        "b_ee": 1.0,
        "c_electron_nucleus": 1.0,
        "d_electron_nucleus": 1.0,
    }
    scientific_values = {
        "diagnostic.row_kind": row["kind"],
        "diagnostic.protocol": row["protocol"],
        "diagnostic.comparison_kind": row["comparison_kind"],
        "diagnostic.checkpoint_label": row["checkpoint_label"],
        "diagnostic.checkpoint_step": row["checkpoint_step"],
        "diagnostic.n_walkers": row["n_walkers"],
        "diagnostic.n_draws": row["n_draws"],
        "diagnostic.burn_in": row["burn_in"],
        "diagnostic.stride": row["stride"],
        "diagnostic.record_capacity": row["record_capacity"],
        "diagnostic.diagnostic_samples": row["diagnostic_samples"],
        "diagnostic.factor_arm": factor_arm,
        "evaluation.seed": row["seed"],
        "runtime.seed": row["seed"],
        "evaluation_sampler.seed": row["seed"],
        "evaluation_sampler.n_walkers": row["n_walkers"],
        "evaluation_sampler.burn_in": row["burn_in"],
        "evaluation_sampler.n_steps": row["stride"],
    }
    for key, value in scientific_values.items():
        OmegaConf.update(cfg, key, value, merge=False)
    _apply_scale_overrides(cfg, manifest)

    binding = reconcile_checkpoint(row)
    config_sha = _config_identity_hash(cfg, manifest=manifest, row=row)
    result_dir = layout.row_dir(
        results_root,
        layout.STAGE_EVAL,
        str(row["row_id"]),
        plan_attempt_id,
    )
    identity_values = {
        "trajectory_identity.stage": layout.STAGE_EVAL,
        "trajectory_identity.run_id": row["row_id"],
        "trajectory_identity.attempt_id": plan_attempt_id,
        "trajectory_identity.evaluator_id": EVALUATOR_ID,
        "trajectory_identity.checkpoint_file": binding["checkpoint_model_file"],
        "trajectory_identity.config_sha256": config_sha,
        "load.path": binding["checkpoint_dir"],
        "load.replay_semantics.source_git_sha": binding["source_git_sha"],
        "load.replay_semantics.source_tpen_version": binding["source_tpen_version"],
        "load.replay_semantics.checkpoint_schema_version": binding[
            "checkpoint_schema_version"
        ],
        "load.replay_semantics.checkpoint_kind": binding["checkpoint_kind"],
        "load.replay_semantics.checkpoint_model_sha256": binding["hashes"]["model"],
        "run.root": str(result_dir),
        "run.run_id": row["row_id"],
        "run.layout": "flat",
    }
    for key, value in identity_values.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    return cfg, config_sha


def _apply_scale_overrides(cfg: DictConfig, manifest: Mapping[str, Any]) -> None:
    """Apply only the manifest-recorded smoke reductions to the same task graph."""

    if manifest["scale"] != "smoke":
        return
    scale = manifest["scale_overrides"]
    refinement = int(scale["atlas_max_refinement_steps"])
    radii = list(scale["atlas_radii"])
    for task in ("he_en_numerical_atlas", "he_ee_ideal_vs_executed_numerical_atlas"):
        OmegaConf.update(
            cfg,
            f"evaluation_tasks.{task}.generator.max_refinement_steps",
            refinement,
            merge=False,
        )
    for task in (
        "he_one_electron_tail_atlas",
        "he_center_of_mass_tail_atlas",
        "he_angular_shell_atlas",
    ):
        OmegaConf.update(
            cfg,
            f"evaluation_tasks.{task}.generator.radii",
            radii,
            merge=False,
        )


def _config_identity_hash(
    cfg: DictConfig,
    *,
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
) -> str:
    """Hash the pre-identity scientific config without a self-reference."""

    container = OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False)
    return plan_stage.canonical_sha256(
        {
            "config": container,
            "plan_sha256": manifest["plan_sha256"],
            "evaluation_git_sha": manifest["evaluation_git_sha"],
            "row_id": row["row_id"],
            "checkpoint_model_sha256": row["checkpoint_model_sha256"],
        }
    )


def run_row(
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    results_root: str | Path,
    plan_attempt_id: str,
    launch_attempt_id: str,
    device_reader: DeviceReader = torch_device_name,
    runner: ConfigRunner | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Perform all preflight reconciliation and execute exactly one row."""

    environ = os.environ if environ is None else environ
    job_id = require_scheduler(environ)
    require_evaluation_checkout(manifest)
    result_dir = layout.row_dir(
        results_root,
        layout.STAGE_EVAL,
        str(row["row_id"]),
        plan_attempt_id,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir = result_dir / str(row["row_id"])
    if run_dir.exists():
        raise DiagnosticDriverError(f"diagnostic run directory already exists: {run_dir}")
    verify_delivered_device(
        row,
        receipt_dir=result_dir,
        job_id=job_id,
        device_reader=device_reader,
        environ=environ,
    )
    binding = reconcile_checkpoint(row)
    _write_new_json(result_dir / CHECKPOINT_BINDING, binding)
    _write_new_json(
        result_dir / "row.json",
        {
            "schema": "he-v1-diagnostic-row/v1",
            "row": dict(row),
            "plan_attempt_id": plan_attempt_id,
            "plan_sha256": manifest["plan_sha256"],
            "launch_attempt_id": launch_attempt_id,
            "evaluation_git_sha": manifest["evaluation_git_sha"],
            "job_id": job_id,
        },
    )
    cfg, _config_sha = build_diagnostic_config(
        manifest,
        row,
        results_root=results_root,
        plan_attempt_id=plan_attempt_id,
    )
    configured_runner = runner if runner is not None else _run_from_config
    config_path = f"{manifest['base_eval_config']}+{manifest['overlay_config']}"
    return int(
        configured_runner(
            cfg,
            config_path=config_path,
            command=f"he-v1-diagnostic-v1 {row['row_id']}",
        )
    )


def _run_from_config(cfg: DictConfig, *, config_path: str, command: str) -> int:
    """Call the sole sanctioned TPEN entrypoint from experiment code."""

    from tpen.run import run_from_config  # noqa: PLC0415 - sanctioned exception

    return int(run_from_config(cfg, config_path=config_path, command=command))


def _write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the in-allocation row arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--plan-attempt-id", required=True)
    parser.add_argument("--launch-attempt-id", required=True)
    parser.add_argument("--row-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Load one immutable row and execute it."""

    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    manifest = plan_stage.read_manifest(results_root, args.plan_attempt_id)
    row = plan_stage.row_by_id(manifest, args.row_id)
    return run_row(
        manifest,
        row,
        results_root=results_root,
        plan_attempt_id=args.plan_attempt_id,
        launch_attempt_id=args.launch_attempt_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOCATION_RECEIPT",
    "CHECKPOINT_BINDING",
    "DiagnosticDriverError",
    "build_diagnostic_config",
    "reconcile_checkpoint",
    "require_evaluation_checkout",
    "require_scheduler",
    "run_row",
    "torch_device_name",
    "verify_delivered_device",
]
