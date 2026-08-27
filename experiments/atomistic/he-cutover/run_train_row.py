"""Run one He-cutover training row inside Slurm or PBS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cutover_strata
import hev1


def require_allocation(environ: Mapping[str, str] | None = None) -> str:
    """Return the Slurm or PBS allocation id, refusing a login-node run."""

    environ = os.environ if environ is None else environ
    job_id = str(environ.get("SLURM_JOB_ID") or environ.get("PBS_JOBID") or "").strip()
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID and PBS_JOBID are empty; allocation required")
    return job_id


def configure_smoke_training(cfg: Any, row: Mapping[str, Any]) -> Any:
    """Apply only the three declared smoke-scale mutations."""

    cfg.trainer.max_steps = int(row["max_steps"])
    cfg.sampler.n_walkers = int(row["n_walkers"])
    checkpoints = [callback for callback in cfg.callbacks if str(callback.get("_target_")) == "tpen.callback.Checkpoint"]
    if len(checkpoints) != 1:
        raise RuntimeError("training config must declare exactly one Checkpoint callback")
    checkpoints[0].every_n_steps = int(row["max_steps"])
    return cfg


def run(row: Mapping[str, Any], *, plan_attempt_id: str, environ: Mapping[str, str] | None = None, device_reader=hev1.driver.torch_device_name, runner=None) -> int:
    """Validate allocation/device and invoke the sanctioned TPEN entrypoint."""

    environ = os.environ if environ is None else environ
    require_allocation(environ)
    delivered = device_reader()
    cutover_strata.check_delivered_device(facility=str(row["facility"]), stratum=str(row["resources"]["stratum"]), delivered=delivered)
    config_path = Path(__file__).resolve().parents[2] / str(row["config"])
    overrides = [f"runtime.seed={row['seed']}", f"run.root={row['result_dir']}", f"run.run_id={row['row_id']}", "run.layout=flat"]
    cfg = hev1.driver.build_config(config_path, overrides, checked=[overrides[0]])
    configure_smoke_training(cfg, row)
    if runner is None:
        from tpen.run import run_from_config
        runner = run_from_config
    return int(runner(cfg, config_path=str(config_path), command=f"he-cutover train {row['row_id']}"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-json", required=True)
    parser.add_argument("--plan-attempt-id", required=True)
    args = parser.parse_args(argv)
    return run(json.loads(args.row_json), plan_attempt_id=args.plan_attempt_id)


if __name__ == "__main__":
    raise SystemExit(main())
