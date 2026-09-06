"""Run one He-cutover fixed-checkpoint evaluation chain."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

# Siblings are loaded study-scoped, not by bare import: experiments/ has
# several same-named modules and the first study loaded would otherwise own
# the bare name for every study after it. See experiments/toolkit/study_imports.py.
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

_TPEN_REPO_ROOT = _TpenPath(__file__).resolve().parents[3]
if str(_TPEN_REPO_ROOT) not in _tpen_sys.path:
    _tpen_sys.path.insert(0, str(_TPEN_REPO_ROOT))

from experiments.toolkit.study_imports import sibling  # noqa: E402

cutover_strata = sibling(__file__, 'cutover_strata')
_tpen_run_train_row = sibling(__file__, 'run_train_row')
require_allocation = _tpen_run_train_row.require_allocation


def output_overrides(row: Mapping[str, Any]) -> list[str]:
    """Return overrides that realize the plan-owned row directory."""

    result_dir = Path(str(row["result_dir"]))
    if result_dir.name != str(row["row_id"]):
        raise ValueError("planned eval result_dir must end with row_id")
    return [f"run.root={result_dir.parent}", f"run.run_id={row['row_id']}", "run.layout=flat"]


def run(row: Mapping[str, Any], *, plan_attempt_id: str, environ: Mapping[str, str] | None = None, device_reader=None, runner=None) -> int:
    """Reuse He-v1 canary configuration and checkpoint identity machinery."""

    environ = os.environ if environ is None else environ
    require_allocation(environ)
    import hev1

    device_reader = device_reader or hev1.driver.torch_device_name
    cutover_strata.check_delivered_device(facility=str(row["facility"]), stratum=str(row["resources"]["stratum"]), delivered=device_reader())
    checkpoint = hev1.eval_stage.require_complete_checkpoint(row["checkpoint_dir"])
    config_path = Path(__file__).resolve().parents[3] / str(row["config"])
    base_overrides = [*output_overrides(row), f"load.path={checkpoint}"]
    identity_hash = hev1.eval_stage.config_identity_hash(config_path, base_overrides, identity_values={key: row[key] for key in ("task_names", "n_walkers", "n_draws", "burn_in", "discard_draws", "stride", "chunk_size")})
    semantic = hev1.eval_stage.checkpoint_replay_semantics_overrides(checkpoint, binding=None)
    identity = hev1.eval_stage.trajectory_identity_overrides(row, plan_attempt_id=plan_attempt_id, checkpoint_dir=checkpoint, config_sha256=identity_hash)
    cfg = hev1.driver.build_config(config_path, [*base_overrides, *identity, *semantic], checked=[base_overrides[-1], *identity, *semantic])
    hev1.eval_stage.configure_canary_evaluation(cfg, row)
    if runner is None:
        from tpen.run import run_from_config
        runner = run_from_config
    return int(runner(cfg, config_path=str(config_path), command=f"he-cutover eval {row['row_id']}"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-json", required=True)
    parser.add_argument("--plan-attempt-id", required=True)
    args = parser.parse_args(argv)
    return run(json.loads(args.row_json), plan_attempt_id=args.plan_attempt_id)


if __name__ == "__main__":
    raise SystemExit(main())
