"""Run one planned He-v1 fixed-model evaluation chain inside its allocation.

Each row is one independent chain over one predeclared retained checkpoint of
one training seed. The checkpoint it restores is passed explicitly and must
exist as a COMPLETE checkpoint directory before the run starts: an evaluation
that silently restored a different (or partial) checkpoint would produce a
number attributed to the wrong model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import driver  # noqa: E402

#: Marker written by the checkpoint writer when a directory is fully written.
#: Spelled here rather than imported because ``experiments/`` may not import
#: ``tpen``; the drift is guarded by a test.
COMPLETE_MARKER = "COMPLETE"

#: Model weights file inside a checkpoint directory. The A6-C trajectory
#: summary content-hashes THIS FILE, while ``load.path`` restores from the
#: DIRECTORY that contains it. The asymmetry is real and has already cost this
#: lane one job: passing the directory where the file is wanted raises
#: ``IsADirectoryError`` inside the summary, after the chain has been sampled.
CHECKPOINT_MODEL_FILE = "model.pt"

#: Evaluator identity recorded in the trajectory join. Constant for this study.
EVALUATOR_ID = "tpen_he_v1_eval"


def require_checkpoint_model_file(checkpoint_dir: str | Path) -> Path:
    """Return the ``model.pt`` inside a complete checkpoint directory.

    Raises
    ------
    driver.DriverError
        If the weights file is absent. Checked here, before the allocation
        spends anything, because the summary that needs it runs LAST: a missing
        or mis-typed checkpoint path cannot fail early on its own and would
        otherwise surface only after the whole chain has been sampled.
    """

    model_file = Path(checkpoint_dir) / CHECKPOINT_MODEL_FILE
    if not model_file.is_file():
        raise driver.DriverError(
            f"checkpoint {checkpoint_dir} has no {CHECKPOINT_MODEL_FILE}; the trajectory "
            "statistics identity content-hashes that file, and restoring from the "
            "directory does not supply it"
        )
    return model_file


def config_identity_hash(config_path: str | Path, overrides: Sequence[str]) -> str:
    """Return the deterministic config hash recorded in the join identity.

    Computed over the config FILE BYTES plus the row's overrides, both before
    injection. Hashing the resolved document after the identity is injected
    would be self-referential -- the hash would be an input to the thing it
    describes -- so the inputs are deliberately the two things that fully
    determine the run and do not depend on the hash.
    """

    digest = hashlib.sha256()
    digest.update(Path(config_path).read_bytes())
    digest.update(b"\0overrides\0")
    digest.update(json.dumps(sorted(str(item) for item in overrides)).encode("utf-8"))
    return digest.hexdigest()


def trajectory_identity_overrides(
    row: Mapping[str, object],
    *,
    plan_attempt_id: str,
    checkpoint_dir: Path,
    config_sha256: str,
) -> list[str]:
    """Return the six A6-C identity overrides for one evaluation row.

    Every field the producer requires is supplied explicitly. The config
    declares them ``???`` so a forgotten one fails at resolution rather than
    producing a sidecar that cannot be joined.
    """

    return [
        f"trajectory_identity.stage={row['stage']}",
        f"trajectory_identity.run_id={row['row_id']}",
        f"trajectory_identity.attempt_id={plan_attempt_id}",
        f"trajectory_identity.evaluator_id={EVALUATOR_ID}",
        f"trajectory_identity.checkpoint_file={require_checkpoint_model_file(checkpoint_dir)}",
        f"trajectory_identity.config_sha256={config_sha256}",
    ]


def require_complete_checkpoint(path: str | Path) -> Path:
    """Return ``path`` once it is a complete checkpoint directory.

    Raises
    ------
    driver.DriverError
        If the directory is missing, or is missing its manifest or completion
        marker. A partially written checkpoint restores a model nobody planned.
    """

    checkpoint_dir = Path(path)
    if not checkpoint_dir.is_dir():
        raise driver.DriverError(f"checkpoint directory does not exist: {checkpoint_dir}")
    missing = [
        name
        for name in ("manifest.json", COMPLETE_MARKER)
        if not (checkpoint_dir / name).is_file()
    ]
    if missing:
        raise driver.DriverError(
            f"checkpoint {checkpoint_dir} is incomplete; missing: {missing}"
        )
    return checkpoint_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse evaluation-driver arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    driver.add_common_arguments(parser)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Complete checkpoint directory this chain restores.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one evaluation chain."""

    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    row = driver.load_row(results_root, args.plan_attempt_id, args.row_id, kind="eval")
    checkpoint_dir = require_complete_checkpoint(args.checkpoint_dir)
    config_sha256 = config_identity_hash(
        driver.STUDY_DIR.parents[2] / str(row["config"]),
        [str(item) for item in row["overrides"]],
    )
    return driver.run_row(
        row,
        results_root=results_root,
        plan_attempt_id=args.plan_attempt_id,
        launch_attempt_id=args.launch_attempt_id,
        # `load.path` is the DIRECTORY; the identity's `checkpoint_file` is the
        # model.pt inside it. Both are supplied because they are different
        # paths for different consumers, not two spellings of one.
        extra_overrides=[
            f"load.path={checkpoint_dir}",
            *trajectory_identity_overrides(
                row,
                plan_attempt_id=args.plan_attempt_id,
                checkpoint_dir=checkpoint_dir,
                config_sha256=config_sha256,
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
