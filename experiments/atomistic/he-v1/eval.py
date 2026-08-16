"""Run one planned He-v1 fixed-model evaluation chain inside its allocation.

Each row is one independent chain over one predeclared retained checkpoint of
one training seed. The checkpoint it restores is passed explicitly and must
exist as a COMPLETE checkpoint directory before the run starts: an evaluation
that silently restored a different (or partial) checkpoint would produce a
number attributed to the wrong model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import driver  # noqa: E402

#: Marker written by the checkpoint writer when a directory is fully written.
#: Spelled here rather than imported because ``experiments/`` may not import
#: ``tpen``; the drift is guarded by a test.
COMPLETE_MARKER = "COMPLETE"


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
    return driver.run_row(
        row,
        results_root=results_root,
        plan_attempt_id=args.plan_attempt_id,
        launch_attempt_id=args.launch_attempt_id,
        extra_overrides=[f"load.path={checkpoint_dir}"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
