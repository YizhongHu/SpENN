"""Run one planned He-v1 training row inside its allocation.

Thin by construction: it resolves the row from the durable manifest, lets
:mod:`driver` verify the allocation and the delivered card, and starts the
configured run through ``tpen.run.run_from_config``. It expands no grid, picks
no seed, and never resumes -- a row that does not finish inside its wall time
fails, because `production-grid-v0` forbids relying on resume until its
semantics are independently demonstrated.
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse training-driver arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    driver.add_common_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one training row."""

    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    row = driver.load_row(results_root, args.plan_attempt_id, args.row_id, kind="train")
    return driver.run_row(
        row,
        results_root=results_root,
        plan_attempt_id=args.plan_attempt_id,
        launch_attempt_id=args.launch_attempt_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
