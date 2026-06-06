"""Alternate command-line entrypoint for configured SpENN runs."""

from __future__ import annotations

from spenn.training.run import main


if __name__ == "__main__":
    raise SystemExit(main())
