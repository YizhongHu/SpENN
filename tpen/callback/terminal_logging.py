"""The package terminal logging channel and its colouring.

Split out of ``tpen.callback.status`` when `tpen.callback.Status` became a
`tpen.callback.StatefulCallback`. A stateful callback must import its
``state_type`` at class-creation time, and importing `tpen.training.state`
pulls in torch, so ``tpen.callback.status`` can no longer be imported eagerly.
`tpen.run` imports `configure_terminal_logging` at module scope and must stay
torch-free, so the channel configuration lives here, where it also sits closer
to what it owns: the logger, not the callback that writes to it.
"""

from __future__ import annotations

import logging
import os
import sys

_STATUS_COLORS = {
    "run": "\033[36m",
    "train": "\033[34m",
    "completed": "\033[32m",
    "failed": "\033[31m",
}


def configure_terminal_logging(
    *,
    enabled: bool = True,
    level: str = "info",
    color: str = "auto",
    logger_name: str = "tpen",
) -> None:
    """Configure the package terminal logging channel.

    Parameters
    ----------
    enabled : bool, optional
        If ``False``, leave logging configuration unchanged.
    level : str, optional
        Logging level name.
    color : {"auto", "always", "never"}, optional
        Accepted for config validation and consistency with
        `tpen.callback.Status`.
    logger_name : str, optional
        Logger subtree to configure.
    """

    if not enabled:
        return
    validate_terminal_choice(color, name="color")
    logger = logging.getLogger(logger_name)
    logger.setLevel(_logging_level(level))
    for handler in logger.handlers:
        if getattr(handler, "_tpen_terminal_handler", False):
            handler.setLevel(_logging_level(level))
            return
    handler = logging.StreamHandler()
    handler._tpen_terminal_handler = True
    handler.setLevel(_logging_level(level))
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def validate_terminal_choice(value: str, *, name: str) -> str:
    """Return ``value`` if it is a valid terminal tri-state, else raise."""

    if value not in {"auto", "always", "never"}:
        raise ValueError(f"{name} must be one of 'auto', 'always', or 'never', got {value!r}")
    return value


def color_status_line(line: str, *, kind: str, color: str) -> str:
    """Wrap one status line in the ANSI colour for its ``kind``, if enabled."""

    if not _color_enabled(color):
        return line
    prefix = _STATUS_COLORS.get(kind)
    if prefix is None:
        return line
    return f"{prefix}{line}\033[0m"


def _logging_level(level: str) -> int:
    value = getattr(logging, str(level).upper(), None)
    if not isinstance(value, int):
        raise ValueError(f"Unsupported logging level {level!r}")
    return value


def _color_enabled(color: str) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if color == "always":
        return True
    if color == "never":
        return False
    if os.environ.get("SLURM_JOB_ID"):
        return False
    return sys.stderr.isatty()


__all__ = ["color_status_line", "configure_terminal_logging", "validate_terminal_choice"]
