"""Shared pytest configuration and fixtures for the SpENN test suite."""

from __future__ import annotations

import logging

import pytest

# Loggers that the terminal-status callback mutates globally
# (`configure_terminal_logging` adds a StreamHandler and sets ``propagate=False``).
# Without isolation, a test that enables terminal logging leaks
# ``propagate=False`` into later caplog-based unit tests, which then capture no
# records. Reset these loggers to a clean, propagating state before each test
# and restore the pre-test snapshot afterwards.
_ISOLATED_LOGGERS = ("spenn", "spenn.status")


def _drop_terminal_handlers(logger: logging.Logger) -> None:
    logger.handlers[:] = [
        handler for handler in logger.handlers if not getattr(handler, "_spenn_terminal_handler", False)
    ]


@pytest.fixture(autouse=True)
def _restore_spenn_logger_state():
    """Give each test a clean SpENN logger tree and restore it afterwards."""

    saved = {}
    for name in _ISOLATED_LOGGERS:
        logger = logging.getLogger(name)
        saved[name] = (list(logger.handlers), logger.level, logger.propagate)
        # Clear any leaked terminal handler / propagate=False from a prior test.
        _drop_terminal_handlers(logger)
        logger.propagate = True
    try:
        yield
    finally:
        for name, (handlers, level, propagate) in saved.items():
            logger = logging.getLogger(name)
            logger.handlers[:] = handlers
            logger.setLevel(level)
            logger.propagate = propagate
