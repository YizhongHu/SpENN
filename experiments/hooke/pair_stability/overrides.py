"""Command override rewriting helpers for pair-stability launchers."""

from __future__ import annotations

from typing import Sequence


def rewrite_cli_overrides(command: Sequence[str], overrides: dict[str, object]) -> list[str]:
    """Return ``command`` with exact ``key=`` entries replaced by ``overrides``."""

    keys = set(overrides)
    rewritten = []
    for part in command:
        text = str(part)
        key, separator, _value = text.partition("=")
        if separator and key in keys:
            continue
        rewritten.append(text)
    rewritten.extend(f"{key}={value}" for key, value in overrides.items())
    return rewritten
