"""Small naming helpers shared across SpENN."""

from __future__ import annotations

import re

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def camel_to_snake(name: str) -> str:
    """Return the snake_case form of a CamelCase name.

    Inserts boundaries between lower/upper runs and before the final word of an
    acronym (``HTMLParser`` -> ``html_parser``), then lowercases and trims
    surrounding underscores (``KineticEnergy`` -> ``kinetic_energy``,
    ``_ConstantTerm`` -> ``constant_term``).
    """

    return _CAMEL_BOUNDARY.sub("_", name).strip("_").lower()


__all__ = ["camel_to_snake"]
