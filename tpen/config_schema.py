"""Closed-schema preconstruction firewall for TPEN run configurations.

This module owns the *mechanism*: how a run configuration is swept for
forbidden surfaces and unknown fields before anything is constructed. It does
not own any study's policy -- the helium-importance train schema declares which
surfaces are forbidden and which sections are closed, and passes that policy in.

Why the sweep runs over two trees
---------------------------------
A forbidden value can hide in either of two places, and checking one tree
cannot see the other:

``raw``
    The configuration as written, with interpolations left as literal
    ``"${...}"`` text. A key that only *appears* after resolution is invisible
    here, but the interpolation *expressions* themselves are visible -- and an
    expression is how rank-local data enters a configuration
    (``${oc.env:RANK}``). Only the raw tree can catch that.

``resolved``
    The configuration Hydra will actually instantiate from. A forbidden literal
    reached through a chain of interpolations, or supplied by a nested default,
    is visible only here.

So :func:`sweep` runs over both and labels every rejection with the tree it
came from.

Why token matching rather than substring matching
-------------------------------------------------
``"reference" in "preference"`` is true, and ``"band" in "bandwidth"`` is true.
A substring rule would reject innocent keys and -- worse -- would look correct
while doing so, because its rejections are still *rejections* and nobody
inspects a firewall's false positives as closely as its false negatives. Keys
are therefore split into lowercase alphanumeric tokens and matched whole:
``energy_band`` and ``energyBand`` both tokenize to ``("energy", "band")`` and
match, while ``bandwidth`` tokenizes to ``("bandwidth",)`` and does not.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ClosedSchemaError",
    "ForbiddenSurface",
    "Rejection",
    "SchemaPolicy",
    "iter_nodes",
    "sweep",
    "tokens_of",
]


# Split on every run of characters that is not a lowercase letter or digit.
# The name is lowercased first, so ``camelCase`` boundaries are recovered by the
# separate pass in :func:`tokens_of` rather than by this pattern.
_NON_TOKEN = re.compile(r"[^0-9a-z]+")

# Insert a separator at a lower-to-upper transition so ``referenceEnergy``
# tokenizes the same way ``reference_energy`` does. Applied before lowercasing.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# An OmegaConf interpolation expression, e.g. ``${oc.env:RANK}`` or
# ``${system.spatial_dim}``. Only the raw tree contains these; the resolved tree
# has had them replaced by values.
_INTERPOLATION = re.compile(r"\$\{([^}]*)\}")


def tokens_of(name: object) -> tuple[str, ...]:
    """Split a configuration key into lowercase word tokens.

    Parameters
    ----------
    name : object
        A configuration key or other identifier. Coerced with :func:`str`.

    Returns
    -------
    tuple of str
        The non-empty lowercase tokens of ``name``, in order.

    Examples
    --------
    >>> tokens_of("reference_energy")
    ('reference', 'energy')
    >>> tokens_of("referenceEnergy")
    ('reference', 'energy')
    >>> tokens_of("bandwidth")
    ('bandwidth',)
    """

    spaced = _CAMEL_BOUNDARY.sub("_", str(name))
    return tuple(token for token in _NON_TOKEN.split(spaced.lower()) if token)


@dataclass(frozen=True)
class ForbiddenSurface:
    """One named family of configuration keys a closed schema refuses.

    Parameters
    ----------
    name : str
        Short identifier for the family, e.g. ``"reference"``. Reported on every
        rejection so a failure names the rule it broke rather than only the key.
    tokens : frozenset of str
        Key tokens that trigger the rejection. A key matches when *any* of its
        tokens is in this set; see the module docstring for why matching is on
        whole tokens.
    reason : str
        Why the family is forbidden. Included verbatim in the rejection message,
        because the person who trips a firewall is rarely the person who wrote it.
    """

    name: str
    tokens: frozenset[str]
    reason: str

    def matches(self, key: object) -> bool:
        """Return whether ``key`` belongs to this forbidden family."""

        return bool(self.tokens.intersection(tokens_of(key)))


@dataclass(frozen=True)
class Rejection:
    """One reason a configuration was refused.

    Attributes
    ----------
    rule : str
        Which check failed, e.g. ``"forbidden-surface:reference"`` or
        ``"unknown-field"``.
    tree : str
        ``"raw"`` or ``"resolved"`` -- which of the two swept trees the offending
        node was found in. A surface present in only one of them is a real and
        reportable difference, so the tree is part of the finding rather than
        incidental.
    path : str
        Dotted path to the offending node, with list positions as ``[i]``.
    detail : str
        Human-readable explanation.
    """

    rule: str
    tree: str
    path: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.tree}] {self.path}: {self.rule} -- {self.detail}"


class ClosedSchemaError(ValueError):
    """A run configuration violated its closed schema.

    Carries *every* rejection rather than only the first. A firewall that fails
    on one finding at a time turns a single fix-and-rerun cycle into as many
    cycles as there are violations, and each cycle on this project may be a
    cluster job.

    Attributes
    ----------
    rejections : tuple of Rejection
        All findings, in discovery order.
    """

    def __init__(self, rejections: Sequence[Rejection]) -> None:
        self.rejections = tuple(rejections)
        joined = "\n  ".join(str(rejection) for rejection in self.rejections)
        count = len(self.rejections)
        plural = "" if count == 1 else "s"
        super().__init__(f"closed schema rejected the configuration ({count} finding{plural}):\n  {joined}")


@dataclass(frozen=True)
class SchemaPolicy:
    """The study-specific policy a closed schema enforces.

    Parameters
    ----------
    name : str
        Policy identifier, e.g. ``"tpen.hi.train.v1"``. This is the string a
        configuration opts in with, so it is a durable external identifier.
    forbidden_surfaces : tuple of ForbiddenSurface
        Key families refused anywhere at any depth in either tree.
    allowed_sections : frozenset of str
        Closed set of permitted top-level section names. A top-level key outside
        this set is an unknown field.
    forbidden_resolvers : frozenset of str
        Interpolation resolver names refused in the raw tree. These are the
        mechanism by which process-local facts (environment, clock, randomness)
        would otherwise reach a resolved configuration and make it differ
        between ranks.
    """

    name: str
    forbidden_surfaces: tuple[ForbiddenSurface, ...] = ()
    allowed_sections: frozenset[str] = field(default_factory=frozenset)
    forbidden_resolvers: frozenset[str] = field(default_factory=frozenset)


def iter_nodes(tree: Any, prefix: str = "") -> Iterator[tuple[str, Any, Any]]:
    """Walk a plain container tree, yielding every keyed node at every depth.

    Parameters
    ----------
    tree : Any
        A nested structure of mappings, sequences, and scalars -- the output of
        ``OmegaConf.to_container``. Not a ``DictConfig``: containers are walked
        so that no access can trigger interpolation as a side effect.
    prefix : str, optional
        Dotted path prefix for the current node.

    Yields
    ------
    path : str
        Dotted path to the node, with list positions rendered as ``[i]``.
    key : Any
        The mapping key the node was reached through, or ``None`` for a list
        element (which has a position but no name to match rules against).
    value : Any
        The node itself.

    Notes
    -----
    Strings are sequences, so they are excluded from the sequence branch
    explicitly; otherwise every string would be walked character by character.
    """

    if isinstance(tree, Mapping):
        for key, value in tree.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, key, value
            yield from iter_nodes(value, path)
    elif isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
        for index, value in enumerate(tree):
            path = f"{prefix}[{index}]"
            yield path, None, value
            yield from iter_nodes(value, path)


def _sweep_forbidden_surfaces(tree: Any, label: str, policy: SchemaPolicy) -> list[Rejection]:
    """Reject every key belonging to a forbidden family, at any depth."""

    rejections: list[Rejection] = []
    for path, key, _value in iter_nodes(tree):
        if key is None:
            continue
        for surface in policy.forbidden_surfaces:
            if surface.matches(key):
                rejections.append(
                    Rejection(
                        rule=f"forbidden-surface:{surface.name}",
                        tree=label,
                        path=path,
                        detail=f"key {key!r} is a {surface.name} surface; {surface.reason}",
                    )
                )
    return rejections


def _sweep_unknown_sections(tree: Any, label: str, policy: SchemaPolicy) -> list[Rejection]:
    """Reject top-level sections outside the policy's closed set."""

    if not policy.allowed_sections or not isinstance(tree, Mapping):
        return []
    return [
        Rejection(
            rule="unknown-field",
            tree=label,
            path=str(key),
            detail=(
                f"section {str(key)!r} is not in the closed schema; "
                f"permitted sections are {sorted(policy.allowed_sections)}"
            ),
        )
        for key in tree
        if str(key) not in policy.allowed_sections
    ]


def _sweep_forbidden_resolvers(raw_tree: Any, policy: SchemaPolicy) -> list[Rejection]:
    """Reject interpolations that read process-local state, in the raw tree only.

    A resolved tree cannot be checked for this: by then ``${oc.env:RANK}`` has
    become the value it produced, which is indistinguishable from a literal that
    was always there. This is precisely the check that makes a resolved
    configuration a pure function of the file and its overrides, and therefore
    identical on every rank.
    """

    if not policy.forbidden_resolvers:
        return []

    rejections: list[Rejection] = []
    for path, _key, value in iter_nodes(raw_tree):
        if not isinstance(value, str):
            continue
        for expression in _INTERPOLATION.findall(value):
            # ``oc.env:RANK`` -> resolver ``oc.env``; a plain node reference such
            # as ``system.spatial_dim`` has no colon and no resolver name.
            resolver, separator, _argument = expression.partition(":")
            if not separator:
                continue
            resolver = resolver.strip()
            if resolver in policy.forbidden_resolvers:
                rejections.append(
                    Rejection(
                        rule="forbidden-resolver",
                        tree="raw",
                        path=path,
                        detail=(
                            f"interpolation ${{{expression}}} reads process-local state through "
                            f"resolver {resolver!r}; the resolved configuration must be a pure "
                            "function of the config file and its overrides so that every rank "
                            "resolves it identically"
                        ),
                    )
                )
    return rejections


def sweep(raw_tree: Any, resolved_tree: Any, policy: SchemaPolicy) -> tuple[Rejection, ...]:
    """Collect every closed-schema violation in a configuration.

    Parameters
    ----------
    raw_tree : Any
        The configuration as plain containers with interpolations unresolved.
    resolved_tree : Any
        The same configuration with interpolations resolved.
    policy : SchemaPolicy
        The study policy to enforce.

    Returns
    -------
    tuple of Rejection
        Every finding, in discovery order. Empty when the configuration passes.

    Notes
    -----
    This function only *reports*. Raising is the caller's decision, so a caller
    that wants to audit a legacy configuration can list its violations without
    being stopped by the first one.
    """

    rejections: list[Rejection] = []
    rejections.extend(_sweep_unknown_sections(raw_tree, "raw", policy))
    rejections.extend(_sweep_forbidden_surfaces(raw_tree, "raw", policy))
    rejections.extend(_sweep_forbidden_resolvers(raw_tree, policy))

    # The resolved tree is swept for the same surfaces because resolution can
    # introduce a key the raw tree never spelled. Unknown top-level sections are
    # re-checked for the same reason.
    rejections.extend(_sweep_unknown_sections(resolved_tree, "resolved", policy))
    rejections.extend(_sweep_forbidden_surfaces(resolved_tree, "resolved", policy))
    return tuple(rejections)
