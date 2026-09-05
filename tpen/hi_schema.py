"""The closed helium-importance train schema.

This module owns the *policy*: which sections a helium-importance training
configuration may declare, which key families it may never contain, and which
callbacks it may install. The sweep mechanism that enforces it lives in
:mod:`tpen.config_schema`.

Opting in
---------
The firewall is selected by the configuration, through a top-level ``schema``
key holding :data:`HI_TRAIN_SCHEMA`::

    schema: tpen.hi.train.v1

It is deliberately NOT applied to every configuration. The repository retains
frozen historical fixtures -- most directly
``experiments/atomistic/he-v1/configs/train.yaml``, whose ``system`` block
carries a ``reference_energy`` -- and the plan of record designates those
records preserved rather than edited. A globally applied firewall would force a
choice between breaking a completed study's provenance and weakening the rule.
Opt-in keeps the helium-importance schema genuinely closed while leaving every
historical record exactly as it was.

Why a training configuration may not hold a reference
-----------------------------------------------------
A reference energy reachable during training is an arm-selection hazard: any
quantity that can be compared against it mid-run can be used, deliberately or
not, to choose between arms on accuracy. The separately loaded evaluation
manifest is the only permitted reference holder, so the comparison can only
happen after the training configuration is frozen.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

from tpen.config_schema import (
    ClosedSchemaError,
    ForbiddenSurface,
    Rejection,
    SchemaPolicy,
    iter_nodes,
    sweep_environment,
    sweep_raw,
    sweep_resolved,
)

__all__ = [
    "ADMITTED_CALLBACK_TARGETS",
    "HI_TRAIN_POLICY",
    "HI_TRAIN_SCHEMA",
    "SCHEMA_KEY",
    "declared_schema",
    "validate_hi_train_config",
]


# Durable external identifier: it is written into config files, so it is not
# renameable without rewriting every configuration that opts in.
HI_TRAIN_SCHEMA = "tpen.hi.train.v1"

# Top-level key a configuration uses to select its schema.
SCHEMA_KEY = "schema"


# ---------------------------------------------------------------------------
# Forbidden key families
# ---------------------------------------------------------------------------
# Each family names a class of value that must not be reachable from a training
# configuration. Token sets are deliberately narrow: see the token-versus-
# substring discussion in :mod:`tpen.config_schema` for why a wider set would
# reject ordinary words while still looking like a working firewall.

REFERENCE_SURFACE = ForbiddenSurface(
    name="reference",
    tokens=frozenset({"reference", "baseline", "truth"}),
    reason=(
        "a training configuration may not hold or reach a reference value; the "
        "separately loaded evaluation manifest is the only permitted reference holder"
    ),
)

GAP_SURFACE = ForbiddenSurface(
    name="gap",
    tokens=frozenset({"gap", "gaps"}),
    reason=(
        "an energy gap is a derived reference quantity and belongs to evaluation, "
        "not to a training configuration"
    ),
)

BAND_SURFACE = ForbiddenSurface(
    name="band",
    tokens=frozenset({"band", "bands"}),
    reason=(
        "an accuracy band encodes a selection rule; selection belongs to the "
        "confirmation lane, not to a training configuration"
    ),
)

CONTINUATION_SURFACE = ForbiddenSurface(
    name="continuation",
    tokens=frozenset({"continuation", "continue"}),
    reason=(
        "the continuation ladder is retired in favour of train/eval/test; a "
        "decide-then-extend surface must not be configurable here. Note that "
        "checkpoint RESUME is unaffected and remains permitted -- resuming an "
        "interrupted run is recovery, whereas continuation is selection"
    ),
)


# ---------------------------------------------------------------------------
# Closed section set
# ---------------------------------------------------------------------------
HI_TRAIN_SECTIONS = frozenset(
    {
        SCHEMA_KEY,
        "experiment",
        "run",
        "runtime",
        "system",
        "atoms",
        "runner",
        "model",
        "hamiltonian_terms",
        "sampler",
        "optimizer",
        "trainer",
        "callbacks",
        "loggers",
    }
)


# ---------------------------------------------------------------------------
# Admitted callbacks
# ---------------------------------------------------------------------------
# An ALLOWLIST rather than a denylist, because the hazard here is a callback
# nobody thought to forbid. A denylist admits every callback written after it,
# including a future in-training evaluation callback that reports an energy
# against a reference -- exactly the surface this schema exists to close.
#
# Membership means "carries no reference and reports no energy". Notably absent:
# any diagnostic that accepts a ``reference_energy`` (see
# ``tpen/diagnostics/energy.py``), and any in-training evaluation callback,
# which the plan of record defers pending a separate approval of its second
# RNG / dual payload / lag confound.
ADMITTED_CALLBACK_TARGETS = frozenset(
    {
        # Provenance and run bookkeeping.
        "tpen.callback.ConfigSnapshot",
        "tpen.callback.ResolvedConfigSnapshot",
        "tpen.callback.Metadata",
        "tpen.callback.Status",
        "tpen.callback.ArtifactIndex",
        "tpen.callback.FailureLog",
        # Correctness and health guards. None reports an energy value.
        "tpen.callback.DataIntegrity",
        "tpen.callback.GradientStats",
        "tpen.callback.SamplerHealth",
        "tpen.callback.FactorScalars",
        "tpen.callback.RuntimeEquivariance",
        # Durable state and resource accounting.
        "tpen.callback.Checkpoint",
        "tpen.callback.ResourceUsage",
        # Timing observability.
        "tpen.callback.RunTiming",
        "tpen.callback.TrainPhaseTiming",
        "tpen.callback.TrainStepTiming",
        "tpen.callback.DiagnosticTiming",
    }
)


HI_TRAIN_POLICY = SchemaPolicy(
    name=HI_TRAIN_SCHEMA,
    forbidden_surfaces=(
        REFERENCE_SURFACE,
        GAP_SURFACE,
        BAND_SURFACE,
        CONTINUATION_SURFACE,
    ),
    allowed_sections=HI_TRAIN_SECTIONS,
    # ``oc.env`` reads the process environment and ``now`` reads the clock.
    # Either makes the resolved configuration a function of something other
    # than the file and its overrides, so two ranks could resolve the same file
    # differently. ``oc.decode`` is included because it evaluates arbitrary text
    # that may itself be built from environment interpolation.
    forbidden_resolvers=frozenset({"oc.env", "env", "now", "oc.decode"}),
)


def declared_schema(cfg: Any) -> str | None:
    """Return the schema a configuration opts in to, if any.

    Parameters
    ----------
    cfg : Any
        A ``DictConfig`` or plain mapping.

    Returns
    -------
    str or None
        The value of the top-level ``schema`` key, or ``None`` when the
        configuration declares none.
    """

    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, SCHEMA_KEY, default=None)
    elif isinstance(cfg, Mapping):
        value = cfg.get(SCHEMA_KEY)
    else:
        return None
    return None if value is None else str(value)


def _sweep_callbacks(resolved_tree: Any) -> list[Rejection]:
    """Reject any configured callback outside the admitted set.

    Checked on the resolved tree only. A callback's ``_target_`` may itself be
    an interpolation, in which case the raw tree holds ``"${...}"`` rather than
    a class path and has nothing to compare against the allowlist.
    """

    if not isinstance(resolved_tree, Mapping):
        return []
    callbacks = resolved_tree.get("callbacks")
    if callbacks is None:
        return []

    rejections: list[Rejection] = []
    for path, key, value in iter_nodes({"callbacks": callbacks}):
        if key != "_target_" or not isinstance(value, str):
            continue
        # Only the callback's own ``_target_`` is a callback identity; a nested
        # ``_target_`` is a constructor argument (a schedule, a payload, a
        # probe) and is governed by its owning callback, not by this allowlist.
        owner = path.rsplit("._target_", 1)[0]
        if owner.count(".") != 0 or not owner.startswith("callbacks["):
            continue
        if value not in ADMITTED_CALLBACK_TARGETS:
            rejections.append(
                Rejection(
                    rule="unadmitted-callback",
                    tree="resolved",
                    path=path,
                    detail=(
                        f"callback {value!r} is not in the admitted set. Admission means the "
                        "callback carries no reference and reports no energy. Add it to "
                        "tpen.hi_schema.ADMITTED_CALLBACK_TARGETS only after confirming that"
                    ),
                )
            )
    return rejections


def validate_hi_train_config(cfg: DictConfig, *, env: Mapping[str, str] | None = None) -> None:
    """Refuse a helium-importance training configuration that violates its schema.

    Parameters
    ----------
    cfg : DictConfig
        The loaded training configuration, before any construction.
    env : Mapping of str to str, optional
        The launch environment to audit. Defaults to ``os.environ``.

    Raises
    ------
    ClosedSchemaError
        When the configuration declares :data:`HI_TRAIN_SCHEMA` and violates it.
        Every finding is reported, not only the first.

    Notes
    -----
    A configuration that declares no schema, or a different one, is returned
    unvalidated -- see the module docstring on why the firewall is opt-in.

    The launch environment is audited alongside the configuration because the
    reference-energy firewall names it as one of the surfaces a reference must
    not enter, next to training configs, runner inputs, checkpoint metadata and
    decision logic. A config-only check would leave one of the five open.
    """

    if declared_schema(cfg) != HI_TRAIN_SCHEMA:
        return

    environment = os.environ if env is None else env
    raw_tree = OmegaConf.to_container(cfg, resolve=False)

    # The raw sweep runs FIRST and unconditionally. Resolution can fail, and if
    # the raw findings were collected after it, a single broken interpolation
    # would suppress every forbidden reference the raw tree already showed --
    # the finding and the thing that hides it would share a failure domain. A
    # config that both fails to resolve and carries a reference must report the
    # reference, because that is the finding that stops a run from happening.
    rejections = list(sweep_raw(raw_tree, HI_TRAIN_POLICY))
    rejections.extend(sweep_environment(environment, HI_TRAIN_POLICY))

    # Resolution failure is itself a rejection, which keeps every
    # preconstruction failure a single exception type for the caller.
    try:
        resolved_tree = OmegaConf.to_container(cfg, resolve=True)
    except Exception as error:  # noqa: BLE001 - OmegaConf raises several unrelated types
        rejections.append(
            Rejection(
                rule="unresolvable",
                tree="resolved",
                path="<root>",
                detail=f"configuration does not resolve: {type(error).__name__}: {error}",
            )
        )
        raise ClosedSchemaError(rejections) from error

    rejections.extend(sweep_resolved(resolved_tree, HI_TRAIN_POLICY))
    rejections.extend(_sweep_callbacks(resolved_tree))
    if rejections:
        raise ClosedSchemaError(rejections)
