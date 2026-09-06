"""The closed helium-importance train schema.

This module owns the *policy*: which sections a helium-importance training
configuration may declare, which key families it may never contain, and which
callbacks it may install. The sweep mechanism that enforces it lives in
:mod:`tpen.config_schema`.

Declaring the schema
--------------------
The firewall is selected by a top-level ``schema`` key holding
:data:`HI_TRAIN_SCHEMA`::

    schema: tpen.hi.train.v1

For a helium-importance configuration that key is MANDATORY, not optional, and
omitting it is a loud refusal rather than a quiet pass. A closed schema that
applies only when a config remembers to ask for it is not closed -- it is
closed-if-you-remember, and the omission is silent and permanent.

Membership in the family is decided POSITIVELY, by ``experiment.name``, rather
than by defaulting the firewall on for every configuration in the repository
and exempting the rest. See the comment on :data:`HI_EXPERIMENT_NAME` for the
measurement behind that choice, and for the residual hole it leaves.

A configuration outside the family passes through untouched. That is how the
frozen historical fixtures survive without an exemption list:
``experiments/atomistic/he-v1/configs/train.yaml`` carries a
``reference_energy`` in its ``system`` block and the plan of record designates
it preserved rather than edited -- and it is ``tpen_he_v1``, so it is simply not
this family.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from omegaconf import DictConfig, OmegaConf

from tpen.config_schema import (
    ClosedSchemaError,
    canonical_digest,
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
    "ADMITTED_METHOD_TARGETS",
    "REFERENCE_MANIFEST_MODULE",
    "HI_METHOD_ROSTER",
    "MethodAvailability",
    "HI_TRAIN_POLICY",
    "HI_EXPERIMENT_NAME",
    "HI_TRAIN_SCHEMA",
    "SCHEMA_KEY",
    "canonical_train_identity",
    "declared_schema",
    "is_hi_family",
    "validate_hi_train_config",
]


# Durable external identifier: it is written into config files, so it is not
# renameable without rewriting every configuration that opts in.
HI_TRAIN_SCHEMA = "tpen.hi.train.v1"

# Top-level key a configuration uses to select its schema.
SCHEMA_KEY = "schema"

# ---------------------------------------------------------------------------
# Why declaring the schema is MANDATORY for this family, not optional
# ---------------------------------------------------------------------------
# A closed schema enforced only when a configuration remembers to ask for it is
# not closed; it is closed-if-you-remember, and the failure is silent and
# permanent. A new helium-importance train config that omits ``schema:`` would
# get ZERO enforcement and nothing anywhere would go red.
#
# So the marker is required of every configuration that IDENTIFIES ITSELF as
# helium-importance, and its absence is a loud refusal.
#
# The family is detected POSITIVELY, by experiment name, rather than by
# defaulting the firewall on for everything and exempting the rest. MEASURED
# across the repository's configs, the experiment names present are
# tpen_he_importance, tpen_he_v1, tpen_h2_v1, tpen_pair_v1, and four hooke
# configs whose name is the interpolation ``${study.name}``. Defaulting on
# would therefore refuse three other lanes' experiments, and the "narrow
# exemption list" needed to rescue them would be a register of everything that
# is NOT helium-importance -- unbounded, and growing with every new experiment
# anywhere in the repository. A positive detector grows with THIS study
# instead, and needs no exemption entry for the frozen he-v1 fixture at all,
# because that fixture is ``tpen_he_v1`` and simply is not in the family.
#
# RESIDUAL HOLE, stated rather than papered over: a configuration that omits
# BOTH the schema key AND the helium-importance experiment name is not caught
# here. That is evasion, not omission, and omission is the failure this rule
# exists to close. The repo-level test in
# ``tests/unit/test_hi_reference_separation.py`` is the second, independent net
# for the static case; neither net alone covers the other's population, which
# is why there are two.
HI_EXPERIMENT_NAME = "tpen_he_importance"


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
# DELIBERATELY BROADER THAN "target-triggered". The surface this closes is a
# stop rule keyed to a target, but the tokens refuse ANY configurable stop rule,
# because a rule cannot be inspected for what it is keyed to at schema time:
# ``trainer.stop_at: -2.9037`` names no reference yet is target-triggered, while
# ``trainer.stop_at_step: 50000`` is a budget. Refusing the family and requiring
# a schema change to re-admit one is the narrow-token doctrine applied to a
# surface whose hazard lives in the VALUE rather than the key.
#
# The over-restriction risk was measured, not assumed: no configuration under
# ``experiments/atomistic/he-importance/configs/`` declares a key carrying any
# of these tokens, and ``TestEveryHIConfigDeclaresTheSchema`` validates all of
# them, so an over-wide token set here turns the existing suite red rather than
# surfacing later as a run that cannot start.
#
# ``early_stopping`` and ``early_stop`` are both caught -- ``stopping`` and
# ``stop`` are separate tokens, so neither spelling needs its own entry.
STOP_RULE_SURFACE = ForbiddenSurface(
    name="stop-rule",
    tokens=frozenset({"stop", "stopping", "halt", "patience"}),
    reason=(
        "a training configuration may not declare a stop rule; a run that halts "
        "when it reaches a target has read the target, and a stop rule is "
        "arm-selection performed inside training rather than in the "
        "confirmation lane. Every arm runs its declared budget"
    ),
)


# The evaluation manifest's module, named as a STRING and deliberately NOT
# imported. Importing it here would put ``tpen.hi_manifest`` on the training
# path -- the exact reachability ``tests/unit/test_hi_reference_separation.py``
# asserts is absent -- so the firewall would breach the invariant it enforces.
# That test carries its own copy of this name and a control asserts the two
# agree, because a silent divergence would leave the rule below pointing at a
# module nobody loads.
REFERENCE_MANIFEST_MODULE = "tpen.hi_manifest"


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


# ---------------------------------------------------------------------------
# Admitted method coverage
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MethodAvailability:
    """One optimizer method in the study roster, and whether it may run yet.

    Parameters
    ----------
    method : str
        The study's name for the method, e.g. ``"sr"``. This is the vocabulary
        the materializer uses when it marks a cell unavailable.
    admitted : bool
        Whether a training configuration may name this method today.
    target : str or None
        The config ``_target_`` that selects it, when admitted. ``None`` whenever
        the method may not be selected, which covers two different situations
        that must not be conflated:

        - no implementation exists, and inventing a module path would make this
          table claim otherwise;
        - an implementation exists but is EXCLUDED from this study, so naming it
          would invite someone to select it.

        ``requires`` is what distinguishes them, so it has to stay accurate as
        the repository moves underneath it. SR is the live example: its
        implementation landed while this table said it was waiting for that
        implementation.
    requires : str
        What admission is waiting on. Reported verbatim when a configuration
        names the method, so the refusal says what would change it.
    """

    method: str
    admitted: bool
    target: str | None
    requires: str


# The roster is the five methods of the optimizer grid. Only Adam is admitted
# today; the other four are UNAVAILABLE, which is a different thing from absent.
# An unavailable cell must stay visibly unavailable rather than quietly become
# Adam -- a silent substitution would make a run claim it compared a method it
# never ran.
HI_METHOD_ROSTER: tuple[MethodAvailability, ...] = (
    MethodAvailability(
        method="adam",
        admitted=True,
        target="torch.optim.Adam",
        requires="admitted; the first scientific roster runs on Adam",
    ),
    MethodAvailability(
        method="sr",
        admitted=False,
        target=None,
        # EXCLUDED FROM THIS STUDY, not awaiting implementation. SR/minSR LANDED
        # on dev (Lane N, PRs #472 and #476), so `tpen/training/sr.py` exists and
        # works -- and it still cannot be a scan arm here, because it does not
        # run on two-electron models and helium is two electrons.
        #
        # The distinction matters and is why this text is not "waiting for Lane
        # N": that phrasing was true until Lane N merged, and would now send a
        # reader to satisfy a prerequisite that is ALREADY satisfied, from which
        # they would conclude the refusal is a bug. A refusal that survives its
        # stated reason misdirects harder than one with no reason at all.
        requires=(
            "EXCLUDED from the helium-importance scan on scientific grounds: SR/minSR is "
            "implemented and merged (Lane N), but does not run on two-electron models and "
            "helium is two electrons. This is not pending work and no amount of optimizer "
            "work will admit it; admitting it would need a different system"
        ),
    ),
    MethodAvailability(
        method="kfac",
        admitted=False,
        target=None,
        requires="a passing fail-closed kfac-pytorch compatibility gate; a failing gate leaves KFAC unavailable",
    ),
    MethodAvailability(
        method="spring",
        admitted=False,
        target=None,
        requires="admitted minSR first, then the thin projected-history recurrence above it",
    ),
    MethodAvailability(
        method="linear_method",
        admitted=False,
        target=None,
        requires="a separately authorized Hamiltonian-tangent design gate and dense-memory admission",
    ),
)

ADMITTED_METHOD_TARGETS = frozenset(
    entry.target for entry in HI_METHOD_ROSTER if entry.admitted and entry.target
)

# Adam coordinates that §2.7 fixes for every cell. ``lr`` and ``beta2`` are
# deliberately absent: those are the SCAN coordinates (four and two levels
# respectively), and pinning them here would make the study's own grid
# unrunnable. This table holds only what no arm may vary.
_FIXED_ADAM_COORDINATES: dict[str, object] = {
    "eps": 1e-8,
    "weight_decay": 0,
}

# The first Adam moment is fixed at .9; only the second is scanned.
_FIXED_ADAM_BETA1 = 0.9


HI_TRAIN_POLICY = SchemaPolicy(
    name=HI_TRAIN_SCHEMA,
    forbidden_surfaces=(
        REFERENCE_SURFACE,
        GAP_SURFACE,
        BAND_SURFACE,
        CONTINUATION_SURFACE,
        STOP_RULE_SURFACE,
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


def is_hi_family(cfg: Any) -> bool:
    """Return whether a configuration identifies itself as helium-importance.

    Parameters
    ----------
    cfg : Any
        A ``DictConfig`` or plain mapping.

    Returns
    -------
    bool
        ``True`` when ``experiment.name`` is :data:`HI_EXPERIMENT_NAME`.

    Notes
    -----
    Read WITHOUT resolving, so a config whose interpolations are broken is
    still recognised as belonging to the family and still refused for omitting
    the schema key. Resolving here would make an unrelated typo silently
    downgrade a helium-importance config to an unenforced one -- the finding
    and the thing that hides it would share a failure domain.
    """

    if isinstance(cfg, DictConfig):
        try:
            name = OmegaConf.select(cfg, "experiment.name", default=None)
        except Exception:  # noqa: BLE001 - a broken tree must not grant an exemption
            return False
    elif isinstance(cfg, Mapping):
        experiment = cfg.get("experiment")
        name = experiment.get("name") if isinstance(experiment, Mapping) else None
    else:
        return False
    return name == HI_EXPERIMENT_NAME


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
        # THE BOUNDARY IS UNCHANGED AND THE REASON STILL HOLDS. What changed is
        # that a nested target is no longer ungoverned: `_sweep_target_values`
        # refuses an executable target that names the evaluation reference,
        # anywhere in the tree and at any depth. See its docstring for why that
        # is a separate rule rather than a wider allowlist here.
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


def _sweep_target_values(resolved_tree: Any) -> list[Rejection]:
    """Reject an executable ``_target_`` that names the evaluation reference.

    Parameters
    ----------
    resolved_tree : Any
        Resolved configuration tree.

    Returns
    -------
    list of Rejection
        One rejection per offending target, at any depth in any section.

    Notes
    -----
    WHAT THIS CLOSES, MEASURED. :meth:`ForbiddenSurface.matches` tests KEYS, so
    a value is never tokenized. Injecting
    ``_target_: tpen.hi_manifest.reference_energy`` at eight points in the
    control config was ACCEPTED at six of them -- ``loggers``, ``sampler``,
    ``model``, ``runner``, ``trainer`` and the nested ``model.embedding``. Only
    ``callbacks`` and ``optimizer`` refused it, and each of those refuses for
    its own reason (an allowlist, and the method roster) rather than because
    targets were governed. **Five root sections had no target rule at all**, so
    the nested-callback-target finding this rule was filed under was one
    instance of the gap rather than the gap itself. With this rule the same
    eight injections are refused eight of eight, and the six that changed are
    refused by NAME here rather than incidentally by some other rule -- see
    ``TestNoTargetCanNameTheReferenceModule``, which asserts the rule name and
    not merely that something was refused.

    WHY A DENYLIST HERE AND AN ALLOWLIST FOR CALLBACKS. The callback allowlist
    is enumerable: the study installs a fixed set of bookkeeping and health
    callbacks and a new one is a review event. The targets in ``model``,
    ``sampler`` and ``trainer`` are NOT enumerable at schema time -- the scan
    varies producers, activations, update rules and five initializations, so an
    allowlist would have to list every arm the materializer may emit and would
    refuse a legitimate arm the day one is added. That is the over-restriction
    Amendment A warns about, and it surfaces as a run that cannot start rather
    than as a red test. The hazard being closed is narrow and nameable -- an
    executable that loads the evaluation reference -- so it is named.

    TWO RULES, AND NEITHER SUBSUMES THE OTHER. The module rule catches
    ``tpen.hi_manifest.load_evaluation_manifest``, whose tokens are
    ``tpen, hi, manifest, load, evaluation, manifest`` and contain no forbidden
    token at all. The token rule catches
    ``tpen.diagnostics.energy.ReferenceGapProbe``, which lives outside the
    manifest module entirely. Each was checked against the other's example.

    RESOLVED TREE ONLY, for the same reason :func:`_sweep_callbacks` gives: a
    ``_target_`` may itself be an interpolation, and the raw tree then holds
    ``"${...}"`` with nothing to compare. The key-side sweeps run on both trees
    already, so a forbidden KEY is still caught in raw.

    RESIDUAL, stated rather than papered over: this refuses targets that NAME
    the reference. A module that reaches it transitively is not visible here --
    that is what the import separation test covers, and the two nets have
    different populations. A diagnostic that ACCEPTS a reference needs a
    ``reference_energy`` argument key, which the key-side sweep already refuses.
    """

    rejections: list[Rejection] = []
    for path, key, value in iter_nodes(resolved_tree):
        if key != "_target_" or not isinstance(value, str):
            continue
        target = value.strip()
        if target == REFERENCE_MANIFEST_MODULE or target.startswith(
            f"{REFERENCE_MANIFEST_MODULE}."
        ):
            rejections.append(
                Rejection(
                    rule="forbidden-target:reference-module",
                    tree="resolved",
                    path=path,
                    detail=(
                        f"_target_ {target!r} is in {REFERENCE_MANIFEST_MODULE!r}, which holds "
                        "the evaluation reference. Instantiating it would load the reference "
                        "onto the training path, which is what this schema exists to prevent. "
                        "The reference is read after training, by a separate process"
                    ),
                )
            )
            continue
        for surface in HI_TRAIN_POLICY.forbidden_surfaces:
            if surface.matches(target):
                rejections.append(
                    Rejection(
                        rule=f"forbidden-target:{surface.name}",
                        tree="resolved",
                        path=path,
                        detail=(
                            f"_target_ {target!r} names a {surface.name} surface; "
                            f"{surface.reason}. A _target_ is executable, so the token check "
                            "applies to the value here and not only to the key"
                        ),
                    )
                )
    return rejections


# ---------------------------------------------------------------------------
# Frozen architectural coordinates
# ---------------------------------------------------------------------------
# Only coordinates the study does NOT vary appear here. The scan varies the
# producer/path policy, channels, activations, embedding width/depth, the
# feature update rule, and five initializations; pinning any of those would
# make the study's own grid unrunnable. Every key name below was checked
# against experiments/atomistic/he-v1/configs/train.yaml, a real production
# config -- a constraint naming a key that no config spells would never fire
# and would be a silent no-op rather than a check.

# Value must equal the expected one wherever the key appears anywhere under
# ``model``. Depth-independent because the producer/path policy varies the
# nesting: A1/A2 swap tensor, linear and hybrid producers, so a fixed path
# would stop matching on some arms.
_FROZEN_MODEL_KEYS: dict[str, tuple[object, str]] = {
    "max_order": (2, "body order 2 (literal control, stack/path metadata)"),
    "max_virtual_order": (2, "virtual support fixed at 2, no V1 arm (A3)"),
    "implementation": (
        "vectorized",
        "vectorized kernels; the slow implementation is an oracle, not a scientific arm",
    ),
}

# Singular coordinates, given as dotted paths.
_FROZEN_SCALARS: dict[str, tuple[object, str]] = {
    "system.spatial_dim": (3, "spatial dimension 3 (literal control, system/numerics)"),
    "runtime.dtype": ("float64", "float64 (literal control, system/numerics)"),
    "hamiltonian_terms.electron_nucleus.eps": (
        0.0,
        "Coulomb distance floor 0.0; a floor would mask near-nucleus cancellation",
    ),
}

# Trainability must be DECLARED, never inherited. Absence is a violation here,
# unlike the frozen scalars above, because every one of these defaults to the
# WRONG value. This is not hypothetical: he-v1's own config records that
# PfaffianReadout defaults to trainable=False, so passing only `channels`
# pinned the channel weights at a uniform 1/32 for all 300,000 updates with
# nothing in named_parameters(), nothing in state_dict(), and no log line. The
# silence was total. Requiring the declaration is what makes that failure
# impossible to repeat by omission.
_REQUIRED_TRAINABILITY: dict[str, str] = {
    "model.readout.trainable": "trainable weighted Pfaffian readout (literal control)",
}

# Factor trainability, keyed by the trailing component of the factor _target_.
_REQUIRED_FACTOR_TRAINABILITY: dict[str, tuple[str, str]] = {
    "ElectronElectronCusp": (
        "trainable_range",
        "both e-e ranges remain trainable in every arm (A10)",
    ),
    "ElectronNucleusCusp": (
        "law.trainable",
        "the e-n curvature law is trainable (literal control, e-n factor)",
    ),
    # Registered when the factor landed rather than when a config first uses
    # it. The failure this table exists to prevent -- a coefficient silently
    # frozen at its initial value, absent from named_parameters() and
    # state_dict() and therefore unlogged -- is available to this factor the
    # moment someone adds it to a config, and a rule added later would arrive
    # after the run that needed it.
    # NOT ACTIVE IN ANY SHIPPED CONFIGURATION as of this commit. The factor
    # exists, is exported from `tpen.nn`, and has this rule -- which together
    # look exactly like a factor in use. It is composed into nothing:
    # `experiments/atomistic/he-importance/configs/train.yaml` does not list it
    # among `model.factors`, and no other config does either. Composing it
    # changes the model that produces the science, so it is a scan-design
    # decision rather than a code chore; tracked as its own item. Said here
    # because a capability present but unwired is the kind of thing a later
    # reader assumes is on.
    "BoundedTwoCoefficientJastrow": (
        "trainable",
        "both Jastrow coefficients train; they start at zero, so an inherited "
        "false leaves the factor as the identity for the whole run with nothing "
        "to show for it",
    ),
}

# ADMITTED electron-nucleus cusp laws, keyed by the trailing component of the
# law's ``_target_``.
#
# An ALLOWLIST, for the same reason the callback set is one: the hazard is a law
# nobody thought to forbid. A denylist would admit every cusp law written after
# it, including the next unconstrained-tail variant.
#
# What membership MEANS here is a physical property, not a preference: the law's
# outer radial slope is negative for every nucleus BY CONSTRUCTION, so no
# training trajectory can reach a growing, non-normalizable tail.
# `CurvatureElectronNucleusCuspLaw` is deliberately ABSENT. It is not
# deprecated and remains correct for `experiments/atomistic/he-v1`, which is
# written in its coordinates -- but its own docstring records that it does not
# enforce ``c/d < Z``, so an HI arm selecting it could cross the sign change
# mid-run with nothing raising. Refusing it at validation converts that into a
# config error before anything is constructed, which is the only point at which
# it is cheap to notice.
_ADMITTED_ELECTRON_NUCLEUS_LAWS: dict[str, str] = {
    "TailSafeElectronNucleusCuspLaw": (
        "coordinates the curvature as c = d (Z - kappa), so the outer slope is "
        "-kappa < 0 for every nucleus at every point in training"
    ),
}

# Absent or null means no clipping, which is what the study requires, so this
# one is checked for absence rather than for a value.
_GLOBAL_CLIP_PATH = "trainer.gradient_clip_norm"


def _select(tree: Any, dotted: str) -> tuple[bool, Any]:
    """Return ``(found, value)`` for a dotted path in a plain container tree."""

    node: Any = tree
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _sweep_frozen_architecture(resolved_tree: Any) -> list[Rejection]:
    """Reject a configuration that moves a coordinate no arm may move."""

    if not isinstance(resolved_tree, Mapping):
        return []
    rejections: list[Rejection] = []

    model = resolved_tree.get("model")
    if isinstance(model, (Mapping, list)):
        for path, key, value in iter_nodes({"model": model}):
            if key not in _FROZEN_MODEL_KEYS:
                continue
            expected, authority = _FROZEN_MODEL_KEYS[key]
            if value != expected:
                rejections.append(
                    Rejection(
                        rule="frozen-coordinate",
                        tree="resolved",
                        path=path,
                        detail=f"expected {expected!r}, got {value!r}: {authority}",
                    )
                )

    for dotted, (expected, authority) in _FROZEN_SCALARS.items():
        found, value = _select(resolved_tree, dotted)
        if not found:
            # Absent means the code's own default applies. These three have
            # correct defaults, so absence is not a violation -- unlike the
            # trainability declarations below.
            continue
        if isinstance(expected, float) or isinstance(value, (int, float)) and not isinstance(value, bool):
            matches = isinstance(value, (int, float)) and float(value) == float(expected)
        else:
            matches = value == expected
        if not matches:
            rejections.append(
                Rejection(
                    rule="frozen-coordinate",
                    tree="resolved",
                    path=dotted,
                    detail=f"expected {expected!r}, got {value!r}: {authority}",
                )
            )

    found, clip = _select(resolved_tree, _GLOBAL_CLIP_PATH)
    if found and clip is not None:
        rejections.append(
            Rejection(
                rule="frozen-coordinate",
                tree="resolved",
                path=_GLOBAL_CLIP_PATH,
                detail=(
                    f"global gradient clipping is not part of the study's objective/protection "
                    f"policy, got {clip!r}. The update-norm bounds of SR and SPRING are "
                    "parameter-coordinate-dependent protections and are not this knob"
                ),
            )
        )

    rejections.extend(_sweep_trainability(resolved_tree))
    return rejections


def _sweep_trainability(resolved_tree: Mapping[str, Any]) -> list[Rejection]:
    """Require every trainability flag to be declared true, never inherited."""

    rejections: list[Rejection] = []

    for dotted, authority in _REQUIRED_TRAINABILITY.items():
        # Scoped to configurations that actually declare the component. A config
        # with no readout at all is INCOMPLETE, which is a different defect from
        # a readout whose trainability was silently inherited -- and it is the
        # latter this rule exists to catch. Conflating them would make every
        # partial config report a trainability violation it does not have.
        parent, _, _leaf = dotted.rpartition(".")
        if not _select(resolved_tree, parent)[0]:
            continue
        found, value = _select(resolved_tree, dotted)
        if not found:
            rejections.append(
                Rejection(
                    rule="undeclared-trainability",
                    tree="resolved",
                    path=dotted,
                    detail=(
                        f"{dotted} must be declared explicitly and true ({authority}). The "
                        "default is the opposite, and an inherited false is invisible: the "
                        "parameter appears in neither named_parameters() nor state_dict(), so "
                        "nothing logs it and no gradient touches it"
                    ),
                )
            )
        elif value is not True:
            rejections.append(
                Rejection(
                    rule="undeclared-trainability",
                    tree="resolved",
                    path=dotted,
                    detail=f"expected true, got {value!r}: {authority}",
                )
            )

    factors = _select(resolved_tree, "model.factors")[1]
    if isinstance(factors, list):
        for index, factor in enumerate(factors):
            if not isinstance(factor, Mapping):
                continue
            target = factor.get("_target_")
            if not isinstance(target, str):
                continue
            spec = _REQUIRED_FACTOR_TRAINABILITY.get(target.rsplit(".", 1)[-1])
            if spec is None:
                continue
            relative, authority = spec
            found, value = _select(factor, relative)
            if not found or value is not True:
                rejections.append(
                    Rejection(
                        rule="undeclared-trainability",
                        tree="resolved",
                        path=f"model.factors[{index}].{relative}",
                        detail=(
                            f"must be declared explicitly and true ({authority}); "
                            f"{'absent' if not found else repr(value)}"
                        ),
                    )
                )
    return rejections


# ---------------------------------------------------------------------------
# Rank-invariant preconstruction
# ---------------------------------------------------------------------------
# Admissible non-finite local-energy policies, and the one this study defaults
# to nothing. DECLARATION IS REQUIRED: there is no inherited value, because the
# inherited value would be the historical "mask" and masking is a
# known-biased estimator rather than a neutral fallback. Non-finite local
# energies occur where the local energy is pathological -- near nodes, at
# coalescence, in the tail -- so dropping them selects a subsample
# systematically and biases the energy by an uncharacterised amount. A count of
# how many rows were dropped does not recover that bias.
#
# So "mask" stays REACHABLE, because a scientist may knowingly want it, but only
# by writing it down where this schema can see it.
_ADMITTED_NONFINITE_POLICIES: frozenset[str] = frozenset({"fail", "mask"})
_NONFINITE_POLICY_PATH = "trainer.nonfinite_local_energy_policy"


def _sweep_nonfinite_local_energy_policy(resolved_tree: Any) -> list[Rejection]:
    """Require an explicit, admissible non-finite local-energy policy.

    Parameters
    ----------
    resolved_tree : Any
        Resolved configuration tree.

    Returns
    -------
    list of Rejection
        One rejection if the policy is absent or not admissible.
    """

    # Scoped to configurations that actually declare a trainer, matching the
    # trainability rule's precedent that "a config with no readout is not a
    # trainability violation". A configuration with no `trainer` section is not
    # configuring training at all, so there is no estimator for it to declare
    # and demanding one would refuse valid partial configs -- an
    # over-restriction that would surface as a run that cannot start.
    trainer_found, trainer_section = _select(resolved_tree, "trainer")
    if not trainer_found or not isinstance(trainer_section, Mapping):
        return []

    found, value = _select(resolved_tree, _NONFINITE_POLICY_PATH)
    if not found or value is None:
        return [
            Rejection(
                rule="undeclared-nonfinite-policy",
                tree="resolved",
                path=_NONFINITE_POLICY_PATH,
                detail=(
                    "must be declared explicitly; admissible values are "
                    f"{sorted(_ADMITTED_NONFINITE_POLICIES)}. There is no default here on "
                    "purpose: the inherited behaviour is to MASK non-finite local-energy "
                    "rows, which drops a systematically selected subsample and biases the "
                    "energy estimator by an uncharacterised amount. A run must say which "
                    "estimator it is using"
                ),
            )
        ]
    if value not in _ADMITTED_NONFINITE_POLICIES:
        return [
            Rejection(
                rule="undeclared-nonfinite-policy",
                tree="resolved",
                path=_NONFINITE_POLICY_PATH,
                detail=(
                    f"{value!r} is not admissible; expected one of "
                    f"{sorted(_ADMITTED_NONFINITE_POLICIES)}"
                ),
            )
        ]
    return []


def _sweep_electron_nucleus_law(resolved_tree: Any) -> list[Rejection]:
    """Reject an electron-nucleus cusp law outside the admitted set.

    Separate from `_sweep_trainability`, which asks whether the law is TRAINED.
    This asks whether it is the RIGHT LAW, and the two are independent: a
    trainable unconstrained-tail law satisfies that rule completely while still
    being able to train its way into a non-normalizable tail.

    Parameters
    ----------
    resolved_tree : Any
        Resolved configuration tree.

    Returns
    -------
    list of Rejection
        One rejection per factor naming an unadmitted or undeclared law.
    """

    rejections: list[Rejection] = []
    factors = _select(resolved_tree, "model.factors")[1]
    if not isinstance(factors, list):
        return rejections
    for index, factor in enumerate(factors):
        if not isinstance(factor, Mapping):
            continue
        target = factor.get("_target_")
        if not isinstance(target, str) or target.rsplit(".", 1)[-1] != "ElectronNucleusCusp":
            continue
        found, law = _select(factor, "law")
        if not found or not isinstance(law, Mapping):
            # An absent law is already fatal through the trainability rule,
            # which requires `law.trainable` to be declared true. Not repeated
            # here: two rejections for one omission would read as two defects.
            continue
        law_target = law.get("_target_")
        path = f"model.factors[{index}].law._target_"
        if not isinstance(law_target, str):
            rejections.append(
                Rejection(
                    rule="unadmitted-cusp-law",
                    tree="resolved",
                    path=path,
                    detail=(
                        "the electron-nucleus cusp law must declare a _target_; admitted "
                        f"laws are {sorted(_ADMITTED_ELECTRON_NUCLEUS_LAWS)}"
                    ),
                )
            )
            continue
        name = law_target.rsplit(".", 1)[-1]
        if name not in _ADMITTED_ELECTRON_NUCLEUS_LAWS:
            admitted = ", ".join(
                f"{key} ({reason})" for key, reason in sorted(_ADMITTED_ELECTRON_NUCLEUS_LAWS.items())
            )
            rejections.append(
                Rejection(
                    rule="unadmitted-cusp-law",
                    tree="resolved",
                    path=path,
                    detail=(
                        f"{law_target!r} is not admitted for this study. An admitted law must "
                        "guarantee a decaying outer tail for every nucleus by construction, "
                        "rather than leaving the bound to the caller. Admitted: "
                        f"{admitted}"
                    ),
                )
            )
    return rejections


# ---------------------------------------------------------------------------
# Closing over the components that are actually CONSTRUCTED
# ---------------------------------------------------------------------------
# THE HOLE THIS CLOSES, and it is the schema's own claim failing rather than a
# new kind of problem. Every component rule above reads a path from the
# configuration ROOT -- `model`, `trainer.gradient_clip_norm`,
# `model.factors[i].law`, `optimizer`. What `tpen.run._instantiate_runner`
# hands to Hydra is `cfg.runner`, and `instantiate` is RECURSIVE, so what gets
# built is whatever hangs under `runner`. In the shipped control config those
# are interpolations (`model: ${model}`) and the two agree -- but nothing
# required that. A runner section carrying its own literal copies was accepted
# with a frozen readout, a global gradient clip, an unadmitted cusp law and an
# OMITTED non-finite policy, while every root field stayed compliant.
#
# WHY VIEWS RATHER THAN ROOT-EQUALITY. Requiring `runner.model == model` would
# refuse every divergence, including ones nobody cares about, and would forbid
# a runner from ever carrying a component the root does not also spell. That
# over-restriction is invisible until a legitimate run cannot start. Re-running
# the component rules over what is constructed refuses exactly the divergences
# the rules already name, and nothing else -- and it needs no new rule, which
# is why the blast radius is the existing rules' blast radius.
#
# COMPONENT KEYS, not a fixed `runner.*` path list: the view is any mapping
# under `runner` that carries at least one component the rules know how to
# judge. In the shipped config the only such node is `runner` itself, so this
# costs one extra pass; a config that nested a component deeper is covered
# without a second rule.
_COMPONENT_KEYS = frozenset(
    {"model", "trainer", "optimizer", "sampler", "hamiltonian_terms"}
)

# The rules that judge a COMPONENT rather than the run as a whole. Deliberately
# omitted: `_sweep_rank_divergent_fields`, which is about `run.run_id` and is a
# property of the run's identity rather than of anything constructed;
# `_sweep_callbacks`, because `_instantiate_runner` already refuses a runner
# that owns callbacks at all; and `_sweep_target_values`, which already walks
# the whole resolved tree and therefore covers every view for free.
_COMPONENT_SWEEPS = (
    _sweep_frozen_architecture,
    _sweep_electron_nucleus_law,
    _sweep_nonfinite_local_energy_policy,
)


def _reprefix(rejections: list[Rejection], prefix: str) -> list[Rejection]:
    """Re-root each rejection's path so it names where the component was found.

    A finding reported at ``trainer.gradient_clip_norm`` when the offending
    value is at ``runner.trainer.gradient_clip_norm`` sends the reader to a
    field that is compliant, which is worse than no path at all.
    """

    return [replace(rejection, path=f"{prefix}{rejection.path}") for rejection in rejections]


def _sweep_constructed_components(resolved_tree: Any) -> list[Rejection]:
    """Apply the component rules to the root AND to everything under ``runner``.

    Parameters
    ----------
    resolved_tree : Any
        Resolved configuration tree.

    Returns
    -------
    list of Rejection
        Findings from every component view, each path re-rooted at the view.

    Notes
    -----
    The root view keeps its existing semantics exactly, including
    ``require_optimizer=True``: a training configuration that declares no
    method is incomplete. A component view is PRESENCE-SCOPED throughout --
    every rule it runs already skips a component the view does not declare --
    so an evaluation runner carrying no trainer and no optimizer is untouched.

    RESIDUAL, stated rather than papered over: this closes the components the
    rules know how to judge. A component the rules have no rule for is still
    unjudged wherever it appears, at the root as much as under ``runner``, and
    that is a gap in the RULE SET rather than in this traversal.
    """

    rejections: list[Rejection] = []
    for component_sweep in _COMPONENT_SWEEPS:
        rejections.extend(component_sweep(resolved_tree))
    rejections.extend(_sweep_method(resolved_tree))

    if not isinstance(resolved_tree, Mapping):
        return rejections
    runner = resolved_tree.get("runner")
    if not isinstance(runner, Mapping):
        return rejections

    for path, _key, value in iter_nodes({"runner": runner}):
        if not isinstance(value, Mapping) or not (_COMPONENT_KEYS & set(value)):
            continue
        for component_sweep in _COMPONENT_SWEEPS:
            rejections.extend(_reprefix(component_sweep(value), f"{path}."))
        rejections.extend(
            _reprefix(_sweep_method(value, require_optimizer=False), f"{path}.")
        )
    return rejections


def _sweep_rank_divergent_fields(resolved_tree: Any) -> list[Rejection]:
    """Reject fields that would resolve to a different value in each process.

    Notes
    -----
    MEASURED in ``tpen/artifacts.py``: ``generate_run_id`` returns
    ``f"{timestamp}_{slug}_{uuid4().hex[:6]}"``, and ``prepare_run_context``
    calls it whenever ``run.run_id`` resolves to ``None``. The suffix is
    RANDOM, so a null ``run_id`` does not merely risk divergence under clock
    skew -- it produces a different identifier in every process, always. Under
    a distributed launch each rank would then write to its own run directory
    and the run would have no single identity.

    This is the acceptance contract's second falsifier, "rank inputs differ",
    and it is a property of the existing code rather than of anything this
    schema adds. The remedy is for the materializer to assign the run id, which
    is L2's job; the schema's job is to refuse a config that leaves it to
    chance.
    """

    if not isinstance(resolved_tree, Mapping):
        return []
    run = resolved_tree.get("run")
    if not isinstance(run, Mapping):
        return []
    if run.get("run_id") is not None:
        return []
    return [
        Rejection(
            rule="rank-divergent-field",
            tree="resolved",
            path="run.run_id",
            detail=(
                "run.run_id must be assigned explicitly. A null run_id is filled in by "
                "generate_run_id, whose value ends in uuid4().hex[:6] -- so every process "
                "computes a DIFFERENT identifier and each rank would write to its own run "
                "directory. The resolved configuration must be identical on all ranks"
            ),
        )
    ]


def canonical_train_identity(cfg: DictConfig) -> str:
    """Return the digest every rank must agree on for one training run.

    Parameters
    ----------
    cfg : DictConfig
        A resolved training configuration.

    Returns
    -------
    str
        Hex SHA-256 of the canonical rendering of the resolved configuration.

    Notes
    -----
    Two ranks launched from the same file and overrides must produce the same
    value. What makes that true is not this function but the two checks beside
    it: forbidden resolvers keep resolution a pure function of the file, and
    the rank-divergent-field rule keeps ``run.run_id`` from being drawn per
    process. Without those, this digest would faithfully report a difference
    and say nothing about its cause.

    The digest covers the whole resolved configuration deliberately. Excluding
    fields to make ranks agree would be arranging for the answer.
    """

    return canonical_digest(OmegaConf.to_container(cfg, resolve=True))


def _roster_summary() -> str:
    """Render the roster so a refusal states every method's admission status."""

    return "; ".join(
        f"{entry.method}={'admitted' if entry.admitted else 'unavailable'} ({entry.requires})"
        for entry in HI_METHOD_ROSTER
    )


def _sweep_method(resolved_tree: Any, *, require_optimizer: bool = True) -> list[Rejection]:
    """Reject a training configuration whose optimizer method is not admitted.

    Parameters
    ----------
    resolved_tree : Any
        The tree to check, which is either the configuration root or one of the
        component views under ``runner``.
    require_optimizer : bool, optional
        Whether an ABSENT optimizer is itself a violation. True at the root,
        where a training configuration that declares no method is incomplete.
        False on a component view, where absence means the view simply does not
        carry an optimizer -- demanding one there would refuse every runner that
        is not a trainer, which is an over-restriction that surfaces as a run
        that cannot start rather than as a red test.

    Notes
    -----
    An unadmitted method is refused rather than replaced. The plan of record
    requires an unavailable cell to stay visibly unavailable; substituting Adam
    would let a run report that it exercised a method it never ran, which is a
    worse failure than not running at all.
    """

    if not isinstance(resolved_tree, Mapping):
        return []
    optimizer = resolved_tree.get("optimizer")
    if optimizer is None:
        if not require_optimizer:
            return []
        return [
            Rejection(
                rule="missing-method",
                tree="resolved",
                path="optimizer",
                detail=(
                    "a training configuration must declare its optimizer method. "
                    f"Roster: {_roster_summary()}"
                ),
            )
        ]
    if not isinstance(optimizer, Mapping):
        return [
            Rejection(
                rule="missing-method",
                tree="resolved",
                path="optimizer",
                detail=f"optimizer must be a config block, got {type(optimizer).__name__}",
            )
        ]

    target = optimizer.get("_target_")
    if not isinstance(target, str):
        return [
            Rejection(
                rule="missing-method",
                tree="resolved",
                path="optimizer._target_",
                detail=f"optimizer must declare a _target_. Roster: {_roster_summary()}",
            )
        ]

    if target not in ADMITTED_METHOD_TARGETS:
        return [
            Rejection(
                rule="unadmitted-method",
                tree="resolved",
                path="optimizer._target_",
                detail=(
                    f"method {target!r} is not admitted. It is refused rather than replaced: "
                    "an unavailable method must stay visibly unavailable, never silently "
                    f"become Adam. Roster: {_roster_summary()}"
                ),
            )
        ]

    return _sweep_adam_coordinates(optimizer)


def _sweep_adam_coordinates(optimizer: Mapping[str, Any]) -> list[Rejection]:
    """Reject an Adam block that varies a coordinate no arm may vary."""

    rejections: list[Rejection] = []
    for key, expected in _FIXED_ADAM_COORDINATES.items():
        if key not in optimizer:
            # Absent means the library default applies, and the fixed values
            # here ARE the library defaults. Requiring them to be spelled out
            # would reject every config that simply omits them.
            continue
        actual = optimizer[key]
        if float(actual) != float(expected):
            rejections.append(
                Rejection(
                    rule="frozen-coordinate",
                    tree="resolved",
                    path=f"optimizer.{key}",
                    detail=(
                        f"Adam {key} is fixed at {expected} for every cell in the optimizer "
                        f"grid, got {actual!r}. Only lr and beta2 are scanned"
                    ),
                )
            )

    betas = optimizer.get("betas")
    if isinstance(betas, Sequence) and not isinstance(betas, (str, bytes)) and betas:
        if float(betas[0]) != _FIXED_ADAM_BETA1:
            rejections.append(
                Rejection(
                    rule="frozen-coordinate",
                    tree="resolved",
                    path="optimizer.betas[0]",
                    detail=(
                        f"Adam beta1 is fixed at {_FIXED_ADAM_BETA1} for every cell, got "
                        f"{betas[0]!r}. beta2 is the scanned moment, beta1 is not"
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

    declared = declared_schema(cfg)
    if declared != HI_TRAIN_SCHEMA:
        if is_hi_family(cfg):
            # The config says it is helium-importance but did not declare the
            # schema. Refusing loudly here is the whole point: silently
            # returning would give a real HI run zero enforcement.
            raise ClosedSchemaError(
                [
                    Rejection(
                        rule="undeclared-schema",
                        tree="raw",
                        path=SCHEMA_KEY,
                        detail=(
                            f"experiment.name is {HI_EXPERIMENT_NAME!r}, so this is a "
                            f"helium-importance configuration and must declare "
                            f"{SCHEMA_KEY}: {HI_TRAIN_SCHEMA}. Declared: {declared!r}. "
                            "The marker is not optional for this family -- a closed schema "
                            "that applies only when a config remembers to ask for it is not "
                            "closed, and the omission would otherwise be silent"
                        ),
                    )
                ]
            )
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
    rejections.extend(_sweep_target_values(resolved_tree))
    rejections.extend(_sweep_constructed_components(resolved_tree))
    rejections.extend(_sweep_rank_divergent_fields(resolved_tree))
    if rejections:
        raise ClosedSchemaError(rejections)
