"""Typed provenance for faithful fixed-model checkpoint replay.

The record in this module describes the executable semantics that determine a
He-v1 evaluation result.  It is deliberately content-addressed and verified
before checkpoint weights are loaded, so a result cannot claim one source,
checkpoint, configuration, distance treatment, or dtype while executing
another.

The electron-electron fields describe the currently executed physical-
separation calculation and its distinct positivity offset.  They are recorded
as two fields because ``ElectronElectronCusp.eps`` and ``range_eps`` are now
separate parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

from omegaconf import OmegaConf

from .hashing import file_sha256
from .manifest import CheckpointManifest

REPLAY_SEMANTICS_SCHEMA_VERSION: Final = 1
REPLAY_SEMANTICS_FILENAME: Final = "checkpoint_replay_semantics.json"

ELECTRON_ELECTRON_DISTANCE_FORM: Final = "sqrt_squared_distance_plus_eps_squared"
ELECTRON_ELECTRON_RANGE_OFFSET_FORM: Final = "softplus_plus_eps"
ELECTRON_NUCLEUS_COULOMB_DISTANCE_FORM: Final = "euclidean_norm_clamp_min_eps"
INFINITE_MASS_NONRELATIVISTIC_REFERENCE: Final = (
    "infinite_nuclear_mass_nonrelativistic"
)

ElectronElectronDistanceForm: TypeAlias = Literal[
    "sqrt_squared_distance_plus_eps_squared"
]
ElectronElectronRangeOffsetForm: TypeAlias = Literal["softplus_plus_eps"]
ElectronNucleusCoulombDistanceForm: TypeAlias = Literal[
    "euclidean_norm_clamp_min_eps"
]
ReferenceEnergyQualification: TypeAlias = Literal[
    "infinite_nuclear_mass_nonrelativistic"
]

_GIT_SHA_PATTERN = re.compile(r"\A[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class CuspDistanceSemantics:
    """Executed cusp-critical distance and range-offset semantics.

    Parameters
    ----------
    electron_electron_distance_form : str
        Executed pair-distance form.  He-v1 uses
        ``sqrt(||r_i-r_j||^2 + eps^2)``.
    electron_electron_distance_eps : float
        ``eps`` used by that softened pair distance.
    electron_electron_range_offset_form : str
        Executed positive-range form.  He-v1 uses
        ``softplus(raw_range) + eps``.
    electron_electron_range_offset_eps : float
        ``eps`` added after softplus.  It is recorded separately from the
        distance value; the implementation parameters are separate.
    electron_nucleus_coulomb_distance_form : str
        Executed electron-nucleus Coulomb distance form.  He-v1 uses
        ``clamp_min(||r_i-R_A||, eps)``.
    electron_nucleus_coulomb_distance_eps : float
        Configured ``eps`` for the Coulomb clamp.  He-v1 executes it as exactly
        ``0.0``; this is distinct from the softened electron-electron distance.
    """

    electron_electron_distance_form: ElectronElectronDistanceForm
    electron_electron_distance_eps: float
    electron_electron_range_offset_form: ElectronElectronRangeOffsetForm
    electron_electron_range_offset_eps: float
    electron_nucleus_coulomb_distance_form: ElectronNucleusCoulombDistanceForm
    electron_nucleus_coulomb_distance_eps: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "CuspDistanceSemantics":
        """Validate the supported executed forms and their numeric values."""

        expected_forms = {
            "electron_electron_distance_form": ELECTRON_ELECTRON_DISTANCE_FORM,
            "electron_electron_range_offset_form": ELECTRON_ELECTRON_RANGE_OFFSET_FORM,
            "electron_nucleus_coulomb_distance_form": (
                ELECTRON_NUCLEUS_COULOMB_DISTANCE_FORM
            ),
        }
        for name, expected in expected_forms.items():
            value = getattr(self, name)
            if value != expected:
                raise ValueError(f"{name} must be {expected!r}, got {value!r}")
        for name in (
            "electron_electron_distance_eps",
            "electron_electron_range_offset_eps",
            "electron_nucleus_coulomb_distance_eps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
            object.__setattr__(self, name, value)
        return self

    def to_dict(self) -> dict[str, str | float]:
        """Return a JSON-safe mapping with form and value kept adjacent."""

        return {
            "electron_electron_distance_form": self.electron_electron_distance_form,
            "electron_electron_distance_eps": self.electron_electron_distance_eps,
            "electron_electron_range_offset_form": (
                self.electron_electron_range_offset_form
            ),
            "electron_electron_range_offset_eps": (
                self.electron_electron_range_offset_eps
            ),
            "electron_nucleus_coulomb_distance_form": (
                self.electron_nucleus_coulomb_distance_form
            ),
            "electron_nucleus_coulomb_distance_eps": (
                self.electron_nucleus_coulomb_distance_eps
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CuspDistanceSemantics":
        """Construct and validate semantics from a configuration mapping."""

        return cls(
            electron_electron_distance_form=value["electron_electron_distance_form"],
            electron_electron_distance_eps=float(
                value["electron_electron_distance_eps"]
            ),
            electron_electron_range_offset_form=value[
                "electron_electron_range_offset_form"
            ],
            electron_electron_range_offset_eps=float(
                value["electron_electron_range_offset_eps"]
            ),
            electron_nucleus_coulomb_distance_form=value[
                "electron_nucleus_coulomb_distance_form"
            ],
            electron_nucleus_coulomb_distance_eps=float(
                value["electron_nucleus_coulomb_distance_eps"]
            ),
        )


@dataclass(frozen=True)
class CheckpointReplaySemantics:
    """Immutable identity for one faithfully replayed checkpoint result.

    Parameters
    ----------
    source_git_sha : str
        Full Git SHA recorded by the checkpoint writer.
    source_tpen_version : str
        TPEN package version recorded by the checkpoint writer.
    checkpoint_schema_version : int
        Manifest schema version of the restored checkpoint.
    checkpoint_kind : str
        Manifest kind of the restored checkpoint.
    checkpoint_model_sha256 : str
        SHA256 of the restored ``model.pt`` bytes.
    evaluation_config_sha256 : str
        SHA256 identity supplied for the resolved He-v1 evaluation config.
    runtime_dtype : str
        Configured floating dtype used for restore and evaluation.
    cusp_distance : CuspDistanceSemantics
        Executed cusp-critical distance and range-offset forms and values.
    reference_energy_qualification : str
        Qualification of the helium reference: infinite nuclear mass and
        nonrelativistic Hamiltonian.
    record_schema_version : int, optional
        Schema of this replay-semantics record.
    """

    source_git_sha: str
    source_tpen_version: str
    checkpoint_schema_version: int
    checkpoint_kind: str
    checkpoint_model_sha256: str
    evaluation_config_sha256: str
    runtime_dtype: str
    cusp_distance: CuspDistanceSemantics
    reference_energy_qualification: ReferenceEnergyQualification = (
        INFINITE_MASS_NONRELATIVISTIC_REFERENCE
    )
    record_schema_version: int = REPLAY_SEMANTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "CheckpointReplaySemantics":
        """Validate the complete typed identity and return ``self``."""

        if self.record_schema_version != REPLAY_SEMANTICS_SCHEMA_VERSION:
            raise ValueError(
                "unsupported replay-semantics record schema "
                f"{self.record_schema_version}; expected {REPLAY_SEMANTICS_SCHEMA_VERSION}"
            )
        source_git_sha = _stripped("source_git_sha", self.source_git_sha)
        if not _GIT_SHA_PATTERN.match(source_git_sha):
            raise ValueError(
                "source_git_sha must be a full 40-character lowercase Git SHA, "
                f"got {source_git_sha!r}"
            )
        object.__setattr__(self, "source_git_sha", source_git_sha)
        object.__setattr__(
            self,
            "source_tpen_version",
            _stripped("source_tpen_version", self.source_tpen_version),
        )
        object.__setattr__(
            self, "checkpoint_kind", _stripped("checkpoint_kind", self.checkpoint_kind)
        )
        object.__setattr__(self, "runtime_dtype", _stripped("runtime_dtype", self.runtime_dtype))
        if int(self.checkpoint_schema_version) < 1:
            raise ValueError("checkpoint_schema_version must be positive")
        object.__setattr__(
            self, "checkpoint_schema_version", int(self.checkpoint_schema_version)
        )
        for name in ("checkpoint_model_sha256", "evaluation_config_sha256"):
            digest = _stripped(name, getattr(self, name))
            if not _SHA256_PATTERN.match(digest):
                raise ValueError(
                    f"{name} must be 64 lowercase hex characters, got {digest!r}"
                )
            object.__setattr__(self, name, digest)
        if not isinstance(self.cusp_distance, CuspDistanceSemantics):
            raise TypeError("cusp_distance must be a CuspDistanceSemantics")
        self.cusp_distance.validate()
        if self.reference_energy_qualification != INFINITE_MASS_NONRELATIVISTIC_REFERENCE:
            raise ValueError(
                "reference_energy_qualification must identify the "
                "infinite-nuclear-mass nonrelativistic reference"
            )
        return self

    def content_id(self) -> str:
        """Return the SHA256 identity of the canonical semantic fields."""

        encoded = json.dumps(
            self._identity_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe record, including its content identity."""

        return {**self._identity_dict(), "content_id": self.content_id()}

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "record_schema_version": self.record_schema_version,
            "source_git_sha": self.source_git_sha,
            "source_tpen_version": self.source_tpen_version,
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "checkpoint_kind": self.checkpoint_kind,
            "checkpoint_model_sha256": self.checkpoint_model_sha256,
            "evaluation_config_sha256": self.evaluation_config_sha256,
            "runtime_dtype": self.runtime_dtype,
            "cusp_distance": self.cusp_distance.to_dict(),
            "reference_energy_qualification": self.reference_energy_qualification,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckpointReplaySemantics":
        """Construct, validate, and content-check a serialized record."""

        cusp_value = value["cusp_distance"]
        if not isinstance(cusp_value, Mapping):
            raise TypeError("cusp_distance must be a mapping")
        record = cls(
            record_schema_version=int(
                value.get("record_schema_version", REPLAY_SEMANTICS_SCHEMA_VERSION)
            ),
            source_git_sha=str(value["source_git_sha"]),
            source_tpen_version=str(value["source_tpen_version"]),
            checkpoint_schema_version=int(value["checkpoint_schema_version"]),
            checkpoint_kind=str(value["checkpoint_kind"]),
            checkpoint_model_sha256=str(value["checkpoint_model_sha256"]),
            evaluation_config_sha256=str(value["evaluation_config_sha256"]),
            runtime_dtype=str(value["runtime_dtype"]),
            cusp_distance=CuspDistanceSemantics.from_mapping(cusp_value),
            reference_energy_qualification=value.get(
                "reference_energy_qualification",
                INFINITE_MASS_NONRELATIVISTIC_REFERENCE,
            ),
        )
        claimed_content_id = value.get("content_id")
        if claimed_content_id is not None and claimed_content_id != record.content_id():
            raise ValueError(
                "replay-semantics content_id mismatch "
                f"(claimed {claimed_content_id}, computed {record.content_id()})"
            )
        return record


def coerce_checkpoint_replay_semantics(value: Any) -> CheckpointReplaySemantics:
    """Return a typed replay record from a typed value or configuration mapping."""

    if isinstance(value, CheckpointReplaySemantics):
        return value.validate()
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(
            "load.replay_semantics must be a CheckpointReplaySemantics or mapping"
        )
    return CheckpointReplaySemantics.from_mapping(value)


def verify_checkpoint_replay_semantics(
    semantics: CheckpointReplaySemantics,
    *,
    manifest: CheckpointManifest,
    checkpoint_dir: Path,
    model: Any,
    context: Any,
) -> None:
    """Fail closed if declared replay semantics differ from executable state.

    This gate performs only explicit typed/path checks and runs before any
    ``load_state_dict`` call.  A refusal therefore leaves the configured target
    model untouched.
    """

    semantics.validate()
    _require_equal(
        "checkpoint_schema_version",
        semantics.checkpoint_schema_version,
        manifest.schema_version,
    )
    _require_equal("checkpoint_kind", semantics.checkpoint_kind, manifest.kind)
    _require_equal(
        "source_git_sha",
        semantics.source_git_sha,
        manifest.provenance.get("git_sha"),
    )
    _require_equal(
        "source_tpen_version",
        semantics.source_tpen_version,
        manifest.provenance.get("tpen_version"),
    )

    model_file = manifest.files.get("model")
    if not model_file:
        raise ValueError(f"{checkpoint_dir}: checkpoint manifest lacks model file")
    _require_equal(
        "checkpoint_model_sha256",
        semantics.checkpoint_model_sha256,
        file_sha256(checkpoint_dir / model_file),
    )

    cfg = getattr(context, "cfg", None)
    if cfg is None:
        raise ValueError("checkpoint replay verification requires context.cfg")
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    _require_equal(
        "evaluation_config_sha256",
        semantics.evaluation_config_sha256,
        OmegaConf.select(cfg, "trajectory_identity.config_sha256"),
    )
    _require_equal(
        "runtime_dtype",
        semantics.runtime_dtype,
        manifest.runtime.get("dtype"),
    )
    _require_equal(
        "runtime_dtype",
        semantics.runtime_dtype,
        getattr(getattr(context, "metadata", None), "dtype", None),
    )

    from tpen.nn.cusp import ElectronElectronCusp

    try:
        factors = model.factors
    except AttributeError as error:
        raise TypeError(
            "replay-verified model must expose an explicit factors collection"
        ) from error
    electron_electron_cusps = [
        factor for factor in factors if isinstance(factor, ElectronElectronCusp)
    ]
    if len(electron_electron_cusps) != 1:
        raise ValueError(
            "replay-verified model must contain exactly one ElectronElectronCusp, "
            f"got {len(electron_electron_cusps)}"
        )
    cusp = electron_electron_cusps[0]
    cusp_eps = float(cusp.eps)
    cusp_range_eps = float(cusp.range_eps)
    _require_equal(
        "electron_electron_distance_eps",
        semantics.cusp_distance.electron_electron_distance_eps,
        cusp_eps,
    )
    _require_equal(
        "electron_electron_range_offset_eps",
        semantics.cusp_distance.electron_electron_range_offset_eps,
        cusp_range_eps,
    )
    _require_equal(
        "electron_nucleus_coulomb_distance_eps",
        semantics.cusp_distance.electron_nucleus_coulomb_distance_eps,
        OmegaConf.select(cfg, "hamiltonian_terms.electron_nucleus.eps"),
    )


def write_checkpoint_replay_semantics(
    semantics: CheckpointReplaySemantics, run_dir: str | Path
) -> Path:
    """Write one immutable replay-semantics artifact into a run directory."""

    from tpen.artifacts import write_json

    path = Path(run_dir) / REPLAY_SEMANTICS_FILENAME
    payload = semantics.to_dict()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                f"{path} already records different checkpoint replay semantics"
            )
        return path
    temporary = path.with_name(f"{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite incomplete artifact {temporary}")
    write_json(temporary, payload)
    temporary.rename(path)
    return path


def _require_equal(name: str, declared: Any, executed: Any) -> None:
    if declared != executed:
        raise ValueError(
            f"checkpoint replay {name} mismatch (declared {declared!r}, executed {executed!r})"
        )


def _stripped(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must be non-empty")
    return stripped


__all__ = [
    "ELECTRON_ELECTRON_DISTANCE_FORM",
    "ELECTRON_ELECTRON_RANGE_OFFSET_FORM",
    "ELECTRON_NUCLEUS_COULOMB_DISTANCE_FORM",
    "INFINITE_MASS_NONRELATIVISTIC_REFERENCE",
    "REPLAY_SEMANTICS_FILENAME",
    "REPLAY_SEMANTICS_SCHEMA_VERSION",
    "CheckpointReplaySemantics",
    "CuspDistanceSemantics",
    "coerce_checkpoint_replay_semantics",
    "verify_checkpoint_replay_semantics",
    "write_checkpoint_replay_semantics",
]
