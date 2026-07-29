"""Strict V4 experiment identity and completed-bundle contracts."""

from ._codec import CONTRACT_SCHEMA_VERSION, ContractError, canonical_sha256
from .execution import ExecutionProfileV1
from .identities import (
    MetricKeyV1,
    ProducerAttemptV1,
    ProducerV1,
    RunV1,
    SeedAssignmentV1,
    TrialV1,
)
from .io import (
    BUNDLE_SCHEMA_VERSION,
    BundleManifestV1,
    ContractBundleV1,
    SourceDescriptorV1,
    bundle_manifest_sha256,
    publish_bundle,
    read_bundle,
)
from .stages import StageResultV1

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "BundleManifestV1",
    "ContractBundleV1",
    "ContractError",
    "ExecutionProfileV1",
    "MetricKeyV1",
    "ProducerAttemptV1",
    "ProducerV1",
    "RunV1",
    "SeedAssignmentV1",
    "SourceDescriptorV1",
    "StageResultV1",
    "TrialV1",
    "bundle_manifest_sha256",
    "canonical_sha256",
    "publish_bundle",
    "read_bundle",
]
