"""Contract tests for immutable trajectory-statistics receipts."""

from dataclasses import FrozenInstanceError

import pytest

from tpen.statistics.mixing import MixingDiagnostics
from tpen.statistics.receipt import (
    IDENTITY_FIELDS,
    PlateauDiagnostics,
    TrajectoryShape,
    TrajectoryStatisticsIdentity,
    TrajectoryStatisticsPayload,
    TrajectoryStatisticsReceipt,
)


_DIGEST = "a" * 64
_UNSET = object()


def _identity(**overrides: str) -> TrajectoryStatisticsIdentity:
    """Build a valid identity while keeping individual assertions readable."""

    values = {
        "stage": "eval",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "checkpoint_sha256": _DIGEST,
        "config_sha256": "b" * 64,
        "observable": "local_energy",
        "evaluator_id": "local_energy/v1",
    }
    values.update(overrides)
    return TrajectoryStatisticsIdentity(**values)


def _receipt(
    status: str = "available",
    *,
    identity: TrajectoryStatisticsIdentity | None = None,
    shape: TrajectoryShape | None | object = _UNSET,
    plateau: PlateauDiagnostics | None | object = _UNSET,
    mixing: MixingDiagnostics | None | object = _UNSET,
    payload: TrajectoryStatisticsPayload | None | object = _UNSET,
    reason: str | None | object = _UNSET,
) -> TrajectoryStatisticsReceipt:
    """Build a receipt with explicit status-dependent defaults."""

    if status == "available":
        shape = TrajectoryShape(2, 8, 3, 1) if shape is _UNSET else shape
        plateau = PlateauDiagnostics(True, 3, 2, 7) if plateau is _UNSET else plateau
        mixing = MixingDiagnostics(1.01, 4, 4, None) if mixing is _UNSET else mixing
        payload = (
            TrajectoryStatisticsPayload(2.0, 8.0, 0.5, 1.25, 4.0)
            if payload is _UNSET
            else payload
        )
        # `_UNSET` and an explicit None must stay distinguishable: `reason or ...`
        # would substitute a default and make the missing-reason case untestable.
        reason = None if reason is _UNSET else reason
    elif status == "unresolved":
        shape = TrajectoryShape(2, 8, 3, 1) if shape is _UNSET else shape
        plateau = None if plateau is _UNSET else plateau
        mixing = None if mixing is _UNSET else mixing
        payload = None if payload is _UNSET else payload
        reason = "no plateau" if reason is _UNSET else reason
    else:
        shape = None if shape is _UNSET else shape
        plateau = None if plateau is _UNSET else plateau
        mixing = None if mixing is _UNSET else mixing
        payload = None if payload is _UNSET else payload
        reason = "trajectory was not collected" if reason is _UNSET else reason
    return TrajectoryStatisticsReceipt(
        identity=identity or _identity(),
        status=status,
        recorded_at_utc="2026-08-15T12:00:00Z",
        estimator_id="pooled_geyer_ips",
        estimator_version="1",
        tau_convention="tau_int = 1 + 2 * sum_{k>=1} rho_k",
        shape=shape,
        plateau=plateau,
        mixing=mixing,
        payload=payload,
        reason=reason,
        warnings=("caveat", 7),
    )


def test_identity_key_and_dict_follow_canonical_field_order() -> None:
    """The seven fields are the durable join key in exactly one order."""

    identity = _identity()

    assert identity.as_key() == tuple(getattr(identity, field) for field in IDENTITY_FIELDS)
    assert tuple(identity.to_dict()) == IDENTITY_FIELDS
    assert identity.to_dict() == dict(zip(IDENTITY_FIELDS, identity.as_key()))


def test_identity_strips_whitespace_before_joining() -> None:
    """Invisible whitespace must not create a second sidecar identity."""

    padded = TrajectoryStatisticsIdentity(
        stage=" eval ",
        run_id=" run-1 ",
        attempt_id=" attempt-1 ",
        checkpoint_sha256=f" {_DIGEST} ",
        config_sha256=f" {'b' * 64} ",
        observable=" local_energy ",
        evaluator_id=" local_energy/v1 ",
    )

    assert padded == _identity()
    assert padded.as_key() == _identity().as_key()


@pytest.mark.parametrize("field_name", IDENTITY_FIELDS)
def test_identity_rejects_blank_field(field_name: str) -> None:
    """Every component is mandatory because partial joins are unsafe."""

    with pytest.raises(ValueError):
        _identity(**{field_name: " \t "})


@pytest.mark.parametrize("field_name", ("checkpoint_sha256", "config_sha256"))
@pytest.mark.parametrize(
    "digest",
    (
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
    ),
)
def test_identity_rejects_invalid_sha256(field_name: str, digest: str) -> None:
    """Content-addressed fields accept only canonical lowercase sha256 text."""

    with pytest.raises(ValueError):
        _identity(**{field_name: digest})


@pytest.mark.parametrize("field_name", ("checkpoint_sha256", "config_sha256"))
def test_identity_accepts_lowercase_sha256(field_name: str) -> None:
    """Both digest fields accept a valid 64-character lowercase digest."""

    assert getattr(_identity(**{field_name: _DIGEST}), field_name) == _DIGEST


@pytest.mark.parametrize("field_name", IDENTITY_FIELDS)
def test_identity_rejects_non_string_field(field_name: str) -> None:
    """Join-key fields cannot be silently coerced from unrelated types."""

    with pytest.raises(TypeError):
        _identity(**{field_name: 123})


def test_identity_is_frozen() -> None:
    """Changing a published key would invalidate every durable lookup."""

    with pytest.raises(FrozenInstanceError):
        _identity().run_id = "new-run"


def test_shape_total_draws_and_serialization() -> None:
    """Shape exposes total sample count for ESS consumers."""

    shape = TrajectoryShape(3, 5, 2, 4)

    assert shape.total_draws == 15
    assert shape.to_dict()["total_draws"] == 15


@pytest.mark.parametrize(
    "field_name,value",
    (("walker_count", 0), ("draw_count", 0), ("draw_stride", 0), ("burn_in_draws", -1)),
)
def test_shape_rejects_invalid_counts(field_name: str, value: int) -> None:
    """Impossible trajectory layouts must not enter a receipt."""

    values = {"walker_count": 2, "draw_count": 8, "draw_stride": 1, "burn_in_draws": 0}
    values[field_name] = value
    with pytest.raises(ValueError):
        TrajectoryShape(**values)


@pytest.mark.parametrize("field_name", ("tau_int", "ess", "mcse", "mean", "variance"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_payload_rejects_nonfinite_values(field_name: str, value: float) -> None:
    """A published numerical payload must be finite in every field."""

    values = {"tau_int": 2.0, "ess": 8.0, "mcse": 0.5, "mean": 1.0, "variance": 4.0}
    values[field_name] = value
    with pytest.raises(ValueError):
        TrajectoryStatisticsPayload(**values)


@pytest.mark.parametrize("field_name", ("tau_int", "ess", "variance"))
def test_payload_rejects_nonpositive_required_scales(field_name: str) -> None:
    """Positive scales are required for meaningful uncertainty estimates."""

    values = {"tau_int": 2.0, "ess": 8.0, "mcse": 0.5, "mean": 1.0, "variance": 4.0}
    values[field_name] = 0.0
    with pytest.raises(ValueError):
        TrajectoryStatisticsPayload(**values)


def test_payload_rejects_negative_mcse_but_accepts_zero() -> None:
    """Zero MCSE is valid for a degenerate reported estimate; negative is not."""

    with pytest.raises(ValueError):
        TrajectoryStatisticsPayload(2.0, 8.0, -0.1, 1.0, 4.0)
    assert TrajectoryStatisticsPayload(2.0, 8.0, 0.0, 1.0, 4.0).mcse == 0.0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"status": "available", "payload": None, "reason": None},
        {"status": "available", "payload": TrajectoryStatisticsPayload(2, 8, 0, 1, 4), "reason": "why"},
        {"status": "unresolved", "payload": TrajectoryStatisticsPayload(2, 8, 0, 1, 4), "reason": "why"},
        {"status": "unresolved", "payload": None, "reason": None},
        {"status": "absent", "payload": None, "reason": None},
    ),
    ids=("available_without_payload", "available_with_reason", "unresolved_with_payload", "unresolved_without_reason", "absent_without_reason"),
)
def test_receipt_enforces_status_payload_reason_invariant(kwargs: dict[str, object]) -> None:
    """Status is a lossless signal, never a partially populated row."""

    with pytest.raises(ValueError):
        _receipt(**kwargs)


@pytest.mark.parametrize("status", ("available", "unresolved"))
def test_receipt_requires_shape_when_trajectory_exists(status: str) -> None:
    """Consumers need shape to interpret any collected trajectory outcome."""

    with pytest.raises(ValueError):
        _receipt(status, shape=None, payload=None if status == "unresolved" else TrajectoryStatisticsPayload(2, 8, 0, 1, 4), reason="not resolved" if status == "unresolved" else None)


def test_absent_receipt_may_omit_trajectory_details() -> None:
    """Absence means no trajectory existed, so all trajectory details are optional."""

    receipt = _receipt("absent", shape=None, plateau=None, mixing=None)

    assert receipt.shape is None
    assert receipt.plateau is None
    assert receipt.mixing is None
    assert receipt.payload is None


def test_receipt_rejects_unknown_status() -> None:
    """Consumers must not silently accept a status with undefined semantics."""

    with pytest.raises(ValueError):
        _receipt("published")


@pytest.mark.parametrize("status", ("available", "unresolved", "absent"))
def test_receipt_dict_round_trip_preserves_equality(status: str) -> None:
    """The JSON-safe wire form reconstructs the same typed receipt."""

    receipt = _receipt(status)
    record = receipt.to_dict()

    assert TrajectoryStatisticsReceipt.from_dict(record) == receipt
    assert tuple(field for field in IDENTITY_FIELDS if field in record) == IDENTITY_FIELDS
    assert "statistics" in record
    if status == "available":
        assert record["statistics"] is not None
    else:
        assert record["statistics"] is None


def test_receipt_wire_layout_flattens_identity_and_nests_payload() -> None:
    """Sidecar consumers join on flat identity fields, not the statistics payload."""

    receipt = _receipt("available")
    record = receipt.to_dict()

    assert all(record[field] == getattr(receipt.identity, field) for field in IDENTITY_FIELDS)
    assert "source_artifact_sha256" not in record
    assert record["statistics"] == receipt.payload.to_dict()
    assert record["mixing"] == {
        "r_hat": receipt.mixing.r_hat,
        "n_split_chains": receipt.mixing.n_split_chains,
        "draws_per_split_chain": receipt.mixing.draws_per_split_chain,
        "reason": receipt.mixing.reason,
    }
    assert record["plateau"] == receipt.plateau.to_dict()


def test_absent_wire_form_has_no_trajectory_payloads() -> None:
    """An absent receipt serializes all trajectory-dependent sections as null."""

    record = _receipt("absent").to_dict()

    assert record["shape"] is None
    assert record["plateau"] is None
    assert record["mixing"] is None
    assert record["statistics"] is None


def test_receipt_warnings_are_tuple_of_strings() -> None:
    """Warnings have one stable immutable in-memory representation."""

    receipt = _receipt()

    assert receipt.warnings == ("caveat", "7")
    assert isinstance(receipt.warnings, tuple)



def test_available_round_trips_through_the_consumer_boundary() -> None:
    """Keep basic status and payload validation on the sidecar read path."""
    receipt = _receipt("available")

    restored = TrajectoryStatisticsReceipt.from_dict(receipt.to_dict())

    assert restored == receipt
    assert restored.plateau is not None and restored.plateau.plateau_reached
