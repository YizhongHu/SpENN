"""Fail-closed dispatcher for immutable selector verification semantics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import selector_verifier_v1

_VERIFIERS = {
    selector_verifier_v1.VERIFIER_ID: selector_verifier_v1,
}


def build_contract(
    verifier_id: str,
    *,
    grid: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    selection_report: Mapping[str, Any],
    champion_rows: Sequence[Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    producer_replay: Mapping[str, Any],
    producer_source_path: str,
    producer_source_sha256: str,
) -> dict[str, Any]:
    """Build a contract with one explicitly selected immutable verifier."""

    verifier = _require_verifier(verifier_id)
    return verifier.build_contract(
        grid=grid,
        summary_rows=summary_rows,
        selection_report=selection_report,
        champion_rows=champion_rows,
        artifact_sha256=artifact_sha256,
        producer_replay=producer_replay,
        producer_source_path=producer_source_path,
        producer_source_sha256=producer_source_sha256,
    )


def verify_contract(
    contract: object,
    *,
    grid: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    selection_report: Mapping[str, Any],
    champion_rows: Sequence[Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
) -> tuple[str, ...]:
    """Dispatch verification by immutable ID; unknown versions fail closed."""

    if not isinstance(contract, Mapping):
        return ("selection verification contract is not an object",)
    verifier = contract.get("verifier")
    if not isinstance(verifier, Mapping):
        return ("selection verification contract has no verifier identity",)
    verifier_id = str(verifier.get("id") or "")
    try:
        selected = _require_verifier(verifier_id)
    except ValueError as exc:
        return (str(exc),)
    return selected.verify_contract(
        contract,
        grid=grid,
        summary_rows=summary_rows,
        selection_report=selection_report,
        champion_rows=champion_rows,
        artifact_sha256=artifact_sha256,
    )


def _require_verifier(verifier_id: str):
    try:
        return _VERIFIERS[verifier_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown selector verifier id: {verifier_id!r}"
        ) from exc
