"""Graph storage operations for the standard-name pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.models import RefineStopReason
from imas_codex.standard_names.defaults import DEFAULT_REFINE_ROTATIONS

from .graph_foundation import _CLAIM_VERIFY_SETTLE_SECONDS

logger = logging.getLogger(__name__)


@retry_on_deadlock()
def release_generate_name_claims(
    *, source_ids: list[str], claim_token: str, services: Any
) -> int:
    """Release StandardNameSource claims IFF ``claim_token`` matches.

    Clears ``claimed_at`` and ``claim_token`` so items become eligible for
    re-claim.  Used by the generate_name pool's error-recovery path.

    Returns the count of sources actually released.  A count less than
    ``len(source_ids)`` means the orphan sweep (or another worker) already
    cleared the stale claim — the caller can safely ignore this.
    """
    if not source_ids:
        return 0
    with services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (n:StandardNameSource {id: sid})
            WHERE n.claim_token = $token
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=source_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(source_ids):
        logger.debug(
            "release_generate_name_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(source_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_generate_name_failed_claims(
    *, source_ids: list[str], claim_token: str, services: Any
) -> int:
    """Release StandardNameSource claims on worker failure IFF token matches.

    Identical to :func:`release_generate_name_claims`; provided as the
    symmetric "failed" variant so callers can be explicit about intent.

    Returns the count of sources actually released.
    """
    return services.release_generate_name_claims(
        source_ids=source_ids,
        claim_token=claim_token,
    )


@retry_on_deadlock()
def release_enrich_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
    services: Any,
) -> int:
    """Release StandardName enrichment claims IFF token (and stage) match.

    Parameters
    ----------
    sn_ids:
        StandardName node ids to release.
    claim_token:
        Token that was set at claim time; nodes with a different token are
        left untouched.
    expected_stage:
        If provided, also verify ``name_stage = $expected_stage`` before
        clearing.  Pass ``None`` (default) to skip stage verification.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.name_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with services.GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            SET n.updated_at = datetime(), n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_enrich_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_enrich_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    from_stage: str | None = None,
    to_stage: str | None = None,
    services: Any,
) -> int:
    """Release StandardName enrichment claims on worker failure.

    Clears claim state and optionally reverts ``name_stage`` to *to_stage*
    when the worker processing failed and the item should be retried.

    Parameters
    ----------
    sn_ids:
        StandardName node ids to release.
    claim_token:
        Token set at claim time.
    from_stage:
        If provided, verify ``name_stage = $from_stage`` before acting.
    to_stage:
        If provided (and *from_stage* matches), revert ``name_stage`` to
        this value so the item is eligible for a fresh claim.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_where = "AND n.name_stage = $from_stage" if from_stage is not None else ""
    stage_set = "n.name_stage = $to_stage," if to_stage is not None else ""
    params: dict[str, Any] = {"ids": sn_ids, "token": claim_token}
    if from_stage is not None:
        params["from_stage"] = from_stage
    if to_stage is not None:
        params["to_stage"] = to_stage
    with services.GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_where}
            SET n.updated_at = datetime(),
                {stage_set}
                n.claimed_at = null,
                n.claim_token = null
            RETURN count(n) AS released
            """,
            **params,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_enrich_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_names_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
    services: Any,
) -> int:
    """Release StandardName review-names claims IFF token (and stage) match.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.name_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with services.GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            SET n.updated_at = datetime(), n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_names_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_names_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    from_stage: str | None = None,
    to_stage: str | None = None,
    services: Any,
) -> int:
    """Release StandardName review-names claims on worker failure.

    Clears claim state and optionally reverts ``name_stage``.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_where = "AND n.name_stage = $from_stage" if from_stage is not None else ""
    stage_set = "n.name_stage = $to_stage," if to_stage is not None else ""
    params: dict[str, Any] = {"ids": sn_ids, "token": claim_token}
    if from_stage is not None:
        params["from_stage"] = from_stage
    if to_stage is not None:
        params["to_stage"] = to_stage
    with services.GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_where}
            SET n.updated_at = datetime(),
                {stage_set}
                n.claimed_at = null,
                n.claim_token = null
            RETURN count(n) AS released
            """,
            **params,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_names_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_docs_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
    services: Any,
) -> int:
    """Release StandardName review-docs claims IFF token (and stage) match.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.docs_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with services.GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            WITH n  // Preserve name_stage; this release owns the docs axis.
            SET n.updated_at = datetime(), n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_docs_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_docs_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    from_stage: str | None = None,
    to_stage: str | None = None,
    services: Any,
) -> int:
    """Release StandardName review-docs claims on worker failure.

    Clears claim state and optionally reverts ``docs_stage``.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_where = "AND n.docs_stage = $from_stage" if from_stage is not None else ""
    stage_set = "n.docs_stage = $to_stage," if to_stage is not None else ""
    params: dict[str, Any] = {"ids": sn_ids, "token": claim_token}
    if from_stage is not None:
        params["from_stage"] = from_stage
    if to_stage is not None:
        params["to_stage"] = to_stage
    with services.GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_where}
            WITH n  // Preserve name_stage; this release owns the docs axis.
            SET n.updated_at = datetime(),
                {stage_set}
                n.claimed_at = null,
                n.claim_token = null
            RETURN count(n) AS released
            """,
            **params,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_docs_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


REFINE_CLAIM_RELEASE_MANIFEST_SCHEMA = "imas-codex.refine-claim-release-manifest"


REFINE_CLAIM_RELEASE_RECEIPT_SCHEMA = "imas-codex.refine-claim-release-receipt"


REFINE_CLAIM_RELEASE_PROJECTION_VERSION = 1


class RefineClaimReleaseConflict(RuntimeError):
    """An exact stranded-refinement claim could not be safely released."""


_LOCK_REFINE_CLAIM_RELEASE_QUERY = """
// REFINE_CLAIM_RELEASE_LOCK
MATCH (sn:StandardName {id: $target_id})
WHERE sn.name_stage = $expected_stage
  AND sn.claim_token = $claim_token
  AND sn.claimed_at IS NOT NULL
  AND ($scope_run_id IS NULL OR sn.run_id = $scope_run_id)
SET sn._refine_claim_release_lock = true
REMOVE sn._refine_claim_release_lock
RETURN sn.id AS id
"""


_READ_REFINE_CLAIM_RELEASE_QUERY = """
// REFINE_CLAIM_RELEASE_READ
MATCH (sn:StandardName {id: $target_id})
CALL (sn) {
  OPTIONAL MATCH (source:StandardNameSource)-[binding:PRODUCED_NAME]->(sn)
  OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(bound:StandardName)
  WHERE NOT (bound.name_stage IN ['superseded', 'exhausted', 'contested'])
  WITH source, binding,
       collect(DISTINCT CASE WHEN bound IS NULL THEN null ELSE {
         id: bound.id
       } END) AS bound_names
  RETURN collect(CASE WHEN source IS NULL THEN null ELSE {
    source: properties(source),
    relationship: properties(binding),
    bound_names: bound_names
  } END) AS source_bindings
}
CALL (sn) {
  OPTIONAL MATCH (projection)-[edge:HAS_STANDARD_NAME]->(sn)
  WHERE projection:IMASNode OR projection:FacilitySignal
  RETURN collect(CASE WHEN projection IS NULL THEN null ELSE {
    labels: labels(projection),
    node: properties(projection),
    relationship: properties(edge)
  } END) AS source_projections
}
CALL (sn) {
  OPTIONAL MATCH (sn)-[:REFINED_FROM]->(predecessor:StandardName)
  RETURN collect(DISTINCT predecessor.id) AS predecessor_ids
}
CALL (sn) {
  OPTIONAL MATCH (successor:StandardName)-[:REFINED_FROM]->(sn)
  RETURN collect(DISTINCT successor.id) AS successor_ids
}
CALL (sn) {
  OPTIONAL MATCH (sn)-[edge]->(related)
  RETURN collect(CASE WHEN edge IS NULL THEN null ELSE {
    type: type(edge),
    relationship: properties(edge),
    node_labels: labels(related),
    node: properties(related)
  } END) AS outgoing_relationships
}
CALL (sn) {
  OPTIONAL MATCH (related)-[edge]->(sn)
  RETURN collect(CASE WHEN edge IS NULL THEN null ELSE {
    type: type(edge),
    relationship: properties(edge),
    node_labels: labels(related),
    node: properties(related)
  } END) AS incoming_relationships
}
RETURN properties(sn) AS standard_name,
       toString(sn.claimed_at) AS claimed_at_text,
       source_bindings,
       source_projections,
       predecessor_ids,
       successor_ids,
       outgoing_relationships,
       incoming_relationships
"""


_WRITE_REFINE_CLAIM_RELEASE_QUERY = """
// REFINE_CLAIM_RELEASE_WRITE
MATCH (sn:StandardName {id: $target_id})
WHERE sn.name_stage = $expected_stage
  AND sn.claim_token = $claim_token
  AND toString(sn.claimed_at) = $expected_claimed_at
  AND ($scope_run_id IS NULL OR sn.run_id = $scope_run_id)
SET sn.updated_at = datetime(), sn.name_stage = 'reviewed',
    sn.claim_token = null,
    sn.claimed_at = null
RETURN sn.id AS id
"""


def _sorted_refine_claim_projection(values: Any, *, services: Any) -> list[Any]:
    normalized = [services._authority_json_value(value) for value in values or []]
    return sorted(
        normalized,
        key=lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
    )


def _project_refine_claim_release_state(
    row: dict[str, Any], *, services: Any
) -> dict[str, Any]:
    properties = services._authority_json_value(dict(row.get("standard_name") or {}))
    description = str(properties.get("description") or "")
    documentation = str(properties.get("documentation") or "")
    return {
        "target_id": properties.get("id"),
        "name_stage": properties.get("name_stage"),
        "claim": {
            "token": properties.get("claim_token"),
            "claimed_at": row.get("claimed_at_text") or properties.get("claimed_at"),
        },
        "scope": {
            "run_id": properties.get("run_id"),
            "drain_scope_id": properties.get("drain_scope_id"),
            "drain_scope_claimed_at": properties.get("drain_scope_claimed_at"),
            "drain_scope_paths": properties.get("drain_scope_paths"),
            "drain_claim_scope_id": properties.get("drain_claim_scope_id"),
        },
        "content_hashes": {
            "description_sha256": hashlib.sha256(description.encode()).hexdigest(),
            "documentation_sha256": hashlib.sha256(documentation.encode()).hexdigest(),
            "combined_sha256": services._authority_payload_hash(
                {
                    "description": description,
                    "documentation": documentation,
                }
            ),
        },
        "authority": {
            "validation_status": properties.get("validation_status"),
            "unit": properties.get("unit"),
            "kind": properties.get("kind"),
        },
        "node_properties": properties,
        "source_bindings": services._sorted_refine_claim_projection(
            row.get("source_bindings")
        ),
        "source_projections": services._sorted_refine_claim_projection(
            row.get("source_projections")
        ),
        "predecessor_ids": sorted(
            str(value) for value in row.get("predecessor_ids") or []
        ),
        "successor_ids": sorted(str(value) for value in row.get("successor_ids") or []),
        "outgoing_relationships": services._sorted_refine_claim_projection(
            row.get("outgoing_relationships")
        ),
        "incoming_relationships": services._sorted_refine_claim_projection(
            row.get("incoming_relationships")
        ),
    }


def _predicted_refine_claim_release_state(
    before: dict[str, Any], *, services: Any
) -> dict[str, Any]:
    predicted = json.loads(
        json.dumps(before, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    predicted["name_stage"] = "reviewed"
    predicted["claim"] = {"token": None, "claimed_at": None}
    predicted["node_properties"]["name_stage"] = "reviewed"
    predicted["node_properties"].pop("claim_token", None)
    predicted["node_properties"].pop("claimed_at", None)
    return predicted


def _refine_claim_release_manifest(
    *,
    target_id: str,
    claim_token: str,
    expected_stage: str,
    scope_run_id: str | None,
    before: dict[str, Any],
    predicted_after: dict[str, Any],
    services: Any,
) -> dict[str, Any]:
    return {
        "schema": REFINE_CLAIM_RELEASE_MANIFEST_SCHEMA,
        "projection_version": REFINE_CLAIM_RELEASE_PROJECTION_VERSION,
        "operation": "release_stranded_refine_claim",
        "target_id": target_id,
        "expected_stage": expected_stage,
        "claim_token": claim_token,
        "expected_scope_run_id": scope_run_id,
        "before": before,
        "predicted_after": predicted_after,
    }


def _refine_claim_no_change_receipt(
    *,
    target_id: str,
    claim_token: str,
    expected_stage: str,
    scope_run_id: str | None,
    services: Any,
) -> dict[str, Any]:
    return {
        "schema": REFINE_CLAIM_RELEASE_RECEIPT_SCHEMA,
        "projection_version": REFINE_CLAIM_RELEASE_PROJECTION_VERSION,
        "outcome": "no_exact_eligible_claim",
        "dry_run": True,
        "eligible": False,
        "would_release": 0,
        "changed": 0,
        "target_id": target_id,
        "expected_stage": expected_stage,
        "claim_token": claim_token,
        "expected_scope_run_id": scope_run_id,
        "manifest": None,
        "manifest_sha256": None,
    }


def release_stranded_refine_claim(
    standard_name: str,
    *,
    claim_token: str,
    expected_stage: str,
    scope_run_id: str | None = None,
    apply: bool = False,
    manifest_sha256: str | None = None,
    gc: Any | None = None,
    services: Any,
) -> dict[str, Any]:
    """Release one exact stranded name-refinement claim with full CAS fencing.

    The default performs the same lock, compare-and-set write, and postflight
    as apply, then rolls the transaction back. Apply additionally requires the
    SHA-256 emitted by that dry run and an exact ``run_id`` scope. The only
    durable changes are ``name_stage: refining -> reviewed`` and removal of the
    matching main ``claim_token``/``claimed_at`` pair.
    """
    target_id = str(standard_name).strip()
    token = str(claim_token).strip()
    scope = str(scope_run_id).strip() if scope_run_id is not None else None
    if not target_id:
        raise ValueError("standard_name must be non-empty")
    if not token:
        raise ValueError("claim_token must be non-empty")
    if expected_stage != "refining":
        raise ValueError("expected_stage must be 'refining'")
    if scope_run_id is not None and not scope:
        raise ValueError("scope_run_id must be non-empty when supplied")
    if apply and scope is None:
        raise ValueError("apply requires an exact scope_run_id")
    if apply and not manifest_sha256:
        raise ValueError("apply requires manifest_sha256")
    if manifest_sha256 is not None and not services._SHA256_RE.fullmatch(
        manifest_sha256
    ):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")

    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            try:
                params = {
                    "target_id": target_id,
                    "claim_token": token,
                    "expected_stage": expected_stage,
                    "scope_run_id": scope,
                }
                locked = [
                    dict(record)
                    for record in transaction.run(
                        _LOCK_REFINE_CLAIM_RELEASE_QUERY, **params
                    )
                ]
                if not locked:
                    transaction.rollback()
                    if apply:
                        raise services.RefineClaimReleaseConflict(
                            "no exact eligible refine claim matched the target, "
                            "stage, token, claimed_at, and scope fences"
                        )
                    return services._refine_claim_no_change_receipt(**params)
                if len(locked) != 1 or locked[0].get("id") != target_id:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim lock did not resolve exactly one target"
                    )

                before_rows = [
                    dict(record)
                    for record in transaction.run(
                        _READ_REFINE_CLAIM_RELEASE_QUERY, target_id=target_id
                    )
                ]
                if len(before_rows) != 1:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim projection did not resolve exactly one target"
                    )
                before = services._project_refine_claim_release_state(before_rows[0])
                if before["target_id"] != target_id:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim target identity drifted"
                    )
                if before["name_stage"] != expected_stage:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim stage drifted"
                    )
                if before["claim"]["token"] != token:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim token drifted"
                    )
                if before["claim"]["claimed_at"] is None:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim timestamp is absent"
                    )
                if scope is not None and before["scope"]["run_id"] != scope:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim scope drifted"
                    )
                source_bindings = before["source_bindings"]
                if not source_bindings:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim has no current source binding"
                    )
                for binding in source_bindings:
                    if binding.get("bound_names") != [{"id": target_id}]:
                        raise services.RefineClaimReleaseConflict(
                            "refine claim source does not have exactly one current "
                            "binding to the target"
                        )
                binding_ids = sorted(
                    str(binding.get("source", {}).get("id") or "")
                    for binding in source_bindings
                )
                projected_source_ids = sorted(
                    str(source_id)
                    for source_id in before["node_properties"].get("source_paths", [])
                )
                if binding_ids != projected_source_ids:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim source binding and scalar projection drifted"
                    )

                predicted_after = services._predicted_refine_claim_release_state(before)
                manifest = services._refine_claim_release_manifest(
                    target_id=target_id,
                    claim_token=token,
                    expected_stage=expected_stage,
                    scope_run_id=scope,
                    before=before,
                    predicted_after=predicted_after,
                )
                computed_manifest_sha256 = services._authority_payload_hash(manifest)
                if apply and manifest_sha256 != computed_manifest_sha256:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim manifest SHA-256 does not match current authority"
                    )

                write_rows = [
                    dict(record)
                    for record in transaction.run(
                        _WRITE_REFINE_CLAIM_RELEASE_QUERY,
                        **params,
                        expected_claimed_at=str(before["claim"]["claimed_at"]),
                    )
                ]
                if len(write_rows) != 1 or write_rows[0].get("id") != target_id:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim compare-and-set did not change exactly one target"
                    )

                after_rows = [
                    dict(record)
                    for record in transaction.run(
                        _READ_REFINE_CLAIM_RELEASE_QUERY, target_id=target_id
                    )
                ]
                if len(after_rows) != 1:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim postflight did not resolve exactly one target"
                    )
                after = services._project_refine_claim_release_state(after_rows[0])
                if after != predicted_after:
                    raise services.RefineClaimReleaseConflict(
                        "refine claim postflight detected collateral projection drift"
                    )

                if apply:
                    transaction.commit()
                else:
                    transaction.rollback()
                return {
                    "schema": REFINE_CLAIM_RELEASE_RECEIPT_SCHEMA,
                    "projection_version": REFINE_CLAIM_RELEASE_PROJECTION_VERSION,
                    "outcome": "released" if apply else "would_release",
                    "dry_run": not apply,
                    "eligible": True,
                    "would_release": 0 if apply else 1,
                    "changed": 1 if apply else 0,
                    "target_id": target_id,
                    "manifest": manifest,
                    "manifest_sha256": computed_manifest_sha256,
                    "before": before,
                    "predicted_after": predicted_after,
                    "after": after if apply else predicted_after,
                    "postflight_verified": True,
                    "collateral_proof": {
                        "node_properties_except_release_fields_unchanged": True,
                        "relationships_unchanged": True,
                        "source_bindings_unchanged": True,
                        "source_projections_unchanged": True,
                        "lineage_unchanged": True,
                    },
                }
            except BaseException:
                if not transaction.closed:
                    transaction.rollback()
                raise
    finally:
        if own:
            client.close()


@retry_on_deadlock()
def release_refine_name_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
    services: Any,
) -> int:
    """Release StandardName refine-name claims IFF token (and stage) match.

    Clears ``claimed_at`` and ``claim_token`` and reverts
    ``name_stage`` from ``'refining'`` back to ``'reviewed'`` when
    the node is still in the refining stage.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.name_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with services.GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            SET n.updated_at = datetime(),
                n.claimed_at = null,
                n.claim_token = null,
                n.name_stage = CASE
                    WHEN n.name_stage = 'refining' THEN 'reviewed'
                    ELSE n.name_stage
                END
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_refine_name_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


def resubmit_pinned_rename_for_review(
    *,
    sn_id: str,
    token: str,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    services: Any,
) -> str:
    """Route a below-threshold pinned rename to a fresh review quorum.

    A rename edit (``edit_mode='rename'``) carries an operator-chosen name
    string — the name is a fixed decision, not a draft to be rewritten. When
    such a name scores below threshold (quorum variance on a borderline name),
    the refine pool must NOT try to reword it: re-emitting the identical pinned
    name trips the self-referential-refine guard and decomposing a lexicalised
    base trips grammar validation — either way the pinned name is wrongly marked
    ``exhausted`` and its (already-superseded) predecessor's quantity silently
    drops from export.

    Instead, resubmit the SAME name to review for a fresh quorum draw
    (``name_stage`` → ``'drafted'``, reviewer name-score cleared), bounded by
    ``review_resubmit_count < rotation_cap``. A borderline name whose siblings
    accept at 0.96+ typically clears on a fresh draw. When the cap is reached
    the name is left at ``'reviewed'`` (never exhausted) for operator
    resolution; :func:`claim_refine_name_batch` excludes capped pinned renames
    so they do not re-loop.

    Token-and-stage verified (``claim_token = $token AND name_stage =
    'refining'``). Returns ``'resubmitted'``, ``'capped'``, or ``''`` (no-op:
    token/stage mismatch — a concurrent sweep or worker already moved it).
    """
    with services.GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $sn_id})
            WHERE sn.claim_token = $token AND sn.name_stage = 'refining'
            WITH sn, coalesce(sn.review_resubmit_count, 0) AS n
            SET sn.updated_at = datetime(), sn.name_stage = CASE WHEN n < $cap THEN 'drafted' ELSE 'reviewed' END,
                sn.review_resubmit_count = CASE WHEN n < $cap THEN n + 1 ELSE n END,
                sn.reviewer_score_name = CASE WHEN n < $cap
                                              THEN null ELSE sn.reviewer_score_name END,
                sn.claim_token = null,
                sn.claimed_at = null
            RETURN CASE WHEN n < $cap THEN 'resubmitted' ELSE 'capped' END AS outcome
            """,
            sn_id=sn_id,
            token=token,
            cap=rotation_cap,
        )
    return rows[0]["outcome"] if rows else ""


RESCORE_ELIGIBILITY_QUERY = """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.name_stage IN ['exhausted', 'reviewed']
              AND NOT coalesce(sn.status, '') IN ['deprecated', 'superseded']
            WITH sn, sn.name_stage AS prior_stage
            // One compare-and-set transition: if another worker moves the
            // stage before this MATCH acquires the write, no row is returned
            // and the caller refuses the rescore. Dry-run retains the same
            // eligibility query but executes no mutation.
            FOREACH (_ IN CASE WHEN $dry_run THEN [] ELSE [1] END |
                SET sn.updated_at = datetime(), sn.name_stage = 'drafted',
                    sn.reviewer_score_name = null,
                    sn.review_resubmit_count = 0,
                    // The stale diagnosis of the last refinement goes; the
                    // rotations it spent stay. A rescore buys a fresh quorum
                    // draw on the SAME name, never a fresh rewrite budget —
                    // refunding one would re-open the paid loop for a name
                    // that scores low again. `sn edit` carries the new
                    // steering information that reopens rewriting.
                    sn.refine_stop_reason = null,
                    sn.refine_stopped_at = null,
                    sn.refine_collision_name = null,
                    sn.claim_token = null,
                    sn.claimed_at = null,
                    sn.run_id = coalesce($run_id, sn.run_id),
                    sn.validation_status = 'pending',
                    sn.validation_issues = null,
                    sn.validation_layer_summary = null,
                    sn.validated_at = null
            )
            RETURN prior_stage AS prior_stage
            """


def stage_name_for_rescore(
    sn_id: str, *, run_id: str | None = None, dry_run: bool = False, services: Any
) -> dict[str, Any]:
    """Return a stranded non-accepted name to review for a fresh quorum.

    Operator recovery path for names stuck at a terminal-but-unpublished
    name-axis stage — an ``exhausted`` name (refine cap reached, or a
    borderline name wrongly exhausted) or a ``reviewed`` name whose score never
    cleared. Reverts ``name_stage`` to ``'drafted'`` so a review re-scores it
    with a fresh quorum, clears the stale reviewer name-score and the stale
    refinement diagnosis, resets the re-review budget, and clears any claim.
    ``refine_attempts`` is deliberately preserved — see the query. Edit fields
    are left intact so an attached hint (e.g. a "keep this form" steer) rides
    the fresh review.

    When *run_id* is given it is stamped on the node so a scoped review
    (``run_sn_pools(scope_run_id=run_id)``) claims exactly this name — the
    mechanism the ``sn rescore`` inline review uses. A predecessor that was
    already superseded stays superseded — the recovered name keeps its
    REFINED_FROM lineage and, once re-accepted, resolves the export gap it
    left behind.

    Refuses (``{"ok": False, "reason": ...}``) for names that must not be
    force-staged: not found, terminal catalog status, already
    ``accepted``/``approved``/``superseded``, or already live
    (``drafted``/``refining``). A terminal identity needs a signed transition
    before rescore can recover it.

    Returns ``{"ok": True, "sn_id", "prior_stage", "run_id", "dry_run"}`` on
    success.
    """
    with services.GraphClient() as gc:
        rows = gc.query(
            RESCORE_ELIGIBILITY_QUERY,
            id=sn_id,
            run_id=run_id,
            dry_run=dry_run,
        )
        if not rows:
            # Diagnostic only after the guarded transition lost eligibility;
            # this read can refine the refusal message but can never turn the
            # failed transition into success.
            current = gc.query(
                """MATCH (sn:StandardName {id: $id})
                OPTIONAL MATCH (sn)-[:HAS_SUCCESSOR]->(next:StandardName)
                RETURN sn.name_stage AS stage, sn.status AS status,
                       next.id AS successor""",
                id=sn_id,
            )
            if not current:
                return {"ok": False, "reason": f"name {sn_id!r} not found"}
            stage = current[0].get("stage")
            status = current[0].get("status")
            successor = current[0].get("successor")
            reason = f"{sn_id!r} is name_stage={stage!r}, status={status!r} — "
            if status in {"deprecated", "superseded"}:
                reason += "rescore refuses terminal status"
            elif stage == "superseded":
                reason += "rescore refuses a superseded name stage"
            else:
                reason += "rescore only recovers 'exhausted' or 'reviewed' names"
            if status in {"deprecated", "superseded"} or stage == "superseded":
                reason += (
                    "; use a signed transition in "
                    "imas_codex/standard_names/signed_manifest.py to return it "
                    "to a recoverable state"
                )
            if successor:
                reason += f"; successor {successor!r}"
            return {"ok": False, "reason": reason}
        stage = rows[0].get("prior_stage")
    result = {
        "ok": True,
        "sn_id": sn_id,
        "prior_stage": stage,
        "run_id": run_id,
        "dry_run": dry_run,
    }
    logger.info(
        "stage_name_for_rescore: %s (%s) → %s (run_id=%s)",
        sn_id,
        stage,
        stage if dry_run else "drafted",
        run_id,
    )
    return result


@retry_on_deadlock()
def restore_name_after_failed_rescore(
    sn_id: str, *, run_id: str, prior_stage: str, services: Any
) -> bool:
    """Restore a deterministically rejected rescore to its terminal stage.

    The run-id and drafted-stage guards make this a compare-and-set: a late
    failure cannot overwrite a node another worker has already advanced. The
    node remains quarantined, while every rescore-specific scope and claim is
    removed atomically so no later scoped run can pick it up for paid review.
    """
    if prior_stage not in ("exhausted", "reviewed"):
        raise ValueError(f"cannot restore rescore to stage {prior_stage!r}")
    with services.GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.name_stage = 'drafted'
              AND sn.run_id = $run_id
              AND sn.validation_status = 'quarantined'
            SET sn.updated_at = datetime(), sn.name_stage = $prior_stage,
                sn.run_id = null,
                sn.claim_token = null,
                sn.claimed_at = null
            RETURN count(sn) AS restored
            """,
            id=sn_id,
            run_id=run_id,
            prior_stage=prior_stage,
        )
    return bool(rows and rows[0].get("restored"))


_TERMINAL_REFINE_STOP_REASONS = frozenset(
    {
        RefineStopReason.successor_collision.value,
        RefineStopReason.successor_lifecycle_collision.value,
        RefineStopReason.vocabulary_gap.value,
        RefineStopReason.grammar_invalid.value,
    }
)


@retry_on_deadlock()
def stop_refine_name_attempt(
    *,
    sn_id: str,
    token: str,
    reason: str,
    detail: str = "",
    collision_name: str | None = None,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    services: Any,
) -> str:
    """Close one failed refinement attempt and decide whether more may follow.

    Records *reason* (a :class:`RefineStopReason` value) with its timestamp and,
    for a collision, the identity the refiner proposed — so the next attempt can
    report what the last one hit and a parked name can be enumerated by cause.

    The resulting stage is decided from committed state inside the write:
    ``'exhausted'`` when at least one attempt was charged and either the reason
    proves no further attempt can succeed or the charged budget is spent;
    otherwise ``'reviewed'`` so the first or remaining rotation stays
    available. Exhaustion closes an open name-steering edit, mirroring the
    acceptance path.

    Fenced on the claim token and the ``'refining'`` stage. Returns the stage
    written, or ``''`` when the fence did not match (an orphan sweep or another
    worker already moved the row).
    """
    terminal = reason in _TERMINAL_REFINE_STOP_REASONS
    with services.GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $sn_id})
            WHERE sn.claim_token = $token
              AND sn.name_stage = 'refining'
            WITH sn, CASE
                   WHEN coalesce(sn.refine_attempts, 0) > 0
                    AND ($terminal
                     OR coalesce(sn.refine_attempts, 0) >= $rotation_cap)
                   THEN 'exhausted' ELSE 'reviewed' END AS target_stage
            SET sn.updated_at = datetime(), sn.name_stage = target_stage,
                sn.validation_status = CASE
                    WHEN target_stage = 'exhausted'
                     AND $reason IN ['grammar_invalid', 'vocabulary_gap']
                    THEN 'quarantined'
                    ELSE sn.validation_status END,
                sn.validation_issues = CASE
                    WHEN target_stage = 'exhausted'
                     AND $reason IN ['grammar_invalid', 'vocabulary_gap']
                    THEN coalesce(sn.validation_issues, []) + [CASE
                        WHEN $detail = '' THEN $reason
                        ELSE $reason + ': ' + $detail END]
                    ELSE sn.validation_issues END,
                sn.refine_stop_reason = CASE
                    WHEN target_stage = 'exhausted' AND NOT $terminal
                    THEN $attempts_exhausted ELSE $reason END,
                sn.refine_stopped_at = datetime(),
                sn.refine_collision_name = $collision_name,
                sn.claim_token = null,
                sn.claimed_at = null,
                // A name-axis exhaust closes only a name-steering edit.
                sn.edit_status = CASE
                    WHEN target_stage = 'exhausted'
                     AND sn.edit_status = 'open'
                     AND sn.name_hint IS NOT NULL
                    THEN 'exhausted' ELSE sn.edit_status END,
                sn.reviewer_comments_name = CASE
                    WHEN $detail = '' THEN sn.reviewer_comments_name
                    ELSE coalesce(sn.reviewer_comments_name, '')
                         + ' [' + $reason + '] ' + $detail END
            RETURN target_stage AS stage
            """,
            sn_id=sn_id,
            token=token,
            reason=reason,
            terminal=terminal,
            rotation_cap=rotation_cap,
            collision_name=collision_name,
            detail=detail[:300],
            attempts_exhausted=services.RefineStopReason.attempts_exhausted.value,
        )
    if not rows:
        logger.debug(
            "stop_refine_name_attempt: %s no longer holds token %s — no-op",
            sn_id,
            token[:8],
        )
        return ""
    stage: str = rows[0]["stage"]
    logger.info(
        "refine_name: %s stopped at %s (reason=%s%s)",
        sn_id,
        stage,
        reason,
        f", collides with {collision_name}" if collision_name else "",
    )
    return stage


def _mark_refine_docs_exhausted(
    *, sn_id: str, token: str, error_msg: str, services: Any
) -> None:
    """Mark a name's docs exhausted when refine_docs fails deterministically.

    A docs-validation failure (e.g. the LLM consistently returns docs that
    violate the ``RefinedDocs`` length constraints) is deterministic for a
    given item — reverting ``docs_stage`` to 'reviewed' would re-claim and
    re-burn budget forever.  Move to 'exhausted' instead and record the reason.
    """
    with services.GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName {id: $sn_id})
            WHERE sn.claim_token = $token
              AND sn.docs_stage = 'refining'
            SET sn.updated_at = datetime(), sn.docs_stage = 'exhausted',
                sn.claim_token = null,
                sn.claimed_at = null,
                // A docs-axis exhaust closes only a docs-steering edit.
                sn.edit_status = CASE WHEN sn.edit_status = 'open'
                                       AND sn.docs_hint IS NOT NULL
                                      THEN 'exhausted'
                                      ELSE sn.edit_status END,
                sn.reviewer_comments_docs =
                    coalesce(sn.reviewer_comments_docs, '')
                    + ' [docs_refine_exhaust] ' + $error_msg
            """,
            sn_id=sn_id,
            token=token,
            error_msg=error_msg[:300],
        )


def _verify_docs_claim_winners(
    items: list[dict[str, Any]],
    *,
    eligible_stage: str,
    settle_seconds: float = _CLAIM_VERIFY_SETTLE_SECONDS,
    services: Any,
) -> list[dict[str, Any]]:
    """Drop docs-claim items that lost a concurrent-claim race.

    *items* are the rows returned by :func:`_claim_sn_atomic` (all sharing one
    ``claim_token``).  After a short *settle_seconds* pause — long enough for a
    lock-serialised claim burst to finish committing — this re-reads committed
    graph state and keeps only the nodes that STILL hold that token, remain at
    *eligible_stage*, AND carry the exact ``claim_seq`` this worker was
    assigned.  A race loser's ``claim_seq`` was superseded by a later
    committer's higher value, so it is dropped here before any LLM call.

    ``settle_seconds`` is overridable (tests pass ``0``); this function runs in
    a worker thread (claims go through ``asyncio.to_thread``), so the pause does
    not block the event loop.

    Returns the filtered item list (order preserved).
    """
    return services._verify_claim_winners(
        items,
        stage_field="docs_stage",
        eligible_stage=eligible_stage,
        settle_seconds=settle_seconds,
        axis="docs",
    )


def _verify_claim_winners(
    items: list[dict[str, Any]],
    *,
    stage_field: str,
    eligible_stage: str,
    settle_seconds: float,
    axis: str,
    require_validated: bool = False,
    services: Any,
) -> list[dict[str, Any]]:
    """Shared claim-race winner verifier for the docs and names axes.

    Keeps only items that still hold ``claim_token`` at *eligible_stage* with
    the assigned ``claim_seq`` after settling; see the module comment above
    :func:`_verify_docs_claim_winners` for the mechanism.
    """
    if not items:
        return items
    token = items[0].get("claim_token") or ""
    if not token:
        return items
    # Per-item (id, claim_seq). Legacy claims that predate the seq stamp fall
    # back to token+stage only so this never over-drops on missing data.
    seqs = {it["id"]: it.get("claim_seq") for it in items}
    have_seq = all(v is not None for v in seqs.values())
    # The settle only helps the claim_seq dedup — it gives a lock-serialised
    # claim burst time to finish committing so the seq re-read is decisive.
    # Without claim_seq (legacy nodes / mocked claims) there is nothing for the
    # pause to resolve, so skip it and fall through to the token-only check.
    if have_seq and settle_seconds > 0:
        services.time.sleep(settle_seconds)
    ids = [it["id"] for it in items]
    validation_gate = "AND sn.validation_status = 'valid'" if require_validated else ""
    with services.GraphClient() as gc:
        rows = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (sn:StandardName {{id: sid}})
            WHERE sn.claim_token = $token
              AND sn.{stage_field} = $eligible_stage
              {validation_gate}
            RETURN sn.id AS id, sn.claim_seq AS claim_seq
            """,
            ids=ids,
            token=token,
            eligible_stage=eligible_stage,
        )
    if have_seq:
        winners = {r["id"] for r in rows if r.get("claim_seq") == seqs[r["id"]]}
    else:
        winners = {r["id"] for r in rows}
    if len(winners) == len(items):
        return items
    logger.debug(
        "_verify_claim_winners[%s]: %d/%d survived claim-race (token=%s, stage=%s)",
        axis,
        len(winners),
        len(items),
        token[:8],
        eligible_stage,
    )
    return [it for it in items if it["id"] in winners]


def _verify_name_claim_winners(
    items: list[dict[str, Any]],
    *,
    eligible_stage: str,
    settle_seconds: float = _CLAIM_VERIFY_SETTLE_SECONDS,
    services: Any,
) -> list[dict[str, Any]]:
    """Drop name-claim items that lost a concurrent-claim race.

    Names-axis twin of :func:`_verify_docs_claim_winners` — same settle +
    ``claim_seq`` mechanism, gated on ``name_stage`` instead of ``docs_stage``.
    """
    return services._verify_claim_winners(
        items,
        stage_field="name_stage",
        eligible_stage=eligible_stage,
        settle_seconds=settle_seconds,
        axis="name",
        require_validated=True,
    )


def _verify_source_claim_winners(
    items: list[dict[str, Any]],
    *,
    settle_seconds: float = _CLAIM_VERIFY_SETTLE_SECONDS,
    services: Any,
) -> list[dict[str, Any]]:
    """Drop generate_name source-claim items that lost a concurrent-claim race.

    The generate_name pool claims :class:`StandardNameSource` nodes (gate
    ``status='extracted'``) rather than StandardName nodes, so it cannot share
    :func:`_verify_name_claim_winners`.  Same mechanism: re-read committed
    ``claim_token`` ownership and keep only the sources this worker still owns
    at the eligible ``status='extracted'`` gate, dropping race losers before
    the (paid) name-composition LLM call.

    Returns the filtered item list (order preserved).
    """
    if not items:
        return items
    tokens = {item.get("claim_token") for item in items}
    if len(tokens) != 1 or not next(iter(tokens)):
        logger.warning(
            "_verify_source_claim_winners: rejected %d claim(s) without one token",
            len(items),
        )
        return []
    token = next(iter(tokens))
    seqs = {it["id"]: it.get("claim_seq") for it in items}
    if any(seq is None for seq in seqs.values()):
        logger.warning(
            "_verify_source_claim_winners: rejected %d claim(s) without claim_seq",
            len(items),
        )
        return []
    if settle_seconds > 0:
        services.time.sleep(settle_seconds)
    ids = [it["id"] for it in items]
    with services.GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sns:StandardNameSource {id: sid})
            WHERE sns.claim_token = $token
              AND sns.status = 'extracted'
            RETURN sns.id AS id, sns.claim_seq AS claim_seq
            """,
            ids=ids,
            token=token,
        )
    winners = {row["id"] for row in rows if row.get("claim_seq") == seqs[row["id"]]}
    if len(winners) == len(items):
        return items
    logger.debug(
        "_verify_source_claim_winners: %d/%d survived claim-race (token=%s)",
        len(winners),
        len(items),
        token[:8],
    )
    return [it for it in items if it["id"] in winners]
