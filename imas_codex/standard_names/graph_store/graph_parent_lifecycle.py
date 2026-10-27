"""Graph storage operations for the standard-name pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock

from . import graph_foundation as foundation
from .graph_foundation import _CLAIM_TIMEOUT_SECONDS, DEFAULT_POOL_BATCH_SIZE

logger = logging.getLogger(__name__)


@retry_on_deadlock()
def release_all_orphan_claims(*, services: Any) -> dict[str, int]:
    """Release all claimed-but-unreleased StandardName and StandardNameSource nodes.

    Called from the ``run_sn_pools`` finally block after clean shutdown so
    that any batch still marked as claimed at process-exit is unlocked.
    Per-batch release already happens inside each process() try/finally, but
    batches in flight at the 60s grace-period timeout leave ``claimed_at``
    set permanently.  This sweep clears them unconditionally — it is safe
    because the run is over and no other process can be competing for the
    same tokens.

    A StandardName caught mid-refine sits at a transient ``'refining'`` stage.
    Clearing only its claim would strand it there — reviewable/refinable only
    after the periodic orphan sweep reverts the stage. Apply the same
    ``refining -> reviewed`` revert the orphan sweep uses (see
    ``orphan_sweep._SWEEP_QUERIES``) as part of the release, so shutdown leaves
    no node in a transient stage with a cleared claim.

    Returns a dict with ``"sn"`` and ``"sns"`` keys showing the counts
    of released nodes.
    """
    with services.GraphClient() as gc:
        sn_result = gc.query(
            """
            MATCH (n:StandardName)
            WHERE n.claimed_at IS NOT NULL
            SET n.updated_at = datetime(), n.name_stage = CASE WHEN n.name_stage = 'refining'
                                    THEN 'reviewed' ELSE n.name_stage END,
                n.docs_stage = CASE WHEN n.docs_stage = 'refining'
                                    THEN 'reviewed' ELSE n.docs_stage END,
                n.claimed_at = null,
                n.claim_token = null
            RETURN count(n) AS released
            """
        )
        sns_result = gc.query(
            """
            MATCH (n:StandardNameSource)
            WHERE n.claimed_at IS NOT NULL
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """
        )
    sn_count = sn_result[0]["released"] if sn_result else 0
    sns_count = sns_result[0]["released"] if sns_result else 0
    return {"sn": sn_count, "sns": sns_count}


def pool_pending_counts(
    *,
    min_score: float = 0.75,
    rotation_cap: int = 3,
    domain: str | None = None,
    scope_run_id: str | None = None,
    edits_only: bool = False,
    services: Any,
) -> dict[str, int]:
    """Return pending-work counts for all six worker pools in one query.

    The six predicates mirror the eligibility criteria and optional scope in
    the corresponding ``claim_*_batch`` functions but do NOT filter on
    ``claimed_at`` — the counts reflect total eligible work,
    including items currently being processed.  This is correct for
    both throttle decisions (total queue depth matters) and display
    (users want to see total pending, not just unclaimed).

    Returns a dict keyed by pool name::

        {
            "generate_name": int,
            "review_name": int,
            "refine_name": int,
            "generate_docs": int,
            "review_docs": int,
            "refine_docs": int,
        }
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    source_scope = ""
    name_scope = ""
    params: dict[str, Any] = {
        "min_score": min_score,
        "rotation_cap": rotation_cap,
        "max_compose_attempts": services._MAX_COMPOSE_CLAIM_ATTEMPTS,
        "parent_desc_placeholder": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    }
    if domain:
        source_scope += " AND s.physics_domain = $domain"
        name_scope += " AND sn.physics_domain = $domain"
        params["domain"] = domain
    if scope_run_id:
        source_scope += " AND s.run_id = $scope_run_id"
        name_scope += " AND sn.run_id = $scope_run_id"
        params["scope_run_id"] = scope_run_id
    if edits_only:
        # Name-generation claims intentionally have no edits-only source path:
        # rename/docs edits start from an already-composed StandardName.
        source_scope += " AND false"
        name_scope += " AND coalesce(sn.edit_status, '') = 'open'"

    score_gate = (
        ""
        if (scope_run_id or edits_only)
        else (
            " AND (sn.reviewer_score_name IS NOT NULL"
            " OR (coalesce(sn.origin, '') = 'derived'"
            " AND EXISTS { MATCH (kid:StandardName)-[:HAS_PARENT]->(sn)"
            " WHERE NOT (coalesce(kid.name_stage, '') IN"
            " ['superseded', 'exhausted', 'contested']) }))"
        )
    )

    cypher = f"""
    CALL {{ MATCH (s:StandardNameSource {{status: 'extracted'}})
           WHERE coalesce(s.attempt_count, 0) < $max_compose_attempts
             {source_scope}
           RETURN count(s) AS generate_name }}
    CALL {{ MATCH (sn:StandardName {{name_stage: 'drafted'}})
           WHERE sn.description IS NOT NULL
             AND sn.description <> $parent_desc_placeholder
             AND coalesce(sn.origin, '') <> 'derived'
             {name_scope}
           RETURN count(sn) AS review_name }}
    CALL {{ MATCH (sn:StandardName {{name_stage: 'reviewed'}})
           WHERE sn.reviewer_score_name < $min_score
             AND coalesce(sn.chain_length, 0) < $rotation_cap
             AND coalesce(sn.origin, '') <> 'derived'
             AND NOT (coalesce(sn.edit_mode, '') = 'rename'
                      AND coalesce(sn.review_resubmit_count, 0)
                          >= $rotation_cap)
             {name_scope}
           RETURN count(sn) AS refine_name }}
    CALL {{ MATCH (sn:StandardName
                   {{name_stage: 'accepted', docs_stage: 'pending'}})
           WHERE true {score_gate}
             {name_scope}
           RETURN count(sn) AS generate_docs }}
    CALL {{ MATCH (sn:StandardName {{docs_stage: 'drafted'}})
           WHERE NOT (sn.name_stage IN
                      ['superseded', 'exhausted', 'contested'])
             {score_gate}
             {name_scope}
           RETURN count(sn) AS review_docs }}
    CALL {{ MATCH (sn:StandardName {{docs_stage: 'reviewed'}})
           WHERE sn.reviewer_score_docs < $min_score
             AND coalesce(sn.docs_chain_length, 0) < $rotation_cap
             AND NOT (sn.name_stage IN
                      ['superseded', 'exhausted', 'contested'])
             {score_gate}
             {name_scope}
           RETURN count(sn) AS refine_docs }}
    RETURN generate_name, review_name, refine_name,
           generate_docs, review_docs, refine_docs
    """
    with GraphClient() as gc:
        rows = list(gc.query(cypher, **params))

    if not rows:
        return {
            "generate_name": 0,
            "review_name": 0,
            "refine_name": 0,
            "generate_docs": 0,
            "review_docs": 0,
            "refine_docs": 0,
        }
    r = rows[0]
    return {
        "generate_name": int(r.get("generate_name", 0)),
        "review_name": int(r.get("review_name", 0)),
        "refine_name": int(r.get("refine_name", 0)),
        "generate_docs": int(r.get("generate_docs", 0)),
        "review_docs": int(r.get("review_docs", 0)),
        "refine_docs": int(r.get("refine_docs", 0)),
    }


def scoped_terminal_residue(
    *, scope_run_id: str | None = None, drain_scope_id: str | None = None, services: Any
) -> dict[str, Any]:
    """Return claim and transient-stage residue inside one exact run scope."""
    if bool(scope_run_id) == bool(drain_scope_id):
        raise ValueError("exactly one scoped terminal selector is required")
    scope_id = scope_run_id or drain_scope_id
    scope_property = "run_id" if scope_run_id else "drain_scope_id"
    cypher = f"""
    CALL {{
      MATCH (sn:StandardName)
      WHERE sn.{scope_property} = $scope_id
        AND (sn.claim_token IS NOT NULL
             OR sn.claimed_at IS NOT NULL
             OR sn.name_stage = 'refining'
             OR sn.docs_stage = 'refining')
      WITH sn ORDER BY sn.id
      RETURN count(sn) AS name_count,
             collect({{
               id: sn.id,
               name_stage: sn.name_stage,
               docs_stage: sn.docs_stage,
               claim_token: sn.claim_token,
               claimed_at: toString(sn.claimed_at)
             }})[0..50] AS names
    }}
    CALL {{
      MATCH (sns:StandardNameSource)
      WHERE sns.{scope_property} = $scope_id
        AND (sns.claim_token IS NOT NULL OR sns.claimed_at IS NOT NULL)
      WITH sns ORDER BY sns.id
      RETURN count(sns) AS source_count,
             collect({{
               id: sns.id,
               status: sns.status,
               claim_token: sns.claim_token,
               claimed_at: toString(sns.claimed_at)
             }})[0..50] AS sources
    }}
    RETURN name_count, source_count, names, sources
    """
    with services.GraphClient() as gc:
        rows = list(gc.query(cypher, scope_id=scope_id))
    row = dict(rows[0]) if rows else {}
    name_count = int(row.get("name_count") or 0)
    source_count = int(row.get("source_count") or 0)
    return {
        "total": name_count + source_count,
        "name_count": name_count,
        "source_count": source_count,
        "names": list(row.get("names") or []),
        "sources": list(row.get("sources") or []),
    }


def _verify_enrich_parents_claim_winners(
    items: list[dict[str, Any]], *, services: Any
) -> list[dict[str, Any]]:
    """Drop enrich-parents claim items that lost a concurrent-claim race.

    Same two-step claim_token verify mandated for every claim function (see
    :func:`_verify_name_claim_winners`): two replicas bind the same eligible
    derived parent at the shared seed MATCH, the lock-serialised SET lets the
    LAST claim_token win, but BOTH replicas returned the node and would proceed
    to the (paid) enrichment LLM call. This re-reads committed state and keeps
    only nodes that STILL hold our token AND remain enrichment-eligible (a
    placeholder derived parent) — the race loser's token was overwritten, so it
    is dropped before any LLM call.

    Returns the filtered item list (order preserved).
    """
    if not items:
        return items
    token = items[0].get("claim_token") or ""
    if not token:
        return items
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    ids = [it["id"] for it in items]
    with services.GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
              AND sn.origin = 'derived'
              AND sn.description = $placeholder
            RETURN sn.id AS id
            """,
            ids=ids,
            token=token,
            placeholder=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        )
    winners = {r["id"] for r in rows}
    if len(winners) == len(items):
        return items
    logger.debug(
        "_verify_enrich_parents_claim_winners: %d/%d survived claim-race (token=%s)",
        len(winners),
        len(items),
        token[:8],
    )
    return [it for it in items if it["id"] in winners]


@retry_on_deadlock()
def claim_enrich_parents_batch(
    batch_size: int = DEFAULT_POOL_BATCH_SIZE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    domain: str | None = None,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
    edits_only: bool = False,
    *,
    services: Any,
) -> list[dict[str, Any]]:
    """Claim placeholder derived parents that still need a real description.

    Eligibility: ``origin = 'derived'`` AND the description is still the
    deterministic-parent placeholder AND the parent has at least one live
    (non-superseded/exhausted) ``HAS_PARENT`` child to ground on.  Childless
    placeholders (legitimately unscoped) never match and are skipped.

    The claim does NOT transition any stage — only ``claim_token`` /
    ``claimed_at`` are written.  The stage transition to ``'drafted'`` happens
    in :func:`persist_enriched_parent` once a real description + embedding
    exist, so a failed LLM call leaves the parent claimable as a placeholder.

    Returns claimed items as dicts with ``id``, ``name``, ``kind``, ``unit``,
    ``physics_domain``, ``claim_token``.
    """
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    where = (
        "sn.origin = 'derived'"
        " AND sn.description = $parent_desc_placeholder"
        " AND EXISTS { MATCH (child:StandardName)-[:HAS_PARENT]->(sn)"
        "   WHERE NOT coalesce(child.name_stage, '') IN ['superseded', 'exhausted', 'contested'] }"
    )
    query_params: dict[str, Any] = {
        "parent_desc_placeholder": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    }
    items = services._claim_sn_atomic(
        eligibility_where=where,
        query_params=query_params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=", coalesce(sn.name, sn.id) AS name",
        domain=domain,
        scope_run_id=scope_run_id,
        drain_scope_id=drain_scope_id,
        edits_only=edits_only,
    )
    # Two-step claim_token verify (mandated for every claim function): drop nodes
    # a concurrent replica won at the lock-serialised SET, before any enrichment
    # LLM call — otherwise multi-replica runs pay the enrichment cost N× per
    # parent (the docs/name amplification analogue).
    return services._verify_enrich_parents_claim_winners(items)


def fetch_derived_parent_children(
    parent_ids: list[str], *, services: Any
) -> dict[str, list[dict[str, Any]]]:
    """Fetch the grounding children for a batch of derived parents.

    For each parent id, returns its live (non-superseded/exhausted)
    ``HAS_PARENT`` children with the fields the enrichment prompt grounds on:
    the child standard name, its description, unit, and physics domain.  A
    child still carrying the placeholder description contributes its NAME (the
    name itself is meaningful) but an empty description, so the prompt grounds
    on whatever real descriptions exist plus the child name set.

    Returns ``{parent_id: [child_dict, ...]}``; parents with no live children
    map to an empty list.
    """
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    if not parent_ids:
        return {}
    with services.GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $parent_ids AS pid
            MATCH (child:StandardName)-[:HAS_PARENT]->(p:StandardName {id: pid})
            WHERE NOT coalesce(child.name_stage, '') IN ['superseded', 'exhausted', 'contested']
            WITH pid, child
            ORDER BY child.id
            RETURN pid AS parent_id,
                   collect({
                     name: child.id,
                     description: CASE
                         WHEN child.description = $placeholder THEN null
                         ELSE child.description END,
                     unit: child.unit,
                     physics_domain: child.physics_domain,
                     kind: child.kind
                   }) AS children
            """,
            parent_ids=parent_ids,
            placeholder=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        )
    return {r["parent_id"]: list(r["children"]) for r in rows}


_STRUCTURAL_AUTHORITY_MECHANISM = "deterministic-grammar-peel"


_STRUCTURAL_AUTHORITY_SCHEMA = "StructuralNameAuthority"


_STRUCTURAL_AUTHORITY_CANONICALIZATION = "json-sort-keys"


class StructuralAuthorityConflict(RuntimeError):
    """A structural accept no longer matches its exact live child closure."""


def _structural_authority_identities(*, services: Any) -> tuple[str, str]:
    """Return immutable identities for the executing code and schema bytes."""
    code_identity = hashlib.sha256(
        services.Path(services.__file__).read_bytes()
    ).hexdigest()
    schema_path = (
        services.Path(services.__file__).parents[1] / "schemas" / "standard_name.yaml"
    )
    schema_identity = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    return code_identity, schema_identity


def _structural_authority_snapshot(
    gc: Any, sn_id: str, *, services: Any
) -> dict[str, Any] | None:
    """Read one parent and its exact ordered live child rows."""
    rows = gc.query(
        """
        MATCH (parent:StandardName {id: $id})
        OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(parent)
        WHERE child IS NULL OR NOT coalesce(child.name_stage, '') IN
          ['superseded', 'exhausted', 'contested']
        WITH parent, child ORDER BY child.id
        RETURN parent.id AS parent_id,
               elementId(parent) AS parent_element_id,
               parent.name_stage AS name_stage,
               parent.origin AS origin,
               parent.claim_token AS claim_token,
               parent.reviewed_name_at AS reviewed_name_at,
               parent.docs_stage AS docs_stage,
               parent.validation_status AS validation_status,
               parent.embedded_at AS embedded_at,
               parent.chain_length AS chain_length,
               [(parent)-[:HAS_STRUCTURAL_AUTHORITY]->(existing)
                 | existing.id] AS authority_ids,
               [item IN collect(CASE WHEN child IS NULL THEN null ELSE {
                 id: child.id,
                 element_id: elementId(child),
                 name_stage: child.name_stage,
                 reviewer_score_name: child.reviewer_score_name,
                 reviewer_model_name: child.reviewer_model_name
               } END) WHERE item IS NOT NULL] AS children
        """,
        id=sn_id,
    )
    return dict(rows[0]) if rows else None


def _structural_authority_record(
    snapshot: dict[str, Any], *, accepting: bool, services: Any
) -> dict[str, Any]:
    """Build a content-addressed signed authority from a live child snapshot."""
    child_rows = list(snapshot.get("children") or [])
    if not child_rows:
        raise services.StructuralAuthorityConflict(
            f"derived parent {snapshot.get('parent_id')!r} has no live children"
        )
    child_ids = [str(child["id"]) for child in child_rows]
    if child_ids != sorted(child_ids) or len(child_ids) != len(set(child_ids)):
        raise services.StructuralAuthorityConflict(
            "structural authority requires unique children in canonical id order"
        )

    parent_id = str(snapshot["parent_id"])
    created_at_value = services.datetime.now(foundation.UTC)
    created_at = created_at_value.isoformat()
    code_identity, schema_identity = services._structural_authority_identities()
    signed_payload = {
        "accepted_name_id": parent_id,
        "child_rows": child_rows,
        "code_identity": code_identity,
        "created_at": created_at_value,
        "entailment_mechanism": _STRUCTURAL_AUTHORITY_MECHANISM,
        "operation": "accept-and-authorize"
        if accepting
        else "authorize-existing-accept",
        "schema_identity": schema_identity,
    }
    payload_sha256 = services._authority_payload_hash(signed_payload)
    authority_id = f"structural-authority:{payload_sha256}"
    prefix = authority_id + ":"
    participants = [
        {
            "id": prefix + "participant:parent",
            "kind": "node",
            "graph_label": "StandardName",
            "signature_sha256": services._authority_payload_hash(
                {
                    "id": parent_id,
                    "element_id": snapshot.get("parent_element_id"),
                    "name_stage": snapshot.get("name_stage"),
                }
            ),
        },
        *[
            {
                "id": prefix + f"participant:child:{index}",
                "kind": "node",
                "graph_label": "StandardName",
                "signature_sha256": services._authority_payload_hash(child),
            }
            for index, child in enumerate(child_rows)
        ],
    ]
    mutations = [
        {
            "id": prefix + "mutation:authority",
            "order": 0,
            "kind": "add_relationship",
            "participant_id": participants[0]["id"],
            "arguments": json.dumps(
                {
                    "relationship": "HAS_STRUCTURAL_AUTHORITY",
                    "child_relationship": "ENTAILED_FROM_CHILD",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    ]
    if accepting:
        mutations.append(
            {
                "id": prefix + "mutation:accept",
                "order": 1,
                "kind": "set_properties",
                "participant_id": participants[0]["id"],
                "arguments": json.dumps(
                    {"name_stage": "accepted"},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    guards = [
        {
            "id": prefix + "guard:participants",
            "kind": "participant_lock",
            "implementation": "lock-exact-parent-and-live-children",
            "participant_ids": [participant["id"] for participant in participants],
            "expected": len(participants),
        },
        {
            "id": prefix + "guard:children",
            "kind": "closure_digest",
            "implementation": "ordered-live-child-row-digest",
            "participant_ids": [participant["id"] for participant in participants[1:]],
            "expected": payload_sha256,
        },
        {
            "id": prefix + "guard:replay",
            "kind": "replay_postcondition",
            "implementation": "one-structural-authority-per-accept",
            "participant_ids": [participants[0]["id"]],
            "expected": 1,
        },
    ]
    return {
        "id": authority_id,
        "child_rows": child_rows,
        "expected_name_stage": snapshot.get("name_stage"),
        "expected_claim_token": snapshot.get("claim_token"),
        "identity": {
            "id": prefix + "identity",
            "kind": "standard_name",
            "target_id": parent_id,
        },
        "signature": {
            "id": prefix + "signature",
            "profile": "each_row",
            "sha256": payload_sha256,
            "canonicalization": _STRUCTURAL_AUTHORITY_CANONICALIZATION,
        },
        "participants": participants,
        "selection": {
            "id": prefix + "selection",
            "mode": "exact_caller_set",
            "predicate": "exact-live-derived-parent-child-set",
        },
        "mutations": mutations,
        "guards": guards,
        "orphan_policy": "refuse",
        "accepted_name_id": parent_id,
        "child_ids": child_ids,
        "children": child_ids,
        "entailment_mechanism": _STRUCTURAL_AUTHORITY_MECHANISM,
        "created_at": created_at,
        "code_identity": code_identity,
        "schema_identity": schema_identity,
        "payload_sha256": payload_sha256,
    }


def _persist_structural_authority(
    gc: Any,
    record: dict[str, Any],
    *,
    parent_updates: dict[str, Any] | None = None,
    require_derived: bool = True,
    require_source_free: bool = False,
    services: Any,
) -> bool:
    """Persist one authority and optional accept transition in one statement.

    ``require_source_free`` re-checks inside the same guarded statement that no
    ``StandardNameSource`` produces the parent. A caller that admits a parent on
    the strength of it having no producer must hold that fact through the write,
    or a source attached in between would be accepted without its name reviewed.
    """
    rows = gc.query(
        """
        MATCH (parent:StandardName {id: $parent_id})
        WHERE ($require_derived = false OR parent.origin = 'derived')
          AND ($require_source_free = false OR NOT EXISTS {
                MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(parent) })
          AND parent.name_stage = $expected_name_stage
          AND (parent.claim_token = $expected_claim_token
               OR (parent.claim_token IS NULL AND $expected_claim_token IS NULL))
          AND NOT (parent)-[:HAS_STRUCTURAL_AUTHORITY]->(:StructuralNameAuthority)
        OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(parent)
        WHERE child IS NULL OR NOT coalesce(child.name_stage, '') IN
          ['superseded', 'exhausted', 'contested']
        WITH parent, child ORDER BY child.id
        WITH parent, [c IN collect(child) WHERE c IS NOT NULL] AS children
        WHERE [c IN children | c.id] = $child_ids
          AND [c IN children | elementId(c)] = $child_element_ids
          AND size(children) > 0
        SET parent += $parent_updates,
            parent.updated_at = datetime(),
            parent._structural_authority_lock = true
        REMOVE parent._structural_authority_lock
        CREATE (authority:StructuralNameAuthority)
        SET authority = $authority_properties
        CREATE (parent)-[:HAS_STRUCTURAL_AUTHORITY]->(authority)
        CREATE (identity:RepairRowIdentity)
        SET identity = $identity
        CREATE (authority)-[:IDENTITY]->(identity)
        CREATE (signature:RepairAuthorityDigest)
        SET signature = $signature
        CREATE (authority)-[:SIGNATURES]->(signature)
        CREATE (selection:RepairSelection)
        SET selection = $selection
        CREATE (authority)-[:SELECTION]->(selection)
        WITH parent, authority, children
        UNWIND $participants AS participant_data
        CREATE (participant:RepairParticipant)
        SET participant = participant_data
        CREATE (authority)-[:PARTICIPANTS]->(participant)
        WITH DISTINCT parent, authority, children
        UNWIND $mutations AS mutation_data
        CREATE (mutation:RepairMutation)
        SET mutation = mutation_data
        CREATE (authority)-[:MUTATIONS]->(mutation)
        WITH DISTINCT parent, authority, children
        UNWIND $guards AS guard_data
        CREATE (guard:RepairGuard)
        SET guard = guard_data
        CREATE (authority)-[:GUARDS]->(guard)
        WITH DISTINCT parent, authority, children
        UNWIND children AS entailing_child
        CREATE (authority)-[:ENTAILED_FROM_CHILD]->(entailing_child)
        WITH DISTINCT parent, authority
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(parent)
        FOREACH (_ IN CASE WHEN $description IS NULL OR source IS NULL
                           THEN [] ELSE [1] END |
          SET source.description = $description)
        RETURN DISTINCT parent.name_stage AS name_stage,
                        authority.id AS authority_id
        """,
        parent_id=record["accepted_name_id"],
        child_ids=record["child_ids"],
        child_element_ids=[child["element_id"] for child in record["child_rows"]],
        child_rows=record["child_rows"],
        expected_name_stage=record["expected_name_stage"],
        expected_claim_token=record["expected_claim_token"],
        require_derived=require_derived,
        require_source_free=require_source_free,
        parent_updates=parent_updates or {},
        authority_properties={
            key: record[key]
            for key in (
                "id",
                "orphan_policy",
                "accepted_name_id",
                "child_ids",
                "children",
                "entailment_mechanism",
                "created_at",
                "code_identity",
                "schema_identity",
            )
        }
        | {
            "identity": record["identity"]["id"],
            "signatures": [record["signature"]["id"]],
            "participants": [item["id"] for item in record["participants"]],
            "selection": record["selection"]["id"],
            "mutations": [item["id"] for item in record["mutations"]],
            "guards": [item["id"] for item in record["guards"]],
        },
        identity=record["identity"],
        signature=record["signature"],
        selection=record["selection"],
        participants=record["participants"],
        mutations=record["mutations"],
        guards=record["guards"],
        description=(parent_updates or {}).get("description"),
    )
    returned_authority_ids = {row.get("authority_id") for row in rows}
    return returned_authority_ids == {record["id"]}


@retry_on_deadlock()
def persist_enriched_parent(
    *,
    sn_id: str,
    claim_token: str,
    description: str,
    embedding: list[float] | None,
    model: str,
    run_id: str | None = None,
    services: Any,
) -> str:
    """Persist a synthesised derived-parent description and accept structurally.

    A derived parent's NAME is a deterministic grammar peel that already passed
    the two-clause admission gate, and its DESCRIPTION is a faithful
    generalisation over its already-RD-quorum-accepted children — so it inherits
    name validity by construction.  It therefore SKIPS REVIEW_NAME: routing a
    structurally-fixed abstraction through a name quorum that systematically
    penalises it for being less specific than its children only sends it to a
    futile refine→exhaust (measured: ~66% of derived parents scored <0.85,
    mean 0.778, despite clean grammar).  The description's quality is still
    gated downstream on the DOCS axis (generate_docs + review_docs).

    Single write:

    1. Verify the claim (token match OR cleared by the orphan sweep) and that
       the node is still a derived parent.
    2. SET ``description`` + ``embedding`` (+ ``embedded_at``), record the
       enrichment model/time, and transition ``name_stage`` directly to
       ``'accepted'``. The name is NOT scored — a derived parent name is a
       deterministic grammar peel that is never reviewed/refined, so a
       ``reviewer_score_name`` would be meaningless. ``reviewer_model_name`` is
       stamped ``'structural-inheritance'`` (audit marker) and ``reviewed_name_at``
       is set as the "name finalised, docs may proceed" signal. ``docs_stage``
       defaults to ``'pending'`` so it becomes docs-eligible immediately (the
       docs gates admit derived parents via ``origin``, not a name score).
    3. Mirror the description onto the parent's ``StandardNameSource`` so
       consolidation / export see the real text.

    Returns the new ``name_stage`` (``'accepted'``), or ``''`` when the node no
    longer matched (concurrent winner / not a derived parent).
    """
    description = services._sanitize_doc_text(description)
    with services.GraphClient() as gc:
        snapshot = services._structural_authority_snapshot(gc, sn_id)
        if snapshot and "parent_id" not in snapshot:
            return str(snapshot.get("name_stage") or "")
        if (
            snapshot is None
            or snapshot.get("origin") != "derived"
            or snapshot.get("claim_token") not in (claim_token, None)
            or not snapshot.get("children")
            or snapshot.get("authority_ids")
        ):
            logger.debug(
                "persist_enriched_parent: %s no longer matched (concurrent "
                "winner, missing child authority, or not a derived parent) — no-op",
                sn_id,
            )
            return ""
        record = services._structural_authority_record(snapshot, accepting=True)
        now = services.datetime.now(foundation.UTC)
        applied = services._persist_structural_authority(
            gc,
            record,
            parent_updates={
                "description": description,
                "embedding": embedding,
                "embedded_at": (
                    now if embedding is not None else snapshot.get("embedded_at")
                ),
                "parent_enriched_at": now,
                "parent_enrich_model": model,
                "name_stage": "accepted",
                "reviewer_model_name": "structural-inheritance",
                "reviewed_name_at": snapshot.get("reviewed_name_at") or now,
                "docs_stage": snapshot.get("docs_stage") or "pending",
                "validation_status": snapshot.get("validation_status") or "valid",
                "needs_composition": None,
                "claim_token": None,
                "claimed_at": None,
            },
        )
        if not applied:
            raise services.StructuralAuthorityConflict(
                f"derived parent {sn_id!r} changed before atomic acceptance"
            )

    new_stage = "accepted"
    logger.info(
        "\033[32menrich_parents\033[0m: %s → name_stage=%s — %s",
        sn_id,
        new_stage,
        (description or "")[:100],
    )
    # NB: deliberately no bump_sn_run_counter here — the SNRun ``names_enriched``
    # scalar is reconciled against the generate_docs pool total (see
    # run_sn_pools' async-counter drift check); enrich_parents telemetry is
    # surfaced via the per-pool completed-count log instead. The ``run_id``
    # parameter is retained for signature symmetry with the other persist_*
    # functions and possible future per-pool counters.
    return new_stage


_UNBACKED_SOURCE_PATH_SCALARS = ("source_path", "source_paths")


def _structural_accept_route(
    *, origin: str | None, producer_count: int, live_child_count: int, services: Any
) -> str | None:
    """Return how a stuck structural parent may be accepted, or ``None``.

    ``derived``
        The parent already carries the derived-scaffold origin. Unchanged
        behaviour: a derived parent's name is a grammar peel over its children
        and is never name-reviewed, whether or not a source also produces it.

    ``source_free``
        The origin scalar is absent, yet the edges say this name is a structural
        parent that NO source produces. There is nothing to review the name
        against — the review it would need can never be earned — so its children
        entail it exactly as they entail a stamped derived parent. Decided from
        the edges rather than from ``origin``/``source_path``, because a scalar
        that disagrees with the topology is the stale half of the pair.

    A name with a live producing source and no scaffold origin is a reviewable
    name and is refused, as is any name without a live child.
    """
    if live_child_count < 1:
        return None
    if origin == "derived":
        return "derived"
    if origin is None and producer_count == 0:
        return "source_free"
    return None


@retry_on_deadlock()
def structural_accept_derived_parents(gc: Any | None = None, *, services: Any) -> int:
    """Accept any derived parent stuck on the name axis structurally.

    A derived parent's name is a deterministic grammar peel that already passed
    the admission gate and generalises its accepted children by construction —
    so it is NEVER name-reviewed or refined (both claim gates exclude
    ``origin='derived'``). The enrich pool accepts placeholder parents directly
    (:func:`persist_enriched_parent`), but a derived parent can still reach
    ``drafted`` / ``reviewed`` / ``exhausted`` via OTHER paths — a child's
    refine_name producing the parent-equal general name, or a legacy
    materialise routing. Without this they would strand there forever (review
    no longer claims them, refine excludes them).

    This promotes every such parent that has a REAL (non-placeholder)
    description to ``name_stage='accepted'`` (no ``reviewer_score_name`` — the
    name is never reviewed, so scoring it is meaningless), stamping
    ``reviewer_model_name='structural-inheritance'`` and ``reviewed_name_at``
    (the "name finalised" signal the docs axis keys on). Idempotent and
    self-healing — safe to call at every ``sn run`` startup. Placeholder-
    description parents are left for the enrich pool (or are childless zombies).
    Returns the count promoted.

    A parent can also be a structural scaffold WITHOUT carrying the origin —
    a name whose producing source moved to another identity, leaving children
    attached and no ``PRODUCED_NAME`` edge behind. Its name cannot be reviewed
    either (there is no source to review it against), so it strands in exactly
    the same way. :func:`_structural_accept_route` admits that case from the
    edges, and the promotion stamps the scaffold origin and clears the path
    scalars no edge backs. The widening is fenced by ``require_source_free``:
    a name with a live producing source is refused, in the selection and again
    inside the guarded write.
    """
    from imas_codex.standard_names.defaults import (
        DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
    )

    _own_gc = gc is None
    if _own_gc:
        gc = services.GraphClient().__enter__()
    try:
        rows = gc.query(
            """
            MATCH (sn:StandardName)
            WHERE (sn.origin = 'derived' OR sn.origin IS NULL)
              AND sn.name_stage IN ['drafted', 'reviewed', 'exhausted', 'refining']
              AND sn.description IS NOT NULL
              AND sn.description <> $ph
              AND NOT (sn)-[:HAS_STRUCTURAL_AUTHORITY]->(
                :StructuralNameAuthority)
              AND EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(sn) }
            RETURN sn.id AS id,
                   sn.origin AS origin,
                   count {
                     (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
                   } AS producer_count,
                   count {
                     (child:StandardName)-[:HAS_PARENT]->(sn)
                     WHERE NOT coalesce(child.name_stage, '') IN
                       ['superseded', 'exhausted', 'contested']
                   } AS live_child_count
            ORDER BY sn.id
            """,
            ph=DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        )
        promoted = 0
        for row in rows:
            route = services._structural_accept_route(
                origin=row.get("origin"),
                producer_count=int(row.get("producer_count") or 0),
                live_child_count=int(row.get("live_child_count") or 0),
            )
            if route is None:
                continue
            snapshot = services._structural_authority_snapshot(gc, str(row["id"]))
            if snapshot is None or not snapshot.get("children"):
                continue
            record = services._structural_authority_record(snapshot, accepting=True)
            now = services.datetime.now(foundation.UTC)
            parent_updates = {
                "name_stage": "accepted",
                "reviewer_model_name": "structural-inheritance",
                "reviewed_name_at": snapshot.get("reviewed_name_at") or now,
                "docs_stage": snapshot.get("docs_stage") or "pending",
                "chain_length": snapshot.get("chain_length") or 0,
                "claim_token": None,
                "claimed_at": None,
            }
            if route == "source_free":
                # Promotion is the statement that this name is a source-free
                # scaffold, so it stamps the scaffold origin and drops the path
                # scalars no edge backs in the same guarded write. Leaving them
                # would keep a path this name never had readable as provenance.
                parent_updates["origin"] = "derived"
                parent_updates.update(dict.fromkeys(_UNBACKED_SOURCE_PATH_SCALARS))
            if not services._persist_structural_authority(
                gc,
                record,
                parent_updates=parent_updates,
                require_derived=route == "derived",
                require_source_free=route == "source_free",
            ):
                raise services.StructuralAuthorityConflict(
                    f"derived parent {row['id']!r} changed before atomic acceptance"
                )
            promoted += 1
        if promoted:
            logger.info(
                "structural_accept_derived_parents: promoted %d derived parent(s) "
                "to accepted (structural inheritance)",
                promoted,
            )
        return promoted
    finally:
        if _own_gc:
            gc.__exit__(None, None, None)


@retry_on_deadlock()
def backfill_structural_name_authorities(
    *, receipt_path: Path | None = None, gc: Any | None = None, services: Any
) -> dict[str, Any]:
    """Authorize every childful bare structural accept, exactly once.

    The marker selects the historical cohort but grants no authority. Each
    written record is instead signed from the exact ordered live child rows.
    Childless parents remain unscoped and are reported without mutation.
    """
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        candidate_rows = client.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.name_stage = 'accepted'
              AND sn.reviewer_score_name IS NULL
              AND sn.reviewer_model_name = 'structural-inheritance'
              AND NOT (sn)-[:HAS_STRUCTURAL_AUTHORITY]->(
                :StructuralNameAuthority)
            RETURN sn.id AS id
            ORDER BY sn.id
            """
        )
        lacking_before = len(candidate_rows)
        written: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        refused: list[dict[str, str]] = []
        for candidate in candidate_rows:
            sn_id = str(candidate["id"])
            snapshot = services._structural_authority_snapshot(client, sn_id)
            if snapshot is None:
                refused.append({"id": sn_id, "reason": "parent disappeared"})
                continue
            if snapshot.get("authority_ids"):
                refused.append(
                    {"id": sn_id, "reason": "authority appeared during selection"}
                )
                continue
            if not snapshot.get("children"):
                skipped.append({"id": sn_id, "reason": "no live children"})
                continue
            record = services._structural_authority_record(snapshot, accepting=False)
            if not services._persist_structural_authority(
                client, record, parent_updates={}, require_derived=False
            ):
                refused.append(
                    {"id": sn_id, "reason": "authority guard refused changed row"}
                )
                continue
            written.append(
                {
                    "id": sn_id,
                    "authority_id": record["id"],
                    "child_ids": record["child_ids"],
                    "payload_sha256": record["payload_sha256"],
                }
            )

        eligible_before = lacking_before - len(skipped)
        if refused:
            raise services.StructuralAuthorityConflict(
                "structural authority backfill refused rows: "
                + "; ".join(f"{row['id']}: {row['reason']}" for row in refused)
            )
        if len(written) != eligible_before:
            raise services.StructuralAuthorityConflict(
                "structural authority receipt cardinality mismatch: "
                f"eligible={eligible_before}, written={len(written)}"
            )
        receipt = {
            "schema": "imas-codex.structural-authority-backfill-receipt",
            "created_at": services.datetime.now(foundation.UTC).isoformat(),
            "lacking_authority_before": lacking_before,
            "eligible_before": eligible_before,
            "written_count": len(written),
            "written": written,
            "skipped_count": len(skipped),
            "skipped": skipped,
            "refused_count": len(refused),
            "refused": refused,
        }
        if receipt_path is not None:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        return receipt
    finally:
        if own:
            client.close()


def find_orphan_parent_source_candidates(
    gc: Any | None = None, *, services: Any
) -> list[dict[str, Any]]:
    """Return childful parents eligible for structural source reconstruction."""
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        return list(
            client.query(
                """
                MATCH (parent:StandardName)
                WHERE NOT ( (:StandardNameSource)-[:PRODUCED_NAME]->(parent) )
                  AND NOT coalesce(parent.name_stage, '') IN
                    ['superseded', 'exhausted', 'contested']
                OPTIONAL MATCH (structural_source:StandardNameSource {
                  id: 'derived:' + parent.id
                })
                WITH parent, structural_source
                WHERE structural_source IS NULL
                   OR structural_source.status <> 'stale'
                WITH parent, structural_source
                WHERE structural_source IS NULL OR NOT EXISTS {
                  MATCH (structural_source)-[:PRODUCED_NAME]->(bound)
                  WHERE bound <> parent
                    AND NOT coalesce(bound.name_stage, '') IN
                      ['superseded', 'exhausted', 'contested']
                }
                MATCH (child)-[:HAS_PARENT]->(parent)
                WITH parent,
                     count(child) AS total_children,
                     count(CASE WHEN child.name_stage IS NOT NULL THEN 1 END)
                       AS composed
                WHERE total_children >= 1 AND total_children = composed
                MATCH (c)-[:HAS_PARENT]->(parent)
                OPTIONAL MATCH (cs:StandardNameSource)-[:PRODUCED_NAME]->(c)
                OPTIONAL MATCH (cs)-[:FROM_DD_PATH]->(imas:IMASNode)
                WITH parent,
                     [p IN collect(DISTINCT imas.id) WHERE p IS NOT NULL]
                       AS dd_paths
                RETURN parent.id AS parent_id,
                       parent.origin AS origin,
                       dd_paths
                ORDER BY parent.id
                """
            )
        )
    finally:
        if own:
            client.close()


def classify_orphan_parent_source_candidates(
    gc: Any, candidates: list[dict[str, Any]] | None = None, *, services: Any
) -> dict[str, list[dict[str, Any]]]:
    """Partition parent-source candidates by structural admission.

    Only ``origin='derived'`` nodes are structural scaffolds owned by the
    parent admission gate. Pipeline and catalog nodes can also be HAS_PARENT
    targets after refinement, but their identity and lifecycle remain owned by
    those origin-specific paths; provenance recovery must not reinterpret or
    retire them as derived scaffolds.

    Returns the exact ``repairable`` rows safe to pass to source seeding and the
    ``rejected_derived`` rows that must remain visible to fallback/unresolved
    classification until lifecycle normalization retires them.
    """
    rows = (
        services.find_orphan_parent_source_candidates(gc=gc)
        if candidates is None
        else list(candidates)
    )
    from imas_codex.standard_names.parents import is_admissible_parent_name

    repairable: list[dict[str, Any]] = []
    rejected_derived: list[dict[str, Any]] = []
    for row in rows:
        if row.get("origin") != "derived":
            repairable.append(row)
            continue
        result = is_admissible_parent_name(row["parent_id"], gc)
        if result.admit:
            repairable.append(row)
        else:
            rejected_derived.append({**row, "admission_reason": result.reason})
    return {
        "repairable": repairable,
        "rejected_derived": rejected_derived,
    }


def reconcile_orphan_parent_sources(
    gc: Any | None = None,
    *,
    dry_run: bool = False,
    classification: dict[str, list[dict[str, Any]]] | None = None,
    services: Any,
) -> int:
    """Seed the missing structural provenance source for childful parents.

    A structural parent (a ``StandardName`` that is the ``HAS_PARENT`` target of
    at least one child) records its provenance as its own ``derived:<name>``
    source linked via ``PRODUCED_NAME``. A common child DD path does not realize
    the parent, so structural provenance has no ``FROM_DD_PATH`` relationship.
    Both normal materialization and this recovery path use
    :func:`_derived_parent_source_metadata`, so they cannot diverge on source
    identity.

    So a parent that acquires a non-null ``name_stage`` *before* it is seeded is
    skipped forever and left a provenance orphan — the exact leak this repairs.
    Two paths do it: a child's ``REFINE_NAME`` producing the parent-general name
    (which routes that name through ``REVIEW_NAME`` to ``accepted`` as
    ``origin='pipeline'``), and :func:`structural_accept_derived_parents` (which
    flips the stage to ``accepted`` but writes no source). Both leave an accepted
    parent with no ``PRODUCED_NAME`` source — a violation of the ledger invariant
    (:func:`~imas_codex.standard_names.ledger.find_provenance_orphans`).

    This decouples source-seeding from the acceptance path. For every parent with
    at least one child, all children composed, and no ``PRODUCED_NAME`` source, it
    materializes the structural source + ``PRODUCED_NAME`` without a DD
    realization edge. It is origin- and operator-kind-agnostic, but only a live
    parent may receive provenance. The canonical ``derived:<name>`` source must
    also be unbound or bound only to this parent; a source migrated to another
    live target is current authority for that target and must never be reused to
    revive its terminal predecessor. This only records provenance and NEVER
    mutates the parent's ``name_stage`` / ``origin`` / ``description``.
    Idempotent: a parent that gains a source drops out of the selector. A stale
    structural source is a lifecycle tombstone and is never rebound implicitly;
    an explicit lifecycle transition must make it eligible first. Childless
    sourceless names are not parents and are left untouched (a distinct stranded-
    name concern). Every derived candidate is re-checked against the current
    admission gate before seeding; an invalid scaffold remains source-less for
    ledgered lifecycle retirement rather than being legitimized by synthetic
    provenance. Non-derived parents bypass structural admission and retain their
    origin-owned recovery path.

    With ``dry_run=True``, returns the number eligible without writing.
    Otherwise returns the number of parent sources seeded.
    """
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        if classification is None:
            classification = services.classify_orphan_parent_source_candidates(client)
        for row in classification["rejected_derived"]:
            logger.warning(
                "reconcile_orphan_parent_sources: refusing to seed "
                "inadmissible derived parent %s: %s",
                row["parent_id"],
                row["admission_reason"],
            )
        rows = classification["repairable"]
        if dry_run:
            return len(rows)
        seeded = 0
        for r in rows:
            parent_id = r["parent_id"]
            meta = services._derived_parent_source_metadata(parent_id)
            result = client.query(
                """
                MATCH (parent:StandardName {id: $parent_id})
                OPTIONAL MATCH (existing_source:StandardNameSource {
                  id: $source_node_id
                })
                WITH parent, existing_source
                WHERE NOT coalesce(parent.name_stage, '') IN
                        ['superseded', 'exhausted', 'contested']
                  AND NOT EXISTS {
                    MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(parent)
                  }
                  AND (existing_source IS NULL
                       OR existing_source.status <> 'stale')
                  AND (existing_source IS NULL OR NOT EXISTS {
                    MATCH (existing_source)-[:PRODUCED_NAME]->(bound)
                    WHERE bound <> parent
                      AND NOT coalesce(bound.name_stage, '') IN
                        ['superseded', 'exhausted', 'contested']
                  })
                MERGE (sns:StandardNameSource {id: $source_node_id})
                  ON CREATE SET sns.created_at = datetime(), sns.attempt_count = 0
                SET sns.status = coalesce(sns.status, 'composed'),
                    sns.source_type = $source_type,
                    sns.source_id = $source_id,
                    sns.batch_key = coalesce(sns.batch_key, $batch_key),
                    sns.composed_at = coalesce(sns.composed_at, datetime()),
                    sns.produced_sn_id = parent.id,
                    sns.claimed_at = null,
                    sns.claim_token = null
                MERGE (sns)-[:PRODUCED_NAME]->(parent)
                RETURN count(parent) AS seeded
                """,
                parent_id=parent_id,
                **meta,
            )
            seeded += result[0]["seeded"] if result else 1
    finally:
        if own:
            client.close()

    if seeded:
        logger.info(
            "reconcile_orphan_parent_sources: seeded %d missing parent "
            "provenance source(s)",
            seeded,
        )
    return seeded


def reconcile_orphan_parent_sources_batched(
    gc: Any,
    rows: list[dict[str, Any]],
    *,
    event_by_parent: dict[str, dict[str, Any]] | None = None,
    services: Any,
) -> int:
    """Seed an exact accepted-parent cohort in one graph query."""
    items = []
    for row in rows:
        parent_id = str(row["parent_id"])
        items.append(
            {
                "parent_id": parent_id,
                **services._derived_parent_source_metadata(parent_id),
                "event": (event_by_parent or {}).get(parent_id),
            }
        )
    result = gc.query(
        """
        // STRUCTURAL_CLOSURE_BATCH_SEED_SOURCES
        UNWIND $items AS requested_item
        OPTIONAL MATCH (stale_source:StandardNameSource {
          id: requested_item.source_node_id
        })
        WITH collect(requested_item) AS requested_items,
             count(CASE WHEN stale_source.status = 'stale' THEN 1 END)
               AS stale_source_count,
             count(CASE WHEN EXISTS {
               MATCH (stale_source)-[:PRODUCED_NAME]->(bound_target)
               WHERE bound_target.id <> requested_item.parent_id
                 AND NOT coalesce(bound_target.name_stage, '') IN
                   ['superseded', 'exhausted', 'contested']
             } THEN 1 END) AS conflicting_binding_count
        WHERE stale_source_count = 0 AND conflicting_binding_count = 0
        UNWIND requested_items AS item
        MATCH (parent:StandardName {id: item.parent_id})
        WHERE parent.name_stage = 'accepted'
          AND NOT EXISTS {
            MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(parent)
          }
        MERGE (source:StandardNameSource {id: item.source_node_id})
          ON CREATE SET source.created_at = datetime(), source.attempt_count = 0
        SET source.status = coalesce(source.status, 'composed'),
            source.source_type = item.source_type,
            source.source_id = item.source_id,
            source.batch_key = coalesce(source.batch_key, item.batch_key),
            source.composed_at = coalesce(source.composed_at, datetime()),
            source.produced_sn_id = parent.id,
            source.claimed_at = null,
            source.claim_token = null
        MERGE (source)-[:PRODUCED_NAME]->(parent)
        FOREACH (_ IN CASE WHEN item.event IS NULL THEN [] ELSE [1] END |
          CREATE (change:StandardNameChange)
          SET change = item.event
          CREATE (parent)-[:HAS_INTERNAL_CHANGE]->(change)
        )
        RETURN collect(parent.id) AS ids
        """,
        items=items,
    )
    expected = {str(item["parent_id"]) for item in items}
    seeded_ids = set(result[0].get("ids") or []) if len(result) == 1 else set()
    if seeded_ids != expected:
        raise RuntimeError("batched accepted-parent source cardinality changed")
    return len(expected)


@retry_on_deadlock()
def release_enrich_parents_claims(
    *, sn_ids: list[str], claim_token: str, services: Any
) -> int:
    """Release enrich_parents claims (token-verified, no stage change).

    Used both as the pool release adapter (clears any unprocessed items in a
    batch) and on LLM/processing failure — the parent stays a placeholder and
    is re-claimed on a later pass.
    """
    if not sn_ids:
        return 0
    with services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sn_ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
            SET sn.updated_at = datetime(), sn.claim_token = null,
                sn.claimed_at  = null
            RETURN count(sn) AS released
            """,
            sn_ids=sn_ids,
            token=claim_token,
        )
    return result[0]["released"] if result else 0
