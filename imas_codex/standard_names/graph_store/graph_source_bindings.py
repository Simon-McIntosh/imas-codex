"""Graph storage operations for the standard-name pipeline.

Source claims, fences, outcomes and retry state.
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient

from . import graph_foundation as foundation

logger = logging.getLogger(__name__)


def write_enrichment_results(
    results: list[dict[str, Any]],
    *,
    services: Any,
    override: bool = False,
) -> int:
    """Write enrichment results back to graph.

    Only updates doc fields: description, documentation, links,
    validity_domain, constraints. Clears review_input_hash to invalidate
    stale reviews.

    Does NOT touch: id, kind, unit, model, grammar_parse_version, validation_diagnostics_json, etc.

    Parameters
    ----------
    override:
        When ``True``, bypass pipeline protection on catalog-edited names.

    Returns the number of nodes updated.
    """
    if not results:
        return 0

    # Pipeline protection
    from imas_codex.standard_names.protection import filter_protected

    results, skipped = filter_protected(results, override=override)
    if skipped:
        logger.warning(
            "write_enrichment_results: stripped protected fields from %d name(s): %s",
            len(skipped),
            ", ".join(skipped[:5]),
        )
    if not results:
        return 0

    with services.GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id})
            SET sn.updated_at = datetime(), sn.description = b.description,
                sn.documentation = b.documentation,
                sn.links = b.links,
                sn.validity_domain = b.validity_domain,
                sn.constraints = b.constraints,
                sn.link_status = b.link_status,
                sn.enriched_at = datetime(),
                sn.review_input_hash = null
            """,
            batch=[
                {
                    "id": r["id"],
                    "description": r.get("description") or "",
                    "documentation": r.get("documentation") or "",
                    "links": r.get("links") or None,
                    "validity_domain": r.get("validity_domain"),
                    "constraints": r.get("constraints") or None,
                    "link_status": services._compute_link_status(r.get("links")),
                }
                for r in results
            ],
        )

    logger.info("Enriched %d StandardName nodes", len(results))
    return len(results)


# =============================================================================
# StandardNameSource CRUD
# =============================================================================

_VALID_PIPELINE_SOURCE_TYPES = {"dd", "signals"}
_MAX_SOURCE_COMPOSE_HINT_CHARS = 4000
_MAX_SOURCE_COMPOSE_HINT_REASON_CHARS = 2000


def _exact_dd_source_id(dd_path: str, *, services: Any) -> tuple[str, str]:
    """Return canonical source id and bare path for one exact DD source."""
    raw = str(dd_path).strip()
    if raw.startswith("dd:"):
        raw = raw[3:]
    if (
        not raw
        or raw.startswith("signals:")
        or any(char in raw for char in "*?[]{}")
        or any(char.isspace() for char in raw)
    ):
        raise ValueError("source-hint requires one exact DD path without wildcards")
    return f"dd:{raw}", raw


def _source_compose_hint_decision(
    gc: Any,
    *,
    services: Any,
    source_id: str,
    replace: bool,
) -> dict[str, Any]:
    """Classify whether an exact DD source can accept compose steering."""
    rows = gc.query(
        """
        OPTIONAL MATCH (sns:StandardNameSource {id: $source_id})
        WITH sns, CASE WHEN sns IS NULL THEN false ELSE EXISTS {
          MATCH (sns)-[:PRODUCED_NAME]->(live:StandardName)
          WHERE NOT (coalesce(live.name_stage, '') IN $terminal_stages)
        } END AS has_live_binding
        RETURN sns.id AS id,
               sns.source_id AS source_path,
               sns.source_type AS source_type,
               sns.status AS source_status,
               sns.claimed_at AS claimed_at,
               coalesce(sns.attempt_count, 0) AS attempt_count,
               sns.compose_hint_status AS compose_hint_status,
               CASE
                 WHEN sns IS NULL THEN 'missing_source'
                 WHEN sns.source_type <> 'dd' THEN 'not_dd_source'
                 WHEN sns.claimed_at IS NOT NULL THEN 'active_claim'
                 // Only a source proven to be sitting at `extracted` may take
                 // a compose steer; a source without a status is unknown and
                 // is refused here rather than falling through to eligible.
                 WHEN coalesce(sns.status, '') <> 'extracted'
                   THEN 'source_not_extracted'
                 WHEN coalesce(sns.attempt_count, 0) >= $attempt_cap
                   THEN 'attempt_cap_reached'
                 WHEN has_live_binding THEN 'live_name_binding'
                 WHEN sns.compose_hint_status = 'open' AND NOT $replace
                   THEN 'open_hint_exists'
                 ELSE 'eligible'
               END AS decision
        """,
        source_id=source_id,
        replace=replace,
        attempt_cap=services._MAX_COMPOSE_CLAIM_ATTEMPTS,
        terminal_stages=sorted(foundation._TERMINAL_BINDING_NAME_STAGES),
    )
    return dict(rows[0]) if rows else {"decision": "missing_source", "id": None}


@retry_on_deadlock()
def set_source_compose_hint(
    dd_path: str,
    *,
    services: Any,
    hint: str,
    reason: str,
    replace: bool = False,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Open durable composition steering for one exact DD source.

    Eligibility and the write enforce the same claim, lifecycle, attempt, and
    live-binding predicates. A claim turnover between classification and write
    therefore refuses the update rather than steering an in-flight worker.
    """
    source_id, source_path = services._exact_dd_source_id(dd_path)
    hint = str(hint).strip()
    reason = str(reason).strip()
    if not hint:
        raise ValueError("source-hint requires a non-empty hint")
    if not reason:
        raise ValueError("source-hint requires a non-empty reason")
    if len(hint) > _MAX_SOURCE_COMPOSE_HINT_CHARS:
        raise ValueError(
            f"source hint exceeds {_MAX_SOURCE_COMPOSE_HINT_CHARS} characters"
        )
    if len(reason) > _MAX_SOURCE_COMPOSE_HINT_REASON_CHARS:
        raise ValueError(
            "source hint reason exceeds "
            f"{_MAX_SOURCE_COMPOSE_HINT_REASON_CHARS} characters"
        )

    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        current = services._source_compose_hint_decision(
            client,
            source_id=source_id,
            replace=replace,
        )
        result = {
            **current,
            "source_id": source_id,
            "source_path": source_path,
            "eligible": current.get("decision") == "eligible",
            "applied": False,
            "replaced": False,
            "dry_run": dry_run,
        }
        if current.get("decision") != "eligible":
            return result
        replacing_open = current.get("compose_hint_status") == "open"
        if dry_run:
            result["decision"] = "would_replace" if replacing_open else "would_set"
            result["replaced"] = replacing_open
            return result

        rows = client.query(
            """
            MATCH (sns:StandardNameSource {id: $source_id})
            WHERE sns.source_type = 'dd'
              AND sns.status = 'extracted'
              AND sns.claimed_at IS NULL
              AND coalesce(sns.attempt_count, 0) < $attempt_cap
              AND NOT EXISTS {
                MATCH (sns)-[:PRODUCED_NAME]->(live:StandardName)
                WHERE NOT (coalesce(live.name_stage, '') IN $terminal_stages)
              }
              AND ($replace OR coalesce(sns.compose_hint_status, '') <> 'open')
            WITH sns, sns.compose_hint_status = 'open' AS replaced
            SET sns.compose_hint = $hint,
                sns.compose_hint_reason = $reason,
                sns.compose_hint_status = 'open',
                sns.compose_hint_requested_at = datetime(),
                sns.compose_hint_consumed_at = null
            RETURN sns.id AS id, sns.source_id AS source_path, replaced
            """,
            source_id=source_id,
            hint=hint,
            reason=reason,
            replace=replace,
            attempt_cap=services._MAX_COMPOSE_CLAIM_ATTEMPTS,
            terminal_stages=sorted(foundation._TERMINAL_BINDING_NAME_STAGES),
        )
        if rows:
            result.update(
                applied=True,
                replaced=bool(rows[0].get("replaced")),
                decision="replaced" if rows[0].get("replaced") else "set",
            )
            return result

        latest = services._source_compose_hint_decision(
            client,
            source_id=source_id,
            replace=replace,
        )
        latest_decision = latest.get("decision", "compare_and_set_lost")
        if latest_decision == "eligible":
            latest_decision = "compare_and_set_lost"
        result.update(
            latest,
            eligible=False,
            decision=latest_decision,
        )
        return result
    finally:
        if own:
            client.close()


def _pin_dd_source_snapshots(
    gc: GraphClient,
    sources: list[dict],
    *,
    services: Any,
    default_dd_version: str | None = None,
) -> list[dict]:
    """Capture immutable DD semantics while the requested version is current.

    ``IMASNode`` is a mutable current-version projection, so it may only be
    sampled when its current ``DDVersion`` exactly matches the version supplied
    by extraction. Existing source nodes are never backfilled here: their
    original snapshot cannot be reconstructed safely after the fact.

    An already-pinned source is re-seedable without re-supplying its version.
    Re-seeding one (e.g. ``sn run --focus`` / ``--edits`` over a path whose
    source already exists) reuses the stored immutable snapshot: no version is
    required from the caller and ``IMASNode`` is not re-sampled — the ``MERGE
    ON MATCH`` clause preserves every ``dd_*`` field regardless. Only a
    genuinely new source must supply an exact ``dd_version`` to be snapshotted;
    inferring ``latest`` for a new source is still refused.

    A batch seeder that mixes re-seeds with the occasional genuinely-new source
    (a manifest mop-up over a corpus whose pins predate a later-added path)
    cannot know per-source which is which. It passes ``default_dd_version`` —
    the current version — which pins ONLY the genuinely-new sources; re-seeds
    ignore it and keep their stored pin. This is an explicit declaration, not
    ``latest`` inference, so the never-infer-latest contract still holds. A
    per-source ``dd_version`` that disagrees with a stored pin still raises.
    """
    prepared = [dict(source) for source in sources]
    dd_sources = [s for s in prepared if s.get("source_type") == "dd"]
    if not dd_sources:
        return prepared

    # Consult existing nodes FIRST — the pin state decides whether a source is
    # a re-seed (reuse stored snapshot) or a genuinely new one (must snapshot).
    existing = gc.query(
        """
        UNWIND $ids AS id
        OPTIONAL MATCH (source:StandardNameSource {id: id})
        RETURN id, source.id AS existing_id, source.dd_version AS dd_version,
               source.dd_documentation AS dd_documentation,
               source.dd_snapshot_pinned AS dd_snapshot_pinned
        """,
        ids=[source["id"] for source in dd_sources],
    )
    by_existing = {row["id"]: row for row in existing or []}

    requests: list[dict] = []  # only genuinely-new sources need snapshotting
    for source in dd_sources:
        row = by_existing.get(source["id"])
        already_pinned = bool(
            row and row.get("existing_id") and row.get("dd_snapshot_pinned")
        )
        if already_pinned:
            # Re-seed: reuse the stored pin. Validate an explicitly-supplied
            # version, but never require one, and never re-sample IMASNode —
            # the immutable snapshot already exists and ON MATCH preserves it.
            supplied = source.get("dd_version")
            stored = row.get("dd_version")
            if supplied and supplied != stored:
                raise ValueError(
                    f"existing DD source {source['id']!r} is pinned to "
                    f"{stored!r}, not {supplied!r}"
                )
            source["dd_version"] = stored
            continue
        if row and row.get("existing_id") and not row.get("dd_snapshot_pinned"):
            raise ValueError(
                f"existing DD source {source['id']!r} has no provable immutable "
                "snapshot; run the provenance backfill report"
            )
        version = source.get("dd_version") or default_dd_version
        if not version:
            raise ValueError(
                f"DD source {source.get('dd_path')!r} has no exact dd_version; "
                "source creation must never infer latest"
            )
        source["dd_version"] = version
        requests.append(
            {
                "id": source["id"],
                "path": source.get("dd_path"),
                "dd_version": version,
            }
        )

    if not requests:
        return prepared

    snapshots = gc.query(
        """
        UNWIND $requests AS request
        MATCH (version:DDVersion {id: request.dd_version, is_current: true})
        MATCH (node:IMASNode {id: request.path})
        OPTIONAL MATCH (node)-[:HAS_PARENT]->(parent:IMASNode)
        OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
        OPTIONAL MATCH (node)-[:HAS_COORDINATE]->(coordinate)
        RETURN request.id AS id, version.id AS dd_version,
               node.documentation AS dd_documentation,
               parent.id AS dd_parent_path,
               parent.documentation AS dd_parent_documentation,
               node.data_type AS dd_data_type,
               coalesce(node.unit, unit.id) AS dd_unit,
               collect(DISTINCT coordinate.id) AS dd_coordinates,
               node.lifecycle_status AS dd_lifecycle_status,
               node.lifecycle_version AS dd_lifecycle_version,
               node.description AS enhanced_description,
               node.enrichment_source AS enhancement_kind
        """,
        requests=requests,
    )
    by_id = {row["id"]: dict(row) for row in snapshots or []}
    missing = sorted(
        request["id"] for request in requests if request["id"] not in by_id
    )
    if missing:
        raise ValueError(
            "cannot capture exact DD snapshot for source(s): " + ", ".join(missing)
        )
    for source in prepared:
        if source.get("source_type") == "dd" and source["id"] in by_id:
            snapshot = by_id[source["id"]]
            from imas_codex.standard_names.dd_resolutions import resolve_dd_row

            context = resolve_dd_row(
                {
                    "path": source["dd_path"],
                    "unit": snapshot.get("dd_unit"),
                    "documentation": snapshot.get("dd_documentation"),
                    "data_type": snapshot.get("dd_data_type"),
                    "coordinates": snapshot.get("dd_coordinates") or (),
                    "lifecycle_status": snapshot.get("dd_lifecycle_status"),
                    "lifecycle_version": snapshot.get("dd_lifecycle_version"),
                },
                dd_version=snapshot["dd_version"],
            )
            effective = context.as_pipeline_item()
            snapshot.update(
                {
                    "dd_unit": effective["unit"],
                    "dd_documentation": effective["documentation"],
                    "dd_data_type": effective["data_type"],
                    "dd_coordinates": effective["coordinates"],
                    "dd_lifecycle_status": effective["lifecycle_status"],
                    "dd_lifecycle_version": effective["lifecycle_version"],
                    "raw_dd_context": effective["raw_dd_context"],
                    "published_dd_context": effective["published_dd_context"],
                    "dd_resolution_ids": effective["dd_resolution_ids"],
                    "dd_resolution_converged_ids": effective[
                        "dd_resolution_converged_ids"
                    ],
                    "dd_resolution_manifest_digest": effective[
                        "dd_resolution_manifest_digest"
                    ],
                    "_dd_resolution_marker": effective["_dd_resolution_marker"],
                }
            )
            source.update(snapshot)
    return prepared


def merge_standard_name_sources(
    sources: list[dict],
    *,
    services: Any,
    force: bool = False,
    default_dd_version: str | None = None,
) -> int:
    """Batch MERGE StandardNameSource nodes.

    Each source dict must have: id, source_type, source_id, batch_key, status.
    DD sources additionally require ``dd_path`` and the exact ``dd_version``;
    their immutable semantic snapshot is captured centrally here. Optional:
    description and signal (for signal sources).

    ``default_dd_version`` pins genuinely-new DD sources that carry no per-source
    version (a mop-up seeder that mixes re-seeds with occasional new paths passes
    the current version here); re-seeds ignore it and keep their stored pin. See
    :func:`_pin_dd_source_snapshots`.

    On CREATE: sets all fields.
    On MATCH (existing node):
      - If force=True: resets to extracted, clears attempt_count/last_error/failed_at.
      - If status is 'stale': requeues to extracted.
      - Otherwise: preserves existing status (skip already-processed sources).

    Rejects source_type values not in {'dd', 'signals'} with ValueError.
    Returns count of nodes created or updated.
    """
    if not sources:
        return 0

    invalid = {s.get("source_type") for s in sources} - _VALID_PIPELINE_SOURCE_TYPES
    if invalid:
        raise ValueError(
            f"Invalid source_type(s) for pipeline: {invalid}. "
            f"Only {_VALID_PIPELINE_SOURCE_TYPES} are valid."
        )

    # Birth-invariant: a 'dd' source with no dd_path cannot have a FROM_DD_PATH
    # edge and would be an orphan from the moment it is written.  Drop these
    # before they reach the graph and log a WARNING so the caller knows.
    # Also stamp value-provenance (estimator) deterministically from the path:
    # measured/reconstructed/reference facets carry it as link metadata so the
    # estimator collapses to the base quantity's name (provenance-controlled-vocab).
    from imas_codex.standard_names.provenance import detect_value_provenance

    valid_sources: list[dict] = []
    for s in sources:
        if s.get("source_type") == "dd" and not s.get("dd_path"):
            logger.warning(
                "merge_standard_name_sources: skipping dd source %r — "
                "no dd_path supplied; writing would create an orphan "
                "(birth invariant violation).",
                s.get("id"),
            )
            continue
        if "provenance" not in s:
            _path = s.get("dd_path") or s.get("source_id") or ""
            s["provenance"] = detect_value_provenance(_path)[0]
        valid_sources.append(s)
    sources = valid_sources
    if not sources:
        return 0

    with services.GraphClient() as gc:
        sources = services._pin_dd_source_snapshots(
            gc, sources, default_dd_version=default_dd_version
        )
        result = gc.query(
            """
            UNWIND $sources AS src
            MERGE (sns:StandardNameSource {id: src.id})
            ON CREATE SET
                sns.source_type = src.source_type,
                sns.source_id = src.source_id,
                sns.batch_key = src.batch_key,
                sns.status = src.status,
                sns.description = nullIf(src.description, ''),
                sns.physics_domain = src.physics_domain,
                sns.dd_version = src.dd_version,
                sns.provenance = src.provenance,
                sns.dd_documentation = src.dd_documentation,
                sns.dd_snapshot_pinned = true,
                sns.dd_parent_path = src.dd_parent_path,
                sns.dd_parent_documentation = src.dd_parent_documentation,
                sns.dd_data_type = src.dd_data_type,
                sns.dd_unit = src.dd_unit,
                sns.dd_coordinates = src.dd_coordinates,
                sns.dd_lifecycle_status = src.dd_lifecycle_status,
                sns.dd_lifecycle_version = src.dd_lifecycle_version,
                sns.enhanced_description = src.enhanced_description,
                sns.enhancement_kind = src.enhancement_kind,
                sns.attempt_count = 0
            ON MATCH SET
                sns.batch_key = src.batch_key,
                sns.description = coalesce(nullIf(src.description, ''), sns.description),
                sns.physics_domain = coalesce(src.physics_domain, sns.physics_domain),
                sns.provenance = coalesce(src.provenance, sns.provenance),
                sns.status = CASE
                    WHEN $force THEN 'extracted'
                    WHEN sns.status = 'stale' THEN 'extracted'
                    ELSE sns.status
                END,
                sns.attempt_count = CASE
                    WHEN $force THEN 0
                    ELSE sns.attempt_count
                END,
                sns.last_error = CASE
                    WHEN $force THEN null
                    ELSE sns.last_error
                END,
                sns.failed_at = CASE
                    WHEN $force THEN null
                    ELSE sns.failed_at
                END,
                sns.claimed_at = CASE
                    WHEN $force THEN null
                    ELSE sns.claimed_at
                END,
                sns.claim_token = CASE
                    WHEN $force THEN null
                    ELSE sns.claim_token
                END
            WITH sns, src
            // Create typed relationships to source entities
            FOREACH (_ IN CASE WHEN src.source_type = 'dd' AND src.dd_path IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (imas:IMASNode {id: src.dd_path})
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
            )
            FOREACH (_ IN CASE WHEN src.source_type = 'signals' AND src.signal IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (sig:FacilitySignal {id: src.signal})
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
            )
            RETURN count(sns) AS affected
            """,
            sources=sources,
            force=force,
        )
        return result[0]["affected"] if result else 0


@retry_on_deadlock()
def claim_standard_name_source_batch(
    batch_key: str,
    *,
    services: Any,
    limit: int = 50,
    timeout_minutes: int = 30,
) -> tuple[str, list[dict]]:
    """Atomic full-batch claim of StandardNameSource nodes by batch_key.

    Claims up to ``limit`` extracted sources with matching batch_key.
    Uses two-step token verification to prevent double-claiming.
    Reclaims sources with stale claims (older than timeout_minutes).

    Returns (claim_token, claimed_sources) where each source dict has
    id, source_id, source_type, batch_key, description.
    """
    token = str(uuid.uuid4())
    with services.GraphClient() as gc:
        # Step 1: Claim
        gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.batch_key = $batch_key
              AND (
                (sns.status = 'extracted' AND sns.claimed_at IS NULL)
                OR (sns.claimed_at IS NOT NULL
                    AND sns.claimed_at < datetime() - duration({minutes: $timeout}))
              )
            WITH sns ORDER BY rand() LIMIT $limit
            SET sns.claimed_at = datetime(),
                sns.claim_token = $token,
                sns.claim_seq = coalesce(sns.claim_seq, 0) + 1
            """,
            batch_key=batch_key,
            limit=limit,
            timeout=timeout_minutes,
            token=token,
        )
        # Step 2: Verify
        claimed = list(
            gc.query(
                """
                MATCH (sns:StandardNameSource {claim_token: $token})
                RETURN sns.id AS id,
                       sns.source_id AS source_id,
                       sns.source_type AS source_type,
                       sns.batch_key AS batch_key,
                       sns.description AS description,
                       sns.claim_token AS claim_token,
                       sns.claim_seq AS claim_seq
                """,
                token=token,
            )
        )
    return token, claimed


class SourceClaimResult(list[dict[str, Any]]):
    """Claim winners plus conflicts withheld before an attempt is charged."""

    def __init__(
        self,
        claimed: list[dict[str, Any]],
        *,
        withheld: list[dict[str, Any]],
    ) -> None:
        super().__init__(claimed)
        self.withheld = withheld


_EXPLICIT_SOURCE_CLAIM_QUERY = """
UNWIND $ids AS sns_id
MATCH (sns:StandardNameSource {id: sns_id})
OPTIONAL MATCH (sns)-[:PRODUCED_NAME]->(holder:StandardName)
WHERE NOT coalesce(holder.name_stage, '') IN $settled_stages
WITH sns,
     collect(DISTINCT holder {.id, stage: holder.name_stage}) AS holding
WHERE sns.status = 'extracted'
  AND (
    sns.claimed_at IS NULL
    OR sns.claimed_at < datetime() - duration({minutes: $timeout})
  )
FOREACH (_ IN CASE WHEN size(holding) = 0 THEN [1] ELSE [] END |
    SET sns.claimed_at = datetime(),
        sns.claim_token = $token,
        sns.claim_seq = coalesce(sns.claim_seq, 0) + 1,
        sns.attempt_count = coalesce(sns.attempt_count, 0) + 1
)
RETURN sns.id AS id,
       sns.source_id AS source_id,
       sns.source_type AS source_type,
       sns.batch_key AS batch_key,
       sns.description AS description,
       sns.status AS status,
       sns.claim_token AS claim_token,
       sns.claim_seq AS claim_seq,
       sns.attempt_count AS attempt_count,
       size(holding) = 0 AS claimed,
       holding
"""


@retry_on_deadlock()
def claim_explicit_standard_name_sources(
    source_ids: list[str],
    *,
    services: Any,
    timeout_minutes: int = 30,
) -> SourceClaimResult:
    """Atomically claim the exact DD sources requested by a focused run.

    Every returned row carries the committed token and incremented sequence;
    missing, ineligible, or already-claimed sources are omitted. Sources bound
    to a non-settled identity are returned through ``result.withheld`` with the
    holding identity and stage, without consuming an attempt. Bindings only to
    superseded or exhausted identities remain claimable.
    """
    if not source_ids:
        return services.SourceClaimResult([], withheld=[])
    token = str(uuid.uuid4())
    sns_ids = [f"dd:{source_id}" for source_id in dict.fromkeys(source_ids)]
    with services.GraphClient() as gc:
        rows = gc.query(
            _EXPLICIT_SOURCE_CLAIM_QUERY,
            ids=sns_ids,
            token=token,
            timeout=timeout_minutes,
            settled_stages=sorted(foundation._SETTLED_SOURCE_BINDING_STAGES),
        )
    claimed: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        is_claimed = bool(row.pop("claimed"))
        holding = row.pop("holding") or []
        if is_claimed:
            claimed.append(row)
            continue
        for holder in holding:
            withheld.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "holding_identity": holder["id"],
                    "holding_stage": holder["stage"],
                }
            )
    if withheld:
        logger.warning(
            "Withheld %d explicit source claim conflict(s): %s",
            len(withheld),
            ", ".join(
                f"{entry['id']} held by "
                f"{entry['holding_identity']}@{entry['holding_stage']}"
                for entry in withheld
            ),
        )
    return services.SourceClaimResult(claimed, withheld=withheld)


def _complete_source_fences(
    items: list[dict[str, Any]], *, services: Any
) -> list[dict[str, Any]]:
    """Return only outcome items carrying a complete source ownership fence."""
    return [
        item
        for item in items
        if item.get("sns_id")
        and item.get("claim_token")
        and item.get("claim_seq") is not None
    ]


def _require_failure_reason(source_id: Any, reason: Any, *, services: Any) -> str:
    """Return the trimmed failure reason, refusing a null or empty one.

    A source parked at ``failed`` with no ``last_error`` cannot be triaged:
    the reader cannot tell a transient fault from a permanent one, so the row
    is neither returned to the queue nor closed, and it accumulates silently.
    Every write path able to set a source status to ``failed`` therefore
    states its reason or refuses the write, in the same shape as the
    ``vocab_gap`` refusal.
    """
    text = str(reason or "").strip()
    if not text:
        raise ValueError(
            f"failed source outcome for {source_id!r} requires a non-empty reason"
        )
    return text


def _guard_failed_source_reasons(batch: list[dict[str, Any]], *, services: Any) -> None:
    """Refuse any outcome that parks a source at ``failed`` without a reason.

    Runs before the graph is opened, so a reasonless failure never reaches a
    transaction. The write itself coalesces ``last_error`` onto whatever the
    source already carried, which is exactly what makes an absent reason
    invisible at the write site.
    """
    for item in batch:
        if item.get("status") != "failed":
            continue
        item["last_error"] = services._require_failure_reason(
            item.get("sns_id"), item.get("last_error")
        )


def _guard_existing_target_pairings(
    gc: Any, batch: list[dict[str, Any]], *, services: Any, targets_may_be_missing: bool
) -> dict[str, str]:
    """Reject fresh pairings onto an EXISTING name that contradict its sources.

    Returns ``{sns_id: reason}`` for every pairing the attachment guard refuses.

    Scope is deliberate on both sides. It runs **before** the binding query,
    because that query stamps a provisional ``PRODUCED_NAME`` reservation and
    the guard reads exactly that edge to decide a source is already bound — a
    guard run afterwards would wave through every pairing it exists to judge.
    And it runs only where the target name already exists: a name minted by
    this very batch has no other sources to contradict and takes its unit from
    the candidate, so there is nothing to compare, while a source binding onto
    an established name is the case that admits a contradiction — two distinct
    vector fields of one device, or a unit of a different dimension. That
    binding previously reached the graph with no graph-aware check at all,
    because the compose-time gate sees only same-batch siblings and never the
    units.
    """
    from imas_codex.standard_names.attachment_audit import guard_source_pairings

    sources_by_name: dict[str, list[str]] = {}
    for item in batch:
        sn_id = item.get("sn_id")
        sns_id = item.get("sns_id")
        if sn_id and sns_id:
            sources_by_name.setdefault(sn_id, []).append(sns_id)
    if not sources_by_name:
        return {}

    if targets_may_be_missing:
        existing = {
            row["id"]
            for row in gc.query(
                """
                UNWIND $names AS name
                MATCH (sn:StandardName {id: name})
                RETURN sn.id AS id
                """,
                names=sorted(sources_by_name),
            )
        }
    else:
        # The caller binds only to targets that already exist, so asking the
        # graph again would cost a round trip to learn what it guarantees.
        existing = set(sources_by_name)
    refusals: dict[str, str] = {}
    for sn_id in sorted(existing):
        guarded = guard_source_pairings(gc, sn_id, sources_by_name[sn_id])
        for verdict in guarded.rejected:
            refusals[verdict.source_node_id] = verdict.reason
    return refusals


def _supersede_released_bindings(
    gc: Any,
    released: list[dict[str, str]],
    holder_by_source: dict[str, str],
    *,
    operation: str,
    reason: str,
) -> list[str]:
    """Supersede names a rebind drained of their last producing source."""
    from imas_codex.standard_names.source_invariant import (
        supersede_names_losing_last_source,
    )

    moves = [
        (row["old_id"], holder_by_source[row["sns_id"]])
        for row in released
        if holder_by_source.get(row["sns_id"])
        and holder_by_source[row["sns_id"]] != row["old_id"]
    ]
    if not moves:
        return []
    return supersede_names_losing_last_source(
        gc, moves, operation=operation, reason=reason
    )


def _lock_claimed_name_bindings(
    gc: Any,
    batch: list[dict[str, Any]],
    *,
    services: Any,
    allow_missing: bool,
    allow_own_pending_reservation: bool,
    allow_dead_end_revival: bool = False,
    released_out: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Lock source claims and classify their proposed name bindings.

    Stable lifecycle stages can be reused monotonically. A pending name is a
    private reservation: only the source whose exact claim created its
    token-and-sequence stamped ``PRODUCED_NAME`` edge may finish it. In compose
    mode a missing target is created and reserved here, while the source claim
    and target identity are locked by the same caller-owned transaction.
    Attachments require an existing stable target and never create a name.

    A binding is refused for three reasons, all reported the same way: the
    target's lifecycle stage forbids it, its unit disagrees, or the attachment
    guard finds the pairing inconsistent with the sources the name already
    carries (see :func:`_guard_existing_target_pairings`). A compose proposal
    has one governed lifecycle exception: an exact DD source may revive a
    superseded or exhausted spelling when no successor at any depth is live.
    The identity re-enters at ``drafted`` with pending validation and a linked
    ``StandardNameChange`` recording its prior stage and triggering source.
    Existing lineage edges remain intact. Attachments never receive that
    authority. Refusal is per pairing — one bad source never fails its batch —
    so this is the single point at which every claimed-source persister
    inherits the guard.

    A non-bindable target releases the exact source claim back to
    ``extracted`` without resetting ``attempt_count``. The retained attempt
    count remains subject to the normal claim cap, while ``last_error`` records
    the candidate identity, lifecycle state, and attempt number. Because this
    query runs through the caller's transaction, a later rich-write failure
    rolls the release back with every other graph effect.
    """
    attachment_refusals = services._guard_existing_target_pairings(
        gc, batch, targets_may_be_missing=allow_missing
    )
    if attachment_refusals:
        batch = [
            {**item, "attachment_error": attachment_refusals.get(item["sns_id"])}
            for item in batch
        ]
    target_clause = (
        """
        MERGE (target:StandardName {id: b.sn_id})
        ON CREATE SET target.created_at = datetime(),
                      target.name_stage = $pending_stage,
                      target.status = 'draft',
                      target.reservation_source_id = sns.id,
                      target.reservation_claim_token = b.claim_token,
                      target.reservation_claim_seq = b.claim_seq,
                      target.updated_at = datetime()
        """
        if allow_missing
        else "OPTIONAL MATCH (target:StandardName {id: b.sn_id})"
    )
    binding_query = """
        UNWIND $batch AS b
        MATCH (sns:StandardNameSource {id: b.sns_id})
        WHERE sns.claim_token = b.claim_token
          AND sns.claim_seq = b.claim_seq
          AND sns.status = 'extracted'
        __TARGET_CLAUSE__
        WITH b, sns, target
        FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END |
          SET target.binding_lock_token = b.claim_token,
              target.binding_lock_seq = b.claim_seq
        )
        WITH b, sns, target
        CALL (b, sns, target) {
          WITH b, sns, target
          WHERE $allow_own_pending
            AND target.name_stage = $pending_stage
            AND target.reservation_source_id = sns.id
            AND target.reservation_claim_token = b.claim_token
            AND target.reservation_claim_seq = b.claim_seq
          MERGE (sns)-[reservation:PRODUCED_NAME]->(target)
          SET reservation.provisional = true,
              reservation.claim_token = b.claim_token,
              reservation.claim_seq = b.claim_seq,
              reservation.created_target = true
          REMOVE target.reservation_source_id,
                 target.reservation_claim_token,
                 target.reservation_claim_seq
          RETURN count(*) AS reservations
        }
        OPTIONAL MATCH (sns)-[owned:PRODUCED_NAME]->(target)
        WITH b, sns, target, owned,
             CASE
               WHEN $allow_dead_end_revival
                    AND b.source_type = 'dd'
                    AND target.name_stage IN $revivable_stages
                    AND b.attachment_error IS NULL
                    AND (b.unit IS NULL OR target.unit IS NULL
                         OR target.unit = b.unit)
                    AND EXISTS {
                      MATCH (sns)-[:FROM_DD_PATH]->(:IMASNode)
                    }
                    AND NOT EXISTS {
                      MATCH (target)-[:HAS_SUCCESSOR*1..]->(
                        successor:StandardName)
                      WHERE successor.name_stage IS NOT NULL
                        AND NOT (successor.name_stage IN $terminal_stages)
                    }
                    AND NOT EXISTS {
                      MATCH (successor:StandardName)-[:REFINED_FROM*1..]->(
                        target)
                      WHERE successor.name_stage IS NOT NULL
                        AND NOT (successor.name_stage IN $terminal_stages)
                    }
               THEN target.name_stage
               ELSE null
             END AS prior_stage
        FOREACH (_ IN CASE WHEN prior_stage IS NULL THEN [] ELSE [1] END |
          SET target.name_stage = $reviewable_stage,
              target.status = 'draft',
              target.validation_status = 'pending',
              target.validation_issues = null,
              target.validation_layer_summary = null,
              target.validated_at = null,
              target.reviewer_score_name = null,
              target.reviewer_scores_name = null,
              target.reviewer_comments_name = null,
              target.reviewer_comments_per_dim_name = null,
              target.reviewer_model_name = null,
              target.reviewed_name_at = null,
              target.review_quorum_shortfall = null,
              target.review_quorum_shortfall_at = null,
              target.review_resolution_method = null,
              target.refine_stop_reason = null,
              target.refine_stopped_at = null,
              target.refine_collision_name = null,
              target.claim_token = null,
              target.claimed_at = null,
              target.run_id = coalesce(sns.run_id, target.run_id)
        )
        FOREACH (_ IN CASE WHEN prior_stage IS NULL THEN [] ELSE [1] END |
          CREATE (change:StandardNameChange)
          SET change.id = 'sn-change:' + randomUUID(),
              change.from_name = prior_stage,
              change.to_name = $reviewable_stage,
              change.operation = 'revive_dead_end_identity',
              change.reason = 'source ' + sns.id +
                ' independently composed this retired identity while no live ' +
                'successor existed',
              change.origin = 'compose',
              change.run_id = sns.run_id,
              change.changed_at = datetime(),
              change.internal = true
          CREATE (target)-[:HAS_INTERNAL_CHANGE]->(change)
        )
        WITH b, sns, target, owned
        WITH b, sns, target, owned,
             CASE
               WHEN target.name_stage IN $stable_stages THEN true
               WHEN $allow_own_pending
                    AND target.name_stage = $pending_stage
                    AND owned.provisional = true
                    AND owned.claim_token = b.claim_token
                    AND owned.claim_seq = b.claim_seq
               THEN true
               ELSE false
             END AS lifecycle_bindable
        WITH b, sns, target, owned, lifecycle_bindable,
             CASE
               WHEN target IS NULL OR b.unit IS NULL
                    OR target.unit IS NULL OR target.unit = b.unit
               THEN true ELSE false
             END AS unit_bindable,
             b.attachment_error IS NULL AS attachment_bindable,
             CASE
               WHEN target IS NULL THEN '<missing>'
               ELSE coalesce(target.name_stage, '<unset>')
             END AS target_stage
        WITH b, sns, target, owned, lifecycle_bindable, unit_bindable,
             attachment_bindable, target_stage,
             lifecycle_bindable AND unit_bindable
               AND attachment_bindable AS bindable
        SET sns.claimed_at = datetime()
        WITH b, sns, target, owned, target_stage, bindable,
             lifecycle_bindable, unit_bindable, attachment_bindable
        FOREACH (_ IN CASE
          WHEN target.binding_lock_token = b.claim_token
           AND target.binding_lock_seq = b.claim_seq
          THEN [1] ELSE [] END |
          REMOVE target.binding_lock_token, target.binding_lock_seq
        )
        RETURN sns.id AS id,
               CASE
                 WHEN bindable THEN 'winner'
                 WHEN NOT lifecycle_bindable THEN 'lifecycle_collision'
                 WHEN NOT unit_bindable THEN 'unit_collision'
                 ELSE 'attachment_collision'
               END AS outcome,
               CASE
                 WHEN target.name_stage IN $stable_stages THEN 'stable_reuse'
                 WHEN owned.provisional = true
                      AND owned.claim_token = b.claim_token
                      AND owned.claim_seq = b.claim_seq
                 THEN 'owned_reservation'
                 ELSE 'collision'
               END AS binding_kind,
               b.sn_id AS candidate_id,
               target_stage,
               coalesce(sns.attempt_count, 0) AS attempt_count
        """.replace("__TARGET_CLAUSE__", target_clause)
    rows = gc.query(
        binding_query,
        batch=batch,
        allow_own_pending=allow_own_pending_reservation,
        allow_dead_end_revival=allow_dead_end_revival,
        stable_stages=sorted(foundation._STABLE_BINDING_NAME_STAGES),
        revivable_stages=sorted(
            {
                services.NameStage.superseded.value,
                services.NameStage.exhausted.value,
            }
        ),
        terminal_stages=sorted(foundation._TERMINAL_BINDING_NAME_STAGES),
        pending_stage=services.NameStage.pending.value,
        reviewable_stage=services.NameStage.drafted.value,
    )

    outcomes_by_source = {row["id"]: row.get("outcome", "winner") for row in rows}
    fenced_batch = [
        {
            **item,
            "preserve_current": outcomes_by_source[item["sns_id"]] == "winner",
        }
        for item in batch
        if item["sns_id"] in outcomes_by_source
    ]
    if fenced_batch:
        # A reclaimed source can carry abandoned provisional reservations even
        # when its new candidate wins. Only directly stamped relationships are
        # deletable. A winner preserves its current exact reservation; every
        # other provisional edge is stale. Projections and source-path mirrors
        # are never inferred to belong to a reservation because older writers
        # created those assertions without ownership metadata.
        released_rows = gc.query(
            """
            UNWIND $batch AS b
            MATCH (sns:StandardNameSource {id: b.sns_id})
            WHERE sns.claim_token = b.claim_token
              AND sns.claim_seq = b.claim_seq
              AND sns.status = 'extracted'
            MATCH (sns)-[stale:PRODUCED_NAME]->(old:StandardName)
            WHERE stale.provisional = true
              AND stale.claim_token IS NOT NULL
              AND stale.claim_seq IS NOT NULL
              AND NOT (
                b.preserve_current = true
                AND old.id = b.sn_id
                AND stale.claim_token = b.claim_token
                AND stale.claim_seq = b.claim_seq
              )
            WITH old, collect(DISTINCT stale) AS stale_edges,
                 collect(DISTINCT b.sns_id) AS released_source_ids
            WITH old, stale_edges, released_source_ids,
                 any(edge IN stale_edges
                     WHERE edge.created_target = true) AS owns_target
            FOREACH (stale_edge IN stale_edges |
              DELETE stale_edge
            )
            WITH DISTINCT old, owns_target, released_source_ids
            WITH old, owns_target, released_source_ids,
                 owns_target
                   AND old.name_stage = $pending_stage
                   AND coalesce(old.source_paths, []) = []
                   AND NOT EXISTS { MATCH (old)--() }
                   AS delete_owned_orphan
            FOREACH (owned_target IN CASE
              WHEN delete_owned_orphan THEN [old] ELSE [] END |
              DELETE owned_target
            )
            RETURN old.id AS old_id, released_source_ids
            """,
            batch=fenced_batch,
            pending_stage=services.NameStage.pending.value,
        )
        if released_out is not None:
            for row in released_rows or []:
                for source_id in row.get("released_source_ids") or []:
                    released_out.append({"old_id": row["old_id"], "sns_id": source_id})

    collision_ids = {
        row["id"] for row in rows if row.get("outcome", "winner") != "winner"
    }
    collision_batch = [item for item in batch if item["sns_id"] in collision_ids]
    if collision_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sns:StandardNameSource {id: b.sns_id})
            WHERE sns.claim_token = b.claim_token
              AND sns.claim_seq = b.claim_seq
              AND sns.status = 'extracted'
            OPTIONAL MATCH (target:StandardName {id: b.sn_id})
            SET sns.claimed_at = null,
                sns.claim_token = null,
                sns.status = 'extracted',
                sns.composed_at = null,
                sns.last_error = CASE
                  WHEN b.attachment_error IS NOT NULL
                  THEN 'standard-name attachment rejected: candidate "' +
                       b.sn_id + '" ' + b.attachment_error +
                       ' at composition attempt ' +
                       toString(coalesce(sns.attempt_count, 0))
                  WHEN target IS NULL OR NOT (target.name_stage IN $stable_stages)
                  THEN 'standard-name binding collision: candidate "' + b.sn_id +
                       '" has non-bindable name_stage "' +
                       coalesce(target.name_stage, '<missing>') +
                       '" at composition attempt ' +
                       toString(coalesce(sns.attempt_count, 0))
                  ELSE 'standard-name binding collision: candidate "' + b.sn_id +
                       '" unit "' + coalesce(b.unit, '<unset>') +
                       '" conflicts with existing unit "' +
                       coalesce(target.unit, '<unset>') +
                       '" at composition attempt ' +
                       toString(coalesce(sns.attempt_count, 0))
                END
            """,
            batch=collision_batch,
            stable_stages=sorted(foundation._STABLE_BINDING_NAME_STAGES),
        )
    for row in rows:
        if row.get("outcome", "winner") == "winner":
            continue
        reason = attachment_refusals.get(row.get("id") or "")
        logger.warning(
            "Released source %s after %s for candidate %s at name stage %s "
            "(attempt %s retained)%s",
            row.get("id"),
            row.get("outcome"),
            row.get("candidate_id"),
            row.get("target_stage"),
            row.get("attempt_count"),
            f": {reason}" if reason else "",
        )
    return rows


@retry_on_deadlock()
def persist_claimed_source_outcomes(
    outcomes: list[dict[str, Any]], *, services: Any
) -> list[str]:
    """Consume exact source claims and persist terminal/retry diagnostics.

    The claim CAS and source mutation share one transaction. Items without a
    complete fence, and terminal failures without a reason, are rejected
    before opening the graph.
    """
    batch = services._complete_source_fences(outcomes)
    if not batch:
        return []
    services._guard_failed_source_reasons(batch)
    with services.GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $batch AS b
            MATCH (sns:StandardNameSource {id: b.sns_id})
            WHERE sns.claim_token = b.claim_token
              AND sns.claim_seq = b.claim_seq
              AND sns.status = 'extracted'
            SET sns.status = b.status,
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.last_error = coalesce(b.last_error, sns.last_error),
                sns.failed_at = CASE WHEN b.status = 'failed'
                                     THEN datetime() ELSE sns.failed_at END,
                sns.skip_reason = coalesce(b.skip_reason, sns.skip_reason),
                sns.skip_reason_detail = coalesce(
                    b.skip_reason_detail, sns.skip_reason_detail),
                sns.skipped_at = CASE WHEN b.status = 'skipped'
                                      THEN datetime() ELSE sns.skipped_at END
            RETURN sns.id AS id
            """,
            batch=batch,
        )
    return [row["id"] for row in rows]


@retry_on_deadlock()
def persist_claimed_vocab_gaps(
    gaps: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    services: Any,
    source_type: str,
) -> list[str]:
    """Write gaps and consume their exact source claims in one transaction."""
    batch = services._complete_source_fences(outcomes)
    if not batch:
        return []
    services._guard_failed_source_reasons(batch)
    gaps_by_source: dict[str, list[dict[str, Any]]] = {}
    for gap in gaps:
        source_id = gap.get("source_id")
        if source_id:
            gaps_by_source.setdefault(source_id, []).append(gap)

    winner_ids: list[str] = []
    with services.GraphClient() as graph:
        with graph.session() as session:
            tx = session.begin_transaction()
            tx_gc = services._TransactionQuery(tx)
            try:
                locked_rows = tx_gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    WHERE sns.claim_token = b.claim_token
                      AND sns.claim_seq = b.claim_seq
                      AND sns.status = 'extracted'
                    SET sns.claimed_at = datetime()
                    RETURN sns.id AS id, sns.source_id AS source_id
                    """,
                    batch=batch,
                )
                locked_ids = {row["id"] for row in locked_rows}
                locked_source_ids = {row["source_id"] for row in locked_rows}
                winner_outcomes = [
                    dict(item) for item in batch if item["sns_id"] in locked_ids
                ]
                actionable_source_ids = {
                    item["source_id"]
                    for item in winner_outcomes
                    if item.get("status") == "vocab_gap"
                }
                winner_gaps = [
                    gap
                    for source_id in actionable_source_ids
                    for gap in gaps_by_source.get(source_id, [])
                    if source_id in locked_source_ids
                ]
                for item in winner_outcomes:
                    if item.get("status") != "vocab_gap":
                        continue
                    source_id = item["source_id"]
                    source_gaps = gaps_by_source.get(source_id, [])
                    statements: list[str] = []
                    for gap in source_gaps:
                        reason = str(gap.get("reason") or "").strip()
                        if not reason:
                            raise ValueError(
                                "vocab_gap source outcome for "
                                f"{source_id!r} requires a non-empty reason"
                            )
                        segment = str(gap.get("segment") or "").strip()
                        token = str(gap.get("token") or "").strip()
                        if not segment or not token:
                            raise ValueError(
                                "vocab_gap source outcome for "
                                f"{source_id!r} requires a segment and token"
                            )
                        statement = (
                            f"missing {segment} vocabulary token {token!r}: {reason}"
                        )
                        if statement not in statements:
                            statements.append(statement)
                    if not statements:
                        raise ValueError(
                            "vocab_gap source outcome for "
                            f"{source_id!r} requires a non-empty reason"
                        )
                    item["last_error"] = "; ".join(statements)
                if winner_gaps:
                    written = services.write_vocab_gaps(
                        winner_gaps,
                        source_type,
                        gc=tx_gc,
                    )
                    if written == 0:
                        raise RuntimeError(
                            "actionable vocabulary gaps produced no graph records"
                        )

                rows = tx_gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    WHERE sns.claim_token = b.claim_token
                      AND sns.claim_seq = b.claim_seq
                      AND sns.status = 'extracted'
                    SET sns.status = b.status,
                        sns.claimed_at = null,
                        sns.claim_token = null,
                        sns.last_error = coalesce(b.last_error, sns.last_error),
                        sns.failed_at = CASE WHEN b.status = 'failed'
                                             THEN datetime() ELSE sns.failed_at END
                    RETURN sns.id AS id
                    """,
                    batch=winner_outcomes,
                )
                winner_ids = [row["id"] for row in rows]
                if set(winner_ids) != locked_ids:
                    raise RuntimeError(
                        "source claim changed during vocabulary-gap persistence"
                    )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise
    return winner_ids


@retry_on_deadlock()
def persist_claimed_attachments(
    attachments: list[dict[str, Any]], *, services: Any
) -> list[str]:
    """Attach exact source-claim winners to stable lifecycle targets.

    A terminal, transient, missing, or unowned-pending target releases the
    source through the same collision path as generated candidates. No source,
    DD projection, name scalar, or relationship is mutated unless both the
    exact source fence and stable target predicate survive to the final write.
    """
    batch = [
        {**item, "sn_id": item.get("standard_name")}
        for item in services._complete_source_fences(attachments)
        if item.get("standard_name")
    ]
    if not batch:
        return []
    winner_ids: list[str] = []
    with services.GraphClient() as graph:
        with graph.session() as session:
            tx = session.begin_transaction()
            tx_gc = services._TransactionQuery(tx)
            try:
                released: list[dict[str, str]] = []
                locked_rows = services._lock_claimed_name_bindings(
                    tx_gc,
                    batch,
                    allow_missing=False,
                    allow_own_pending_reservation=False,
                    allow_dead_end_revival=False,
                    released_out=released,
                )
                locked_ids = {
                    row["id"]
                    for row in locked_rows
                    if row.get("outcome", "winner") == "winner"
                }
                winner_batch = [item for item in batch if item["sns_id"] in locked_ids]
                if winner_batch:
                    rows = tx_gc.query(
                        """
                        UNWIND $batch AS b
                        MATCH (sns:StandardNameSource {id: b.sns_id})
                        WHERE sns.claim_token = b.claim_token
                          AND sns.claim_seq = b.claim_seq
                          AND sns.status = 'extracted'
                        MATCH (sn:StandardName {id: b.sn_id})
                        WHERE sn.name_stage IN $stable_stages
                        MATCH (src:IMASNode {id: b.source_id})
                        FOREACH (_ IN CASE
                          WHEN sns.compose_hint_status = 'open' THEN [1] ELSE [] END |
                          SET sns.compose_hint_status = 'consumed',
                              sns.compose_hint_consumed_at = datetime()
                        )
                        SET sns.status = 'attached',
                            sns.composed_at = datetime(),
                            sns.claimed_at = null,
                            sns.claim_token = null,
                            sns.produced_sn_id = sn.id,
                            sns.last_error = null
                        MERGE (sns)-[produced:PRODUCED_NAME]->(sn)
                        REMOVE produced.provisional,
                               produced.claim_token,
                               produced.claim_seq,
                               produced.created_target
                        MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                        WITH sns, sn, b, 'dd:' + b.source_id AS uri
                        SET sn.source_paths = CASE
                            WHEN uri IN coalesce(sn.source_paths, [])
                            THEN sn.source_paths
                            ELSE coalesce(sn.source_paths, []) + uri END,
                            sn.updated_at = datetime()
                        RETURN sns.id AS id
                        """,
                        batch=winner_batch,
                        stable_stages=sorted(foundation._STABLE_BINDING_NAME_STAGES),
                    )
                    winner_ids = [row["id"] for row in rows]
                    if set(winner_ids) != locked_ids:
                        raise RuntimeError(
                            "source claim or target lifecycle changed during "
                            "transactional attachment persistence"
                        )
                    _supersede_released_bindings(
                        tx_gc,
                        released,
                        {item["sns_id"]: item["sn_id"] for item in winner_batch},
                        operation="compose_rebind",
                        reason="compose re-bound the source of a drained name",
                    )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise
    return winner_ids


@retry_on_deadlock()
def stage_claimed_generated_candidates(
    candidates: list[dict[str, Any]], *, services: Any
) -> list[str]:
    """Reserve candidate source claims while staging names and source edges.

    A missing/stale fence produces no name, edge, stage, source, or counter
    mutation. The source stays claimed and ``extracted`` until the rich writer
    succeeds and :func:`_finalize_generated_name_stage` consumes the same fence.
    A rich-write crash therefore leaves a pending, automatically retryable
    reservation rather than a composed source stranded on an incomplete name.
    """
    batch = services._complete_source_fences(candidates)
    if not batch:
        return []
    winner_ids: list[str] = []
    with services.GraphClient() as graph:
        with graph.session() as session:
            tx = session.begin_transaction()
            tx_gc = services._TransactionQuery(tx)
            try:
                released: list[dict[str, str]] = []
                locked_rows = services._lock_claimed_name_bindings(
                    tx_gc,
                    batch,
                    allow_missing=True,
                    allow_own_pending_reservation=True,
                    allow_dead_end_revival=True,
                    released_out=released,
                )
                locked_ids = {
                    row["id"]
                    for row in locked_rows
                    if row.get("outcome", "winner") == "winner"
                }
                winner_ids = list(locked_ids)
                _supersede_released_bindings(
                    tx_gc,
                    released,
                    {
                        item["sns_id"]: item["sn_id"]
                        for item in batch
                        if item["sns_id"] in locked_ids
                    },
                    operation="compose_rebind",
                    reason="compose re-bound the source of a drained name",
                )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise
    return winner_ids


def fetch_claimed_source_metadata(token: str, *, services: Any) -> list[dict]:
    """Fetch full metadata for claimed sources, joining source entities.

    For DD sources: joins IMASNode for documentation, unit, cluster info.
    For signal sources: joins FacilitySignal for description, unit, diagnostic.

    Returns list of dicts with source + joined metadata.
    """
    with services.GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (sns:StandardNameSource {claim_token: $token})
                OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(imas:IMASNode)
                OPTIONAL MATCH (imas)-[:HAS_UNIT]->(u:Unit)
                OPTIONAL MATCH (imas)<-[:CONTAINS_PATH]-(c:SemanticCluster)
                OPTIONAL MATCH (sns)-[:FROM_SIGNAL]->(sig:FacilitySignal)
                RETURN sns.id AS id,
                       sns.source_id AS source_id,
                       sns.source_type AS source_type,
                       sns.batch_key AS batch_key,
                       sns.description AS description,
                       imas.id AS dd_path,
                       imas.documentation AS dd_documentation,
                       u.id AS unit,
                       c.label AS cluster_label,
                       c.scope AS cluster_scope,
                       sig.id AS signal_id,
                       sig.description AS signal_description
                """,
                token=token,
            )
        )


_MOVE_SOURCE_BINDING_CYPHER = """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid})
            WHERE sns.claim_token = $token
              AND sns.claim_seq = $claim_seq
              AND sns.status = 'extracted'
            MATCH (sn:StandardName {id: $sn_id})
            WITH sns, sn,
                 [(sns)-[:PRODUCED_NAME]->(prior:StandardName)
                  WHERE prior.id <> sn.id | prior.id] AS prior_ids
            SET sns.status = $bound_status,
                sns.composed_at = datetime(),
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.produced_sn_id = sn.id
            MERGE (sns)-[:PRODUCED_NAME]->(sn)
            WITH DISTINCT sns, sn, prior_ids
            CALL (sns, sn, prior_ids) {
              UNWIND prior_ids AS prior_id
              MATCH (sns)-[move:PRODUCED_NAME]->(old:StandardName {id: prior_id})
              DELETE move
              WITH DISTINCT sn, old
              WITH sn, old,
                   COUNT {
                     (source:StandardNameSource)-[:PRODUCED_NAME]->(old)
                     WHERE coalesce(source.status, '') <> 'stale'
                   } AS producers
              WHERE producers = 0
                AND NOT coalesce(old.name_stage, '') IN
                    ['superseded', 'exhausted', 'contested']
                AND NOT coalesce(old.status, '') IN
                    ['deprecated', 'superseded']
                AND NOT (sn)-[:REFINED_FROM*1..]->(old)
              SET old.updated_at = datetime(),
                  old.name_stage = 'superseded',
                  old.status = 'superseded',
                  old.claimed_at = null,
                  old.claim_token = null
              MERGE (sn)-[:REFINED_FROM]->(old)
              CREATE (change:StandardNameChange {
                id: 'sn-change:' + randomUUID(),
                from_name: old.id, to_name: sn.id,
                operation: $supersede_operation,
                reason: $supersede_reason, origin: 'compose',
                run_id: null, changed_at: datetime(), internal: true})
              CREATE (sn)-[:HAS_INTERNAL_CHANGE]->(change)
            }
            RETURN count(DISTINCT sns) AS affected
            """


def mark_sources_composed(
    token: str,
    claim_seq: int,
    source_ids: list[str],
    standard_name_id: str,
    *,
    services: Any,
) -> int:
    """Mark sources as composed and link to the produced StandardName.

    Fence-verified: only updates extracted sources matching the claim token and
    sequence. The source moves to the name that now holds it, and a holder left
    with no producer is superseded in the same statement.
    Returns count of updated sources.
    """
    with services.GraphClient() as gc:
        result = gc.query(
            _MOVE_SOURCE_BINDING_CYPHER,
            source_ids=source_ids,
            token=token,
            claim_seq=claim_seq,
            sn_id=standard_name_id,
            bound_status="composed",
            supersede_operation="composed_source_moved",
            supersede_reason=(
                "compose moved this source to the composed name, leaving the "
                "former holder without a producer"
            ),
        )
        return result[0]["affected"] if result else 0


def mark_sources_attached(
    token: str,
    claim_seq: int,
    source_ids: list[str],
    standard_name_id: str,
    *,
    services: Any,
) -> int:
    """Mark sources as auto-attached to an existing StandardName.

    Used when a source matches an existing name without needing LLM composition.
    Exact-fence verified. Creates PRODUCED_NAME relationship.
    Returns count of updated sources.
    """
    with services.GraphClient() as gc:
        result = gc.query(
            _MOVE_SOURCE_BINDING_CYPHER,
            source_ids=source_ids,
            token=token,
            claim_seq=claim_seq,
            sn_id=standard_name_id,
            bound_status="attached",
            supersede_operation="attached_source_moved",
            supersede_reason=(
                "attach moved this source to the matched name, leaving the "
                "former holder without a producer"
            ),
        )
        return result[0]["affected"] if result else 0


#: Reason recorded when a source spends its claim budget without a name. It
#: states the budget and nothing about cause, so a parked cohort carrying only
#: this string cannot be triaged from it.
_COMPOSE_CAP_REASON = "compose claim-attempt cap reached"

#: Closed set of dispositions a source parked at the compose claim-attempt cap
#: can carry. Every source at or past ``_MAX_COMPOSE_CLAIM_ATTEMPTS`` lands in
#: exactly one, so the cohort reads as a triage list rather than as a uniform
#: failure and each class states the next operation rather than the symptom.
CAP_PARKED_DISPOSITIONS = frozenset(
    {
        # A name already exists, so the cap bounds further claims rather than
        # the outcome and no composition is owed.
        "name_produced",
        # The data-dictionary quantity behind the source is gone, so no name can
        # ever be composed for it and the parked state is terminal.
        "upstream_quantity_removed",
        # The path names a coordinate or geometry component rather than a
        # physical quantity, so composition is not the applicable operation.
        "compose_not_applicable",
        # The recorded reason names a vocabulary gap: the concept is nameable
        # only once the grammar covers it, which no retry can produce.
        "vocabulary_gap",
        # A nameable quantity with a spent budget, a recorded cause and no
        # structural blocker: a candidate for revival once compose changes.
        "attempt_budget_exhausted",
        # The row records nothing about its own failure, so it cannot be triaged
        # until a reason exists. That absence is the finding, not a classification
        # in its own right: the class counts rows whose cause was lost, and the
        # only repair is to make the write path record one.
        "cause_not_recorded",
    }
)


def classify_parked_source(evidence: dict[str, Any], *, services: Any) -> str:
    """Return the triage disposition for one source parked at the cap.

    ``evidence`` carries what the parked row records about itself: ``produced``
    (the number of live produced names), the backing data-dictionary node's
    ``lifecycle_status`` and ``node_category``, and ``last_error``.

    Total by construction: every input returns a member of
    :data:`CAP_PARKED_DISPOSITIONS`, including the row that records nothing, so
    a census of the cohort has no residue to interpret and the answer cannot be
    silently wrong through an unhandled shape.

    A recorded cause outranks the node category, and the category outranks
    silence: it exists to classify the rows whose cause was lost.
    """
    if int(evidence.get("produced") or 0) > 0:
        return "name_produced"
    if str(evidence.get("lifecycle_status") or "").strip() == "removed":
        return "upstream_quantity_removed"
    reason = str(evidence.get("last_error") or "").strip()
    if reason and reason != _COMPOSE_CAP_REASON:
        if "vocabulary gap" in reason.lower():
            return "vocabulary_gap"
        return "attempt_budget_exhausted"
    if str(evidence.get("node_category") or "").strip() in {"geometry", "coordinate"}:
        return "compose_not_applicable"
    return "cause_not_recorded"


def mark_sources_failed(
    token: str,
    source_ids: list[str],
    error: str,
    *,
    services: Any,
    max_attempts: int = 3,
) -> int:
    """Mark sources as failed with durable retry.

    Increments attempt_count. If below max_attempts, returns to 'extracted'
    for retry. At max_attempts, transitions to terminal 'failed' status.
    Token-verified. Returns count of updated sources.

    ``error`` is the reason the terminal state will carry, so a null or empty
    one is refused before the graph is opened rather than parking an
    untriageable source.
    """
    error = services._require_failure_reason(source_ids, error)
    with services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid, claim_token: $token})
            SET sns.attempt_count = coalesce(sns.attempt_count, 0) + 1,
                sns.last_error = $error,
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.status = CASE
                    WHEN coalesce(sns.attempt_count, 0) + 1 >= $max_attempts
                    THEN 'failed'
                    ELSE 'extracted'
                END,
                sns.failed_at = CASE
                    WHEN coalesce(sns.attempt_count, 0) + 1 >= $max_attempts
                    THEN datetime()
                    ELSE sns.failed_at
                END
            RETURN count(sns) AS affected
            """,
            source_ids=source_ids,
            token=token,
            error=error,
            max_attempts=max_attempts,
        )
        return result[0]["affected"] if result else 0


_CAP_PARKED_QUERY = """
MATCH (sns:StandardNameSource)
WHERE coalesce(sns.attempt_count, 0) >= $cap
OPTIONAL MATCH (node:IMASNode {id: sns.source_id})
RETURN sns.id AS id, sns.last_error AS last_error,
       node.lifecycle_status AS lifecycle_status,
       size([(sns)-[:PRODUCED_NAME]->(:StandardName) | 1]) AS produced,
       node.node_category AS node_category
ORDER BY sns.id
"""


def record_parked_dispositions(
    rows: list[dict[str, Any]],
    *,
    services: Any,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Classify parked rows and write the disposition onto each source.

    The single classify-and-write path, shared by the reconcile pass that reads
    the whole cohort and by the writer that parks a source, so the precedence in
    :func:`classify_parked_source` exists once. ``rows`` carry the evidence
    aliases returned by the parking statement. A classification outside
    :data:`CAP_PARKED_DISPOSITIONS` refuses before any write, so no caller can
    end with a row carrying an unclassified disposition.
    """
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        written: list[dict[str, str]] = []
        for row in rows:
            disposition = services.classify_parked_source(row)
            if disposition not in CAP_PARKED_DISPOSITIONS:
                raise ValueError(f"row {row['id']!r} has no triage disposition")
            written.append({"id": row["id"], "disposition": disposition})
        if not dry_run and written:
            client.query(_CAP_DISPOSITION_WRITE, rows=written)
        counts = Counter(row["disposition"] for row in written)
        return {
            "parked": len(written),
            "written": 0 if dry_run else len(written),
            "counts": dict(sorted(counts.items())),
        }
    finally:
        if own:
            client.close()


def disposition_parked_sources(
    *,
    services: Any,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Give every source parked at the compose cap a triage disposition.

    Reads the parked cohort with the evidence each row carries and delegates to
    :func:`record_parked_dispositions`. This is the reconcile net: the parking
    writer records a disposition as it parks, and this pass heals any row that
    predates that write path or was parked around it.
    """
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        rows = [
            dict(r)
            for r in client.query(
                _CAP_PARKED_QUERY, cap=services._MAX_COMPOSE_CLAIM_ATTEMPTS
            )
        ]
        return services.record_parked_dispositions(rows, dry_run=dry_run, gc=client)
    finally:
        if own:
            client.close()


#: The rows the parking writer has already parked under the cap reason and left
#: without a disposition. Scoped by the reason the writer itself writes, so the
#: read can only see rows a parking event produced, and by the absence of the
#: disposition, so it is empty once the writer has recorded one.
_CAP_UNDISPOSED_QUERY = """
MATCH (sns:StandardNameSource)
WHERE sns.status = 'failed'
  AND sns.last_error = $cap_reason
  AND sns.parked_disposition IS NULL
OPTIONAL MATCH (node:IMASNode {id: sns.source_id})
RETURN sns.id AS id, sns.last_error AS last_error,
       node.lifecycle_status AS lifecycle_status,
       size([(sns)-[:PRODUCED_NAME]->(:StandardName) | 1]) AS produced,
       node.node_category AS node_category
ORDER BY sns.id
"""


def record_dispositions_for_undisposed_parks(
    *,
    services: Any,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Stamp a disposition on the rows a parking event left undisposed.

    The parking writer calls this in the same tick as the write that parks, so
    a source arrives at ``failed`` already carrying a member of
    :data:`CAP_PARKED_DISPOSITIONS` and no later classification pass is needed
    for it. Rows already carrying a disposition are not in scope, so a tick that
    parks nothing and a tick that parks an already-disposed row both do nothing.
    """
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        rows = [
            dict(r)
            for r in client.query(_CAP_UNDISPOSED_QUERY, cap_reason=_COMPOSE_CAP_REASON)
        ]
        return services.record_parked_dispositions(rows, dry_run=dry_run, gc=client)
    finally:
        if own:
            client.close()


_CAP_DISPOSITION_WRITE = """
UNWIND $rows AS row
MATCH (sns:StandardNameSource {id: row.id})
SET sns.parked_disposition = row.disposition
RETURN count(sns) AS written
"""


_CAP_DISPOSITION_READBACK = """
MATCH (sns:StandardNameSource)
WHERE coalesce(sns.attempt_count, 0) >= $cap
RETURN sns.id AS id, sns.parked_disposition AS disposition
ORDER BY sns.id
"""


def census_parked_dispositions(
    *, services: Any, gc: Any | None = None
) -> dict[str, Any]:
    """Re-read the parked cohort and report the disposition it now carries.

    Reads the stored property back rather than the classification this process
    just computed, so the census tests the write instead of re-running the
    classifier that produced the values. It returns the count per class, the
    rows carrying no disposition and the rows carrying one outside
    :data:`CAP_PARKED_DISPOSITIONS`; a census reporting zero unclassified is
    then a statement about the graph and not about the classifier.
    """
    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        rows = [
            dict(r)
            for r in client.query(
                _CAP_DISPOSITION_READBACK,
                cap=services._MAX_COMPOSE_CLAIM_ATTEMPTS,
            )
        ]
    finally:
        if own:
            client.close()
    counts = Counter(v for v in (r["disposition"] for r in rows) if v)
    unclassified = sorted(r["id"] for r in rows if not r["disposition"])
    unexpected = sorted(
        r["id"]
        for r in rows
        if r["disposition"] and r["disposition"] not in CAP_PARKED_DISPOSITIONS
    )
    return {
        "parked": len(rows),
        "counts": dict(sorted(counts.items())),
        "unclassified": unclassified,
        "unexpected": unexpected,
    }


def retry_failed_sources(
    source_paths: list[str],
    *,
    services: Any,
    reason: str,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Return terminal or attempt-capped sources to the composition queue.

    This is the sanctioned recovery operation for two otherwise-unclaimable
    source states:

    * ``status='failed'`` after the composition retry budget was spent;
    * ``status='extracted'`` with ``attempt_count`` already at the claim cap.

    Before resetting the source, the operation creates one
    ``StandardNameSourceRetry`` audit event containing the prior status,
    attempt count, error, and operator reason. The source stores both the
    event-id scalar mirror and the ``HAS_RETRY_EVENT`` relationship. The write
    uses the read status and attempt count as a compare-and-set guard, so a
    concurrent worker cannot be rewound after it has already advanced.

    ``source_paths`` accepts either full ``StandardNameSource.id`` values
    (``dd:<path>`` / ``signals:<id>``) or their bare ``source_id`` values.
    Returns requested/eligible/retried/refused counts plus the eligible source
    ids and created event ids.
    """
    requested = sorted(
        {str(path).strip() for path in source_paths if str(path).strip()}
    )
    if not requested:
        raise ValueError("at least one source path is required")
    reason = reason.strip()
    if not reason:
        raise ValueError("a non-empty retry reason is required")

    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        candidates = list(
            client.query(
                """
                MATCH (sns:StandardNameSource)
                WHERE sns.id IN $requested OR sns.source_id IN $requested
                WITH sns, coalesce(sns.attempt_count, 0) AS attempts
                WHERE sns.status = 'failed'
                   OR (sns.status = 'extracted' AND attempts >= $attempt_cap)
                RETURN sns.id AS id, sns.source_id AS source_path,
                       sns.status AS status,
                       attempts AS attempt_count, sns.last_error AS last_error
                ORDER BY sns.id
                """,
                requested=requested,
                attempt_cap=services._MAX_COMPOSE_CLAIM_ATTEMPTS,
            )
        )
        eligible_ids = [row["id"] for row in candidates]
        matched_requests = {
            requested_path
            for requested_path in requested
            for row in candidates
            if requested_path
            in {
                row["id"],
                row.get("source_path") or row["id"].partition(":")[2],
            }
        }
        result: dict[str, Any] = {
            "requested": len(requested),
            "eligible": len(candidates),
            "retried": 0,
            "refused": len(requested) - len(matched_requests),
            "source_ids": eligible_ids,
            "event_ids": [],
            "dry_run": dry_run,
        }
        if dry_run or not candidates:
            return result

        items = [
            {
                "source_id": row["id"],
                "previous_status": row["status"],
                "previous_attempt_count": int(row["attempt_count"]),
                "previous_error": row.get("last_error"),
                "event_id": f"source-retry:{uuid.uuid4()}",
            }
            for row in candidates
        ]
        rows = list(
            client.query(
                """
                UNWIND $items AS item
                MATCH (sns:StandardNameSource {id: item.source_id})
                WHERE sns.status = item.previous_status
                  AND coalesce(sns.attempt_count, 0)
                      = item.previous_attempt_count
                  AND (
                    sns.status = 'failed'
                    OR (sns.status = 'extracted'
                        AND coalesce(sns.attempt_count, 0) >= $attempt_cap)
                  )
                CREATE (event:StandardNameSourceRetry {id: item.event_id})
                SET event.source_id = sns.id,
                    event.previous_status = item.previous_status,
                    event.previous_attempt_count = item.previous_attempt_count,
                    event.previous_error = item.previous_error,
                    event.reason = $reason,
                    event.retried_at = datetime()
                MERGE (sns)-[:HAS_RETRY_EVENT]->(event)
                SET sns.retry_events =
                        coalesce(sns.retry_events, []) + item.event_id,
                    sns.status = 'extracted',
                    sns.attempt_count = 0,
                    sns.claimed_at = null,
                    sns.claim_token = null,
                    sns.failed_at = null,
                    sns.last_error = null
                RETURN sns.id AS source_id, event.id AS event_id
                """,
                items=items,
                reason=reason,
                attempt_cap=services._MAX_COMPOSE_CLAIM_ATTEMPTS,
            )
        )
        result["retried"] = len(rows)
        result["refused"] += len(candidates) - len(rows)
        result["source_ids"] = [row["source_id"] for row in rows]
        result["event_ids"] = [row["event_id"] for row in rows]
        return result
    finally:
        if own:
            client.close()


def retry_skipped_sources(
    source_paths: list[str],
    *,
    services: Any,
    reason: str,
    dry_run: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Return explicitly selected skipped sources to the composition queue.

    Eligibility is deliberately exact and fail-closed: the source must still
    be ``status='skipped'``, unclaimed, and have no scalar or relationship
    binding to a StandardName. Before resetting the source, the operation
    creates a ``StandardNameSourceRetry`` event. Its required
    ``previous_status='skipped'`` field is the schema-defined recovery-kind
    discriminator; the event also retains the prior attempt count, error, and
    operator reason.

    The compare-and-set write rechecks every eligibility predicate in the same
    transaction that records the event and changes the lifecycle state. It
    clears only exclusion/failure state needed for normal composition. Source
    identity, DD/signal relationships, and exact-source compose hints are not
    written and therefore remain intact.

    ``source_paths`` accepts either full ``StandardNameSource.id`` values or
    their bare ``source_id`` values. No patterns, siblings, or families are
    expanded.
    """
    requested = sorted(
        {str(path).strip() for path in source_paths if str(path).strip()}
    )
    if not requested:
        raise ValueError("at least one source path is required")
    reason = reason.strip()
    if not reason:
        raise ValueError("a non-empty retry reason is required")

    own = gc is None
    client = services.GraphClient() if own else gc
    try:
        candidates = list(
            client.query(
                """
                MATCH (sns:StandardNameSource)
                WHERE sns.id IN $requested OR sns.source_id IN $requested
                WITH sns, coalesce(sns.attempt_count, 0) AS attempts
                WHERE sns.status = 'skipped'
                  AND sns.claimed_at IS NULL
                  AND sns.claim_token IS NULL
                  AND sns.produced_sn_id IS NULL
                  AND NOT (sns)-[:PRODUCED_NAME]->(:StandardName)
                RETURN sns.id AS id, sns.source_id AS source_path,
                       sns.status AS status,
                       attempts AS attempt_count, sns.last_error AS last_error,
                       sns.skip_reason AS skip_reason,
                       sns.skip_reason_detail AS skip_reason_detail
                ORDER BY sns.id
                """,
                requested=requested,
            )
        )
        eligible_ids = [row["id"] for row in candidates]
        matched_requests = {
            requested_path
            for requested_path in requested
            for row in candidates
            if requested_path
            in {
                row["id"],
                row.get("source_path") or row["id"].partition(":")[2],
            }
        }
        result: dict[str, Any] = {
            "requested": len(requested),
            "eligible": len(candidates),
            "retried": 0,
            "refused": len(requested) - len(matched_requests),
            "source_ids": eligible_ids,
            "event_ids": [],
            "dry_run": dry_run,
        }
        if dry_run or not candidates:
            return result

        items = [
            {
                "source_id": row["id"],
                "previous_status": row["status"],
                "previous_attempt_count": int(row["attempt_count"]),
                "previous_error": services._skipped_recovery_error(row),
                "previous_skip_reason": row.get("skip_reason"),
                "previous_skip_reason_detail": row.get("skip_reason_detail"),
                "event_id": f"source-retry:{uuid.uuid4()}",
            }
            for row in candidates
        ]
        rows = list(
            client.query(
                """
                UNWIND $items AS item
                MATCH (sns:StandardNameSource {id: item.source_id})
                WHERE sns.status = 'skipped'
                  AND item.previous_status = 'skipped'
                  AND coalesce(sns.attempt_count, 0)
                      = item.previous_attempt_count
                  AND coalesce(sns.skip_reason, '')
                      = coalesce(item.previous_skip_reason, '')
                  AND coalesce(sns.skip_reason_detail, '')
                      = coalesce(item.previous_skip_reason_detail, '')
                  AND sns.claimed_at IS NULL
                  AND sns.claim_token IS NULL
                  AND sns.produced_sn_id IS NULL
                  AND NOT (sns)-[:PRODUCED_NAME]->(:StandardName)
                CREATE (event:StandardNameSourceRetry {id: item.event_id})
                SET event.source_id = sns.id,
                    event.previous_status = item.previous_status,
                    event.previous_attempt_count = item.previous_attempt_count,
                    event.previous_error = item.previous_error,
                    event.reason = $reason,
                    event.retried_at = datetime()
                MERGE (sns)-[:HAS_RETRY_EVENT]->(event)
                SET sns.retry_events =
                        coalesce(sns.retry_events, []) + item.event_id,
                    sns.status = 'extracted',
                    sns.attempt_count = 0,
                    sns.claimed_at = null,
                    sns.claim_token = null,
                    sns.skip_reason = null,
                    sns.skip_reason_detail = null,
                    sns.skipped_at = null,
                    sns.failed_at = null,
                    sns.last_error = null
                RETURN sns.id AS source_id, event.id AS event_id
                """,
                items=items,
                reason=reason,
            )
        )
        result["retried"] = len(rows)
        result["refused"] += len(candidates) - len(rows)
        result["source_ids"] = [row["source_id"] for row in rows]
        result["event_ids"] = [row["event_id"] for row in rows]
        return result
    finally:
        if own:
            client.close()


def _skipped_recovery_error(source: dict[str, Any], *, services: Any) -> str | None:
    """Compose all durable explanations for a skipped recovery."""
    last_error = str(source.get("last_error") or "").strip()
    skip_reason = str(source.get("skip_reason") or "").strip()
    detail = str(source.get("skip_reason_detail") or "").strip()
    explanations = [last_error.rstrip(" ;")] if last_error.rstrip(" ;") else []
    if skip_reason:
        classification = f"skip_reason={skip_reason}"
        if detail:
            classification += f"; detail={detail}"
        explanations.append(classification)
    return "; ".join(explanations) or None


def mark_source_skipped(
    gc: GraphClient,
    source_id: str,
    *,
    services: Any,
    reason: str,
    detail: str | None = None,
    source_type: str = "dd",
) -> int:
    """Mark a single StandardNameSource as skipped with audit trail.

    Used by the compose worker when a candidate cannot be promoted to a
    StandardName because the source DD path lacks a usable unit string
    (user invariant — the LLM never decides units, the DD source must
    provide one). The graph node id is derived as ``{source_type}:{source_id}``
    matching the convention in :func:`merge_standard_name_sources` and
    :func:`write_skipped_sources`.

    Args:
        gc: Open GraphClient. The caller is responsible for the connection
            lifecycle so this can be called from inside an existing
            ``with GraphClient() as gc:`` block.
        source_id: Bare DD path (or signal id). The full StandardNameSource
            id is derived as ``{source_type}:{source_id}``.
        reason: Machine-readable skip classification (e.g.
            ``'dd_unit_unresolvable'``).
        detail: Free-text detail (e.g. the raw DD unit string that
            triggered the skip).
        source_type: ``'dd'`` (default) or ``'signals'``.

    Returns:
        Number of StandardNameSource nodes updated (0 if no matching node
        exists; that is expected for sources that pre-date the
        StandardNameSource pipeline).
    """
    if not source_id:
        return 0
    sns_id = f"{source_type}:{source_id}"
    result = gc.query(
        """
        MATCH (sns:StandardNameSource {id: $sns_id})
        SET sns.status = 'skipped',
            sns.skip_reason = $reason,
            sns.skip_reason_detail = $detail,
            sns.claimed_at = null,
            sns.claim_token = null
        RETURN count(sns) AS affected
        """,
        sns_id=sns_id,
        reason=reason,
        detail=detail or "",
    )
    return result[0]["affected"] if result else 0


def mark_sources_stale(source_ids: list[str], *, services: Any) -> int:
    """Mark sources as stale (source entity no longer exists).

    Not token-verified — can be called from reconciliation outside claim context.
    Returns count of updated sources.
    """
    if not source_ids:
        return 0
    with services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid})
            SET sns.status = 'stale',
                sns.claimed_at = null,
                sns.claim_token = null
            RETURN count(sns) AS affected
            """,
            source_ids=source_ids,
        )
        return result[0]["affected"] if result else 0


def write_skipped_sources(records: list[dict], *, services: Any) -> int:
    """Record DD paths that cannot be resolved to a standard name source.

    Each record must have: source_type, source_id, skip_reason,
    skip_reason_detail. Optional: description, dd_path (auto-derived from
    source_id for DD sources), signal (for signal sources), status
    (defaults to ``'skipped'`` for backward compatibility; may be
    ``'not_physical_quantity'`` for configuration metadata).

    The ``id`` is derived as ``{source_type}:{source_id}`` (matches the
    existing ``merge_standard_name_sources`` key convention).

    Idempotent — subsequent writes for the same id refresh skip_reason/
    skip_reason_detail but do not re-enqueue for composition.

    Returns count of nodes written.
    """
    if not records:
        return 0

    sources = []
    for r in records:
        source_type = r["source_type"]
        source_id = r["source_id"]
        sources.append(
            {
                "id": f"{source_type}:{source_id}",
                "source_type": source_type,
                "source_id": source_id,
                "skip_reason": r["skip_reason"],
                "skip_reason_detail": r.get("skip_reason_detail", ""),
                "description": r.get("description", ""),
                "status": r.get("status", "skipped"),
                "dd_path": r.get("dd_path")
                or (source_id if source_type == "dd" else None),
                "signal": r.get("signal")
                or (source_id if source_type == "signals" else None),
            }
        )

    with services.GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sources AS src
            MERGE (sns:StandardNameSource {id: src.id})
            ON CREATE SET
                sns.source_type = src.source_type,
                sns.source_id = src.source_id,
                sns.status = src.status,
                sns.skip_reason = src.skip_reason,
                sns.skip_reason_detail = src.skip_reason_detail,
                sns.description = src.description,
                sns.attempt_count = 0
            ON MATCH SET
                sns.status = src.status,
                sns.skip_reason = src.skip_reason,
                sns.skip_reason_detail = src.skip_reason_detail,
                sns.description = coalesce(src.description, sns.description),
                sns.claimed_at = null,
                sns.claim_token = null
            WITH sns, src
            FOREACH (_ IN CASE WHEN src.source_type = 'dd' AND src.dd_path IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (imas:IMASNode {id: src.dd_path})
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
            )
            FOREACH (_ IN CASE WHEN src.source_type = 'signals' AND src.signal IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (sig:FacilitySignal {id: src.signal})
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
            )
            RETURN count(sns) AS affected
            """,
            sources=sources,
        )
        return result[0]["affected"] if result else 0


def list_skipped_sources(
    limit: int = 100, reason: str | None = None, *, services: Any
) -> list[dict]:
    """Query skipped/not_physical_quantity StandardNameSource records.

    Returns a list of dicts with keys: id, source_type, source_id,
    skip_reason, skip_reason_detail, description.

    Args:
        limit: Maximum rows to return.
        reason: Optional skip_reason filter.
    """
    where = "sns.status IN ['skipped', 'not_physical_quantity']"
    params: dict = {"limit": limit}
    if reason is not None:
        where += " AND sns.skip_reason = $reason"
        params["reason"] = reason

    with services.GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (sns:StandardNameSource)
            WHERE {where}
            RETURN sns.id AS id,
                   sns.source_type AS source_type,
                   sns.source_id AS source_id,
                   sns.skip_reason AS skip_reason,
                   sns.skip_reason_detail AS skip_reason_detail,
                   sns.description AS description
            ORDER BY sns.skip_reason, sns.id
            LIMIT $limit
            """,
            **params,
        )
        return [dict(r) for r in result]


def get_skipped_source_counts(
    source_type: str | None = None, *, services: Any
) -> dict[str, int]:
    """Return counts of skipped StandardNameSource records grouped by skip_reason."""
    where = "sns.status = 'skipped'"
    params: dict = {}
    if source_type is not None:
        where += " AND sns.source_type = $source_type"
        params["source_type"] = source_type

    with services.GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (sns:StandardNameSource)
            WHERE {where}
            RETURN coalesce(sns.skip_reason, 'unknown') AS skip_reason,
                   count(sns) AS cnt
            ORDER BY cnt DESC
            """,
            **params,
        )
        return {r["skip_reason"]: r["cnt"] for r in result}


def release_standard_name_source_claims(token: str, *, services: Any) -> int:
    """Release all claims held by this token without changing status.

    Used for error recovery — release claims so other workers can pick them up.
    Returns count of released sources.
    """
    with services.GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sns:StandardNameSource {claim_token: $token})
            SET sns.claimed_at = null,
                sns.claim_token = null
            RETURN count(sns) AS affected
            """,
            token=token,
        )
        return result[0]["affected"] if result else 0


# =============================================================================
# Polling-based work claiming — compose and review
# =============================================================================

# Settle window inserted before the post-claim winner re-read (see
# _verify_docs_claim_winners / _verify_name_claim_winners). Concurrent replicas
# that bind the same eligible node commit their claim SETs in a lock-serialised
# burst; the last committer's claim_seq wins the node. Re-reading immediately
# after our own commit can observe our claim_seq as current before a later
# racer commits, so every racer would momentarily look like the winner and each
# would fire a paid LLM call. Waiting out this short window lets the burst's
# final committer land, so exactly one replica survives the re-read. Kept small
# (batch LLM calls dwarf it) and overridable to 0 in tests.

# In-process tally of paid-call outcomes at the persist step, keyed by
# (run_id, pool). ``attempts`` counts every persist call that follows a paid
# LLM call; ``wasted`` counts those that no-oped because a concurrent replica
# already advanced the node past our claim (the claim-race residue). The run
# summary reads a snapshot to surface a wasted-paid-call ratio and warn when it
# exceeds the tripwire threshold. Populated by persist functions that can
# silently no-op (currently the docs axis — the names axis raises instead).
_persist_outcomes: dict[tuple[str, str], list[int]] = {}


def _record_persist_outcome(
    run_id: str | None, pool: str, *, services: Any, persisted: bool
) -> None:
    """Record one paid-call persist outcome for the (run_id, pool) tallies."""
    key = (run_id or "", pool)
    entry = services._persist_outcomes.setdefault(key, [0, 0])
    entry[0] += 1
    if not persisted:
        entry[1] += 1


def persist_outcome_snapshot(
    run_id: str | None, *, services: Any
) -> dict[str, dict[str, int]]:
    """Return ``{pool: {"attempts": a, "wasted": w}}`` for *run_id*.

    ``wasted`` is the count of paid LLM calls whose persist no-oped because a
    concurrent replica had already claimed and advanced the node — the direct
    measure of claim-race waste. Empty when nothing was recorded for the run.
    """
    rid = run_id or ""
    return {
        pool: {"attempts": a, "wasted": w}
        for (r, pool), (a, w) in services._persist_outcomes.items()
        if r == rid
    }


def reset_persist_outcomes(run_id: str | None = None, *, services: Any) -> None:
    """Clear persist-outcome tallies (all runs, or just *run_id*)."""
    if run_id is None:
        services._persist_outcomes.clear()
        return
    for key in [k for k in services._persist_outcomes if k[0] == (run_id or "")]:
        del services._persist_outcomes[key]
