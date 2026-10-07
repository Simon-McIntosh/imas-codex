"""Shared status reset infrastructure for all discovery domains.

Each domain defines a ``ResetSpec`` describing valid target states and the
fields that must be cleared when resetting nodes to that state.  The generic
:func:`reset_to_status` function builds a Cypher query from the spec and
executes it within a single transaction.

Usage from a CLI command::

    from imas_codex.discovery.base.reset import reset_to_status

    count = reset_to_status(
        spec=SIGNALS_RESET_SPECS["discovered"],
        facility=facility,
        extra_filter="AND s.discovery_source IN $sources",
        extra_params={"sources": scanner_types},
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResetSpec:
    """Describes how to reset nodes of a given label to a target status."""

    label: str
    """Neo4j node label, e.g. ``FacilitySignal``."""

    target_status: str
    """Status value to set, e.g. ``discovered``."""

    source_statuses: list[str]
    """Statuses that are eligible for reset (nodes *not* in this list are
    left untouched)."""

    clear_fields: list[str]
    """Node properties to SET to ``null`` when resetting."""

    source_filter: str | None = None
    """Optional Cypher predicate (alias ``n``) that admits rows whose status is
    *not* in ``source_statuses``.  It is combined as
    ``n.status IN $source_statuses OR (<source_filter>)``, so it can widen the
    reset to a specific subset of one extra status without matching every row
    of that status."""

    facility_key: str = "facility_id"
    """Property on the node that holds the facility identifier.  Most domains
    use ``facility_id``; paths use a relationship ``-[:AT_FACILITY]->``."""

    facility_via_rel: bool = False
    """If ``True``, the facility is matched via a relationship
    ``(n)-[:AT_FACILITY]->(f:Facility {id: $facility})`` instead of a
    direct property on the node."""

    post_cypher: str | None = None
    """Optional Cypher fragment appended *after* the main SET clause.
    Useful for relationship cleanup (e.g., deleting ``CHECKED_WITH``)."""


def reset_to_status(
    spec: ResetSpec,
    facility: str,
    *,
    path_prefixes: list[str] | None = None,
    extra_filter: str = "",
    extra_params: dict | None = None,
) -> int:
    """Reset nodes matching *spec* back to ``spec.target_status``.

    Args:
        spec: Reset specification for the target state.
        facility: Facility identifier.
        path_prefixes: When given, restrict the reset to nodes whose ``path``
            starts with one of these prefixes; the clause and its Cypher
            parameter name are rendered here so callers pass the list only.
            When ``None``, the reset covers the whole facility.
        extra_filter: Additional Cypher WHERE fragment (e.g., scanner filter).
            Must start with ``AND``.  Use ``n`` as the node alias.
        extra_params: Additional query parameters referenced by *extra_filter*.

    Returns:
        Number of nodes reset.
    """
    from imas_codex.graph import GraphClient
    from imas_codex.graph.query_builder import render_path_prefix_clause

    # Build MATCH clause
    if spec.facility_via_rel:
        match = f"MATCH (n:{spec.label})-[:AT_FACILITY]->(f:Facility {{id: $facility}})"
    else:
        match = f"MATCH (n:{spec.label} {{{spec.facility_key}: $facility}})"

    # Build WHERE clause — the prefix scope is rendered here so callers never
    # spell the predicate or name its parameter themselves.
    status_predicate = "n.status IN $source_statuses"
    if spec.source_filter:
        status_predicate = f"({status_predicate} OR ({spec.source_filter}))"
    where_parts = [f"WHERE {status_predicate}"]
    if extra_filter:
        where_parts.append(extra_filter)
    if path_prefixes:
        where_parts.append(render_path_prefix_clause("n", "path_prefixes"))
    where = " ".join(where_parts)

    # Build SET clause
    set_parts = ["n.status = $target_status", "n.claimed_at = null"]
    for fld in spec.clear_fields:
        set_parts.append(f"n.{fld} = null")
    set_clause = "SET " + ",\n    ".join(set_parts)

    # Optional post-processing
    post = ""
    if spec.post_cypher:
        post = f"\nWITH n\n{spec.post_cypher}"

    query = f"""
        {match}
        {where}
        {set_clause}
        {post}
        RETURN count(n) AS reset_count
    """

    params = {
        "facility": facility,
        "source_statuses": spec.source_statuses,
        "target_status": spec.target_status,
    }
    if path_prefixes:
        params["path_prefixes"] = path_prefixes
    params.update(extra_params or {})

    with GraphClient() as gc:
        result = gc.query(query, **params)
        rows = list(result)
        count = rows[0]["reset_count"] if rows else 0

    logger.info(
        "Reset %d %s nodes to '%s' for %s",
        count,
        spec.label,
        spec.target_status,
        facility,
    )
    return count


# ═══════════════════════════════════════════════════════════════════════════
# Domain-specific reset specs
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

SIGNAL_RESET_SPECS: dict[str, ResetSpec] = {
    "discovered": ResetSpec(
        label="FacilitySignal",
        target_status="discovered",
        source_statuses=["enriched", "underspecified", "checked"],
        clear_fields=[
            "enrichment_source",
            "enriched_at",
            "name",
            "description",
            "physics_domain",
            "keywords",
            "sign_convention",
            "diagnostic",
            "analysis_code",
            "embedding",
            "embedded_at",
            "embed_failed_at",
            "checked",
            "checked_at",
            "check_retries",
            "context_quality",
            "enrichment_model",
            "enrichment_prompt_hash",
        ],
        post_cypher=("OPTIONAL MATCH (n)-[r:MEMBER_OF]->() DELETE r"),
    ),
    "enriched": ResetSpec(
        label="FacilitySignal",
        target_status="enriched",
        source_statuses=["checked"],
        clear_fields=[
            "checked",
            "checked_at",
            "check_retries",
        ],
        post_cypher=("OPTIONAL MATCH (n)-[r:CHECKED_WITH]->() DELETE r"),
    ),
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PATH_SCORE_FIELDS = [
    "scored_at",
    "score_composite",
    "score_percentile",
    "score_reason",
    "primary_evidence",
    "evidence_summary",
    "score_cost",
    "score_modeling_code",
    "score_analysis_code",
    "score_operations_code",
    "score_modeling_data",
    "score_experimental_data",
    "score_data_access",
    "score_workflow",
    "score_visualization",
    "score_documentation",
    "score_imas",
    "score_convention",
]

_PATH_TRIAGE_FIELDS = [
    "triaged_at",
    "triage_composite",
    "triage_modeling_code",
    "triage_analysis_code",
    "triage_operations_code",
    "triage_modeling_data",
    "triage_experimental_data",
    "triage_data_access",
    "triage_workflow",
    "triage_visualization",
    "triage_documentation",
    "triage_imas",
    "triage_convention",
    "description",
    "path_purpose",
    "keywords",
    "physics_domain",
    "evidence_id",
    "should_expand",
    "should_enrich",
    "expansion_reason",
    "skip_reason",
    "enrich_skip_reason",
]

_PATH_ENRICH_FIELDS = [
    "is_enriched",
    "enriched_at",
    "total_bytes",
    "total_lines",
    "language_breakdown",
    "is_multiformat",
    "enrich_warnings",
]

PATH_RESET_SPECS: dict[str, ResetSpec] = {
    "triaged": ResetSpec(
        label="FacilityPath",
        target_status="triaged",
        source_statuses=["scored"],
        clear_fields=_PATH_SCORE_FIELDS + ["expanded_at"],
        facility_via_rel=True,
    ),
    "scanned": ResetSpec(
        label="FacilityPath",
        target_status="scanned",
        source_statuses=["triaged", "scored"],
        # A seeded root can sit at ``discovered`` while carrying a triage score
        # (a ``--root`` re-seed may overwrite its status as its scores survive),
        # leaving it unclaimable: the scan takes only discovered rows whose
        # triage is null and the triage claim only scanned rows.  The bound
        # admits that stranded subset only — a never-scanned discovered row has
        # no triage score and stays untouched, so an unscoped reset cannot push
        # it past enumeration.
        source_filter="n.status = 'discovered' AND n.triage_composite IS NOT NULL",
        clear_fields=(
            _PATH_TRIAGE_FIELDS
            + _PATH_ENRICH_FIELDS
            + _PATH_SCORE_FIELDS
            + ["expanded_at"]
        ),
        facility_via_rel=True,
    ),
}

# ---------------------------------------------------------------------------
# Wiki
# ---------------------------------------------------------------------------

_WIKI_SCORE_FIELDS = [
    "score_composite",
    "purpose",
    "description",
    "reasoning",
    "keywords",
    "physics_domain",
    "preview_text",
    "score_data_documentation",
    "score_physics_content",
    "score_code_documentation",
    "score_data_access",
    "score_calibration",
    "score_imas_relevance",
    "should_ingest",
    "skip_reason",
    "is_physics_content",
    "score_cost",
    "scored_at",
    "preview_fetched_at",
]

WIKI_RESET_SPECS: dict[str, ResetSpec] = {
    "scanned": ResetSpec(
        label="WikiPage",
        target_status="scanned",
        source_statuses=["scored", "skipped", "ingested"],
        clear_fields=_WIKI_SCORE_FIELDS + ["ingested_at", "chunk_count"],
    ),
    "scored": ResetSpec(
        label="WikiPage",
        target_status="scored",
        source_statuses=["ingested"],
        clear_fields=["ingested_at", "chunk_count"],
    ),
}

# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------

# The relevance fields a CodeFile carries, in one registry: the five noul fields
# the names arm writes, the role distribution and its confidence from the role
# question, and the bookkeeping that records which arm wrote them.  The code
# scorer imports this tuple as ``RELEVANCE_FIELDS`` so the fields a reset clears
# and the fields the decision arms write cannot drift apart.  It lives here, not
# in ``code/scorer.py``, because this module is imported before the code package
# and importing the code package from here would close an import cycle.
CODE_RELEVANCE_FIELDS = (
    "relevance_loads",
    "relevance_processes",
    "relevance_describes",
    "relevance_imas",
    "relevance_simulation",
    "relevance_role",
    "relevance_role_probs",
    "relevance_role_confidence",
    "relevance_stage",
    "relevance_model",
    "triaged_at",
)

_CODE_RELEVANCE_FIELDS = list(CODE_RELEVANCE_FIELDS)

_CODE_ENRICH_FIELDS = [
    "is_enriched",
    "enriched_at",
    "pattern_categories",
    "total_pattern_matches",
    "line_count",
    "preview_text",
    "content_hash",
]

_CODE_SCORE_FIELDS = [
    "score_composite",
    "score_reason",
    "score_data_access",
    "score_data_access_probs",
    "score_data_access_confidence",
    "score_signal_processing",
    "score_signal_processing_probs",
    "score_signal_processing_confidence",
    "score_machine_description",
    "score_machine_description_probs",
    "score_machine_description_confidence",
    "score_imas_mapping",
    "score_imas_mapping_probs",
    "score_imas_mapping_confidence",
    "relevance_grade",
    "relevance_grade_probs",
    "relevance_grade_confidence",
    "scored_at",
    "score_cost",
]

# A file reset below ``ingested`` must not leave the example and chunk nodes the
# ingestion stage wrote for it.  The traversal is the one ``clear_facility_code``
# uses (``CodeFile <-FROM_FILE- CodeExample -HAS_CHUNK-> CodeChunk``), scoped to
# the files this reset touches.  ``collect`` drops the nulls an optional match
# yields for a file with no examples, so the delete never sees a null; the
# ``DISTINCT n`` collapses the example fan-out so the trailing ``count(n)``
# counts files, not deleted examples.
_CODE_CHUNK_CASCADE = """OPTIONAL MATCH (n)<-[:FROM_FILE]-(ce:CodeExample)-[:HAS_CHUNK]->(cc:CodeChunk)
WITH n, collect(ce) + collect(cc) AS doomed
FOREACH (d IN doomed | DETACH DELETE d)
WITH DISTINCT n"""


CODE_RESET_SPECS: dict[str, ResetSpec] = {
    "discovered": ResetSpec(
        label="CodeFile",
        target_status="discovered",
        source_statuses=["triaged", "scored", "ingested", "enriched", "skipped"],
        clear_fields=(
            _CODE_RELEVANCE_FIELDS
            + _CODE_ENRICH_FIELDS
            + _CODE_SCORE_FIELDS
            + ["ingested_at", "skip_reason", "error", "evidence_linked"]
        ),
        post_cypher=_CODE_CHUNK_CASCADE,
    ),
    "triaged": ResetSpec(
        label="CodeFile",
        target_status="triaged",
        source_statuses=["scored", "ingested", "enriched"],
        clear_fields=(
            _CODE_ENRICH_FIELDS
            + _CODE_SCORE_FIELDS
            + ["ingested_at", "skip_reason", "error"]
        ),
        post_cypher=_CODE_CHUNK_CASCADE,
    ),
    "scored": ResetSpec(
        label="CodeFile",
        target_status="scored",
        source_statuses=["ingested"],
        clear_fields=["ingested_at"],
    ),
}

# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

_DOC_SCORE_FIELDS = [
    "score_composite",
    "document_purpose",
    "description",
    "reasoning",
    "keywords",
    "physics_domain",
    "preview_text",
    "score_data_documentation",
    "score_physics_content",
    "score_code_documentation",
    "score_data_access",
    "score_calibration",
    "score_imas_relevance",
    "should_ingest",
    "skip_reason",
    "score_cost",
    "scored_at",
    "defer_reason",
]

DOCUMENT_RESET_SPECS: dict[str, ResetSpec] = {
    "discovered": ResetSpec(
        label="Document",
        target_status="discovered",
        source_statuses=["scored", "skipped", "deferred", "ingested"],
        clear_fields=_DOC_SCORE_FIELDS + ["ingested_at", "chunk_count"],
    ),
    "scored": ResetSpec(
        label="Document",
        target_status="scored",
        source_statuses=["ingested"],
        clear_fields=["ingested_at", "chunk_count"],
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Registry — maps domain name to its reset specs
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_RESET_SPECS: dict[str, dict[str, ResetSpec]] = {
    "signals": SIGNAL_RESET_SPECS,
    "paths": PATH_RESET_SPECS,
    "wiki": WIKI_RESET_SPECS,
    "code": CODE_RESET_SPECS,
    "documents": DOCUMENT_RESET_SPECS,
}


def get_valid_targets(domain: str) -> list[str]:
    """Return valid ``--reset-to`` target states for a domain."""
    specs = DOMAIN_RESET_SPECS.get(domain, {})
    return sorted(specs.keys())
