"""MCP tools for standard name search, fetch, and listing.

Functions are prefixed with ``_`` — they are registered as MCP tools
in ``server.py`` via ``@self.mcp.tool()``.

Search delegates to :mod:`imas_codex.standard_names.search` which performs
3-stream RRF fusion (vector + keyword + tiered grammar matching).
"""

from __future__ import annotations

import dataclasses
import logging

from neo4j.exceptions import ServiceUnavailable

from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import EditScope
from imas_codex.standard_names.search import (
    check_names as _check_names_backing,
    fetch_standard_names as _fetch_sn_backing,
    find_related as _find_related_backing,
    search_standard_names as _search_sn_backing,
    summarise_family as _summarise_family_backing,
)

logger = logging.getLogger(__name__)

NEO4J_NOT_RUNNING_MSG = (
    "Neo4j is not running. Check service with: systemctl --user status imas-codex-neo4j"
)


def _neo4j_error_message(e: Exception) -> str:
    """Format Neo4j errors with helpful instructions."""
    if isinstance(e, ServiceUnavailable):
        return NEO4J_NOT_RUNNING_MSG
    msg = str(e)
    if "Connection refused" in msg or "ServiceUnavailable" in msg:
        return NEO4J_NOT_RUNNING_MSG
    return msg


# ---------------------------------------------------------------------------
# _search_standard_names
# ---------------------------------------------------------------------------


def _search_standard_names(
    query: str,
    *,
    kind: str | None = None,
    name_stage: str | None = None,
    cocos_type: str | None = None,
    physics_domain: str | None = None,
    physical_base: str | None = None,
    subject: str | None = None,
    aggregation: str | None = None,
    orbit: str | None = None,
    population: str | None = None,
    transformation: str | None = None,
    component: str | None = None,
    coordinate: str | None = None,
    process: str | None = None,
    position: str | None = None,
    region: str | None = None,
    device: str | None = None,
    geometric_base: str | None = None,
    k: int = 20,
    gc: GraphClient | None = None,
) -> str:
    """Search standard names by physics concept.

    Delegates to :func:`~imas_codex.standard_names.search.search_standard_names`
    which performs 3-stream RRF fusion (vector + keyword + tiered grammar).
    Quarantined, exhausted, and superseded names are excluded automatically.

    When grammar-segment filters are provided (``physical_base``,
    ``subject``, etc.), results are filtered via ``sn.<segment>`` property
    matching — the canonical source of truth for both open- and closed-
    vocabulary segments.
    """
    has_filters = any(
        v is not None
        for v in (
            kind,
            name_stage,
            cocos_type,
            physics_domain,
            physical_base,
            subject,
            aggregation,
            orbit,
            population,
            transformation,
            component,
            coordinate,
            process,
            position,
            region,
            device,
            geometric_base,
        )
    )
    if (not query or not query.strip()) and not has_filters:
        return (
            "## Standard Name Search Results\n\n"
            "No query provided. Pass a physics concept "
            "(e.g. `electron temperature`, `magnetic flux`) "
            "or a grammar-segment filter (e.g. `physical_base=magnetic_flux`)."
        )

    try:
        if gc is None:
            gc = GraphClient()
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error connecting to graph: {e}"

    segment_filters: dict[str, str] = {}
    _seg_map = {
        "physical_base": physical_base,
        "subject": subject,
        "aggregation": aggregation,
        "orbit": orbit,
        "population": population,
        "transformation": transformation,
        "component": component,
        "coordinate": coordinate,
        "process": process,
        "position": position,
        "region": region,
        "device": device,
        "geometric_base": geometric_base,
    }
    for seg, val in _seg_map.items():
        if val is not None:
            segment_filters[seg] = val

    try:
        rows = _search_sn_backing(
            query or "",
            k=k,
            segment_filters=segment_filters or None,
            kind=kind,
            name_stage=name_stage,
            cocos_type=cocos_type,
            physics_domain=physics_domain,
            gc=gc,
        )
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Search failed: {_neo4j_error_message(e)}"

    return _format_search_report(query or "", rows)


def _format_search_report(query: str, rows: list[dict]) -> str:
    """Format search results as a text report."""
    if not rows:
        return (
            f"## Standard Name Search Results\n\nNo standard names found matching "
            f'"{query}".'
        )

    lines = [
        f'## Standard Name Search Results\n\nFound {len(rows)} standard names matching "{query}"\n'
    ]
    for i, row in enumerate(rows, 1):
        name = row.get("name") or "unknown"
        score = row.get("score", 0.0)
        kind = row.get("kind") or ""
        unit = row.get("unit") or ""
        name_stage = row.get("name_stage") or ""
        description = row.get("description") or ""
        documentation = row.get("documentation") or ""
        cocos_transformation_type = row.get("cocos_transformation_type") or ""
        cocos = row.get("cocos")

        lines.append(f"### {i}. {name} (score: {score:.2f})")
        if kind:
            lines.append(f"- **Kind:** {kind}")
        if unit:
            lines.append(f"- **Unit:** {unit}")
        if name_stage:
            lines.append(f"- **Stage:** {name_stage}")
        if cocos_transformation_type:
            lines.append(f"- **COCOS Transformation:** {cocos_transformation_type}")
        if cocos is not None:
            lines.append(f"- **COCOS:** {cocos}")
        if description:
            lines.append(f"- **Description:** {description}")
        if documentation:
            lines.append(
                f"- **Documentation:** {documentation[:200]}{'...' if len(documentation) > 200 else ''}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _fetch_standard_names
# ---------------------------------------------------------------------------


def _fetch_standard_names(
    names: str,
    *,
    gc: GraphClient | None = None,
) -> str:
    """Fetch full entries for known standard names.

    Args:
        names: Space- or comma-separated standard name IDs.
    """
    try:
        if gc is None:
            gc = GraphClient()
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error connecting to graph: {e}"

    # Parse names (split on space or comma)
    import re

    name_list = [n.strip() for n in re.split(r"[,\s]+", names) if n.strip()]
    if not name_list:
        return "No names provided."

    try:
        rows = _fetch_sn_backing(
            name_list,
            include_grammar=True,
            include_documentation=True,
            gc=gc,
        )
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Fetch failed: {_neo4j_error_message(e)}"

    if not rows:
        not_found = ", ".join(name_list)
        return f"No standard names found for: {not_found}"

    return _format_fetch_report(rows, name_list)


def _format_fetch_report(rows: list[dict], requested: list[str]) -> str:
    """Format fetch results as a detailed report."""
    found_names = {r.get("name") for r in rows}
    not_found = [n for n in requested if n not in found_names]

    lines = ["## Standard Name Details\n"]

    for row in rows:
        name = row.get("name") or "unknown"
        lines.append(f"### {name}")
        lines.append("")

        description = row.get("description") or ""
        documentation = row.get("documentation") or ""
        kind = row.get("kind") or ""
        unit = row.get("unit") or ""
        links = row.get("links") or []
        dd_paths = row.get("source_paths") or []
        constraints = row.get("constraints") or []
        validity_domain = row.get("validity_domain") or ""
        name_stage = row.get("name_stage") or ""
        cocos_transformation_type = row.get("cocos_transformation_type") or ""
        cocos = row.get("cocos")
        dd_version = row.get("dd_version") or ""
        source_ids = row.get("source_ids") or []
        source_ids_names = row.get("source_ids_names") or []

        if description:
            lines.append(f"**Description:** {description}")
        if documentation:
            lines.append(f"\n**Documentation:**\n{documentation}")
        lines.append("")

        if kind:
            lines.append(f"- **Kind:** {kind}")
        if unit:
            lines.append(f"- **Unit:** {unit}")
        if name_stage:
            lines.append(f"- **Stage:** {name_stage}")
        if cocos_transformation_type:
            lines.append(f"- **COCOS Transformation:** {cocos_transformation_type}")
        if cocos is not None:
            lines.append(f"- **COCOS:** {cocos}")
        if dd_version:
            lines.append(f"- **DD Version:** {dd_version}")

        if links:
            link_str = ", ".join(links) if isinstance(links, list) else str(links)
            lines.append(f"- **Links:** {link_str}")
        if dd_paths:
            path_str = (
                "\n  - " + "\n  - ".join(dd_paths)
                if isinstance(dd_paths, list)
                else str(dd_paths)
            )
            lines.append(f"- **Sources:**{path_str}")
        if constraints:
            c_str = (
                ", ".join(constraints)
                if isinstance(constraints, list)
                else str(constraints)
            )
            lines.append(f"- **Constraints:** {c_str}")
        if validity_domain:
            lines.append(f"- **Validity Domain:** {validity_domain}")
        if source_ids:
            src_str = ", ".join(s for s in source_ids if s)
            if src_str:
                lines.append(f"- **Source Nodes:** {src_str}")
        if source_ids_names:
            ids_str = ", ".join(s for s in source_ids_names if s)
            if ids_str:
                lines.append(f"- **Source IDS:** {ids_str}")

        lines.append("")

    if not_found:
        lines.append(f"**Not found:** {', '.join(not_found)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _list_standard_names
# ---------------------------------------------------------------------------


def _list_standard_names(
    *,
    kind: str | None = None,
    name_stage: str | None = None,
    cocos_type: str | None = None,
    physics_domain: str | None = None,
    include_superseded: bool = False,
    gc: GraphClient | None = None,
) -> str:
    """List standard names with optional filters.

    Superseded names (``name_stage='superseded'``) are excluded by default.
    Pass ``include_superseded=True`` to include them.
    """
    try:
        if gc is None:
            gc = GraphClient()
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error connecting to graph: {e}"

    # Build WHERE clause
    conditions = []
    params: dict = {}

    # Always exclude superseded names unless explicitly requested.
    if not include_superseded:
        conditions.append("coalesce(sn.name_stage, '') <> 'superseded'")

    if kind:
        conditions.append("toLower(sn.kind) = toLower($kind)")
        params["kind"] = kind
    if name_stage:
        conditions.append("toLower(sn.name_stage) = toLower($name_stage)")
        params["name_stage"] = name_stage
    if cocos_type:
        conditions.append("sn.cocos_transformation_type = $cocos_type")
        params["cocos_type"] = cocos_type
    if physics_domain:
        # Match either the promoted scalar or any source_domain.
        conditions.append(
            "(sn.physics_domain = $physics_domain "
            "OR $physics_domain IN coalesce(sn.source_domains, []))"
        )
        params["physics_domain"] = physics_domain

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    cypher = f"""
MATCH (sn:StandardName)
{where_clause}
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
RETURN sn.id AS name, sn.kind AS kind,
       coalesce(u.id, sn.unit) AS unit,
       sn.name_stage AS name_stage,
       sn.cocos_transformation_type AS cocos_transformation_type,
       sn.cocos AS cocos,
       sn.description AS description
ORDER BY sn.id
"""

    try:
        rows = gc.query(cypher, **params)
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"List failed: {_neo4j_error_message(e)}"

    return _format_list_report(
        rows,
        kind=kind,
        name_stage=name_stage,
        cocos_type=cocos_type,
        physics_domain=physics_domain,
    )


def _format_list_report(
    rows: list[dict],
    *,
    kind: str | None = None,
    name_stage: str | None = None,
    cocos_type: str | None = None,
    physics_domain: str | None = None,
) -> str:
    """Format list results as a markdown table."""
    filter_parts = []
    if kind:
        filter_parts.append(f"kind={kind}")
    if name_stage:
        filter_parts.append(f"stage={name_stage}")
    if cocos_type:
        filter_parts.append(f"cocos_type={cocos_type}")
    if physics_domain:
        filter_parts.append(f"physics_domain={physics_domain}")
    filter_str = f" (filtered by: {', '.join(filter_parts)})" if filter_parts else ""

    if not rows:
        return f"## Standard Names\n\nNo standard names found{filter_str}."

    lines = [
        f"## Standard Names ({len(rows)} total{filter_str})\n",
        "| Name | Kind | Unit | Status | COCOS Type | COCOS | Description |",
        "|------|------|------|--------|------------|-------|-------------|",
    ]

    for row in rows:
        name = row.get("name") or ""
        row_kind = row.get("kind") or ""
        unit = row.get("unit") or ""
        status = row.get("name_stage") or ""
        cocos_type_val = row.get("cocos_transformation_type") or ""
        cocos_val = row.get("cocos") if row.get("cocos") is not None else ""
        desc = row.get("description") or ""
        # Truncate long descriptions
        if len(desc) > 80:
            desc = desc[:77] + "..."
        lines.append(
            f"| {name} | {row_kind} | {unit} | {status} | {cocos_type_val} | {cocos_val} | {desc} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _list_grammar_vocabulary
# ---------------------------------------------------------------------------


def _list_grammar_vocabulary(
    segment: str,
) -> str:
    """List the valid vocabulary for a grammar segment.

    Queries ``imas_standard_names.grammar.constants.SEGMENT_TOKEN_MAP``
    to return the ISN-defined token list for the requested segment.
    All vocabulary segments are closed — every token must come from the
    installed ISN vocabulary.

    Args:
        segment: Segment name (e.g. ``"physical_base"``, ``"component"``,
            ``"subject"``). Case-insensitive. Valid values come from the
            installed ISN package's ``SEGMENT_TOKEN_MAP`` keys.

    Returns:
        Markdown table of tokens for the requested segment.
    """
    try:
        from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP
    except ImportError:
        return "ISN package unavailable — cannot list grammar vocabulary."

    seg_lc = (segment or "").strip().lower()
    valid_segments = tuple(sorted(SEGMENT_TOKEN_MAP.keys()))
    if seg_lc not in valid_segments:
        valid = ", ".join(valid_segments)
        return f"Unknown grammar segment '{segment}'. Valid segments: {valid}."

    tokens = SEGMENT_TOKEN_MAP.get(seg_lc, ())
    if not tokens:
        return (
            f"## Grammar Vocabulary: `{seg_lc}`\n\n"
            f"Segment `{seg_lc}` has no tokens defined in the installed ISN "
            f"package. This may indicate a packaging issue."
        )

    sorted_tokens = sorted(tokens)
    lines = [
        f"## Grammar Vocabulary: `{seg_lc}`",
        "",
        f"{len(sorted_tokens)} tokens in closed vocabulary (ISN source of truth).",
        "",
        "| Token |",
        "|-------|",
    ]
    for tok in sorted_tokens:
        lines.append(f"| {tok} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _find_related_standard_names
# ---------------------------------------------------------------------------


def _find_related_standard_names(
    name: str,
    *,
    relationship_types: str = "all",
    max_results: int = 20,
    gc: GraphClient | None = None,
) -> str:
    """Find standard names related to *name* across multiple relationships.

    Args:
        name: StandardName.id to centre the discovery on.
        relationship_types: Comma-separated list or ``"all"``. Recognised
            tokens: ``grammar, unit, cocos, cluster, lineage, source``.
        max_results: Maximum results per bucket.

    Returns:
        Bucketed markdown report. Empty buckets suppressed;
        deterministic order via ``RELATED_BUCKET_ORDER``.
    """
    try:
        if gc is None:
            gc = GraphClient()
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error connecting to graph: {e}"

    try:
        exists_rows = gc.query(
            "MATCH (sn:StandardName {id: $name}) RETURN sn.id AS id LIMIT 1",
            name=name,
        )
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"find_related failed: {_neo4j_error_message(e)}"
    if not exists_rows:
        return (
            f"## Related Standard Names — `{name}`\n\n"
            f"Standard name `{name}` not found in the catalogue. "
            "Use `check_standard_names` for spelling suggestions."
        )

    try:
        buckets = _find_related_backing(
            name,
            relationship_types=relationship_types,
            max_results=max_results,
            gc=gc,
        )
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"find_related failed: {_neo4j_error_message(e)}"

    if not buckets:
        return f"## Related Standard Names — `{name}`\n\nNo related names found."

    total = sum(len(rows) for rows in buckets.values())
    lines = [
        f"## Related Standard Names — `{name}` ({total} hits)",
        "",
    ]
    for bucket, rows in buckets.items():
        lines.append(f"### {bucket} ({len(rows)})")
        lines.append("")
        for r in rows:
            related_name = r.get("name") or ""
            desc = r.get("description") or ""
            if desc and len(desc) > 120:
                desc = desc[:117] + "..."
            if desc:
                lines.append(f"- **{related_name}** — {desc}")
            else:
                lines.append(f"- **{related_name}**")
        lines.append("")
    return "\n".join(lines)


def _trace_standard_name_provenance(
    name: str,
    *,
    include_reviews: bool = False,
    max_depth: int = 10,
    gc: GraphClient | None = None,
) -> str:
    """Render explicitly requested semantic sources and internal history."""
    owns_gc = gc is None
    try:
        gc = gc or GraphClient()
        from imas_codex.standard_names.provenance_lifecycle import (
            trace_standard_name_provenance,
        )

        trace = trace_standard_name_provenance(
            gc,
            name,
            include_reviews=include_reviews,
            max_depth=max_depth,
        )
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as exc:
        return f"Provenance trace failed: {_neo4j_error_message(exc)}"
    finally:
        if owns_gc and gc is not None:
            gc.close()

    lines = [f"## Standard Name Provenance — `{name}`", "", "### Semantic sources"]
    sources = trace["semantic_sources"]
    if not sources:
        lines.append("No semantic sources recorded.")
    for source in sources:
        source_id = source.get("dd_path") or source.get("signal_id") or "unknown"
        facet = source.get("provenance")
        suffix = f" ({facet})" if facet else ""
        lines.append(f"- `{source_id}`{suffix}")
    lines.extend(["", "### Internal change history"])
    changes = trace["internal_changes"]
    if not changes:
        lines.append("No compacted internal changes recorded.")
    for change in changes:
        reason = f" — {change['reason']}" if change.get("reason") else ""
        lines.append(
            f"- `{change.get('from_name')}` → `{change.get('to_name')}` "
            f"[{change.get('operation')}] {reason}".rstrip()
        )
    if include_reviews:
        lines.extend(["", "### Review summaries"])
        for review in trace.get("reviews", []):
            lines.append(
                f"- {review.get('axis')}: score={review.get('score')} "
                f"tier={review.get('tier')}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _check_standard_names
# ---------------------------------------------------------------------------


def _check_standard_names(
    names: str,
    *,
    gc: GraphClient | None = None,
) -> str:
    """Validate names against the catalogue with Levenshtein suggestions.

    Args:
        names: Space- or comma-separated StandardName ids.

    Returns:
        Markdown table with columns ``name | exists | suggestion | reason``.
    """
    try:
        if gc is None:
            gc = GraphClient()
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error connecting to graph: {e}"

    import re

    name_list = [n.strip() for n in re.split(r"[,\s]+", names) if n.strip()]
    if not name_list:
        return "No names provided."

    try:
        results = _check_names_backing(name_list, gc=gc)
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"check_names failed: {_neo4j_error_message(e)}"

    if not results:
        return "## Standard-Name Validation\n\nNo results."

    n_missing = sum(1 for r in results if not r.get("exists"))
    lines = [
        f"## Standard-Name Validation ({len(results)} checked, {n_missing} missing)",
        "",
        "| Name | Exists | Suggestion | Reason |",
        "|------|:------:|------------|--------|",
    ]
    for r in results:
        nm = r.get("name") or ""
        exists = "✓" if r.get("exists") else "✗"
        sugg = r.get("suggestion") or ""
        reason = r.get("reason") or ""
        lines.append(f"| `{nm}` | {exists} | `{sugg}` | {reason} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _get_standard_name_summary
# ---------------------------------------------------------------------------


def _get_standard_name_summary(
    physical_base: str,
    *,
    gc: GraphClient | None = None,
) -> str:
    """Family overview for *physical_base*.

    Args:
        physical_base: The Tier-1 anchor segment value (e.g. ``"temperature"``).

    Returns:
        Markdown report with member count, distinct segment values per
        secondary segment, units, COCOS types, physics domains, sample
        names, and lineage counts.
    """
    try:
        if gc is None:
            gc = GraphClient()
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"Error connecting to graph: {e}"

    try:
        summary = _summarise_family_backing(physical_base, gc=gc)
    except ServiceUnavailable:
        return NEO4J_NOT_RUNNING_MSG
    except Exception as e:
        return f"summarise_family failed: {_neo4j_error_message(e)}"

    count = summary.get("count", 0)
    if count == 0:
        return (
            f"## Family Summary — `physical_base = {physical_base}`\n\n"
            f"No standard names with this physical_base."
        )

    lines = [
        f"## Family Summary — `physical_base = {physical_base}` ({count} members)",
        "",
    ]

    seg_distinct = summary.get("segment_distinct") or {}
    if seg_distinct:
        lines.append("### Segment Diversity")
        lines.append("")
        lines.append("| Segment | Distinct Values |")
        lines.append("|---------|-----------------|")
        for seg, vals in seg_distinct.items():
            shown = ", ".join(vals[:8]) + (
                f" *(+{len(vals) - 8} more)*" if len(vals) > 8 else ""
            )
            lines.append(f"| {seg} | {shown or '—'} |")
        lines.append("")

    units = summary.get("unit_distinct") or []
    if units:
        lines.append(f"### Units ({len(units)})")
        lines.append("")
        lines.append(", ".join(f"`{u}`" for u in units[:20]))
        if len(units) > 20:
            lines.append(f" *(+{len(units) - 20} more)*")
        lines.append("")

    cocos = summary.get("cocos_distinct") or []
    if cocos:
        lines.append("### COCOS Transformations")
        lines.append("")
        lines.append(", ".join(f"`{c}`" for c in cocos))
        lines.append("")

    domains = summary.get("physics_domain_distinct") or []
    if domains:
        lines.append("### Physics Domains")
        lines.append("")
        lines.append(", ".join(f"`{d}`" for d in domains))
        lines.append("")

    samples = summary.get("sample_names") or []
    if samples:
        lines.append("### Sample Members")
        lines.append("")
        for nm in samples:
            lines.append(f"- `{nm}`")
        lines.append("")

    lineage = summary.get("lineage") or {}
    if lineage:
        lines.append("### Lineage Counts")
        lines.append("")
        lines.append("| Relation | Members | Max Chain Depth |")
        lines.append("|----------|--------:|-----------------:|")
        lines.append(
            f"| Predecessors | {lineage.get('predecessors_count', 0)} | "
            f"{lineage.get('predecessors_max_depth', 0)} |"
        )
        lines.append(
            f"| Successors | {lineage.get('successors_count', 0)} | "
            f"{lineage.get('successors_max_depth', 0)} |"
        )
        lines.append(
            f"| Refined-from | {lineage.get('refined_from_count', 0)} | "
            f"{lineage.get('refined_from_max_depth', 0)} |"
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# _edit_standard_name
# ---------------------------------------------------------------------------

#: agent-facing --scope value -> internal EditScope enum value.
_EDIT_SCOPE_MAP = {
    "self": EditScope.only_self.value,
    "family": EditScope.family.value,
    "subtree": EditScope.subtree.value,
}


def _edit_standard_name(
    standard_name: str,
    reason: str,
    *,
    hint: str | None = None,
    rename: str | None = None,
    docs: str | None = None,
    axis: str | None = None,
    scope: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Attach a steered edit proposal to a StandardName (WRITE tool).

    Mirrors the ``imas-codex sn edit`` CLI — same underlying engine
    (:func:`imas_codex.standard_names.edit.apply_edit`), always called with
    ``origin="agent"`` (MCP callers are agents, never humans). The proposal
    rides the existing generate -> review -> score pipeline; it is **not**
    applied to the catalogue immediately. Exactly one of ``hint`` / ``rename``
    / ``docs`` selects the mode; ``reason`` is mandatory.

    Args:
        standard_name: Target StandardName id.
        reason: Mandatory justification shown to the reviewer as intent
            context (neutralises the reviewer's revert-to-original pull).
            Ground it in physics/DD-path facts, not preference.
        hint: Steering direction (hint mode) — the pipeline still composes
            the candidate. Mutually exclusive with rename/docs.
        rename: Full replacement name (rename mode) — skips generation,
            enters name review directly.
        docs: Full replacement documentation (docs mode) — skips
            generation, enters docs review directly.
        axis: Which slot(s) a hint steers: ``"name" | "docs" | "both"``
            (hint mode only — rename/docs modes imply their own axis).
        scope: Blast radius: ``"self" | "family" | "subtree"``. Mapped to
            the internal ``EditScope`` enum (``self`` -> ``only_self``).
        dry_run: Preview the plan (and cascade, if any) without writing to
            the graph.

    Returns:
        Dict rendering of the resulting ``EditPlan`` (target, mode, axis,
        scope, entry, successor, cascade_planned, blocked, actions,
        applied) plus a ``"summary"`` key with a short human-readable
        recap. A malformed call (wrong argument combination, missing
        reason, invalid axis/scope) returns ``{"error": "..."}`` instead
        of raising.
    """
    scope_value: str | None = None
    if scope is not None:
        if scope not in _EDIT_SCOPE_MAP:
            return {"error": f"scope={scope!r} invalid (self|family|subtree)"}
        scope_value = _EDIT_SCOPE_MAP[scope]

    # Function-local import: a module-level import of standard_names.edit
    # closes an import cycle (edit -> graph_ops -> discovery -> cli.logging
    # -> cli/__init__ register_commands -> cli.sn) and breaks any
    # standard_names-first import of the package.
    from imas_codex.standard_names.edit import apply_edit

    try:
        plan = apply_edit(
            target=standard_name,
            hint=hint,
            rename=rename,
            docs=docs,
            reason=reason,
            axis=axis,
            scope=scope_value,
            origin="agent",
            dry_run=dry_run,
        )
    except ValueError as e:
        return {"error": str(e)}
    except ServiceUnavailable:
        return {"error": NEO4J_NOT_RUNNING_MSG}
    except Exception as e:  # noqa: BLE001 — surface as tool error, not a crash
        return {"error": f"apply_edit failed: {_neo4j_error_message(e)}"}

    result = dataclasses.asdict(plan)
    if plan.blocked:
        summary = f"BLOCKED: {plan.blocked}"
    elif not plan.applied:
        summary = (
            f"[dry-run] {plan.mode} edit on {plan.target!r} would enter "
            f"{plan.entry} ({len(plan.cascade_planned)} cascade step(s))"
        )
    else:
        successor_note = f" -> successor {plan.successor!r}" if plan.successor else ""
        summary = (
            f"{plan.mode} edit attached to {plan.target!r}{successor_note}, "
            f"entering {plan.entry} (edit_status=open)"
        )
    result["summary"] = summary
    return result
