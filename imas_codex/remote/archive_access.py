"""Paged graph metadata for shell users preparing bounded archive signal reads.

These lookups never contact an archive or fetch samples. Sample arrays come
from ``ArchiveSignalLoader`` in ``imas_codex.remote.codac_uda``.
"""

from __future__ import annotations

from typing import Any

CATEGORY_MATCH = """
MATCH (c:PulseCategory {facility_id: $facility})
WHERE toLower(c.name) CONTAINS toLower($text)
   OR toLower(coalesce(c.description, '')) CONTAINS toLower($text)
"""
PULSE_MATCH = """
MATCH (p:Pulse {facility_id: $facility})-[:IN_PULSE_CATEGORY]->
      (c:PulseCategory {id: $category_id, facility_id: $facility})
"""
SIGNAL_MATCH = """
MATCH (s:FacilitySignal {facility_id: $facility})-[:DATA_ACCESS]->(a:DataAccess)
WHERE a.method_type = 'codac_uda' AND (s.name = $name OR s.accessor = $name)
"""


def _page(
    graph,
    match: str,
    symbol: str,
    projection: str,
    *,
    limit: int,
    offset: int,
    **parameters,
) -> dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("page limit must be between 1 and 100")
    if type(offset) is not int or offset < 0:
        raise ValueError("page offset must be a nonnegative integer")
    (count,) = graph.query(
        match + f"RETURN count(DISTINCT {symbol}) AS total", **parameters
    )
    total = count["total"]
    items = graph.query(
        match
        + f"RETURN DISTINCT {projection} ORDER BY name, id SKIP $offset LIMIT $limit",
        **parameters,
        limit=limit,
        offset=offset,
    )
    return {
        "items": items,
        "total": total,
        "next_offset": offset + limit if offset + limit < total else None,
    }


def find_categories(
    graph, text: str, *, facility: str = "iter", limit: int = 100, offset: int = 0
) -> dict[str, Any]:
    """Match a literal case-insensitive substring of category name or description.

    An unmatched description returns ``items=[]``, ``total=0`` and no next page.
    Results describe the imported catalogue, not a fresh archive listing.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("category search requires a nonempty description")
    return _page(
        graph,
        CATEGORY_MATCH,
        "c",
        "c.id AS id, c.name AS name, c.archive AS archive, "
        "c.description AS description",
        facility=facility,
        text=text,
        limit=limit,
        offset=offset,
    )


def list_pulses(
    graph,
    category_id: str,
    *,
    facility: str = "iter",
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """List exact category members with epoch-nanosecond bounds and descriptions.

    A null ``end_ns`` remains open-ended. Category and pulse IDs are opaque
    graph strings; retain the archive and every category-qualified path segment.
    """
    if not isinstance(category_id, str) or not category_id.strip():
        raise ValueError("pulse listing requires a category identity")
    return _page(
        graph,
        PULSE_MATCH,
        "p",
        "p.id AS id, p.name AS name, p.archive AS archive, "
        "p.start_ns AS start_ns, p.end_ns AS end_ns, p.description AS description",
        facility=facility,
        category_id=category_id,
        limit=limit,
        offset=offset,
    )


def find_signals(
    graph, name: str, *, facility: str = "iter", limit: int = 100, offset: int = 0
) -> dict[str, Any]:
    """Resolve an exact graph name or archive selector to loadable signal IDs."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("signal lookup requires an exact nonempty name")
    return _page(
        graph,
        SIGNAL_MATCH,
        "s",
        "s.id AS id, s.name AS name, s.accessor AS selector, s.unit AS unit, "
        "s.description AS description",
        facility=facility,
        name=name,
        limit=limit,
        offset=offset,
    )
