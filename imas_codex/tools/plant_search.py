"""Bounded, typed catalogue reads for plant signals, test categories and pulses."""

from __future__ import annotations

import json
from typing import Any

from imas_codex.graph.client import GraphClient

PAGE_MAX_BYTES = 256_000
PAGE_MAX_ITEMS = 100
_DESCRIPTION_CHARS = 2048

_SIGNAL_FIELDS = """s { .id, .name, .accessor, .unit, .control_path,
    .component_category, .component_sequence, .field_code, .field_discriminator,
    .name_parse_outcome, .candidate_plant_component,
    description: substring(s.description, 0, $description_chars),
    description_truncated: coalesce(size(s.description), 0) > $description_chars }"""


def _pagination(limit: int, offset: int) -> None:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= PAGE_MAX_ITEMS
    ):
        raise ValueError(f"limit must be an integer between 1 and {PAGE_MAX_ITEMS}")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{name} must contain between 1 and 512 characters")
    return value.strip()


def _page_json(
    items: list[dict],
    total: int,
    offset: int,
    limit: int,
    *,
    tool: str,
    arguments: dict,
    **context: Any,
) -> str:
    """Keep identities intact and advance only past rows actually returned."""
    items = list(items)
    while True:
        next_offset = (
            offset + len(items) if items and offset + len(items) < total else None
        )
        result = {
            **context,
            "items": items,
            "total": total,
            "count": len(items),
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset,
            "continuation": {
                "tool": tool,
                "arguments": {**arguments, "offset": next_offset},
            }
            if next_offset is not None
            else None,
        }
        encoded = json.dumps(result, ensure_ascii=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) <= PAGE_MAX_BYTES:
            return encoded
        if not items:
            raise ValueError("resource metadata exceeds the response byte bound")
        items.pop()
        if not items:
            raise ValueError(
                "one catalogue item exceeds the response byte bound; identity was not truncated"
            )


def _read_page(
    graph: GraphClient,
    match: str,
    predicate: str,
    projection: str,
    order: str,
    parameters: dict,
) -> dict:
    """Count all matches while collecting only the requested, stably ordered page."""
    rows = graph.query(
        f"""{match} WHERE {predicate}
        WITH count(*) AS total
        CALL {{
            {match} WHERE {predicate}
            WITH * ORDER BY {order} SKIP $offset LIMIT $limit
            RETURN collect({projection}) AS items
        }}
        RETURN total, items""",
        **parameters,
        description_chars=_DESCRIPTION_CHARS,
    )
    return rows[0]


def list_plant_signals(
    graph: GraphClient,
    facility: str,
    *,
    plant_system: str | None = None,
    function: str | None = None,
    component_category: str | None = None,
    field: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> str:
    """List syntactic plant codes without inferring PBS identities or code meanings.

    A plant system is a control-path prefix with a hyphen boundary. A function
    is an exact FunctionID (the complete control path). Category and field are
    exact parsed codes. All supplied filters are combined with AND.
    """
    _pagination(limit, offset)
    facility = _text(facility, "facility")
    filters = {
        name: _text(value, name).upper()
        for name, value in {
            "plant_system": plant_system,
            "function": function,
            "component_category": component_category,
            "field": field,
        }.items()
        if value is not None
    }
    if not filters:
        raise ValueError(
            "a plant_system, function, component_category or field filter is required"
        )
    predicates = ["s.control_path IS NOT NULL"]
    if "plant_system" in filters:
        predicates.append(
            "(s.control_path = $plant_system OR s.control_path STARTS WITH $plant_system + '-')"
        )
    for name, prop in (
        ("function", "control_path"),
        ("component_category", "component_category"),
        ("field", "field_code"),
    ):
        if name in filters:
            predicates.append(f"s.{prop} = ${name}")
    page = _read_page(
        graph,
        "MATCH (s:FacilitySignal {facility_id: $facility})",
        " AND ".join(predicates),
        _SIGNAL_FIELDS,
        "s.id",
        {"facility": facility, **filters, "limit": limit, "offset": offset},
    )
    return _page_json(
        **page,
        offset=offset,
        limit=limit,
        tool="search_signals",
        arguments={"facility": facility, **filters, "k": limit},
        kind="signals",
        facility=facility,
        filters=filters,
    )


def search_pulse_catalogue(
    graph: GraphClient,
    facility: str,
    *,
    designation: str | None = None,
    category_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> str:
    """Search category names/descriptions, or list one category's exact pulse windows."""
    _pagination(limit, offset)
    facility = _text(facility, "facility")
    if (designation is None) == (category_id is None):
        raise ValueError("supply exactly one of designation or category_id")
    arguments: dict[str, Any] = {"facility": facility, "limit": limit}
    if category_id is None:
        designation = _text(designation, "designation")
        arguments["designation"] = designation
        page = _read_page(
            graph,
            "MATCH (c:PulseCategory {facility_id: $facility})",
            "(toLower(c.name) CONTAINS $designation OR toLower(coalesce(c.description, '')) CONTAINS $designation)",
            """c { .id, .name, .archive,
                description: substring(c.description, 0, $description_chars),
                description_truncated: coalesce(size(c.description), 0) > $description_chars,
                pulse_count: COUNT { MATCH (p:Pulse)-[:IN_PULSE_CATEGORY]->(c) WHERE p.facility_id = $facility } }""",
            "c.id",
            {
                "facility": facility,
                "designation": designation.lower(),
                "limit": limit,
                "offset": offset,
            },
        )
        context = {"kind": "pulse_categories", "designation": designation}
    else:
        category_id = _text(category_id, "category_id")
        arguments["category_id"] = category_id
        page = _read_page(
            graph,
            "MATCH (p:Pulse {facility_id: $facility})",
            "EXISTS { MATCH (p)-[:IN_PULSE_CATEGORY]->(:PulseCategory {id: $category_id, facility_id: $facility}) }",
            """p { .id, .name, .archive, .start_ns, .end_ns,
                open_ended: p.end_ns IS NULL,
                description: substring(p.description, 0, $description_chars),
                description_truncated: coalesce(size(p.description), 0) > $description_chars }""",
            "p.start_ns, p.id",
            {
                "facility": facility,
                "category_id": category_id,
                "limit": limit,
                "offset": offset,
            },
        )
        context = {
            "kind": "pulses",
            "category_id": category_id,
            "time_base": "unix_epoch_nanoseconds",
        }
    return _page_json(
        **page,
        offset=offset,
        limit=limit,
        tool="search_pulses",
        arguments=arguments,
        facility=facility,
        **context,
    )


def fetch_signal_context(
    graph: GraphClient,
    resource: str,
    *,
    limit: int = 20,
    offset: int = 0,
) -> str | None:
    """Resolve an exact signal id and page its candidate component's siblings."""
    _pagination(limit, offset)
    resource = _text(resource, "resource")
    rows = graph.query(
        f"""MATCH (s:FacilitySignal {{id: $resource}})
        OPTIONAL MATCH (s)-[:CANDIDATE_PLANT_COMPONENT]->(c:PlantComponent {{facility_id: s.facility_id}})
        OPTIONAL MATCH (s)-[:AT_CONTROL_NODE]->(control:ControlNode {{facility_id: s.facility_id}})
        OPTIONAL MATCH (control)-[:PARENT_CONTROL_NODE*0..4]->(ancestor:ControlNode {{facility_id: s.facility_id}})
        WITH s, c, control, ancestor ORDER BY ancestor.level
        RETURN {_SIGNAL_FIELDS} AS signal, s.facility_id AS facility,
            c {{ .id, .code, .name, .component_category, .component_sequence,
                 description: substring(c.description, 0, $description_chars) }} AS component,
            collect(ancestor {{ .id, .code, .name, .level,
                 description: substring(ancestor.description, 0, $description_chars) }}) AS control_hierarchy
        LIMIT 1""",
        resource=resource,
        description_chars=_DESCRIPTION_CHARS,
    )
    if not rows:
        return None
    context = rows[0]
    component = context["component"]
    page = _read_page(
        graph,
        "MATCH (s:FacilitySignal {facility_id: $facility})",
        "s.id <> $resource AND EXISTS { MATCH (s)-[:CANDIDATE_PLANT_COMPONENT]->(:PlantComponent {id: $component_id, facility_id: $facility}) }",
        _SIGNAL_FIELDS,
        "s.id",
        {
            "facility": context["facility"],
            "resource": resource,
            "component_id": component["id"] if component else None,
            "limit": limit,
            "offset": offset,
        },
    )
    hierarchy = context["control_hierarchy"]
    return _page_json(
        **page,
        offset=offset,
        limit=limit,
        tool="fetch_content",
        arguments={"resource": resource, "limit": limit},
        kind="signal_context",
        **context,
        component_identity_status="candidate" if component else None,
        system=next((node for node in hierarchy if node["level"] == 1), None),
        items_kind="sibling_signals",
    )
