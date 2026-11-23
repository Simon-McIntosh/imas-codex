"""Read the field-code sources from one facility's catalogue and build the table.

The catalogue is every archive signal the facility holds in the graph. The
sources are the signal-list rows already imported and the controls metadata a
signal carries, and this module includes the functional specification as a
declared source that establishes no entry when no line of it names a code.

Every source is checked before an absence is reported: a source that returns no
text at all raises, because an empty result is a claim about the instrument until
the instrument is shown to see something known present.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from iter_codex.discovery.signals.field_codes import (
    FieldCodeConflict,
    FieldCodeEntry,
    FieldCodeRules,
    FieldCodeStatement,
    FieldCodeTable,
    build_field_code_table,
    statements_from_texts,
)
from iter_codex.graph.models import ArchiveFieldCodeState, SignalVocabularySource

CONTROLS_SOURCE_PREFIX = "controlsmetadata:"

CODES_QUERY = """
MATCH (s:FacilitySignal {facility_id: $facility})
WHERE s.field_code IS NOT NULL AND s.field_code <> ''
RETURN s.field_code AS code, count(*) AS signals
ORDER BY code
"""

ROWS_QUERY = """
MATCH (r:PlantSignalRow)
WHERE r.field_code IS NOT NULL AND r.field_code <> ''
RETURN r.field_code AS code, r.component_category AS component_kind,
       r.description AS description
ORDER BY r.field_code
"""

CONTROLS_QUERY = """
MATCH (s:FacilitySignal {facility_id: $facility})
WHERE s.description_source STARTS WITH $prefix AND s.field_code IS NOT NULL
RETURN s.field_code AS code, s.component_category AS component_kind,
       s.description AS description
ORDER BY s.field_code
"""


def meanings_from_config(
    entries: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str | None], str]:
    """Read the declared one-line meanings, keyed by code and component kind."""
    meanings: dict[tuple[str, str | None], str] = {}
    for entry in entries:
        code = str(entry.get("code") or "").strip()
        if not code:
            raise ValueError("a declared meaning has no code")
        kind = entry.get("component_kind") or None
        meaning = str(entry.get("meaning") or "").strip()
        if not meaning:
            raise ValueError(f"declared meaning for {code!r} is empty")
        if (code, kind) in meanings:
            raise ValueError(f"declared meaning repeated for {code!r}, kind {kind!r}")
        meanings[(code, kind)] = meaning
    return meanings


def snapshot_sources(
    graph,
    facility: str,
    *,
    limit: int = 100000,
) -> tuple[list[FieldCodeStatement], dict[str, int], dict[str, int]]:
    """Read every source and turn what it states into field-code statements.

    Returns the statements, the catalogue's signal count per code, and the
    number of texts each source contributed, so a reader can see the instrument
    read something before trusting a code's absence.
    """
    if not 1 <= limit <= 1000000:
        raise ValueError("Use a finite catalogue bound")
    codes = graph.query(CODES_QUERY, facility=facility)
    if not codes:
        raise RuntimeError("The catalogue holds no signal carrying a field code")
    signals_by_code = {row["code"]: row["signals"] for row in codes}

    row_records = graph.query(ROWS_QUERY)
    control_records = graph.query(
        CONTROLS_QUERY, facility=facility, prefix=CONTROLS_SOURCE_PREFIX
    )
    if not row_records:
        raise RuntimeError("No signal-list row is imported")
    if not control_records:
        raise RuntimeError("No signal carries controls metadata")

    rules = FieldCodeRules()
    statements: list[FieldCodeStatement] = []
    texts = Counter()
    for record in row_records:
        texts["signal_list_row"] += 1
        statements.extend(
            statements_from_texts(
                record["code"],
                record["component_kind"],
                SignalVocabularySource.signal_list_row.value,
                [record["description"]],
                rules,
            )
        )
    for record in control_records:
        texts["controls_metadata"] += 1
        statements.extend(
            statements_from_texts(
                record["code"],
                record["component_kind"],
                SignalVocabularySource.controls_metadata.value,
                [record["description"]],
                rules,
            )
        )

    if not statements:
        raise RuntimeError(
            "No source states a role for any code, so the reading is empty"
        )
    texts["functional_specification"] = 0
    return statements, signals_by_code, dict(texts)


def meanings_from_graph(graph, facility: str) -> dict[tuple[str, str | None], str]:
    """Read the declared meanings the graph holds, keyed by code and kind.

    The graph is the store: each ``ArchiveFieldCode`` node carries the reading a
    source established and the row it came from, so a build no longer reads the
    facility's private configuration.
    """
    rows = graph.query(
        """
        MATCH (a:ArchiveFieldCode {facility_id: $facility})
        WHERE a.meaning IS NOT NULL AND a.meaning <> ''
        RETURN a.code AS code, a.component_kind AS component_kind,
               a.meaning AS meaning
        ORDER BY code, component_kind
        """,
        facility=facility,
    )
    return meanings_from_config(rows)


FIELD_CODE_NODE_QUERY = """
UNWIND $codes AS entry
MERGE (a:ArchiveFieldCode {id: entry.id})
SET a += entry.props, a.facility_id = entry.facility_id
WITH a, entry
MATCH (f:Facility {id: entry.facility_id})
MERGE (a)-[:AT_FACILITY]->(f)
RETURN count(a) AS written
"""

FIELD_CODE_SIGNAL_QUERY = """
MATCH (a:ArchiveFieldCode {facility_id: $facility})
MATCH (s:FacilitySignal {facility_id: $facility})
WHERE s.field_code = a.code
MERGE (s)-[:HAS_FIELD_CODE]->(a)
RETURN count(DISTINCT a) AS codes, count(DISTINCT s) AS signals
"""

FIELD_CODE_SOURCE_ROW_QUERY = """
MATCH (a:ArchiveFieldCode {facility_id: $facility})
CALL (a) {
    MATCH (r:PlantSignalRow {facility_id: $facility, field_code: a.code})
    RETURN r ORDER BY r.id LIMIT 1
}
MERGE (a)-[:FIELD_CODE_SOURCE_ROW]->(r)
RETURN count(a) AS linked
"""


def field_code_nodes(
    graph,
    facility: str,
    table: FieldCodeTable,
    *,
    catalogue: set[str] | None = None,
) -> list[dict[str, Any]]:
    """One node per catalogue code, carrying what the table establishes for it.

    A code with a single agreed role is ``established``. A code whose kinds
    decide different roles, or whose sources disagree, is ``conflicting`` with
    every role and source recorded. A catalogue code no source interprets is
    retained ``unestablished`` so a signal still reaches its code.
    """
    entries: dict[str, list[FieldCodeEntry]] = {}
    for entry in table.entries:
        entries.setdefault(entry.code, []).append(entry)
    conflicts: dict[str, list[FieldCodeConflict]] = {}
    for conflict in table.conflicts:
        conflicts.setdefault(conflict.code, []).append(conflict)
    unestablished: dict[str, list[dict[str, Any]]] = {}
    for row in table.unestablished:
        unestablished.setdefault(row["code"], []).append(row)
    nodes: list[dict[str, Any]] = []
    for code in sorted(set(entries) | set(conflicts) | set(unestablished)):
        if not code:
            continue
        if catalogue is not None and code not in catalogue:
            continue
        code_entries = entries.get(code, [])
        code_conflicts = conflicts.get(code, [])
        stated = unestablished.get(code, [])
        props: dict[str, Any] = {"code": code}
        decided_roles = {entry.role for entry in code_entries}
        if code_conflicts or len(decided_roles) > 1:
            roles = set(decided_roles)
            sources = {entry.source for entry in code_entries}
            for conflict in code_conflicts:
                roles.update(conflict.roles)
                sources.update(conflict.sources)
            props["state"] = ArchiveFieldCodeState.conflicting.value
            props["conflict_roles"] = sorted(
                conflict_role for conflict_role in roles if conflict_role
            )
            props["conflict_sources"] = sorted(source for source in sources if source)
        elif code_entries:
            entry = code_entries[0]
            props["state"] = ArchiveFieldCodeState.established.value
            props["role"] = entry.role
            props["meaning"] = entry.meaning
            props["meaning_source"] = entry.source
            props["support"] = entry.support
            kinds = {item.component_kind for item in code_entries}
            if len(kinds) == 1 and None not in kinds:
                props["component_kind"] = next(iter(kinds))
        else:
            props["state"] = ArchiveFieldCodeState.unestablished.value
            roles = {row["role"] for row in stated if row.get("role")}
            if len(roles) == 1:
                props["role"] = next(iter(roles))
            kinds = {
                row["component_kind"] for row in stated if row.get("component_kind")
            }
            if len(kinds) == 1:
                props["component_kind"] = next(iter(kinds))
        nodes.append(
            {
                "id": f"{facility}:archive_field_code:{code}",
                "facility_id": facility,
                "props": props,
            }
        )
    return nodes


FIELD_CODE_STORED_QUERY = """
MATCH (a:ArchiveFieldCode {facility_id: $facility})
RETURN a.id AS id, properties(a) AS props ORDER BY id
"""


def pending_field_codes(graph, facility: str, nodes: list[dict[str, Any]]) -> list[str]:
    """Ids a write would create or change, so a repeated build can report none."""
    stored = {
        row["id"]: row["props"]
        for row in graph.query(FIELD_CODE_STORED_QUERY, facility=facility)
    }
    pending = []
    for node in nodes:
        current = stored.get(node["id"])
        wanted = dict(node["props"], facility_id=node["facility_id"])
        if current is None or any(
            current.get(key) != value for key, value in wanted.items()
        ):
            pending.append(node["id"])
    return pending


def persist_field_code_table(
    graph, facility: str, table: FieldCodeTable
) -> dict[str, Any]:
    """Write the table into the graph and link every signal to its code."""
    catalogue = {row["code"] for row in graph.query(CODES_QUERY, facility=facility)}
    nodes = field_code_nodes(graph, facility, table, catalogue=catalogue)
    written = graph.query(FIELD_CODE_NODE_QUERY, codes=nodes)[0]["written"]
    links = graph.query(FIELD_CODE_SIGNAL_QUERY, facility=facility)[0]
    rows = graph.query(FIELD_CODE_SOURCE_ROW_QUERY, facility=facility)[0]["linked"]
    return {
        "codes": written,
        "codes_linked": links["codes"],
        "signals_linked": links["signals"],
        "source_rows": rows,
    }


def build_from_graph(
    graph,
    facility: str,
    meanings: Mapping[tuple[str, str | None], str],
    *,
    limit: int = 100000,
) -> dict[str, Any]:
    """Build the table and its counts from the catalogue and declared meanings."""
    statements, signals_by_code, texts = snapshot_sources(graph, facility, limit=limit)
    table = build_field_code_table(
        statements, meanings, codes_in_catalogue=signals_by_code
    )
    summary = table.summary(signals_by_code)
    distinct = len(signals_by_code)
    summary.update(
        {
            "distinct_codes": distinct,
            "source_texts_read": texts,
            "role_statements": len(statements),
            "codes_with_role_statement": len({item.code for item in statements}),
            "setpoint": _setpoint_report(table, statements),
        }
    )
    return {"table": table, "summary": summary}


def _setpoint_report(
    table: FieldCodeTable, statements: Sequence[FieldCodeStatement]
) -> dict[str, Any]:
    """Name the codes examined for the setpoint role and why none establishes it."""
    examined = sorted({item.code for item in statements})
    setpoint_codes = sorted(
        {entry.code for entry in table.entries if entry.role == "setpoint"}
    )
    report: dict[str, Any] = {
        "codes_examined": len(examined),
        "codes_decided_setpoint": len(setpoint_codes),
        "codes": setpoint_codes,
    }
    if not setpoint_codes:
        report["reason"] = (
            "no examined code has a source text that names a setpoint, a set value "
            "or a threshold"
        )
    return report
