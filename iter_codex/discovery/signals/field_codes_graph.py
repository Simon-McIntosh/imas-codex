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
    FieldCodeRules,
    FieldCodeStatement,
    FieldCodeTable,
    build_field_code_table,
    statements_from_texts,
)
from iter_codex.graph.models import SignalVocabularySource

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
