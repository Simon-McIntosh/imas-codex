"""Read a signal cohort, assign its closed vocabulary, and store it idempotently.

The cohort is the pilot system's catalogue as the plant-context import assessed
it: every ITER signal carrying a ``plant_context_state``. Sources come from what
the graph already holds — the linked signal-list rows, the controls metadata the
signal carries, and the archive's own metadata document — so this module
contacts no archive and reads no document.

Every stored property belongs to the declared schema, every node reached is
scoped to the ITER facility, and a write is acknowledged by a count and then
verified by a read-back of every stored value.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from imas_codex.discovery.signals.vocabulary import (
    AXES,
    SOURCE_ORDER,
    UNIT_SPELLINGS,
    SignalSources,
    VocabularyRules,
    assign_vocabulary,
    summarise,
    vocabulary_properties,
)
from imas_codex.units import normalize_unit_symbol, units_are_equivalent

CONTROLS_SOURCE_PREFIX = "controlsmetadata:"

VOCABULARY_KEYS = tuple(
    f"vocabulary_{axis}{suffix}"
    for axis in AXES
    for suffix in ("", "_source", "_state")
)

COHORT_QUERY = """
MATCH (s:FacilitySignal {facility_id: $facility})
WHERE s.plant_context_state IS NOT NULL
RETURN s.id AS id, s.field_code AS field_code, s.unit AS unit,
       s.description AS description, s.description_source AS description_source,
       s.archive_metadata AS archive_metadata,
       s.vocabulary_quantity AS vocabulary_quantity,
       s.vocabulary_quantity_source AS vocabulary_quantity_source,
       s.vocabulary_quantity_state AS vocabulary_quantity_state,
       s.vocabulary_unit AS vocabulary_unit,
       s.vocabulary_unit_source AS vocabulary_unit_source,
       s.vocabulary_unit_state AS vocabulary_unit_state,
       s.vocabulary_role AS vocabulary_role,
       s.vocabulary_role_source AS vocabulary_role_source,
       s.vocabulary_role_state AS vocabulary_role_state
ORDER BY s.id
LIMIT $limit
"""

COHORT_ROWS_QUERY = """
MATCH (s:FacilitySignal {facility_id: $facility})-[:HAS_PLANT_CONTEXT]
      ->(:SignalContextLink)-[:CONTEXT_SOURCE_ROW]->(row:PlantSignalRow)
RETURN DISTINCT s.id AS id, row.description AS description
ORDER BY id, description
"""

CONTROL_QUERY = """
MATCH (s:FacilitySignal {facility_id: $facility})
RETURN count(s) AS total, count(s.plant_context_state) AS assessed
"""

WRITE_QUERY = """
MATCH (f:Facility {id: $facility})
UNWIND $rows AS row
MATCH (s:FacilitySignal {id: row.id, facility_id: $facility})
SET s.vocabulary_quantity = row.vocabulary_quantity,
    s.vocabulary_quantity_source = row.vocabulary_quantity_source,
    s.vocabulary_quantity_state = row.vocabulary_quantity_state,
    s.vocabulary_unit = row.vocabulary_unit,
    s.vocabulary_unit_source = row.vocabulary_unit_source,
    s.vocabulary_unit_state = row.vocabulary_unit_state,
    s.vocabulary_role = row.vocabulary_role,
    s.vocabulary_role_source = row.vocabulary_role_source,
    s.vocabulary_role_state = row.vocabulary_role_state
MERGE (s)-[:AT_FACILITY]->(f)
RETURN count(s) AS written
"""


def archive_unit(document: str | None) -> str | None:
    """The unit the archive itself reports, read from its stored metadata document."""
    if not document:
        return None
    try:
        decoded = json.loads(document)
    except (TypeError, ValueError):
        return None
    for field in decoded.get("full") or []:
        if isinstance(field, Mapping) and field.get("name") == "Units":
            value = field.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def cohort_sources(
    signals: Sequence[Mapping[str, Any]], rows_by_signal: Mapping[str, Sequence[str]]
) -> list[SignalSources]:
    """Turn the cohort snapshot into one source record per signal."""
    sources = []
    for signal in signals:
        controls = str(signal.get("description_source") or "").startswith(
            CONTROLS_SOURCE_PREFIX
        )
        sources.append(
            SignalSources(
                signal_id=signal["id"],
                field_code=signal.get("field_code"),
                row_texts=tuple(rows_by_signal.get(signal["id"], ())),
                controls_text=signal.get("description") if controls else None,
                controls_unit=signal.get("unit") if controls else None,
                archive_unit=archive_unit(signal.get("archive_metadata")),
            )
        )
    return sources


def snapshot_cohort(
    graph, facility: str, *, limit: int = 20000
) -> tuple[list[dict], list[dict]]:
    """Read the assessed cohort with a positive control that the selector sees data."""
    if not 1 <= limit <= 100000:
        raise ValueError("Use a finite cohort bound")
    control = graph.query(CONTROL_QUERY, facility=facility)[0]
    if not control["total"] or not control["assessed"]:
        raise RuntimeError("Plant-context cohort mark selects no signal")
    signals = graph.query(COHORT_QUERY, facility=facility, limit=limit + 1)
    if not signals or len(signals) > limit:
        raise RuntimeError("Cohort is empty or exceeds its declared bound")
    if graph.query(COHORT_QUERY, facility=facility, limit=limit + 1) != signals:
        raise RuntimeError("Cohort changed while taking its snapshot")
    rows = graph.query(COHORT_ROWS_QUERY, facility=facility)
    return signals, rows


def _rows_by_signal(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        description = row.get("description")
        if description:
            grouped.setdefault(row["id"], []).append(str(description))
    return grouped


def _changed(stored: Mapping[str, Any], desired: Mapping[str, Any]) -> bool:
    return any(
        (stored.get(key) or None) != (desired.get(key) or None)
        for key in VOCABULARY_KEYS
    )


def assign_cohort(
    signals: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    rules: VocabularyRules | None = None,
) -> dict[str, Any]:
    """Assign every signal and describe what the write would change."""
    sources = cohort_sources(signals, _rows_by_signal(rows))
    ledger = assign_vocabulary(sources, rules)
    stored = {signal["id"]: signal for signal in signals}
    pending = [
        {"id": record["id"], **vocabulary_properties(record)}
        for record in ledger
        if _changed({**stored[record["id"]]}, vocabulary_properties(record))
    ]
    return {
        "ledger": ledger,
        "pending": pending,
        "summary": summarise(ledger),
        "signals": len(ledger),
        "changed": len(pending),
    }


def apply_assignment(
    graph, facility: str, pending: Sequence[Mapping[str, Any]]
) -> dict:
    """Write one complete desired state per signal; null removes a stale member."""
    if not pending:
        return {"written": 0, "batches": 0}
    if len({row["id"] for row in pending}) != len(pending):
        raise ValueError("A signal may appear once in one assignment")
    batches = 0
    written = 0
    batch_size = 200
    for offset in range(0, len(pending), batch_size):
        batch = [dict(row) for row in pending[offset : offset + batch_size]]
        for row in batch:
            row.setdefault("vocabulary_quantity", None)
            row.setdefault("vocabulary_quantity_source", None)
            row.setdefault("vocabulary_quantity_state", None)
            row.setdefault("vocabulary_unit", None)
            row.setdefault("vocabulary_unit_source", None)
            row.setdefault("vocabulary_unit_state", None)
            row.setdefault("vocabulary_role", None)
            row.setdefault("vocabulary_role_source", None)
            row.setdefault("vocabulary_role_state", None)
        result = graph.query(WRITE_QUERY, facility=facility, rows=batch)
        if len(result) != 1 or result[0]["written"] != len(batch):
            raise RuntimeError("Vocabulary write was not fully acknowledged")
        written += result[0]["written"]
        batches += 1
    return {"written": written, "batches": batches}


def verify_assignment(
    graph, facility: str, ledger: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Read every stored value back and compare it with the assignment."""
    signals, _ = snapshot_cohort(graph, facility)
    stored = {signal["id"]: signal for signal in signals}
    mismatched = []
    for record in ledger:
        desired = vocabulary_properties(record)
        current = stored.get(record["id"])
        if current is None:
            mismatched.append({"id": record["id"], "reason": "signal left the cohort"})
            continue
        for key in VOCABULARY_KEYS:
            if (current.get(key) or None) != (desired.get(key) or None):
                mismatched.append(
                    {
                        "id": record["id"],
                        "key": key,
                        "stored": current.get(key),
                        "assigned": desired.get(key),
                    }
                )
    assigned = sum(
        sum(1 for axis in AXES if record[f"vocabulary_{axis}_state"] == "decided")
        for record in ledger
    )
    return {
        "signals": len(stored),
        "assigned_values": assigned,
        "mismatched": len(mismatched),
        "mismatches": mismatched[:20],
    }


def unit_agreement(
    signals: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check each decided unit against every source spelling that gave it.

    The comparison is the shared unit authority's, not string equality, so a
    source that wrote the same unit in another spelling is counted as agreeing.
    """
    by_id = {signal["id"]: signal for signal in signals}
    compared = sources_compared = 0
    disagreements = []
    for record in records:
        member = record["vocabulary_unit"]
        if record["vocabulary_unit_state"] != "decided" or member is None:
            continue
        symbol = UNIT_SPELLINGS[member][0]
        signal = by_id[record["id"]]
        raw_units = [
            raw
            for source, _ in record["unit_candidates"]
            if source != SOURCE_ORDER[3]
            for raw in (
                [_source_unit(signal, source)] if _source_unit(signal, source) else []
            )
        ]
        compared += 1
        for raw in raw_units:
            sources_compared += 1
            if not units_are_equivalent(normalize_unit_symbol(raw), symbol):
                disagreements.append(
                    {"id": record["id"], "source_unit": raw, "stored": symbol}
                )
    return {
        "signals_compared": compared,
        "source_units_compared": sources_compared,
        "disagreements": len(disagreements),
        "examples": disagreements[:10],
    }


def _source_unit(signal: Mapping[str, Any], source: str) -> str | None:
    if source == SOURCE_ORDER[1]:
        return (
            signal.get("unit")
            if str(signal.get("description_source") or "").startswith(
                CONTROLS_SOURCE_PREFIX
            )
            else None
        )
    if source == SOURCE_ORDER[2]:
        return archive_unit(signal.get("archive_metadata"))
    return None
