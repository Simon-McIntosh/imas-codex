"""Import archive-declared categories and exact pulse windows with graph receipts."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from imas_codex.discovery.signals.scanners import codac_uda as catalogue
from imas_codex.graph.models import ArchiveEndpoint, Pulse, PulseCategory

CATEGORY_QUERY = """
MATCH (f:Facility {id: $facility})
MATCH (a:ArchiveEndpoint {id: $archive, facility_id: $facility, excluded: false})
UNWIND $items AS props
MERGE (c:PulseCategory {id: props.id})
WITH f, a, c, props, any(key IN keys(props) WHERE
    coalesce(c[key] <> props[key], c[key] IS NOT NULL OR props[key] IS NOT NULL)) AS changed
FOREACH (_ IN CASE WHEN changed THEN [1] ELSE [] END | SET c += props)
MERGE (c)-[:AT_FACILITY]->(f)
MERGE (c)-[:IN_ARCHIVE]->(a)
RETURN count(c) AS written, sum(CASE WHEN changed THEN 1 ELSE 0 END) AS nodes_changed
"""
PULSE_QUERY = """
MATCH (f:Facility {id: $facility})
MATCH (a:ArchiveEndpoint {id: $archive, facility_id: $facility, excluded: false})
UNWIND $items AS props
MERGE (p:Pulse {id: props.id})
WITH f, a, p, props, any(key IN keys(props) WHERE
    coalesce(p[key] <> props[key], p[key] IS NOT NULL OR props[key] IS NOT NULL)) AS changed
FOREACH (_ IN CASE WHEN changed THEN [1] ELSE [] END | SET p += props)
MERGE (p)-[:AT_FACILITY]->(f)
MERGE (p)-[:IN_ARCHIVE]->(a)
WITH p, props, changed
UNWIND props.categories AS category_id
MATCH (c:PulseCategory {id: category_id, archive: $archive, facility_id: $facility})
MERGE (p)-[:IN_PULSE_CATEGORY]->(c)
WITH DISTINCT p, changed
RETURN count(p) AS written, sum(CASE WHEN changed THEN 1 ELSE 0 END) AS nodes_changed
"""


class PulseKindRule(BaseModel):
    """A private, explicit evidence match; no vocabulary is inferred from names."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1)
    source: Literal["category_name", "pulse_description", "document"]
    pattern: str = Field(min_length=1)
    reference: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rule(self):
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(
                "kind evidence pattern must be a valid regular expression"
            ) from exc
        if not self.kind.strip() or not self.reference.strip():
            raise ValueError("kind and evidence reference must be nonempty")
        self.kind = self.kind.strip().lower()
        return self


def classify_categories(
    categories: Sequence[PulseCategory],
    pulses: Sequence[Pulse],
    rules: Sequence[PulseKindRule],
) -> list[PulseCategory]:
    """Label only supported, unambiguous kinds, clearing unsupported stale labels.

    Document rules match category identities and cite the document that supports
    that mapping. Pulse-description rules consider only members of the category.
    Conflicting kinds leave the category unlabelled rather than choosing a winner.
    """
    descriptions: dict[str, list[str]] = {}
    for pulse in pulses:
        if pulse.description:
            for category in pulse.categories or []:
                descriptions.setdefault(category, []).append(pulse.description)
    result = []
    for category in categories:
        supported = []
        for rule in rules:
            texts = (
                descriptions.get(category.id, [])
                if rule.source == "pulse_description"
                else [category.name]
            )
            if any(re.search(rule.pattern, text, re.IGNORECASE) for text in texts):
                supported.append(rule)
        kinds = {rule.kind for rule in supported}
        chosen = supported[0] if len(kinds) == 1 else None
        result.append(
            category.model_copy(
                update={
                    "kind": chosen.kind if chosen else None,
                    "kind_source": f"{chosen.source}:{chosen.reference}"
                    if chosen
                    else None,
                }
            )
        )
    return result


def pulse_metadata_counts(
    categories: Sequence[PulseCategory], pulses: Sequence[Pulse]
) -> dict:
    """Count both supported categories and the unlabelled remainder."""
    return {
        "pulses": len(pulses),
        "with_description": sum(bool(p.description) for p in pulses),
        "with_status": sum(bool(p.status) for p in pulses),
        "categories_with_kind_by_source": dict(
            sorted(
                Counter(
                    c.kind_source.split(":", 1)[0]
                    for c in categories
                    if c.kind and c.kind_source
                ).items()
            )
        ),
        "categories_without_kind": sum(not c.kind for c in categories),
        "kinds": dict(sorted(Counter(c.kind for c in categories if c.kind).items())),
    }


def pulse_distribution(
    categories: Sequence[PulseCategory], pulses: Sequence[Pulse]
) -> dict:
    """Summarize membership, retaining the zero-pulse categories explicitly."""
    members = Counter(
        category for pulse in pulses for category in pulse.categories or []
    )
    distribution = Counter(members[category.id] for category in categories)
    ended = sum(pulse.end_ns is not None for pulse in pulses)
    return {
        "categories": len(categories),
        "pulses": len(pulses),
        "pulses_per_category": dict(sorted(distribution.items())),
        "empty_categories": distribution[0],
        "with_end": ended,
        "without_end": len(pulses) - ended,
    }


def import_pulses(
    graph,
    archive: ArchiveEndpoint,
    categories: Sequence[PulseCategory],
    pulses: Sequence[Pulse],
    *,
    batch_size: int = 100,
) -> dict:
    """Upsert a validated snapshot and acknowledge each bounded transaction.

    Counts of created nodes come from transaction counters, not a count of
    processed rows. Null end times are included in SET so an open-ended source
    record cannot retain a stale graph end time. Missing parents roll back.
    """
    if batch_size < 1 or batch_size > 1000:
        raise ValueError("pulse import batch size must be between 1 and 1000")
    if archive.excluded:
        raise ValueError("excluded archive cannot own an imported pulse catalogue")
    category_ids = {category.id for category in categories}
    if len(category_ids) != len(categories) or len({p.id for p in pulses}) != len(
        pulses
    ):
        raise ValueError("pulse catalogue identities must be unique")
    for item in [*categories, *pulses]:
        if (
            item.archive != archive.id
            or item.facility_id != archive.facility_id
            or item.id != f"{archive.id}:{item.name}"
        ):
            raise ValueError("pulse catalogue identity disagrees with its archive")
    for pulse in pulses:
        if not pulse.categories or not set(pulse.categories) <= category_ids:
            raise ValueError("pulse memberships must reference listed categories")
        if pulse.end_ns is not None and pulse.end_ns < pulse.start_ns:
            raise ValueError("pulse end precedes its start")
    created = catalogue.persist_archive_endpoints(graph, archive.facility_id, [archive])
    changed = 0

    def write(tx, statement, expected, **parameters):
        result = tx.run(statement, **parameters)
        row = result.single()
        summary = result.consume()
        if row is None or row["written"] != expected:
            raise RuntimeError(
                "pulse import not acknowledged; check facility and category parents"
            )
        return summary.counters.nodes_created, row["nodes_changed"]

    with graph.session() as session:
        for statement, items in [(CATEGORY_QUERY, categories), (PULSE_QUERY, pulses)]:
            for offset in range(0, len(items), batch_size):
                batch = items[offset : offset + batch_size]
                properties = [item.model_dump(mode="json") for item in batch]
                for props in properties:
                    props["description"] = props.get("description") or None
                    if statement == PULSE_QUERY:
                        props["status"] = props.get("status") or None
                new, updated = session.execute_write(
                    write,
                    statement,
                    len(batch),
                    facility=archive.facility_id,
                    archive=archive.id,
                    items=properties,
                )
                created += new
                changed += updated
    return {
        **pulse_distribution(categories, pulses),
        "nodes_created": created,
        "nodes_changed": changed,
    }
