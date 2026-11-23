"""Move the ITER archive's plant-signal structure onto the reviewed ontology.

The reviewed structure renames the plant breakdown class, merges the plant
context document class into the shared document class, retires two duplicate
memberships, adds the field-code class and binds first-level control nodes to
the plant nodes a source already states. Every operation is confined to one
facility's own nodes and relationships and is idempotent, so a repeated run
changes nothing.

Invoke with ``uv run python -m iter_codex.graph.plant_ontology_migration``. A run
without ``--apply`` reports what it would change and writes nothing. A run that
names a label or relationship outside the approved sets is refused before
anything is read or written.

``--capture DIR`` writes a private before-image and nothing else; ``--apply``
with ``--before-image DIR`` writes the image and then migrates; ``--restore
PATH`` undoes a run from an image. The image is private and belongs under the
private cache, because it carries node properties that must not enter the
repository. ``--apply --repeat`` reruns every step over a graph already in the
migrated form and writes no image; it exists to show a repeated run changes
nothing, and it refuses a graph that is not yet migrated.
"""

# Cypher property maps and node patterns are full of braces, so a statement that
# interpolates a label or relationship type uses percent formatting rather than
# a format specifier.
# ruff: noqa: UP031

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iter_codex.discovery.base.facility import get_facility_infrastructure
from iter_codex.discovery.signals.field_codes_graph import (
    CODES_QUERY,
    build_from_graph,
    field_code_nodes,
    meanings_from_config,
    pending_field_codes,
    persist_field_code_table,
)
from iter_codex.graph.client import GraphClient

MEANINGS_KEY = "field_code_meanings"

PLANT_STRUCTURE = "PlantStructure"
PLANT_BREAKDOWN = "PlantBreakdownNode"
PLANT_DOCUMENT = "PlantContextDocument"
DOCUMENT = "Document"
CONTROL_NODE = "ControlNode"
SIGNAL = "FacilitySignal"
FIELD_CODE = "ArchiveFieldCode"

PARENT_PLANT_OLD = "PARENT_PLANT_STRUCTURE"
PARENT_PLANT_NEW = "PARENT_PLANT_NODE"
CONTEXT_FOR_PLANT = "CONTEXT_FOR_PLANT"
DOCUMENT_PLANT_STRUCTURE = "DOCUMENT_PLANT_STRUCTURE"
CANDIDATE_PLANT_COMPONENT = "CANDIDATE_PLANT_COMPONENT"
IN_ARCHIVE_CONTROL_NODE = "IN_ARCHIVE_CONTROL_NODE"
ROW_PLANT_STRUCTURE = "ROW_PLANT_STRUCTURE"
HAS_FIELD_CODE = "HAS_FIELD_CODE"
FIELD_CODE_SOURCE_ROW = "FIELD_CODE_SOURCE_ROW"
CONTROL_PLANT_LINK = "CONTROL_NODE_PLANT_STRUCTURE"

APPROVED_LABELS = frozenset(
    (
        "Facility",
        PLANT_STRUCTURE,
        PLANT_BREAKDOWN,
        PLANT_DOCUMENT,
        DOCUMENT,
        CONTROL_NODE,
        SIGNAL,
        "PlantComponent",
        "PlantSignalRow",
        FIELD_CODE,
    )
)

APPROVED_RELATIONSHIPS = frozenset(
    (
        "AT_FACILITY",
        "AT_CONTROL_NODE",
        "PARENT_CONTROL_NODE",
        "IN_ARCHIVE_COMPONENT",
        PARENT_PLANT_OLD,
        PARENT_PLANT_NEW,
        CONTEXT_FOR_PLANT,
        DOCUMENT_PLANT_STRUCTURE,
        CANDIDATE_PLANT_COMPONENT,
        IN_ARCHIVE_CONTROL_NODE,
        CONTROL_PLANT_LINK,
        ROW_PLANT_STRUCTURE,
        HAS_FIELD_CODE,
        FIELD_CODE_SOURCE_ROW,
    )
)


class UnapprovedNameError(ValueError):
    """A label or relationship outside the reviewed decision was named."""


def assert_approved(names: list[str] | tuple[str, ...]) -> None:
    """Refuse a run that names a label or relationship outside the decision."""
    unknown = [
        name
        for name in names
        if name not in APPROVED_LABELS and name not in APPROVED_RELATIONSHIPS
    ]
    if unknown:
        raise UnapprovedNameError(
            f"{sorted(unknown)!r} is outside the classes and relationships this "
            "migration may touch"
        )


# --- counting -------------------------------------------------------------


def _count(graph: GraphClient, label: str, facility: str) -> int:
    query = f"MATCH (n:`{label}` {{facility_id: $facility}}) RETURN count(n) AS c"
    return graph.query(query, facility=facility)[0]["c"]


def _rel_count(graph: GraphClient, rel: str, facility: str) -> int:
    query = (
        f"MATCH (a)-[r:`{rel}`]->(b) "
        "WHERE a.facility_id = $facility AND b.facility_id = $facility "
        "RETURN count(r) AS c"
    )
    return graph.query(query, facility=facility)[0]["c"]


def _foreign(graph: GraphClient, label: str, facility: str) -> int:
    """Count the label's nodes outside this facility, which must be zero."""
    query = (
        f"MATCH (n:`{label}`) WHERE coalesce(n.facility_id, '') <> $facility "
        "RETURN count(n) AS c"
    )
    return graph.query(query, facility=facility)[0]["c"]


def census(graph: GraphClient, facility: str) -> dict[str, Any]:
    """Count every class and relationship this migration can change."""
    labels = [
        PLANT_STRUCTURE,
        PLANT_BREAKDOWN,
        PLANT_DOCUMENT,
        DOCUMENT,
        FIELD_CODE,
    ]
    relationships = [
        PARENT_PLANT_OLD,
        PARENT_PLANT_NEW,
        CONTEXT_FOR_PLANT,
        DOCUMENT_PLANT_STRUCTURE,
        CANDIDATE_PLANT_COMPONENT,
        IN_ARCHIVE_CONTROL_NODE,
        CONTROL_PLANT_LINK,
        HAS_FIELD_CODE,
        FIELD_CODE_SOURCE_ROW,
    ]
    return {
        "labels": {label: _count(graph, label, facility) for label in labels},
        "relationships": {
            rel: _rel_count(graph, rel, facility) for rel in relationships
        },
    }


def _migrated(graph: GraphClient, facility: str) -> bool:
    """Whether any part of the migrated form is already present."""
    return bool(
        _count(graph, PLANT_BREAKDOWN, facility)
        or _count(graph, FIELD_CODE, facility)
        or _rel_count(graph, PARENT_PLANT_NEW, facility)
    )


# --- before-image ---------------------------------------------------------


def capture(graph: GraphClient, facility: str, directory: Path) -> Path:
    """Write every node, property and relationship this run can change."""
    for label in (PLANT_STRUCTURE, PLANT_DOCUMENT, FIELD_CODE):
        foreign = _foreign(graph, label, facility)
        if foreign:
            raise RuntimeError(f"{foreign} {label} nodes belong to another facility")

    occupied = {
        PLANT_BREAKDOWN: _count(graph, PLANT_BREAKDOWN, facility),
        FIELD_CODE: _count(graph, FIELD_CODE, facility),
        PARENT_PLANT_NEW: _rel_count(graph, PARENT_PLANT_NEW, facility),
    }
    if any(occupied.values()):
        raise RuntimeError(
            "the graph already holds the migrated form (%s); a before-image "
            "must be taken against the reviewed base" % occupied
        )
    plant_nodes = graph.query(
        f"MATCH (n:`{PLANT_STRUCTURE}`) "
        "RETURN n.id AS id, properties(n) AS props ORDER BY id"
    )
    documents = graph.query(
        f"MATCH (n:`{PLANT_DOCUMENT}`) "
        "RETURN n.id AS id, properties(n) AS props ORDER BY id"
    )
    plant_parents = graph.query(
        f"MATCH (a)-[r:`{PARENT_PLANT_OLD}`]->(b) "
        "WHERE a.facility_id = $facility AND b.facility_id = $facility "
        "RETURN a.id AS a, b.id AS b ORDER BY a, b",
        facility=facility,
    )
    context_links = graph.query(
        f"MATCH (a:`{PLANT_DOCUMENT}`)-[:`{CONTEXT_FOR_PLANT}`]->(b) "
        "WHERE a.facility_id = $facility AND b.facility_id = $facility "
        "RETURN a.id AS a, b.id AS b ORDER BY a, b",
        facility=facility,
    )
    candidate_links = graph.query(
        f"MATCH (a)-[:`{CANDIDATE_PLANT_COMPONENT}`]->(b) "
        "WHERE a.facility_id = $facility AND b.facility_id = $facility "
        "RETURN a.id AS a, b.id AS b ORDER BY a, b",
        facility=facility,
    )
    control_links = graph.query(
        f"MATCH (a)-[:`{IN_ARCHIVE_CONTROL_NODE}`]->(b) "
        "WHERE a.facility_id = $facility AND b.facility_id = $facility "
        "RETURN a.id AS a, b.id AS b ORDER BY a, b",
        facility=facility,
    )
    signal_props = graph.query(
        f"MATCH (s:`{SIGNAL}` {{facility_id: $facility}}) "
        "WHERE s.candidate_plant_component IS NOT NULL "
        "   OR s.archive_control_node IS NOT NULL "
        "RETURN s.id AS id, s.candidate_plant_component AS candidate, "
        "s.archive_control_node AS control ORDER BY id",
        facility=facility,
    )
    image = {
        "facility": facility,
        "taken_at": datetime.now(UTC).isoformat(),
        "plant_nodes": plant_nodes,
        "documents": documents,
        "plant_parents": plant_parents,
        "context_links": context_links,
        "candidate_links": candidate_links,
        "control_links": control_links,
        "signal_props": signal_props,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "before.json"
    path.write_text(json.dumps(image, indent=1, sort_keys=True))
    (directory / "RESTORE.txt").write_text(
        "Restore the graph from this image with:\n\n"
        "    uv run python -m iter_codex.graph.plant_ontology_migration \\\n"
        f"        --facility {facility} --restore {path}\n\n"
        f"Image taken {image['taken_at']} from facility {facility}.\n"
    )
    return path


def _link(graph: GraphClient, rel: str, rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        count += graph.query(
            "MATCH (a {id: $a}) MATCH (b {id: $b}) MERGE (a)-[r:`"
            + rel
            + "`]->(b) RETURN count(r) AS c",
            a=row["a"],
            b=row["b"],
        )[0]["c"]
    return count


def _restore_signal_props(graph: GraphClient, rows: list[dict[str, Any]]) -> int:
    return graph.query(
        "UNWIND $rows AS row MATCH (s:FacilitySignal {id: row.id}) "
        "SET s.candidate_plant_component = row.candidate, "
        "    s.archive_control_node = row.control "
        "RETURN count(s) AS c",
        rows=rows,
    )[0]["c"]


def restore(graph: GraphClient, facility: str, image_path: Path) -> dict[str, Any]:
    """Undo a migration from the image it captured before writing."""
    image = json.loads(image_path.read_text())
    if image["facility"] != facility:
        raise RuntimeError(
            "the image was taken for " + image["facility"] + ", not " + facility
        )
    removed: dict[str, int] = {}
    removed[FIELD_CODE] = graph.query(
        "MATCH (n:`" + FIELD_CODE + "` {facility_id: $facility}) "
        "DETACH DELETE n RETURN count(n) AS c",
        facility=facility,
    )[0]["c"]
    for rel in (
        HAS_FIELD_CODE,
        FIELD_CODE_SOURCE_ROW,
        CONTROL_PLANT_LINK,
        DOCUMENT_PLANT_STRUCTURE,
        PARENT_PLANT_NEW,
    ):
        removed[rel] = graph.query(
            "MATCH (a)-[r:`" + rel + "`]->(b) WHERE a.facility_id = $facility "
            "AND b.facility_id = $facility DELETE r RETURN count(r) AS c",
            facility=facility,
        )[0]["c"]
    for node in image["plant_nodes"]:
        graph.query(
            "MATCH (n {id: $id}) REMOVE n:`"
            + PLANT_BREAKDOWN
            + "` SET n:`"
            + PLANT_STRUCTURE
            + "` SET n = $props",
            id=node["id"],
            props=node["props"],
        )
    for node in image["documents"]:
        graph.query(
            "MATCH (n {id: $id}) REMOVE n:`"
            + DOCUMENT
            + "` SET n:`"
            + PLANT_DOCUMENT
            + "` SET n = $props",
            id=node["id"],
            props=node["props"],
        )
    restored: dict[str, int] = {}
    restored[PARENT_PLANT_OLD] = _link(graph, PARENT_PLANT_OLD, image["plant_parents"])
    restored[CONTEXT_FOR_PLANT] = _link(
        graph, CONTEXT_FOR_PLANT, image["context_links"]
    )
    restored[CANDIDATE_PLANT_COMPONENT] = _link(
        graph, CANDIDATE_PLANT_COMPONENT, image["candidate_links"]
    )
    restored[IN_ARCHIVE_CONTROL_NODE] = _link(
        graph, IN_ARCHIVE_CONTROL_NODE, image["control_links"]
    )
    signals = 0
    batch: list[dict[str, Any]] = []
    for row in image["signal_props"]:
        batch.append(row)
        if len(batch) >= 1000:
            signals += _restore_signal_props(graph, batch)
            batch = []
    if batch:
        signals += _restore_signal_props(graph, batch)
    restored["signal_properties"] = signals
    return {"restored": restored, "removed_created": removed}


# --- operations -----------------------------------------------------------


def relabel_plant_structure(
    graph: GraphClient, facility: str, *, apply: bool
) -> dict[str, Any]:
    """Rename the plant breakdown class and its parent relationship."""
    foreign = _foreign(graph, PLANT_STRUCTURE, facility)
    if foreign:
        raise RuntimeError(
            "%s %s nodes belong to another facility" % (foreign, PLANT_STRUCTURE)
        )
    if not apply:
        return {
            "would_relabel": _count(graph, PLANT_STRUCTURE, facility),
            "would_rename_parent_edges": _rel_count(graph, PARENT_PLANT_OLD, facility),
        }
    relabelled = graph.query(
        "MATCH (n:`%s` {facility_id: $facility}) SET n:`%s` REMOVE n:`%s` "
        "RETURN count(n) AS c" % (PLANT_STRUCTURE, PLANT_BREAKDOWN, PLANT_STRUCTURE),
        facility=facility,
    )[0]["c"]
    renamed = graph.query(
        "MATCH (a)-[r:`%s`]->(b) "
        "WHERE a.facility_id = $facility AND b.facility_id = $facility "
        "MERGE (a)-[:`%s`]->(b) DELETE r RETURN count(r) AS c"
        % (PARENT_PLANT_OLD, PARENT_PLANT_NEW),
        facility=facility,
    )[0]["c"]
    return {"relabelled": relabelled, "parent_edges_renamed": renamed}


def merge_plant_documents(
    graph: GraphClient, facility: str, *, apply: bool
) -> dict[str, Any]:
    """Merge the plant context document class into the shared document class."""
    foreign = _foreign(graph, PLANT_DOCUMENT, facility)
    if not apply:
        return {
            "would_migrate": _count(graph, PLANT_DOCUMENT, facility),
            "would_rename_links": _rel_count(graph, CONTEXT_FOR_PLANT, facility),
            "left_foreign": foreign,
        }
    written = 0
    for document in graph.query(
        "MATCH (d:`%s` {facility_id: $facility}) "
        "RETURN d.id AS id, properties(d) AS props ORDER BY id" % PLANT_DOCUMENT,
        facility=facility,
    ):
        props = dict(document["props"])
        for old, new in (
            ("role", "content_role"),
            ("system_code", "context_system_code"),
            ("system_binding_note", "source_binding_note"),
            ("description", "title"),
        ):
            if old in props:
                props.setdefault(new, props[old])
                props.pop(old)
        props.setdefault("version_state", "current")
        props.setdefault("status", "discovered")
        graph.query(
            "MATCH (d:`%s` {id: $id}) REMOVE d:`%s` SET d:`%s` SET d = $props"
            % (PLANT_DOCUMENT, PLANT_DOCUMENT, DOCUMENT),
            id=document["id"],
            props=props,
        )
        written += 1
    links = graph.query(
        "MATCH (a:`%s`)-[r:`%s`]->(b) "
        "WHERE a.facility_id = $facility AND b.facility_id = $facility "
        "MERGE (a)-[:`%s`]->(b) DELETE r RETURN count(r) AS c"
        % (DOCUMENT, CONTEXT_FOR_PLANT, DOCUMENT_PLANT_STRUCTURE),
        facility=facility,
    )[0]["c"]
    return {
        "documents_migrated": written,
        "plant_links_renamed": links,
        "documents_left_foreign": foreign,
    }


def retire_duplicate_memberships(
    graph: GraphClient, facility: str, *, apply: bool
) -> dict[str, Any]:
    """Retire the parse-time component link and the archive control link."""
    if not apply:
        return {
            "would_remove_component_links": _rel_count(
                graph, CANDIDATE_PLANT_COMPONENT, facility
            ),
            "would_remove_control_links": _rel_count(
                graph, IN_ARCHIVE_CONTROL_NODE, facility
            ),
            "would_remove_signal_properties": _stale_signal_count(graph, facility),
        }
    removed: dict[str, Any] = {}
    for rel in (CANDIDATE_PLANT_COMPONENT, IN_ARCHIVE_CONTROL_NODE):
        removed[rel] = graph.query(
            "MATCH (a)-[r:`%s`]->(b) WHERE a.facility_id = $facility "
            "AND b.facility_id = $facility DELETE r RETURN count(r) AS c" % rel,
            facility=facility,
        )[0]["c"]
    removed["signal_properties"] = graph.query(
        "MATCH (s:`%s` {facility_id: $facility}) "
        "WHERE s.candidate_plant_component IS NOT NULL "
        "   OR s.archive_control_node IS NOT NULL "
        "REMOVE s.candidate_plant_component, s.archive_control_node "
        "RETURN count(s) AS c" % SIGNAL,
        facility=facility,
    )[0]["c"]
    return removed


def _stale_signal_count(graph: GraphClient, facility: str) -> int:
    return graph.query(
        "MATCH (s:`%s` {facility_id: $facility}) "
        "WHERE s.candidate_plant_component IS NOT NULL "
        "   OR s.archive_control_node IS NOT NULL "
        "RETURN count(s) AS c" % SIGNAL,
        facility=facility,
    )[0]["c"]


def field_code_table(graph: GraphClient, facility: str) -> dict[str, Any]:
    """Build the field-code table from the graph's sources and declarations."""
    private = get_facility_infrastructure(facility) or {}
    declarations = private.get(MEANINGS_KEY) or []
    meanings = meanings_from_config(declarations)
    built = build_from_graph(graph, facility, meanings)
    return {
        "table": built["table"],
        "summary": built["summary"],
        "declarations": declarations,
    }


def populate_field_codes(graph: GraphClient, facility: str, *, apply: bool) -> dict:
    """Write one node per field code and link every signal to its code."""
    built = field_code_table(graph, facility)
    catalogue = {row["code"] for row in graph.query(CODES_QUERY, facility=facility)}
    nodes = field_code_nodes(graph, facility, built["table"], catalogue=catalogue)
    pending = pending_field_codes(graph, facility, nodes)
    if not apply:
        return {
            "code_nodes": len(nodes),
            "would_write_codes": len(pending),
            "summary": built["summary"],
            "declarations": built["declarations"],
        }
    wrote = persist_field_code_table(graph, facility, built["table"])
    wrote["summary"] = built["summary"]
    wrote["declarations"] = built["declarations"]
    wrote["pending_before_write"] = len(pending)
    return wrote


def bind_control_nodes(
    graph: GraphClient, facility: str, *, apply: bool
) -> dict[str, Any]:
    """Bind first-level control nodes to the plant nodes a source states."""
    plant_label = PLANT_BREAKDOWN
    if not _count(graph, PLANT_BREAKDOWN, facility):
        plant_label = PLANT_STRUCTURE
    rows = graph.query(
        "MATCH (r:PlantSignalRow {facility_id: $facility}) "
        "      -[:`%s`]->(p:`%s`) "
        "WHERE r.control_path IS NOT NULL AND r.control_path <> '' "
        "RETURN r.control_path AS control_path, p.id AS plant, "
        "r.id AS row_id, r.source_document AS document "
        "ORDER BY control_path, plant, row_id" % (ROW_PLANT_STRUCTURE, plant_label),
        facility=facility,
    )
    plants_of: dict[str, set[str]] = defaultdict(set)
    rows_of: dict[str, set[str]] = defaultdict(set)
    docs_of: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        root = row["control_path"].split("-")[0]
        plants_of[root].add(row["plant"])
        rows_of[root].add(row["row_id"])
        if row["document"]:
            docs_of[root].add(row["document"])
    bound: list[dict[str, Any]] = []
    for root in sorted(plants_of):
        control_id = facility + ":control_node:" + root
        existing = graph.query(
            "MATCH (c:`%s` {id: $id, facility_id: $facility}) "
            "RETURN c.level AS level" % CONTROL_NODE,
            id=control_id,
            facility=facility,
        )
        if len(existing) != 1 or existing[0]["level"] != 1:
            raise RuntimeError(
                root + " is stated by a pilot row but is not one first-level node"
            )
        for plant in sorted(plants_of[root]):
            if apply:
                graph.query(
                    "MATCH (c:`%s` {id: $control, facility_id: $facility}) "
                    "MATCH (p:`%s` {id: $plant, facility_id: $facility}) "
                    "MERGE (c)-[r:`%s`]->(p) "
                    "SET r.stated_by_rows = $rows, r.stated_by_documents = $docs"
                    % (CONTROL_NODE, PLANT_BREAKDOWN, CONTROL_PLANT_LINK),
                    control=control_id,
                    plant=plant,
                    facility=facility,
                    rows=sorted(rows_of[root]),
                    docs=sorted(docs_of[root]),
                )
        bound.append(
            {
                "control_node": control_id,
                "plant_nodes": sorted(plants_of[root]),
                "source_rows": len(rows_of[root]),
                "source_documents": sorted(docs_of[root]),
            }
        )
    stated = set(plants_of)
    unbound = [
        row["id"]
        for row in graph.query(
            "MATCH (c:`%s` {facility_id: $facility}) WHERE c.level = 1 "
            "RETURN c.id AS id ORDER BY id" % CONTROL_NODE,
            facility=facility,
        )
        if row["id"].rsplit(":", 1)[1] not in stated
    ]
    existing = {
        (row["control"], row["plant"])
        for row in graph.query(
            "MATCH (c:`%s` {facility_id: $facility})-[:`%s`]->(p:`%s` "
            "{facility_id: $facility}) "
            "RETURN c.id AS control, p.id AS plant"
            % (CONTROL_NODE, CONTROL_PLANT_LINK, plant_label),
            facility=facility,
        )
    }
    would_bind = sum(
        1
        for item in bound
        for plant in item["plant_nodes"]
        if (item["control_node"], plant) not in existing
    )
    return {
        "bound_roots": bound,
        "unbound_level_one_roots": unbound,
        "would_bind": would_bind,
        "applied": apply,
    }


# --- fingerprint ----------------------------------------------------------


def fingerprint(graph: GraphClient, facility: str) -> str:
    """Digest the state this migration owns, for a no-op comparison."""
    parts: list[Any] = [census(graph, facility)]
    parts.append(
        graph.query(
            "MATCH (n:`%s` {facility_id: $facility}) "
            "OPTIONAL MATCH (n)-[:`%s`]->(p) "
            "RETURN n.id AS id, properties(n) AS props, p.id AS parent ORDER BY id"
            % (PLANT_BREAKDOWN, PARENT_PLANT_NEW),
            facility=facility,
        )
    )
    parts.append(
        graph.query(
            "MATCH (n:`%s` {facility_id: $facility}) "
            "WHERE n.content_role IS NOT NULL "
            "OPTIONAL MATCH (n)-[:`%s`]->(p) "
            "RETURN n.id AS id, properties(n) AS props, "
            "collect(p.id) AS plants ORDER BY id"
            % (DOCUMENT, DOCUMENT_PLANT_STRUCTURE),
            facility=facility,
        )
    )
    parts.append(
        graph.query(
            "MATCH (n:`%s` {facility_id: $facility}) "
            "OPTIONAL MATCH (n)-[:`%s`]->(r) "
            "RETURN n.id AS id, properties(n) AS props, r.id AS source_row ORDER BY id"
            % (FIELD_CODE, FIELD_CODE_SOURCE_ROW),
            facility=facility,
        )
    )
    parts.append(
        graph.query(
            "MATCH (c:`%s` {facility_id: $facility})-[:`%s`]->(p) "
            "RETURN c.id AS id, p.id AS plant ORDER BY id, plant"
            % (CONTROL_NODE, CONTROL_PLANT_LINK),
            facility=facility,
        )
    )
    parts.append(_stale_signal_count(graph, facility))
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# --- orchestration --------------------------------------------------------


def migrate(
    graph: GraphClient,
    facility: str = "iter",
    *,
    apply: bool = False,
    capture_directory: Path | None = None,
    repeat: bool = False,
) -> dict[str, Any]:
    """Apply every decided change, reporting what each step changed.

    A repeated run over a graph already in the migrated form writes no image,
    because there is nothing left to restore; it must be asked for explicitly so
    a first migration is never mistaken for one.
    """
    assert_approved((PLANT_STRUCTURE, PLANT_DOCUMENT, FIELD_CODE))
    before = census(graph, facility)
    before_digest = fingerprint(graph, facility)
    image: str | None = None
    if apply and repeat:
        if not _migrated(graph, facility):
            raise RuntimeError(
                "the graph is not in the migrated form; run --apply without --repeat"
            )
    elif apply:
        directory = capture_directory
        if directory is None:
            directory = (
                Path.home()
                / ".cache/imas-codex-codac-uda/ontology-migration"
                / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            )
        image = str(capture(graph, facility, directory))
    steps = {
        "relabel": relabel_plant_structure(graph, facility, apply=apply),
        "documents": merge_plant_documents(graph, facility, apply=apply),
        "duplicates": retire_duplicate_memberships(graph, facility, apply=apply),
        "field_codes": populate_field_codes(graph, facility, apply=apply),
        "control_bindings": bind_control_nodes(graph, facility, apply=apply),
    }
    after = census(graph, facility)
    return {
        "facility": facility,
        "applied": apply,
        "taken_at": datetime.now(UTC).isoformat(),
        "before_image": image,
        "before": before,
        "after": after,
        "before_digest": before_digest,
        "after_digest": fingerprint(graph, facility) if apply else before_digest,
        "steps": steps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facility", default="iter")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--repeat", action="store_true", help="rerun over the migrated form, no image"
    )
    parser.add_argument("--label", action="append", default=[])
    for flag in ("--before-image", "--capture", "--restore", "--report"):
        parser.add_argument(flag, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.repeat and not arguments.apply:
        parser.error("--repeat is meaningful only with --apply")
    assert_approved(arguments.label)
    with GraphClient() as graph:
        if arguments.capture:
            path = capture(graph, arguments.facility, arguments.capture)
            print(json.dumps({"captured": str(path)}))
            return 0
        if arguments.restore:
            result = restore(graph, arguments.facility, arguments.restore)
            print(json.dumps(result, default=str))
            return 0
        report = migrate(
            graph,
            arguments.facility,
            apply=arguments.apply,
            capture_directory=arguments.before_image,
            repeat=arguments.repeat,
        )
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, indent=1, sort_keys=True, default=str)
        )
    print(
        json.dumps(
            {
                "facility": report["facility"],
                "applied": report["applied"],
                "repeat": bool(arguments.repeat),
                "before_digest": report["before_digest"],
                "after_digest": report["after_digest"],
                "before_image": report["before_image"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
