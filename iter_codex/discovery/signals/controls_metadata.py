"""Attach a controls feed snapshot to exact members of a private archive census."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from iter_codex.discovery.signals.scanners.codac_uda import (
    CodacUDAScanner,
    _access,
    _persist_signals,
)
from iter_codex.discovery.signals.variable_names import parse_variable_name
from iter_codex.embeddings.encoder import Encoder
from iter_codex.graph.client import GraphClient
from iter_codex.graph.models import FacilitySignal, FacilitySignalStatus
from iter_codex.graph.signal_descriptions import description_is_name

FEED_SOURCE = "controlsmetadata:/variable_hmi"


class ControlsEntry(BaseModel):
    """Source values remain verbatim, including the source's unzoned timestamp."""

    model_config = ConfigDict(extra="ignore", strict=True)
    name: str
    description: str
    unit: str
    data_type: str
    update_time: str

    @field_validator("name", "update_time")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source identity and update time must be nonempty")
        return value

    @field_validator("update_time")
    @classmethod
    def timestamp(cls, value: str) -> str:
        datetime.fromisoformat(value)
        return value


def read_controls_feed(path: Path) -> tuple[list[ControlsEntry], str]:
    """Validate the whole snapshot before any graph mutation."""
    body = path.read_bytes()
    rows = json.loads(body)
    if not isinstance(rows, list) or not rows:
        raise ValueError("controls feed must be a nonempty entry list")
    entries = [ControlsEntry.model_validate(row) for row in rows]
    if len({row.name for row in entries}) != len(entries):
        raise ValueError("controls feed contains duplicate variable identities")
    return entries, hashlib.sha256(body).hexdigest()


def catalogue_members(
    path: Path, names: list[str], *, positive_control: str
) -> dict[str, list[tuple[str, dict | None]]]:
    """Read only the named cohort through the private census's stored format.

    A sealed census and an independently known member distinguish a genuinely
    absent feed entry from a wrong or empty catalogue. No archive is contacted.
    """
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        if not db.execute("SELECT 1 FROM receipts WHERE key='census'").fetchone():
            raise ValueError("controls import requires a sealed archive census")
        if not db.execute(
            "SELECT 1 FROM census WHERE name=? LIMIT 1", (positive_control,)
        ).fetchone():
            raise ValueError("known archive name positive control was not found")
        result = {}
        for name in names:
            rows = db.execute(
                "SELECT role,document FROM census WHERE name=? ORDER BY role", (name,)
            ).fetchall()
            if rows:
                result[name] = [
                    (role, json.loads(document) if document else None)
                    for role, document in rows
                ]
        return result


EXISTING_QUERY = """
UNWIND $ids AS id
MATCH (s:FacilitySignal {id: id})
RETURN s.id AS id
"""

UPDATE_QUERY = """
MATCH (f:Facility {id: $facility})
UNWIND $rows AS row
MATCH (s:FacilitySignal {id: row.id})
WHERE s.facility_id = $facility
WITH f, s, row,
     any(key IN keys(row.props) WHERE s[key] IS NULL OR s[key] <> row.props[key])
       OR (row.embedding IS NULL AND s.embedding IS NOT NULL)
       OR (row.embedding IS NOT NULL AND s.embedding IS NULL)
       OR s.embedded_text_hash IS NULL OR s.embedded_text_hash <> row.text_hash AS changed
MERGE (s)-[:AT_FACILITY]->(f)
RETURN s.id AS id, changed
"""


def _new_signal(
    facility: str, entry: ControlsEntry, memberships: list[tuple[str, dict | None]]
) -> FacilitySignal:
    # Prefer a successful metadata document while retaining every listing role.
    role, document = next(
        (
            (role, doc)
            for role, doc in memberships
            if doc and doc.get("archive_metadata")
        ),
        memberships[0],
    )
    archive = f"{facility}:codac_uda:{role}"
    access = _access(facility, role).id
    if document:
        document = {**document, "name": entry.name}
        document["listed_archives"] = [
            f"{facility}:codac_uda:{member_role}" for member_role, _ in memberships
        ]
        return CodacUDAScanner._signal(
            facility, archive, access, document, "controls_metadata"
        )
    return FacilitySignal(
        id=f"{facility}:general/codac_uda/{entry.name}",
        facility_id=facility,
        name=entry.name,
        accessor=entry.name,
        status=FacilitySignalStatus.discovered,
        data_access=access,
        data_source_name="codac_uda",
        data_source_path=entry.name,
        metadata_archive=archive,
        listed_archives=[f"{facility}:codac_uda:{role}" for role, _ in memberships],
        discovery_source="codac_uda",
        archive_cohorts=["controls_metadata"],
        **parse_variable_name(entry.name).properties(facility),
    )


def import_controls_metadata(
    response: Path,
    catalogue: Path,
    *,
    positive_control: str,
    facility: str = "iter",
    graph: GraphClient,
    encoder: Encoder,
    max_members: int = 2898,
    batch_size: int = 100,
) -> dict:
    """Grow only the feed-covered pilot and update source values idempotently.

    Archive metadata and data-serving confirmation remain owned by the scanner.
    Only newly admitted signals go through that writer; existing enrichment
    lifecycle and claims are preserved. The returned per-entry receipt is private.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    entries, digest = read_controls_feed(response)
    members = catalogue_members(
        catalogue, [entry.name for entry in entries], positive_control=positive_control
    )
    if len(members) > max_members:
        raise ValueError("feed-covered pilot exceeds its declared member limit")
    if graph.query(
        "MATCH (f:Facility {id: $facility}) RETURN count(f) AS total", facility=facility
    ) != [{"total": 1}]:
        raise ValueError("target facility must exist exactly once")
    included = [entry for entry in entries if entry.name in members]
    ids = [f"{facility}:general/codac_uda/{entry.name}" for entry in included]
    existing = {row["id"] for row in graph.query(EXISTING_QUERY, ids=ids)}
    new = [
        _new_signal(facility, entry, members[entry.name])
        for entry, signal_id in zip(included, ids, strict=True)
        if signal_id not in existing
    ]
    # Validate graph bindings before the first write, so a missing/excluded
    # archive cannot create a partially admitted pilot.
    bindings = {signal.metadata_archive for signal in new} | {
        archive for signal in new for archive in signal.listed_archives or []
    }
    if bindings:
        rows = graph.query(
            "UNWIND $ids AS id MATCH (a:ArchiveEndpoint {id:id})"
            "-[:AT_FACILITY]->(:Facility {id:$facility}) "
            "WHERE a.facility_id=$facility AND a.excluded=false "
            "MATCH (da:DataAccess {id:$facility + ':codac_uda:access:' + a.role}) "
            "RETURN DISTINCT a.id AS id",
            ids=sorted(bindings),
            facility=facility,
        )
        if {row["id"] for row in rows} != bindings:
            raise ValueError("readable archive and data-access bindings are required")
    attached = changed = added = 0
    for offset in range(0, len(included), batch_size):
        batch = included[offset : offset + batch_size]
        rows = []
        embed_rows = []
        for entry in batch:
            searchable = bool(entry.description.strip()) and not description_is_name(
                entry.description, entry.name, entry.name
            )
            row = {
                "id": f"{facility}:general/codac_uda/{entry.name}",
                "props": {
                    "description": entry.description,
                    "unit": entry.unit,
                    "data_type": entry.data_type,
                    "description_source": FEED_SOURCE,
                    "description_source_updated_at": entry.update_time,
                },
                "embedding": None,
                "text_hash": hashlib.sha256(b"").hexdigest(),
            }
            rows.append(row)
            if searchable:
                embed_rows.append(row)
        if embed_rows:
            vectors = encoder.embed_texts(
                [row["props"]["description"] for row in embed_rows]
            )
            if len(vectors) != len(embed_rows):
                raise RuntimeError(
                    "embedding receipt does not cover the requested batch"
                )
            for row, vector in zip(embed_rows, vectors, strict=True):
                row["embedding"] = (
                    vector.tolist() if hasattr(vector, "tolist") else list(vector)
                )
                if not row["embedding"]:
                    raise RuntimeError("description embedding is empty")
                row["text_hash"] = hashlib.sha256(
                    row["props"]["description"].encode()
                ).hexdigest()
        pending = [signal for signal in new if signal.id in {row["id"] for row in rows}]
        if pending:
            count = _persist_signals(facility, "controls_metadata", pending)
            if count != len(pending):
                raise RuntimeError("new feed-covered signal batch was not acknowledged")
            added += count
        # A signal carrying a generated description keeps that text as its
        # embed text, so its stored hash is the hash of the generated text. A
        # source update must not read as a description change against it, or
        # every import re-reports the row as changed and re-embeds it forever.
        generated_state = {
            record["id"]: record["embedded_text_hash"] or ""
            for record in graph.query(
                "UNWIND $ids AS id MATCH (s:FacilitySignal {id:id}) "
                "WHERE s.generated_description IS NOT NULL "
                "RETURN s.id AS id, s.embedded_text_hash AS embedded_text_hash",
                ids=[row["id"] for row in rows],
            )
        }
        for row in rows:
            if row["id"] in generated_state:
                row["text_hash"] = generated_state[row["id"]]
        receipt = graph.query(UPDATE_QUERY, facility=facility, rows=rows)
        if len(receipt) != len(rows):
            raise RuntimeError("controls metadata update was not acknowledged")
        changed_ids = {item["id"] for item in receipt if item["changed"]}
        updates = [row for row in rows if row["id"] in changed_ids]
        if updates:
            from iter_codex.discovery.signals.parallel import write_signal_descriptions

            row_by_id = {row["id"]: row for row in updates}
            generated_ids = set(generated_state)
            # A signal's embed text is its generated description where it has
            # one, so a source update must not re-vector that row: replacing
            # the vector would leave it describing text other than the indexed
            # generated description. Those rows take the no-vector arm, which
            # rewrites the source text and leaves the stored vector and hash
            # untouched. A feed-only row with no vector and no text to vectorise
            # records the empty text's hash, so it stops reading as changed.
            count = write_signal_descriptions(
                [{"id": row["id"], **row["props"]} for row in updates],
                gc=graph,
                text_property="description",
                embed_text=lambda item, row_by_id=row_by_id, generated_ids=generated_ids: (
                    item["description"]
                    if item["id"] in generated_ids
                    or row_by_id[item["id"]]["embedding"] is not None
                    else ""
                ),
                precomputed_embeddings={
                    row["id"]: (
                        None if row["id"] in generated_ids else row["embedding"]
                    )
                    for row in updates
                },
            )
            if count != len(updates):
                raise RuntimeError("controls description write was not acknowledged")
        attached += len(receipt)
        changed += len(updates)
    excluded = [entry.name for entry in entries if entry.name not in members]
    return {
        "feed_sha256": digest,
        "entries_read": len(entries),
        "attached": attached,
        "signals_added": added,
        "nodes_changed": changed,
        "left_out": {"absent_from_catalogue": len(excluded)},
        "excluded_names": excluded,
        "with_unit": sum(bool(entry.unit.strip()) for entry in included),
        "name_only_descriptions": sum(
            description_is_name(entry.description, entry.name, entry.name)
            for entry in included
        ),
        "empty_descriptions": sum(not entry.description.strip() for entry in included),
        "source_update_times": dict(Counter(entry.update_time for entry in included)),
    }
