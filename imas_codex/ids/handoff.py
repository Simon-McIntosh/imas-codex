"""Export mapping bindings as facility signal hand-off documents."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.ids.models import CocosLabel
from imas_codex.ids.tools import search_existing_mappings

FORMAT = "imas-codex-mapping-handoff"
DOCUMENT_KEYS = frozenset(
    {"format", "format_version", "facility", "dd_version", "exported_at", "ids"}
)
IDS_KEYS = frozenset({"ids_name", "mapping_id", "status", "signals", "unexpanded"})
SIGNAL_KEYS = frozenset(
    {
        "signal_id",
        "source_id",
        "data_source",
        "source_group",
        "source_array",
        "member_identifier",
        "source_property",
        "target_path",
        "transform_expression",
        "source_units",
        "target_units",
        "cocos_label",
        "cocos_label_source",
        "confidence",
        "evidence",
    }
)
UNEXPANDED_KEYS = frozenset({"source_id", "target_path", "reason"})


def _check_keys(value: Any, expected: frozenset[str], location: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{location} must contain exactly {sorted(expected)}")


def check_handoff_document(document: Any) -> None:
    """Reject documents that differ from the shared hand-off contract."""
    _check_keys(document, DOCUMENT_KEYS, "document")
    if document["format"] != FORMAT or document["format_version"] != 1:
        raise ValueError("unsupported mapping hand-off format")
    if not isinstance(document["facility"], str) or not document["facility"]:
        raise ValueError("facility must be a nonempty string")
    if not isinstance(document["dd_version"], str) or not document["dd_version"]:
        raise ValueError("dd_version must be a nonempty string")
    timestamp = document["exported_at"]
    if not isinstance(timestamp, str):
        raise ValueError("exported_at must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("exported_at must be an ISO 8601 timestamp") from exc
    if parsed.utcoffset() != UTC.utcoffset(None):
        raise ValueError("exported_at must be in UTC")
    if not isinstance(document["ids"], list):
        raise ValueError("ids must be a list")

    for index, entry in enumerate(document["ids"]):
        location = f"ids[{index}]"
        _check_keys(entry, IDS_KEYS, location)
        for name in ("ids_name", "mapping_id", "status"):
            if not isinstance(entry[name], str) or not entry[name]:
                raise ValueError(f"{location}.{name} must be a nonempty string")
        for name, keys in (("signals", SIGNAL_KEYS), ("unexpanded", UNEXPANDED_KEYS)):
            if not isinstance(entry[name], list):
                raise ValueError(f"{location}.{name} must be a list")
            for row_index, row in enumerate(entry[name]):
                row_location = f"{location}.{name}[{row_index}]"
                _check_keys(row, keys, row_location)
                required = (
                    ("signal_id", "source_id", "target_path")
                    if name == "signals"
                    else ("source_id", "target_path", "reason")
                )
                for key in required:
                    if not isinstance(row[key], str) or not row[key]:
                        raise ValueError(
                            f"{row_location}.{key} must be a nonempty string"
                        )


_TRAILING_NUMBER = re.compile(r"(\d+)$")


def _member_patterns(facility: str) -> dict[str, re.Pattern[str]]:
    """Compile the facility's per-source-group member patterns.

    Which part of an array name is the member is facility knowledge, so each
    source group may declare a regular expression whose ``member`` named group
    captures it. A group with no declared pattern is absent from the result and
    keeps the default rule in :func:`_member_identifier`.
    """
    from imas_codex.discovery.base.facility import get_facility

    config = get_facility(facility)
    declared = config.get("signal_member_patterns") or {}
    patterns: dict[str, re.Pattern[str]] = {}
    for group, entry in declared.items():
        expression = entry.get("pattern") if isinstance(entry, dict) else entry
        if expression:
            patterns[group] = re.compile(expression)
    return patterns


def _member_identifier(
    source_array: str,
    arrays: list[str],
    member_pattern: re.Pattern[str] | None = None,
) -> str | None:
    """The segment that tells this member apart within its source.

    A source group that declares a member pattern (facility configuration) has
    its pattern matched against the array name; the ``member`` named group wins,
    and an array the pattern does not match keeps a null identifier. Otherwise a
    grouped source keeps the numeric runs that vary across its members, and a
    singleton source's instance number is the array name's trailing number, so
    an array without one has no identifier.
    """
    if member_pattern is not None:
        match = member_pattern.search(source_array)
        if not match:
            return None
        return match.group("member") or None

    from imas_codex.discovery.signals.parallel import extract_member_identifier

    if len(set(arrays)) > 1:
        identifier = extract_member_identifier(source_array, arrays)
        if identifier and identifier != source_array:
            return identifier
    match = _TRAILING_NUMBER.search(source_array)
    return match.group(1) if match else None


def _source_parts(path: str | None) -> tuple[str, str] | None:
    if not path or "/" not in path:
        return None
    group, array = path.split("/", 1)
    if not group or not array:
        return None
    return group, array


def _nearest_cocos_labels(
    gc: GraphClient, target_ids: Sequence[str]
) -> dict[str, tuple[str, str]]:
    """Map each target path to the nearest labelled node on its ancestor chain.

    A DD transformation class is stored on the structure that carries it, so a
    data target reads the class from the target itself (hop 0) or the closest
    ancestor along ``HAS_PARENT``. A time coordinate is never transformed, so a
    target whose final segment is ``time`` carries the absence token whatever its
    ancestors hold. Any other target with no labelled node on that chain maps to
    the explicit absence token for both the label and its source.
    """
    unique = list(dict.fromkeys(target_ids))
    if not unique:
        return {}
    time_targets = {name for name in unique if name.rsplit("/", 1)[-1] == "time"}
    queried = [name for name in unique if name not in time_targets]
    labels: dict[str, tuple[str, str]] = {}
    if queried:
        rows = gc.query(
            """
            MATCH (t:IMASNode)
            WHERE t.id IN $target_ids
            MATCH path = (t)-[:HAS_PARENT*0..]->(a:IMASNode)
            WHERE a.cocos_transformation_type IS NOT NULL
            WITH t.id AS target_id, a.cocos_transformation_type AS label,
                 a.cocos_label_source AS label_source, length(path) AS hops
            ORDER BY target_id, hops ASC
            WITH target_id, collect(label)[0] AS cocos_label,
                 collect(label_source)[0] AS cocos_label_source
            RETURN target_id, cocos_label, cocos_label_source
            """,
            target_ids=queried,
        )
        labels = {
            row["target_id"]: (
                row["cocos_label"] or CocosLabel.NONE,
                row["cocos_label_source"] or CocosLabel.NONE,
            )
            for row in rows
        }
    for name in time_targets:
        labels[name] = (CocosLabel.NONE, CocosLabel.NONE)
    return labels


def build_mapping_handoff(
    facility: str,
    ids_names: Sequence[str],
    *,
    gc: GraphClient | None = None,
) -> dict[str, Any]:
    """Expand each bound SignalSource to its FacilitySignal members."""
    if not ids_names:
        raise ValueError("at least one IDS name is required")
    if gc is None:
        gc = GraphClient()

    member_patterns = _member_patterns(facility)

    document: dict[str, Any] = {
        "format": FORMAT,
        "format_version": 1,
        "facility": facility,
        "dd_version": None,
        "exported_at": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "ids": [],
    }
    for ids_name in dict.fromkeys(ids_names):
        existing = search_existing_mappings(facility, ids_name, gc=gc)
        mapping = existing["mapping"]
        if mapping is None:
            raise ValueError(f"no mapping found for {facility}/{ids_name}")
        version = mapping.get("dd_version")
        if not version:
            raise ValueError(f"mapping {mapping['id']} has no DD version")
        if document["dd_version"] is None:
            document["dd_version"] = version
        elif document["dd_version"] != version:
            raise ValueError("selected mappings have different DD versions")

        # The schema stores the member's source identity in data_source_name
        # and data_source_path; MEMBER_OF points from FacilitySignal to source.
        member_rows = gc.query(
            """
            MATCH (m:IMASMapping {id: $mapping_id})-[:USES_SIGNAL_SOURCE]->
                  (source:SignalSource)-[binding:MAPS_TO_IMAS]->(target:IMASNode)
            OPTIONAL MATCH (signal:FacilitySignal)-[:MEMBER_OF]->(source)
            RETURN source.id AS source_id, target.id AS target_id,
                   signal.id AS signal_id,
                   signal.data_source_name AS data_source,
                   signal.data_source_path AS data_source_path,
                   binding.source_property AS source_property,
                   binding.mapping_type AS mapping_type,
                   binding.derived_from AS derived_from,
                   binding.confidence AS confidence,
                   binding.evidence AS evidence
            ORDER BY source.id, target.id, signal.id
            """,
            mapping_id=mapping["id"],
        )
        members: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in member_rows:
            members[(row["source_id"], row["target_id"])].append(row)

        entry: dict[str, Any] = {
            "ids_name": ids_name,
            "mapping_id": mapping["id"],
            "status": mapping["status"],
            "signals": [],
            "unexpanded": [],
        }
        bound_targets = defaultdict(set)
        for binding in existing["bindings"]:
            bound_targets[binding["source_id"]].add(binding["target_id"])
        cocos_labels = _nearest_cocos_labels(
            gc, [binding["target_id"] for binding in existing["bindings"]]
        )
        for binding in existing["bindings"]:
            source_id = binding["source_id"]
            target_path = binding["target_id"]
            rows = members.get((source_id, target_path), [])
            derived_from = rows[0].get("derived_from") if rows else None
            if (
                rows
                and rows[0].get("mapping_type") == "error_derived"
                and derived_from in bound_targets[source_id]
            ):
                # The error field was filled from the same signal that feeds
                # its data field, so no error signal stands behind it.
                entry["unexpanded"].append(
                    {
                        "source_id": source_id,
                        "target_path": target_path,
                        "reason": (
                            f"Error mapping derived from the value signal bound to "
                            f"{derived_from}; no error signal exists"
                        ),
                    }
                )
                continue
            arrays = [
                parts[1]
                for row in rows
                if (parts := _source_parts(row.get("data_source_path")))
            ]
            if not rows or all(row.get("signal_id") is None for row in rows):
                entry["unexpanded"].append(
                    {
                        "source_id": source_id,
                        "target_path": target_path,
                        "reason": "No FacilitySignal member is linked to this source",
                    }
                )
                continue
            for row in rows:
                parts = _source_parts(row.get("data_source_path"))
                if not row.get("signal_id") or not row.get("data_source") or not parts:
                    entry["unexpanded"].append(
                        {
                            "source_id": source_id,
                            "target_path": target_path,
                            "reason": f"FacilitySignal {row.get('signal_id') or '<missing>'} lacks source identity",
                        }
                    )
                    continue
                source_group, source_array = parts
                identifier = _member_identifier(
                    source_array, arrays, member_patterns.get(source_group)
                )
                cocos_label, cocos_label_source = cocos_labels.get(
                    target_path, (CocosLabel.NONE, CocosLabel.NONE)
                )
                entry["signals"].append(
                    {
                        "signal_id": row["signal_id"],
                        "source_id": source_id,
                        "data_source": row["data_source"],
                        "source_group": source_group,
                        "source_array": source_array,
                        "member_identifier": identifier,
                        "source_property": row.get("source_property"),
                        "target_path": target_path,
                        "transform_expression": binding.get("transform_expression"),
                        "source_units": binding.get("source_units"),
                        "target_units": binding.get("target_units"),
                        "cocos_label": cocos_label,
                        "cocos_label_source": cocos_label_source,
                        "confidence": row.get("confidence"),
                        "evidence": row.get("evidence"),
                    }
                )
        document["ids"].append(entry)

    check_handoff_document(document)
    return document
