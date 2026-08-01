"""Graph-backed contract for exact skipped-source recovery."""

from __future__ import annotations

import os
import uuid
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.graph


def _one(rows):
    return list(rows)[0]


def test_skipped_source_retry_is_exact_audited_and_context_preserving(
    graph_client,
) -> None:
    """Only the exact unfenced source is reset, with its context intact."""
    from imas_codex.standard_names.graph_ops import retry_skipped_sources

    test_token = os.environ.get("IMAS_CODEX_SKIPPED_RETRY_TEST_TOKEN", uuid.uuid4().hex)
    prefix = f"test_skipped_source_retry__{test_token}__"
    paths = {
        role: f"{prefix}{role}"
        for role in (
            "eligible_error",
            "eligible_skip_only",
            "claimed",
            "bound",
            "collateral",
        )
    }
    source_ids = {role: f"dd:{path}" for role, path in paths.items()}
    unit_ids = {role: f"{prefix}unit_{role}" for role in paths}
    bound_name_id = f"{prefix}bound_name"
    retry_reason = "the exact synthetic source is now nameable"
    expected_event_ids = [
        f"source-retry:{uuid.uuid5(uuid.NAMESPACE_URL, f'{prefix}{role}')}"
        for role in ("eligible_error", "eligible_skip_only")
    ]
    event_ids: list[str] = []

    sources = [
        {
            "id": source_ids[role],
            "path": path,
            "unit_id": unit_ids[role],
            "attempt_count": attempts,
            "claimed": role == "claimed",
            "claim_token": f"{prefix}claim" if role == "claimed" else None,
            "hint": f"preserve {role} source meaning",
            "hint_reason": f"synthetic {role} steering",
            "skip_reason": skip_reason,
            "skip_reason_detail": skip_reason_detail,
            "last_error": last_error,
        }
        for role, path, attempts, skip_reason, skip_reason_detail, last_error in (
            (
                "eligible_error",
                paths["eligible_error"],
                4,
                "dd_unit_unresolvable",
                "synthetic unresolved unit",
                "synthetic prior error",
            ),
            (
                "eligible_skip_only",
                paths["eligible_skip_only"],
                3,
                "dd_unit_context_dependent",
                "synthetic contextual unit",
                None,
            ),
            (
                "claimed",
                paths["claimed"],
                5,
                "dd_unit_unresolvable",
                "synthetic unresolved unit",
                "synthetic prior error",
            ),
            (
                "bound",
                paths["bound"],
                6,
                "dd_unit_unresolvable",
                "synthetic unresolved unit",
                "synthetic prior error",
            ),
            (
                "collateral",
                paths["collateral"],
                7,
                "dd_unit_unresolvable",
                "synthetic unresolved unit",
                "synthetic prior error",
            ),
        )
    ]

    try:
        graph_client.query(
            """
            UNWIND $sources AS row
            CREATE (dd:IMASNode {id: row.path})
            SET dd.unit = 'A', dd.node_category = 'quantity',
                dd.lifecycle_status = 'active'
            CREATE (unit:Unit {id: row.unit_id})
            CREATE (dd)-[:HAS_UNIT]->(unit)
            CREATE (sns:StandardNameSource {id: row.id})
            SET sns.source_type = 'dd',
                sns.source_id = row.path,
                sns.status = 'skipped',
                sns.attempt_count = row.attempt_count,
                sns.claimed_at = CASE WHEN row.claimed THEN datetime() ELSE null END,
                sns.claim_token = row.claim_token,
                sns.skip_reason = row.skip_reason,
                sns.skip_reason_detail = row.skip_reason_detail,
                sns.skipped_at = datetime(),
                sns.last_error = row.last_error,
                sns.failed_at = datetime(),
                sns.dd_unit = 'A',
                sns.compose_hint = row.hint,
                sns.compose_hint_reason = row.hint_reason,
                sns.compose_hint_status = 'open',
                sns.compose_hint_requested_at = datetime()
            CREATE (sns)-[:FROM_DD_PATH]->(dd)
            """,
            sources=sources,
        )
        graph_client.query(
            """
            MATCH (sns:StandardNameSource {id: $source_id})
            CREATE (sn:StandardName {id: $name_id})
            SET sn.name_stage = 'accepted', sn.docs_stage = 'accepted'
            SET sns.produced_sn_id = sn.id
            CREATE (sns)-[:PRODUCED_NAME]->(sn)
            """,
            source_id=source_ids["bound"],
            name_id=bound_name_id,
        )

        requested = [
            paths["eligible_error"],
            source_ids["eligible_skip_only"],
            source_ids["claimed"],
            source_ids["bound"],
            f"dd:{prefix}missing",
        ]
        dry_run = retry_skipped_sources(
            requested,
            reason=retry_reason,
            dry_run=True,
            gc=graph_client,
        )
        assert dry_run == {
            "requested": 5,
            "eligible": 2,
            "retried": 0,
            "refused": 3,
            "source_ids": [
                source_ids["eligible_error"],
                source_ids["eligible_skip_only"],
            ],
            "event_ids": [],
            "dry_run": True,
        }
        before_apply = _one(
            graph_client.query(
                """
                MATCH (sns:StandardNameSource {id: $source_id})
                OPTIONAL MATCH (sns)-[:HAS_RETRY_EVENT]->(event)
                RETURN sns.status AS status, sns.attempt_count AS attempts,
                       count(event) AS events
                """,
                source_id=source_ids["eligible_error"],
            )
        )
        assert before_apply == {"status": "skipped", "attempts": 4, "events": 0}

        with patch(
            "imas_codex.standard_names.graph_ops.uuid.uuid4",
            side_effect=[
                uuid.UUID(value.partition(":")[2]) for value in expected_event_ids
            ],
        ):
            applied = retry_skipped_sources(
                requested,
                reason=retry_reason,
                gc=graph_client,
            )
        assert applied["requested"] == 5
        assert applied["eligible"] == 2
        assert applied["retried"] == 2
        assert applied["refused"] == 3
        assert applied["source_ids"] == [
            source_ids["eligible_error"],
            source_ids["eligible_skip_only"],
        ]
        assert applied["event_ids"] == expected_event_ids
        event_ids.extend(applied["event_ids"])

        recovered = _one(
            graph_client.query(
                """
                MATCH (sns:StandardNameSource {id: $source_id})
                      -[:FROM_DD_PATH]->(dd:IMASNode {id: $path})
                      -[:HAS_UNIT]->(unit:Unit {id: $unit_id})
                MATCH (sns)-[:HAS_RETRY_EVENT]->(event:StandardNameSourceRetry)
                RETURN sns.id AS id, sns.source_id AS source_id,
                       sns.source_type AS source_type, sns.status AS status,
                       sns.attempt_count AS attempts,
                       sns.skip_reason AS skip_reason,
                       sns.skip_reason_detail AS skip_reason_detail,
                       sns.skipped_at AS skipped_at,
                       sns.last_error AS last_error,
                       sns.failed_at AS failed_at,
                       sns.dd_unit AS source_unit,
                       sns.compose_hint AS hint,
                       sns.compose_hint_reason AS hint_reason,
                       sns.compose_hint_status AS hint_status,
                       sns.compose_hint_requested_at AS hint_requested_at,
                       dd.unit AS dd_unit, unit.id AS unit_id,
                       event.id AS event_id,
                       event.source_id AS event_source_id,
                       event.previous_status AS previous_status,
                       event.previous_attempt_count AS previous_attempt_count,
                       event.previous_error AS previous_error,
                       event.reason AS reason,
                       event.retried_at IS NOT NULL AS has_retried_at,
                       event.id IN sns.retry_events AS event_mirrored
                """,
                source_id=source_ids["eligible_error"],
                path=paths["eligible_error"],
                unit_id=unit_ids["eligible_error"],
            )
        )
        assert recovered == {
            "id": source_ids["eligible_error"],
            "source_id": paths["eligible_error"],
            "source_type": "dd",
            "status": "extracted",
            "attempts": 0,
            "skip_reason": None,
            "skip_reason_detail": None,
            "skipped_at": None,
            "last_error": None,
            "failed_at": None,
            "source_unit": "A",
            "hint": "preserve eligible_error source meaning",
            "hint_reason": "synthetic eligible_error steering",
            "hint_status": "open",
            "hint_requested_at": recovered["hint_requested_at"],
            "dd_unit": "A",
            "unit_id": unit_ids["eligible_error"],
            "event_id": recovered["event_id"],
            "event_source_id": source_ids["eligible_error"],
            "previous_status": "skipped",
            "previous_attempt_count": 4,
            "previous_error": (
                "synthetic prior error; skip_reason=dd_unit_unresolvable; "
                "detail=synthetic unresolved unit"
            ),
            "reason": retry_reason,
            "has_retried_at": True,
            "event_mirrored": True,
        }
        assert recovered["hint_requested_at"] is not None
        assert recovered["event_id"] in event_ids

        fallback = _one(
            graph_client.query(
                """
                MATCH (sns:StandardNameSource {id: $source_id})
                      -[:HAS_RETRY_EVENT]->(event:StandardNameSourceRetry)
                RETURN sns.status AS status,
                       sns.compose_hint AS hint,
                       sns.compose_hint_reason AS hint_reason,
                       sns.compose_hint_status AS hint_status,
                       event.previous_status AS previous_status,
                       event.previous_attempt_count AS previous_attempt_count,
                       event.previous_error AS previous_error,
                       event.reason AS reason
                """,
                source_id=source_ids["eligible_skip_only"],
            )
        )
        assert fallback == {
            "status": "extracted",
            "hint": "preserve eligible_skip_only source meaning",
            "hint_reason": "synthetic eligible_skip_only steering",
            "hint_status": "open",
            "previous_status": "skipped",
            "previous_attempt_count": 3,
            "previous_error": (
                "skip_reason=dd_unit_context_dependent; "
                "detail=synthetic contextual unit"
            ),
            "reason": retry_reason,
        }

        peers = list(
            graph_client.query(
                """
                UNWIND $ids AS source_id
                MATCH (sns:StandardNameSource {id: source_id})
                OPTIONAL MATCH (sns)-[:HAS_RETRY_EVENT]->(event)
                OPTIONAL MATCH (sns)-[:PRODUCED_NAME]->(name:StandardName)
                RETURN sns.id AS id, sns.status AS status,
                       sns.attempt_count AS attempts,
                       sns.claim_token AS claim_token,
                       sns.skip_reason AS skip_reason,
                       sns.produced_sn_id AS produced_sn_id,
                       count(DISTINCT event) AS retry_events,
                       count(DISTINCT name) AS produced_edges
                ORDER BY sns.id
                """,
                ids=[
                    source_ids["claimed"],
                    source_ids["bound"],
                    source_ids["collateral"],
                ],
            )
        )
        by_id = {row["id"]: row for row in peers}
        assert by_id[source_ids["claimed"]] == {
            "id": source_ids["claimed"],
            "status": "skipped",
            "attempts": 5,
            "claim_token": f"{prefix}claim",
            "skip_reason": "dd_unit_unresolvable",
            "produced_sn_id": None,
            "retry_events": 0,
            "produced_edges": 0,
        }
        assert by_id[source_ids["bound"]] == {
            "id": source_ids["bound"],
            "status": "skipped",
            "attempts": 6,
            "claim_token": None,
            "skip_reason": "dd_unit_unresolvable",
            "produced_sn_id": bound_name_id,
            "retry_events": 0,
            "produced_edges": 1,
        }
        assert by_id[source_ids["collateral"]] == {
            "id": source_ids["collateral"],
            "status": "skipped",
            "attempts": 7,
            "claim_token": None,
            "skip_reason": "dd_unit_unresolvable",
            "produced_sn_id": None,
            "retry_events": 0,
            "produced_edges": 0,
        }
    finally:
        linked_events = list(
            graph_client.query(
                """
                MATCH (sns:StandardNameSource)-[:HAS_RETRY_EVENT]->(event)
                WHERE sns.id IN $source_ids
                RETURN event.id AS id
                """,
                source_ids=list(source_ids.values()),
            )
        )
        event_ids.extend(
            row["id"] for row in linked_events if row["id"] not in event_ids
        )
        graph_client.query(
            """
            MATCH (event:StandardNameSourceRetry)
            WHERE event.id IN $ids
            DETACH DELETE event
            """,
            ids=event_ids,
        )
        for label, ids in (
            ("StandardNameSource", list(source_ids.values())),
            ("IMASNode", list(paths.values())),
            ("Unit", list(unit_ids.values())),
            ("StandardName", [bound_name_id]),
        ):
            graph_client.query(
                f"MATCH (node:{label}) WHERE node.id IN $ids DETACH DELETE node",
                ids=ids,
            )
        remaining = _one(
            graph_client.query(
                """
                OPTIONAL MATCH (source:StandardNameSource)
                WHERE source.id IN $source_ids
                WITH count(source) AS sources
                OPTIONAL MATCH (dd:IMASNode)
                WHERE dd.id IN $dd_ids
                WITH sources, count(dd) AS dd_nodes
                OPTIONAL MATCH (unit:Unit)
                WHERE unit.id IN $unit_ids
                WITH sources, dd_nodes, count(unit) AS units
                OPTIONAL MATCH (name:StandardName {id: $name_id})
                WITH sources, dd_nodes, units, count(name) AS names
                OPTIONAL MATCH (event:StandardNameSourceRetry)
                WHERE event.id IN $event_ids
                RETURN sources, dd_nodes, units, names, count(event) AS events
                """,
                source_ids=list(source_ids.values()),
                dd_ids=list(paths.values()),
                unit_ids=list(unit_ids.values()),
                name_id=bound_name_id,
                event_ids=event_ids,
            )
        )
        assert remaining == {
            "sources": 0,
            "dd_nodes": 0,
            "units": 0,
            "names": 0,
            "events": 0,
        }
