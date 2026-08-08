"""Contracts for governed semantic source-binding reconciliation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names import semantic_source_reconciliation as reconciliation


def _target(target_id: str, element_id: str, *, stage: str = "accepted") -> dict:
    return {
        "element_id": element_id,
        "labels": ["StandardName"],
        "properties": {
            "id": target_id,
            "name_stage": stage,
            "validation_status": "valid",
            "status": "active",
        },
    }


def _row(
    source_id: str = "dd:diagnostic/channel/value",
    *,
    target_id: str = "voltage_at_diagnostic_channel",
    source_type: str = "dd",
    protected: str | None = None,
) -> dict:
    old = _target("voltage_at_channel", "old-target", stage="superseded")
    target = _target(target_id, "target-element")
    backing_label = "IMASNode" if source_type == "dd" else "FacilitySignal"
    backing_key = "dd_backings" if source_type == "dd" else "signal_backings"
    backing_relation = "backing-link"
    source = {
        "element_id": "source-element",
        "labels": ["StandardNameSource"],
        "properties": {
            "id": source_id,
            "source_type": source_type,
            "source_id": source_id.split(":", 1)[1],
            "status": "attached",
            "validation_status": "valid",
            "produced_sn_id": target_id,
            "claimed_at": None,
            "claim_token": None,
            "drain_scope_id": None,
            "drain_scope_claimed_at": None,
            "drain_claim_scope_id": None,
            "open_edit_id": None,
        },
        "bindings": [
            {
                "relationship_element_id": "old-binding",
                "properties": {},
                "target_element_id": old["element_id"],
                "target_id": old["properties"]["id"],
                "target_properties": old["properties"],
            }
        ],
        "dd_backings": [],
        "signal_backings": [],
        "events": [],
    }
    source[backing_key] = [
        {
            "relationship_element_id": backing_relation,
            "relationship_properties": {},
            "element_id": "backing-element",
            "labels": [backing_label],
            "properties": {
                "id": source_id.split(":", 1)[1],
                "facility_id": protected,
                "cocos_transformation_type": "psi_like",
            },
            "projections": [
                {
                    "relationship_element_id": "projection-link",
                    "properties": {},
                    "target_element_id": target["element_id"],
                    "target_id": target_id,
                    "target_properties": target["properties"],
                }
            ],
        }
    ]
    return {
        "source_id": source_id,
        "prospective_target_id": target_id,
        "sources": [source],
        "prospective_targets": [target],
        "prospective_producers": [],
        "protected_names": [],
        "dd_versions": [
            {
                "element_id": "version-element",
                "properties": {"id": "4.1.1", "is_current": True, "cocos": 17},
            }
        ],
    }


def _manifest_payload(rows: list[dict]) -> dict:
    return {
        "schema": reconciliation.MANIFEST_SCHEMA,
        "schema_version": 1,
        "rows": rows,
    }


def _write_manifest(path: Path, rows: list[dict]) -> str:
    raw = json.dumps(_manifest_payload(rows), sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class _Transaction:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = copy.deepcopy(rows)
        self.extra_global_events: dict[str, list[dict]] = {}
        self.query_markers: list[str] = []
        self.commits = 0
        self.rollbacks = 0

    def run(self, query: str, **params):
        marker = query.strip().splitlines()[0]
        self.query_markers.append(marker)
        if "_CLOSURE" in marker:
            return copy.deepcopy(self.rows)
        if "_PARTICIPANT_LOCK" in marker:
            return [{"locked": len(set(params["element_ids"]))}]
        if "_RELATIONSHIP_LOCK" in marker:
            return [{"locked": len(params["locks"])}]
        if "_EVENT_COLLISIONS" in marker:
            result = []
            for event_id in sorted(params["event_ids"]):
                matches = copy.deepcopy(self.extra_global_events.get(event_id, []))
                for row in self.rows:
                    for source in row["sources"]:
                        for event in source["events"]:
                            properties = event.get("event_properties") or {}
                            if properties.get("id") != event_id:
                                continue
                            matches.append(
                                {
                                    "element_id": event.get("event_element_id"),
                                    "properties": copy.deepcopy(properties),
                                    "links": [
                                        {
                                            "source_id": source["properties"]["id"],
                                            "relationship_element_id": event.get(
                                                "relationship_element_id"
                                            ),
                                        }
                                    ],
                                }
                            )
                result.append({"event_id": event_id, "matches": matches})
            return result
        if "_APPLY" in marker:
            source_ids = []
            event_ids = []
            by_source = {row["source_id"]: row for row in self.rows}
            for item in params["items"]:
                row = by_source[item["source_id"]]
                source = row["sources"][0]
                target = row["prospective_targets"][0]
                source["properties"]["produced_sn_id"] = item["target_id"]
                source["bindings"] = [
                    {
                        "relationship_element_id": "new-binding:" + item["source_id"],
                        "properties": {},
                        "target_element_id": target["element_id"],
                        "target_id": item["target_id"],
                        "target_properties": target["properties"],
                    }
                ]
                backing = (source["dd_backings"] or source["signal_backings"])[0]
                backing["projections"] = [
                    {
                        "relationship_element_id": "new-projection:"
                        + item["source_id"],
                        "properties": {},
                        "target_element_id": target["element_id"],
                        "target_id": item["target_id"],
                        "target_properties": target["properties"],
                    }
                ]
                event = copy.deepcopy(item["event"])
                source["events"] = [
                    {
                        "relationship_element_id": "event-link:" + item["source_id"],
                        "properties": {},
                        "event_element_id": "event-element:" + item["source_id"],
                        "event_properties": event,
                    }
                ]
                row["prospective_producers"] = [
                    {
                        "source_element_id": source["element_id"],
                        "source_properties": source["properties"],
                        "relationship_element_id": source["bindings"][0][
                            "relationship_element_id"
                        ],
                        "target_element_id": target["element_id"],
                        "target_properties": target["properties"],
                    }
                ]
                source_ids.append(item["source_id"])
                event_ids.append(event["id"])
            return [{"source_ids": source_ids, "event_ids": event_ids}]
        raise AssertionError(marker)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class _Session:
    def __init__(self, transaction: _Transaction) -> None:
        self.transaction = transaction

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def begin_transaction(self):
        return self.transaction


class _Client:
    def __init__(self, rows: list[dict]) -> None:
        self.transaction = _Transaction(rows)

    def session(self):
        return _Session(self.transaction)


def _manifest_row(row: dict) -> dict:
    return reconciliation.build_semantic_source_manifest_row(
        row, prospective_target_id=row["prospective_target_id"]
    )


def test_manifest_is_validated_before_graph_access(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    client = Mock()
    with pytest.raises(ValueError, match="top-level"):
        reconciliation.reconcile_semantic_sources(path, reason="audit", gc=client)
    client.session.assert_not_called()


def test_manifest_rejects_derived_sources(tmp_path):
    path = tmp_path / "manifest.json"
    row = _manifest_row(_row())
    row["source_id"] = "derived:quantity"
    _write_manifest(path, [row])
    with pytest.raises(ValueError, match="unsupported source"):
        reconciliation.load_semantic_source_manifest(path)


def test_manifest_rejects_duplicate_sources(tmp_path):
    path = tmp_path / "manifest.json"
    row = _manifest_row(_row())
    _write_manifest(path, [row, row])
    with pytest.raises(ValueError, match="unique"):
        reconciliation.load_semantic_source_manifest(path)


def test_manifest_rejects_unapproved_override(tmp_path):
    path = tmp_path / "manifest.json"
    row = _manifest_row(_row())
    row["reviewed_override"] = True
    _write_manifest(path, [row])
    with pytest.raises(ValueError, match="non-empty string approval"):
        reconciliation.load_semantic_source_manifest(path)


def test_manifest_rejects_untyped_approval_and_nonoverride_approval(tmp_path):
    path = tmp_path / "manifest.json"
    row = _manifest_row(_row())
    row["reviewed_override"] = True
    row["review_approval"] = {"claimed_by": "untyped-object"}
    _write_manifest(path, [row])
    with pytest.raises(ValueError, match="non-empty string approval"):
        reconciliation.load_semantic_source_manifest(path)
    row["reviewed_override"] = False
    row["review_approval"] = "must be null"
    _write_manifest(path, [row])
    with pytest.raises(ValueError, match="require null"):
        reconciliation.load_semantic_source_manifest(path)


def test_manifest_allows_shared_backing_with_identical_target_intent(tmp_path):
    first = _row("dd:diagnostic/channel/first")
    second = _row("dd:diagnostic/channel/second")
    second["sources"][0]["element_id"] = "second-source"
    second["sources"][0]["bindings"][0]["relationship_element_id"] = "second-binding"
    second["sources"][0]["dd_backings"][0]["relationship_element_id"] = "second-backing"
    path = tmp_path / "manifest.json"
    _write_manifest(path, [_manifest_row(first), _manifest_row(second)])
    manifest = reconciliation.load_semantic_source_manifest(path)
    assert len(manifest.rows) == 2


def test_manifest_rejects_shared_relationship_with_conflicting_target(tmp_path):
    first = _manifest_row(_row())
    second_row = _row(
        "dd:diagnostic/channel/second", target_id="current_at_diagnostic_channel"
    )
    second = _manifest_row(second_row)
    second["participant_relationship_ids"] = first["participant_relationship_ids"]
    path = tmp_path / "manifest.json"
    _write_manifest(path, [first, second])
    with pytest.raises(ValueError, match="conflicting intent"):
        reconciliation.load_semantic_source_manifest(path)


@pytest.mark.parametrize("source_type", ["dd", "facility_signal"])
def test_planner_includes_supported_projection_parity(source_type):
    source_id = (
        "dd:diagnostic/channel/value"
        if source_type == "dd"
        else "signals:iter:diagnostic:value"
    )
    row = _row(source_id, source_type=source_type)
    assert len(reconciliation.plan_semantic_source_rows([row])) == 1


def test_planner_excludes_multiple_live_identities():
    row = _row()
    row["sources"][0]["bindings"][0]["target_properties"]["name_stage"] = "accepted"
    assert reconciliation.plan_semantic_source_rows([row]) == []


def test_planner_admits_stale_scalar_mirror():
    row = _row()
    row["sources"][0]["properties"]["produced_sn_id"] = "stale_scalar"
    planned = reconciliation.plan_semantic_source_rows([row])
    assert len(planned) == 1
    assert planned[0]["prospective_target_id"] == row["prospective_target_id"]


def test_planner_excludes_derived_source():
    row = _row()
    row["source_id"] = "derived:quantity"
    row["sources"][0]["properties"]["id"] = "derived:quantity"
    assert reconciliation.plan_semantic_source_rows([row]) == []


@pytest.mark.parametrize("facility", ["west", "WEST"])
def test_facility_batch_membership_stays_actionable(facility):
    """Facility batch membership is repairable, so it refuses no binding."""
    row = _row(protected=facility)
    manifest_row = _manifest_row(row)
    plan = reconciliation._plan_row(row, manifest_row)
    assert plan["status"] == "planned"
    assert plan["unresolved"] == []


@pytest.mark.parametrize(
    "field",
    ["claimed_at", "claim_token", "drain_scope_id", "open_edit_id"],
)
def test_current_claim_or_edit_refuses(field):
    row = _row()
    row["sources"][0]["properties"][field] = "occupied"
    manifest_row = _manifest_row(row)
    plan = reconciliation._plan_row(row, manifest_row)
    assert plan["status"] == "refused"
    assert any("claim" in reason for reason in plan["unresolved"])


def test_prospective_target_produced_by_protected_source_refuses():
    row = _row()
    row["prospective_producers"] = [
        {
            "source_element_id": "fixture-source",
            "source_properties": {
                "id": "fixture:diagnostic:value",
                "origin": "fixture",
            },
            "relationship_element_id": "fixture-binding",
            "target_element_id": "target-element",
            "target_properties": row["prospective_targets"][0]["properties"],
        }
    ]
    manifest_row = _manifest_row(row)
    plan = reconciliation._plan_row(row, manifest_row)
    assert plan["status"] == "refused"
    assert any("fixtures" in reason for reason in plan["unresolved"])


def test_transitive_parent_producer_protection_refuses():
    row = _row()
    row["protected_names"] = [
        {
            "element_id": "parent-name",
            "properties": {"id": "voltage"},
            "path_relationships": [
                {"relationship_element_id": "parent-link", "properties": {}}
            ],
            "producers": [
                {
                    "source_element_id": "fixture-source",
                    "source_properties": {"id": "fixture:protected"},
                    "relationship_element_id": "fixture-binding",
                }
            ],
        }
    ]
    manifest_row = _manifest_row(row)
    plan = reconciliation._plan_row(row, manifest_row)
    assert plan["status"] == "refused"
    assert any("fixtures" in reason for reason in plan["unresolved"])


def test_catalog_cocos_is_global_and_path_label_is_untouched():
    row = _row()
    manifest_row = _manifest_row(row)
    before = row["sources"][0]["dd_backings"][0]["properties"][
        "cocos_transformation_type"
    ]
    assert reconciliation._plan_row(row, manifest_row)["status"] == "planned"
    assert before == "psi_like"
    row["dd_versions"][0]["properties"]["cocos"] = 11
    assert reconciliation._plan_row(row, manifest_row)["status"] == "refused"


def test_nonoverride_event_is_neo4j_storable_and_hash_exact():
    manifest_row = _manifest_row(_row())
    event_id = reconciliation._event_id(manifest_row)
    event = reconciliation._event_record(manifest_row, event_id, reason="exact reason")
    assert "review_approval" not in event
    assert all(value is not None for value in event.values())
    assert reconciliation._valid_event(
        {"event_properties": event},
        manifest_row,
        event_id,
        reason="exact reason",
    )


def test_dd_version_authority_is_a_locked_participant():
    row = _row()
    manifest_row = _manifest_row(row)
    assert row["dd_versions"][0]["element_id"] in manifest_row["participant_ids"]


def test_property_and_relationship_drift_refuse():
    row = _row()
    manifest_row = _manifest_row(row)
    row["sources"][0]["properties"]["status"] = "failed"
    row["sources"][0]["bindings"][0]["relationship_element_id"] = "competitor"
    plan = reconciliation._plan_row(row, manifest_row)
    assert plan["status"] == "refused"
    assert any("drifted" in reason for reason in plan["unresolved"])


def test_apply_is_atomic_and_idempotent(tmp_path):
    row = _row()
    manifest_row = _manifest_row(row)
    path = tmp_path / "manifest.json"
    digest = _write_manifest(path, [manifest_row])
    client = _Client([row])
    receipt = reconciliation.reconcile_semantic_sources(
        path,
        reason="repair exact redundant binding",
        apply=True,
        expected_manifest_hash=digest,
        gc=client,
    )
    assert receipt["mode"] == "applied"
    assert receipt["query_count"] == 8
    assert client.transaction.commits == 1
    assert client.transaction.rollbacks == 0
    source = client.transaction.rows[0]["sources"][0]
    assert source["properties"]["produced_sn_id"] == row["prospective_target_id"]
    assert (
        source["dd_backings"][0]["properties"]["cocos_transformation_type"]
        == "psi_like"
    )
    replay = reconciliation.reconcile_semantic_sources(
        path,
        reason="repair exact redundant binding",
        apply=True,
        expected_manifest_hash=digest,
        gc=client,
    )
    assert replay["mode"] == "already_current"
    assert replay["query_count"] == 2


def test_apply_rolls_back_on_concurrent_closure_drift(tmp_path):
    row = _row()
    path = tmp_path / "manifest.json"
    digest = _write_manifest(path, [_manifest_row(row)])
    client = _Client([row])
    original_run = client.transaction.run
    closure_reads = 0

    def drift(query, **params):
        nonlocal closure_reads
        result = original_run(query, **params)
        if "_CLOSURE" in query:
            closure_reads += 1
            if closure_reads == 2:
                result[0]["sources"][0]["properties"]["status"] = "failed"
        return result

    client.transaction.run = drift
    with pytest.raises(
        reconciliation.SemanticSourceConflict, match="changed after locks"
    ):
        reconciliation.reconcile_semantic_sources(
            path, reason="audit", apply=True, expected_manifest_hash=digest, gc=client
        )
    assert client.transaction.commits == 0
    assert client.transaction.rollbacks == 1


def test_apply_rolls_back_on_concurrent_dd_authority_drift(tmp_path):
    row = _row()
    path = tmp_path / "manifest.json"
    digest = _write_manifest(path, [_manifest_row(row)])
    client = _Client([row])
    original_run = client.transaction.run
    closure_reads = 0

    def drift(query, **params):
        nonlocal closure_reads
        result = original_run(query, **params)
        if "_CLOSURE" in query:
            closure_reads += 1
            if closure_reads == 2:
                result[0]["dd_versions"][0]["properties"]["cocos"] = 11
        return result

    client.transaction.run = drift
    with pytest.raises(
        reconciliation.SemanticSourceConflict, match="changed after locks"
    ):
        reconciliation.reconcile_semantic_sources(
            path, reason="audit", apply=True, expected_manifest_hash=digest, gc=client
        )
    assert client.transaction.commits == 0


def test_replay_refuses_unlinked_duplicate_global_event(tmp_path):
    row = _row()
    manifest_row = _manifest_row(row)
    path = tmp_path / "manifest.json"
    digest = _write_manifest(path, [manifest_row])
    client = _Client([row])
    reconciliation.reconcile_semantic_sources(
        path,
        reason="exact reason",
        apply=True,
        expected_manifest_hash=digest,
        gc=client,
    )
    event_id = reconciliation._event_id(manifest_row)
    event = client.transaction.rows[0]["sources"][0]["events"][0]["event_properties"]
    client.transaction.extra_global_events[event_id] = [
        {"element_id": "unrelated-event", "properties": event, "links": []}
    ]
    replay = reconciliation.reconcile_semantic_sources(
        path,
        reason="exact reason",
        apply=True,
        expected_manifest_hash=digest,
        gc=client,
    )
    assert replay["mode"] == "refused"
    assert "cardinality" in replay["rows"][0]["unresolved"][0]


def test_event_corruption_and_duplicates_refuse():
    row = _row()
    manifest_row = _manifest_row(row)
    source = row["sources"][0]
    target = row["prospective_targets"][0]
    source["bindings"] = [
        {
            "relationship_element_id": "binding",
            "properties": {},
            "target_element_id": target["element_id"],
            "target_id": target["properties"]["id"],
            "target_properties": target["properties"],
        }
    ]
    event_id = reconciliation._event_id(manifest_row)
    event = reconciliation._event_record(manifest_row, event_id)
    event["record_hash"] = "0" * 64
    entry = {"relationship_element_id": "event-link", "event_properties": event}
    source["events"] = [entry, copy.deepcopy(entry)]
    plan = reconciliation._plan_row(row, manifest_row)
    assert plan["status"] == "refused"
    assert any("event" in reason for reason in plan["unresolved"])


def test_constant_query_count_for_one_and_large_cohorts(tmp_path):
    counts = []
    for size in (1, 40):
        rows = []
        manifest_rows = []
        for index in range(size):
            source_id = f"dd:diagnostic/channel/{index}/value"
            row = _row(source_id)
            suffix = str(index)
            source = row["sources"][0]
            source["element_id"] += suffix
            source["bindings"][0]["relationship_element_id"] += suffix
            source["dd_backings"][0]["relationship_element_id"] += suffix
            source["dd_backings"][0]["element_id"] += suffix
            source["dd_backings"][0]["projections"][0]["relationship_element_id"] += (
                suffix
            )
            row["prospective_targets"][0]["element_id"] += suffix
            source["dd_backings"][0]["projections"][0]["target_element_id"] += suffix
            rows.append(row)
            manifest_rows.append(_manifest_row(row))
        path = tmp_path / f"manifest-{size}.json"
        digest = _write_manifest(path, manifest_rows)
        client = _Client(rows)
        receipt = reconciliation.reconcile_semantic_sources(
            path,
            reason="bounded batch",
            apply=True,
            expected_manifest_hash=digest,
            gc=client,
        )
        counts.append(receipt["query_count"])
    assert counts == [8, 8]


def test_queries_are_batched_and_relationship_locks_are_anchored():
    assert "UNWIND $candidates" in reconciliation.CLOSURE_QUERY
    assert "UNWIND $items" in reconciliation.APPLY_QUERY
    assert "UNWIND $element_ids AS element_id" in (
        reconciliation.PARTICIPANT_LOCK_QUERY
    )
    assert "elementId(participant) = element_id" in (
        reconciliation.PARTICIPANT_LOCK_QUERY
    )
    assert "elementId(relationship) IN $relationship_element_ids" in (
        reconciliation.RELATIONSHIP_LOCK_QUERY
    )
    assert "UNWIND $start_element_ids AS start_element_id" in (
        reconciliation.RELATIONSHIP_LOCK_QUERY
    )
    assert "elementId(start) = start_element_id" in (
        reconciliation.RELATIONSHIP_LOCK_QUERY
    )


def test_cli_requires_hash_for_apply(tmp_path):
    path = tmp_path / "manifest.json"
    _write_manifest(path, [_manifest_row(_row())])
    result = CliRunner().invoke(
        sn,
        [
            "reconcile-semantic-sources",
            "--manifest",
            str(path),
            "--reason",
            "audit",
            "--apply",
        ],
    )
    assert result.exit_code == 2
    assert "--manifest-sha256" in result.output


def test_cli_dry_run_routes_to_operator(tmp_path):
    path = tmp_path / "manifest.json"
    _write_manifest(path, [_manifest_row(_row())])
    with patch.object(
        reconciliation, "reconcile_semantic_sources", return_value={"mode": "planned"}
    ) as operation:
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-semantic-sources",
                "--manifest",
                str(path),
                "--reason",
                "audit",
            ],
        )
    assert result.exit_code == 0
    operation.assert_called_once_with(
        str(path), reason="audit", apply=False, expected_manifest_hash=None
    )


@pytest.fixture(scope="module")
def ephemeral_graph() -> Iterator[GraphClient]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("semantic-source tests require an ephemeral graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("semantic-source tests refuse the configured project graph")
    graph = GraphClient(
        uri=uri,
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", ""),
        graph_name="ephemeral-semantic-source-reconciliation",
    )
    graph.get_stats()
    try:
        yield graph
    finally:
        graph.close()


def _clean_ephemeral_graph(graph: GraphClient) -> None:
    with graph.session() as session:
        session.run(
            "MATCH (node {semantic_reconciliation_fixture: true}) DETACH DELETE node"
        ).consume()


def _seed_graph_cohort(
    graph: GraphClient, *, source_type: str, size: int
) -> list[dict[str, str]]:
    _clean_ephemeral_graph(graph)
    rows = []
    for index in range(size):
        path = f"diagnostic/semantic_source_reconciliation/{source_type}/{index}/value"
        source_id = path if source_type == "dd" else f"iter:{path}"
        rows.append(
            {
                "source_id": f"dd:{path}"
                if source_type == "dd"
                else f"signals:{source_id}",
                "backing_id": source_id,
                "old_id": f"old_semantic_quantity_{index}",
                "target_id": f"semantic_quantity_{index}",
            }
        )
    backing_label = "IMASNode" if source_type == "dd" else "FacilitySignal"
    backing_relationship = (
        "FROM_DD_PATH" if source_type == "dd" else "FROM_FACILITY_SIGNAL"
    )
    with graph.session() as session:
        session.run(
            """
            CREATE (:DDVersion {
              id: '4.1.1', is_current: true, cocos: 17,
              semantic_reconciliation_fixture: true})
            """
        ).consume()
        session.run(
            f"""
            UNWIND $rows AS item
            CREATE (source:StandardNameSource {{
              id: item.source_id, source_type: $source_type,
              source_id: item.backing_id, status: 'attached',
              validation_status: 'valid', produced_sn_id: item.target_id,
              semantic_reconciliation_fixture: true}})
            CREATE (old:StandardName {{
              id: item.old_id, name_stage: 'superseded',
              validation_status: 'valid', status: 'superseded',
              semantic_reconciliation_fixture: true}})
            CREATE (target:StandardName {{
              id: item.target_id, name_stage: 'accepted',
              validation_status: 'valid', status: 'active',
              semantic_reconciliation_fixture: true}})
            CREATE (backing:{backing_label} {{
              id: item.backing_id, cocos_transformation_type: 'psi_like',
              semantic_reconciliation_fixture: true}})
            CREATE (source)-[:PRODUCED_NAME]->(old)
            CREATE (source)-[:{backing_relationship}]->(backing)
            CREATE (backing)-[:HAS_STANDARD_NAME]->(target)
            """,
            rows=rows,
            source_type=source_type,
        ).consume()
    return rows


def _write_graph_manifest(
    graph: GraphClient, path: Path, rows: list[dict[str, str]]
) -> str:
    candidates = [
        {
            "source_id": row["source_id"],
            "prospective_target_id": row["target_id"],
        }
        for row in rows
    ]
    with graph.session() as session:
        closures = [
            dict(record)
            for record in session.run(
                reconciliation.CLOSURE_QUERY, candidates=candidates
            )
        ]
    manifest_rows = [
        reconciliation.build_semantic_source_manifest_row(
            closure,
            prospective_target_id=str(closure["prospective_target_id"]),
        )
        for closure in closures
    ]
    return _write_manifest(path, manifest_rows)


def _graph_counts(graph: GraphClient, rows: list[dict[str, str]]) -> dict[str, Any]:
    source_ids = [row["source_id"] for row in rows]
    with graph.session() as session:
        result = session.run(
            """
            MATCH (source:StandardNameSource) WHERE source.id IN $source_ids
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(produced:StandardName)
            OPTIONAL MATCH (source)-[:FROM_DD_PATH|FROM_FACILITY_SIGNAL]->(backing)
            OPTIONAL MATCH (backing)-[:HAS_STANDARD_NAME]->(projected:StandardName)
            OPTIONAL MATCH (source)-[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
            WITH collect(DISTINCT source.produced_sn_id) AS scalar_ids,
              collect(DISTINCT produced.id) AS produced_ids,
              collect(DISTINCT projected.id) AS projected_ids,
              collect(DISTINCT change) AS changes,
              collect(DISTINCT backing.cocos_transformation_type) AS labels
            MATCH (version:DDVersion {is_current: true})
            RETURN scalar_ids, produced_ids, projected_ids,
              [change IN changes | properties(change)] AS changes,
              labels, collect(properties(version)) AS versions
            """,
            source_ids=source_ids,
        ).single(strict=True)
    return dict(result)


@pytest.mark.graph
@pytest.mark.parametrize("source_type", ["dd", "facility_signal"])
@pytest.mark.parametrize("size", [1, 40])
def test_graph_apply_replay_is_constant_and_preserves_dd_authority(
    ephemeral_graph: GraphClient,
    tmp_path: Path,
    source_type: str,
    size: int,
) -> None:
    rows = _seed_graph_cohort(ephemeral_graph, source_type=source_type, size=size)
    manifest = tmp_path / f"{source_type}-{size}.json"
    digest = _write_graph_manifest(ephemeral_graph, manifest, rows)

    receipt = reconciliation.reconcile_semantic_sources(
        manifest,
        reason="repair exact semantic source projection",
        apply=True,
        expected_manifest_hash=digest,
        gc=ephemeral_graph,
    )
    replay = reconciliation.reconcile_semantic_sources(
        manifest,
        reason="repair exact semantic source projection",
        apply=True,
        expected_manifest_hash=digest,
        gc=ephemeral_graph,
    )
    state = _graph_counts(ephemeral_graph, rows)

    expected_targets = sorted(row["target_id"] for row in rows)
    assert receipt["mode"] == "applied"
    assert receipt["query_count"] == 8
    assert replay["mode"] == "already_current"
    assert replay["query_count"] == 2
    assert sorted(state["scalar_ids"]) == expected_targets
    assert sorted(state["produced_ids"]) == expected_targets
    assert sorted(state["projected_ids"]) == expected_targets
    assert state["labels"] == ["psi_like"]
    assert state["versions"] == [
        {
            "id": "4.1.1",
            "is_current": True,
            "cocos": 17,
            "semantic_reconciliation_fixture": True,
        }
    ]
    assert len(state["changes"]) == size
    assert all("review_approval" not in event for event in state["changes"])


class _DDDriftTransaction:
    def __init__(self, transaction: Any) -> None:
        self.transaction = transaction
        self.drifted = False

    def run(self, query: str, **params: Any) -> Any:
        result = self.transaction.run(query, **params)
        if "_PARTICIPANT_LOCK" not in query or self.drifted:
            return result
        rows = list(result)
        self.transaction.run(
            "MATCH (version:DDVersion {is_current: true}) SET version.cocos = 11"
        ).consume()
        self.drifted = True
        return rows

    def commit(self) -> None:
        self.transaction.commit()

    def rollback(self) -> None:
        self.transaction.rollback()


class _DDDriftSession:
    def __init__(self, session: Any) -> None:
        self.session = session

    def begin_transaction(self) -> _DDDriftTransaction:
        return _DDDriftTransaction(self.session.begin_transaction())


class _DDDriftClient:
    def __init__(self, graph: GraphClient) -> None:
        self.graph = graph

    @contextmanager
    def session(self) -> Iterator[_DDDriftSession]:
        with self.graph.session() as session:
            yield _DDDriftSession(session)


@pytest.mark.graph
def test_graph_dd_authority_drift_rolls_back_complete_cohort(
    ephemeral_graph: GraphClient, tmp_path: Path
) -> None:
    rows = _seed_graph_cohort(ephemeral_graph, source_type="dd", size=1)
    manifest = tmp_path / "dd-drift.json"
    digest = _write_graph_manifest(ephemeral_graph, manifest, rows)
    with pytest.raises(
        reconciliation.SemanticSourceConflict, match="changed after locks"
    ):
        reconciliation.reconcile_semantic_sources(
            manifest,
            reason="verify global authority lock",
            apply=True,
            expected_manifest_hash=digest,
            gc=_DDDriftClient(ephemeral_graph),
        )
    state = _graph_counts(ephemeral_graph, rows)
    assert state["versions"][0]["cocos"] == 17
    assert state["produced_ids"] == [rows[0]["old_id"]]
    assert state["projected_ids"] == [rows[0]["target_id"]]
    assert state["changes"] == []


@pytest.mark.graph
def test_graph_replay_refuses_unlinked_duplicate_event(
    ephemeral_graph: GraphClient, tmp_path: Path
) -> None:
    rows = _seed_graph_cohort(ephemeral_graph, source_type="facility_signal", size=1)
    manifest = tmp_path / "event-collision.json"
    digest = _write_graph_manifest(ephemeral_graph, manifest, rows)
    reconciliation.reconcile_semantic_sources(
        manifest,
        reason="verify global event identity",
        apply=True,
        expected_manifest_hash=digest,
        gc=ephemeral_graph,
    )
    with ephemeral_graph.session() as session:
        session.run(
            """
            MATCH (:StandardNameSource {id: $source_id})
              -[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
            CREATE (duplicate:StandardNameChange)
            SET duplicate = properties(change),
              duplicate.semantic_reconciliation_fixture = true
            """,
            source_id=rows[0]["source_id"],
        ).consume()
    replay = reconciliation.reconcile_semantic_sources(
        manifest,
        reason="verify global event identity",
        apply=True,
        expected_manifest_hash=digest,
        gc=ephemeral_graph,
    )
    assert replay["mode"] == "refused"
    assert replay["rows"][0]["unresolved"] == [
        "deterministic event cardinality is not exactly one"
    ]


def _plan_operator_names(plan: Any) -> list[str]:
    operators: list[str] = []
    pending = [plan]
    while pending:
        operator = pending.pop()
        if isinstance(operator, dict):
            operators.append(str(operator.get("operatorType", "")).partition("@")[0])
            pending.extend(operator.get("children") or [])
        else:
            operators.append(str(operator.operator_type).partition("@")[0])
            pending.extend(operator.children)
    return operators


@pytest.mark.graph
@pytest.mark.parametrize("size", [1, 40])
def test_graph_lock_plans_reject_global_scans(
    ephemeral_graph: GraphClient, size: int
) -> None:
    rows = _seed_graph_cohort(ephemeral_graph, source_type="dd", size=size)
    candidates = [
        {
            "source_id": row["source_id"],
            "prospective_target_id": row["target_id"],
        }
        for row in rows
    ]
    with ephemeral_graph.session() as session:
        closures = [
            dict(record)
            for record in session.run(
                reconciliation.CLOSURE_QUERY, candidates=candidates
            )
        ]
        participant_ids = sorted(
            {
                element_id
                for closure in closures
                for element_id in reconciliation._participant_ids(closure)
            }
        )
        relationship_locks = reconciliation._relationship_lock_items(closures)
        participant_plan = (
            session.run(
                f"EXPLAIN {reconciliation.PARTICIPANT_LOCK_QUERY}",
                element_ids=participant_ids,
            )
            .consume()
            .plan
        )
        relationship_plan = (
            session.run(
                f"EXPLAIN {reconciliation.RELATIONSHIP_LOCK_QUERY}",
                locks=relationship_locks,
                relationship_element_ids=[
                    item["relationship_element_id"] for item in relationship_locks
                ],
                start_element_ids=[
                    item["start_element_id"] for item in relationship_locks
                ],
            )
            .consume()
            .plan
        )

    forbidden = {
        "AllNodesScan",
        "AllRelationshipsScan",
        "DirectedAllRelationshipsScan",
        "UndirectedAllRelationshipsScan",
    }
    for label, plan in (
        ("participant", participant_plan),
        ("relationship", relationship_plan),
    ):
        assert plan is not None
        operators = _plan_operator_names(plan)
        assert forbidden.isdisjoint(operators), f"{label}: {operators}"
        assert any("ElementIdSeek" in operator for operator in operators), (
            f"{label}: {operators}"
        )
