"""Contracts for governed semantic source-binding reconciliation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn
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
            return [
                {"event_id": event_id, "matches": []}
                for event_id in params["event_ids"]
            ]
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
    with pytest.raises(ValueError, match="independent approval"):
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


@pytest.mark.parametrize("protected", ["west", "WEST"])
def test_transitive_west_protection_refuses(protected):
    row = _row(protected=protected)
    manifest_row = _manifest_row(row)
    plan = reconciliation._plan_row(row, manifest_row)
    assert plan["status"] == "refused"
    assert any("WEST" in reason for reason in plan["unresolved"])


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
            "source_element_id": "west-source",
            "source_properties": {
                "id": "signals:west:diagnostic:value",
                "facility_id": "west",
            },
            "relationship_element_id": "west-binding",
            "target_element_id": "target-element",
            "target_properties": row["prospective_targets"][0]["properties"],
        }
    ]
    manifest_row = _manifest_row(row)
    plan = reconciliation._plan_row(row, manifest_row)
    assert plan["status"] == "refused"


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
    assert receipt["query_count"] == 7
    assert client.transaction.commits == 1
    assert client.transaction.rollbacks == 0
    source = client.transaction.rows[0]["sources"][0]
    assert source["properties"]["produced_sn_id"] == row["prospective_target_id"]
    assert (
        source["dd_backings"][0]["properties"]["cocos_transformation_type"]
        == "psi_like"
    )


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
    assert counts == [7, 7]


def test_queries_are_batched_and_relationship_locks_are_anchored():
    assert "UNWIND $candidates" in reconciliation.CLOSURE_QUERY
    assert "UNWIND $items" in reconciliation.APPLY_QUERY
    assert "UNWIND $locks" in reconciliation.RELATIONSHIP_LOCK_QUERY
    assert "MATCH ()-[relationship]" not in reconciliation.RELATIONSHIP_LOCK_QUERY
    assert (
        "elementId(anchor) = item.anchor_element_id"
        in reconciliation.RELATIONSHIP_LOCK_QUERY
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
