"""Convention grouping with invented selectors and exact/category field sets."""

import json
import sqlite3
from contextlib import closing
from itertools import chain
from unittest.mock import patch

import pytest

from iter_codex.discovery.signals.archive_grouping import (
    ArchiveGrouping,
    group_private_catalogue,
)
from iter_codex.remote.archive_catalogue import ArchiveCatalogue
from iter_codex.remote.catalogue_grouping import run_grouping


@pytest.fixture
def grouping(tmp_path):
    with closing(ArchiveGrouping(tmp_path / "groups.sqlite")) as store:
        yield store


def records(*names):
    return [{"name": name} for name in names]


def test_component_identity_field_patterns_and_category_unions(grouping):
    names = [
        "DEMO-UNIT:QAZ0001-ABCD",
        "DEMO-UNIT:QAZ0001-EFGH",
        "DEMO-UNIT:QAZ0002-ABCD",
        "DEMO-UNIT:QAZ0002-EFGH",
        "DEMO-UNIT:QAZ0003-ABCD",
        "DEMO-UNIT:QAZ0004-ABCD1",
        "DEMO-UNIT:QAZ0004-ABCD2",
        "DEMO-NEXT:QAZ0001-ABCD",
        "DEMO-NEXT:QAZ0001-EFGH",
        "DEMO-UNIT:HJK0001-ABCD",
        "DEMO-UNIT:STATUS",
        "DEMO-NEXT:MODE",
        "unparseable fixture",
    ]
    receipt = grouping.rebuild("demo", records(*reversed(names)))
    assert receipt["variables"] == 13
    assert receipt["parse_outcomes"] == {
        "component_variable": 10,
        "function_variable": 2,
        "not_conforming": 1,
    }
    assert receipt["grouped"] == 12
    assert receipt["component_groups"] == 6
    assert receipt["control_node_groups"] == 2
    assert receipt["field_patterns"] == 3
    assert receipt["category_patterns"] == 2
    assert receipt["group_size_distribution"] == {"1": 2, "2": 4}
    assert receipt["control_group_size_distribution"] == {"1": 2}
    assert receipt["shared_pattern_component_variable_share"] == {
        "category_fields": 0.9,
        "category": 0.9,
    }
    assert receipt["descriptions_needed"]["category_fields"] == {
        "component": 6,
        "control_node": 2,
        "pattern": 3,
        "total": 11,
    }
    assert receipt["descriptions_needed"]["category"]["total"] == 10
    first = grouping.component(grouping.member("demo", names[0])["component"])
    second = grouping.component(grouping.member("demo", names[2])["component"])
    assert first["id"] != second["id"]
    assert first["archive_field_pattern"] == second["archive_field_pattern"]
    assert grouping.pattern(first["archive_field_pattern"])["fields"] == [
        "ABCD",
        "EFGH",
    ]
    discriminated = grouping.component(grouping.member("demo", names[5])["component"])
    assert discriminated["archive_member_count"] == 2
    assert grouping.pattern(discriminated["archive_field_pattern"])["field_count"] == 1
    assert grouping.pattern(discriminated["archive_category_pattern"])["fields"] == [
        "ABCD",
        "EFGH",
    ]
    repeat = grouping.rebuild("demo", grouping.snapshot_records("demo"))
    assert (
        repeat["groups_added"]
        == repeat["control_groups_added"]
        == repeat["patterns_added"]
        == 0
    )
    assert repeat["largest_patterns"] == receipt["largest_patterns"]


def test_function_variables_belong_only_to_their_control_node(grouping):
    receipt = grouping.rebuild(
        "demo", records("DEMO-UNIT:STATUS", "DEMO-UNIT:MODE", "DEMO-UNIT:QAZ0001-ABCD")
    )
    assert receipt["component_groups"] == receipt["control_node_groups"] == 1
    member = grouping.member("demo", "DEMO-UNIT:STATUS")
    assert member["state"] == "control_node"
    assert member["component"] is None
    assert member["field"] is None
    assert grouping.control_group(member["control_node"])["archive_member_count"] == 2
    assert receipt["component_variables"] == 1
    assert receipt["function_variables"] == 2


@pytest.mark.parametrize(
    "name",
    ["unparseable fixture", "DEMOX-UNIT:QAZ0001-ABCD", "DEMO-UNIT:HAS_UNDERSCORE"],
)
def test_unparseable_name_is_counted_without_component_membership(grouping, name):
    receipt = grouping.rebuild("demo", records("DEMO-UNIT:QAZ0001-ABCD", name))
    assert receipt["variables"] == 2
    assert receipt["component_variables"] == receipt["ungrouped"] == 1
    assert receipt["component_groups"] == receipt["field_patterns"] == 1
    member = grouping.member("demo", name)
    assert member["state"] == "ungrouped"
    assert member["reason"]
    assert member["component"] is member["control_node"] is member["field"] is None


def test_growth_changes_exact_pattern_and_refreshes_category_union(grouping):
    initial = records("DEMO-UNIT:QAZ0001-ABCD")
    grouping.rebuild("demo", initial)
    component = grouping.member("demo", initial[0]["name"])["component"]
    category_pattern = grouping.component(component)["archive_category_pattern"]
    grown = grouping.rebuild("demo", initial + records("DEMO-UNIT:QAZ0001-EFGH"))
    assert grouping.member("demo", initial[0]["name"])["component"] == component
    assert grown["groups_added"] == grown["groups_removed"] == 0
    assert grown["patterns_added"] == grown["patterns_removed"] == 1
    assert grouping.component(component)["archive_category_pattern"] == category_pattern
    assert grouping.pattern(category_pattern)["fields"] == ["ABCD", "EFGH"]


def test_failed_or_duplicate_snapshot_preserves_committed_groups(grouping):
    initial = records("DEMO-UNIT:QAZ0001-ABCD")
    grouping.rebuild("demo", initial)
    before = grouping.summary("demo")
    with pytest.raises(sqlite3.IntegrityError):
        grouping.rebuild("demo", initial * 2)
    assert grouping.summary("demo") == before

    def interrupted():
        yield {"name": "unparseable fixture"}
        raise RuntimeError("interrupted source")

    with pytest.raises(RuntimeError, match="interrupted source"):
        grouping.rebuild("demo", interrupted())
    assert grouping.summary("demo") == before


def test_facilities_have_independent_groups(grouping):
    names = records("DEMO-UNIT:QAZ0001-ABCD")
    alpha = grouping.rebuild("alpha", names)
    beta = grouping.rebuild("beta", names)
    assert alpha["largest_patterns"][0]["id"] != beta["largest_patterns"][0]["id"]
    grouping.rebuild("alpha", [])
    assert grouping.summary("beta")["component_variables"] == 1


def test_name_schema_counts_separate_field_values_from_path_prefixes(grouping):
    grouping.rebuild(
        "demo",
        records(
            "DEMO-UNIT:QAZ0001-ABCD",
            "DEMO-UNIT:STATUS",
            "SYN1-UNIT:QAZ0002-ABCD",
            "SYN1:MODE",
            "unparseable fixture",
        ),
    )
    counts = grouping.name_schema_counts("demo")
    assert counts["control_path_fields"]["1"]["distinct_values"] == 2
    assert counts["control_path_fields"]["1"]["variables"] == 4
    assert counts["control_path_fields"]["2"]["values"] == [
        {"code": "UNIT", "variables": 3}
    ]
    assert counts["control_path_prefixes"]["2"]["distinct_values"] == 2
    assert counts["function_categories"] == {
        "distinct_values": 1,
        "variables": 2,
        "values": [{"code": "QAZ", "variables": 2}],
    }


def test_largest_patterns_are_ranked_and_bounded(grouping):
    names = [f"DEMO-UNIT:QAZ{index:04d}-{chr(65 + index)}" for index in range(25)]
    names.append("DEMO-UNIT:QAZ9999-A")
    receipt = grouping.rebuild("demo", records(*names))
    assert receipt["field_patterns"] == 25
    assert len(receipt["largest_patterns"]) == 20
    assert receipt["largest_patterns"][0]["member_count"] == 2


def test_private_catalogue_union_keeps_pending_refused_and_graph_only_variables(
    tmp_path, grouping
):
    source = tmp_path / "catalogue.sqlite"
    measured = "DEMO-COOL:QAZ0001-ABCD"
    pending = "DEMO-COOL:QAZ0001-EFGH"
    graph_only = "DEMO-HEAT:QAZ0001-ABCD"
    with closing(ArchiveCatalogue(source)) as catalogue:
        catalogue.seed(
            {"slow": [measured, pending, "unparseable fixture"], "fast": [measured]},
            timestamp="invented",
        )
        catalogue.save_batch(
            "slow",
            [measured],
            {
                "entries": [
                    {
                        "name": measured,
                        "data_type": "float64",
                        "unit": {"value": "K"},
                        "archive_metadata": {"type": {"datatypes": []}, "full": []},
                    }
                ]
            },
        )
        catalogue.save_batch(
            "fast",
            [measured],
            {
                "entries": [
                    {
                        "name": measured,
                        "metadata_state": "unavailable",
                        "metadata_error_code": 17,
                        "metadata_error": "fixture refusal",
                    }
                ]
            },
        )
        before = catalogue.db.execute(
            "SELECT * FROM census ORDER BY role,name"
        ).fetchall()
        receipt = group_private_catalogue(
            source,
            grouping,
            "demo",
            graph_signals=[
                {"selector": measured},
                {"selector": graph_only},
            ],
        )
        assert (
            catalogue.db.execute("SELECT * FROM census ORDER BY role,name").fetchall()
            == before
        )
    assert receipt["archive_variables"] == receipt["listed_variables"] == 3
    assert receipt["archive_fraction"] == 1
    assert receipt["metadata_completed"] == 1
    assert receipt["metadata_completed_fraction"] == 1 / 3
    assert receipt["metadata_scan_complete"] is False
    assert receipt["graph_only_variables"] == 1
    assert (receipt["variables"], receipt["grouped"], receipt["ungrouped"]) == (
        4,
        3,
        1,
    )
    member = grouping.member("demo", measured)
    assert member["archives"]["slow"]["unit"] == "K"
    assert member["archives"]["slow"]["data_type"] == "float64"
    assert member["archives"]["fast"]["metadata_state"] == "unavailable"
    assert grouping.member("demo", pending)["state"] == "component"


def test_invalid_source_census_cannot_replace_grouping(tmp_path, grouping):
    grouping.rebuild("demo", records("DEMO-COOL:QAZ0001-ABCD"))
    before = grouping.summary("demo")
    source = tmp_path / "catalogue.sqlite"
    with closing(ArchiveCatalogue(source)) as catalogue:
        catalogue.seed({"slow": []}, timestamp="invented")
    with pytest.raises(RuntimeError, match="census is empty"):
        group_private_catalogue(source, grouping, "demo")
    assert grouping.summary("demo") == before


def test_grouping_snapshot_allows_metadata_writer_to_commit(
    tmp_path, grouping, monkeypatch
):
    source = tmp_path / "catalogue.sqlite"
    name = "DEMO-COOL:QAZ0001-ABCD"
    rebuild = grouping.rebuild
    with closing(ArchiveCatalogue(source)) as catalogue:
        catalogue.db.execute("PRAGMA busy_timeout=0")
        catalogue.seed({"slow": [name]}, timestamp="invented")

        def update_while_grouping(facility, items):
            items = iter(items)
            first = next(items)
            catalogue.save_batch(
                "slow",
                [name],
                {
                    "entries": [
                        {
                            "name": name,
                            "metadata_state": "unavailable",
                            "metadata_error_code": 17,
                            "metadata_error": "fixture refusal",
                        }
                    ]
                },
            )
            return rebuild(facility, chain([first], items))

        monkeypatch.setattr(grouping, "rebuild", update_while_grouping)
        receipt = group_private_catalogue(source, grouping, "demo")
        assert receipt["metadata_completed"] == 0
        assert grouping.member("demo", name)["archives"]["slow"]["completed"] is False
        assert catalogue.record(name)["archives"]["slow"]["completed"] is True


def test_runner_uses_store_accessor_and_publishes_failure_receipts(tmp_path):
    source, store, receipt = (
        tmp_path / name for name in ("source.sqlite", "groups.sqlite", "receipt.json")
    )
    with closing(ArchiveCatalogue(source)) as catalogue:
        catalogue.seed(
            {"slow": ["DEMO-UNIT:QAZ0001-ABCD", "unparseable fixture"]},
            timestamp="invented",
        )
    original = ArchiveGrouping.snapshot_records
    calls = []

    def snapshot(self, facility):
        calls.append(facility)
        return original(self, facility)

    with patch.object(ArchiveGrouping, "snapshot_records", snapshot):
        result = run_grouping(
            catalogue=source, store=store, facility="demo", receipt_path=receipt
        )
    assert calls == ["demo"]
    assert json.loads(receipt.read_text()) == result
    assert result["status"] == "complete"
    assert result["first"]["ungrouped"] == 1
    assert (
        result["repeat"]["groups_added"]
        == result["repeat"]["control_groups_added"]
        == result["repeat"]["patterns_added"]
        == 0
    )
    with pytest.raises(ValueError, match="separate"):
        run_grouping(
            catalogue=source, store=source, facility="demo", receipt_path=receipt
        )
    with pytest.raises(sqlite3.OperationalError):
        run_grouping(
            catalogue=tmp_path / "missing.sqlite",
            store=store,
            facility="demo",
            receipt_path=receipt,
        )
    assert json.loads(receipt.read_text())["status"] == "failed"
