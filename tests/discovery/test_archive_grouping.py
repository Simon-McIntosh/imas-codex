"""Component grouping with invented names and exact, complete field sets."""

import json
import sqlite3
from contextlib import closing
from itertools import chain

import pytest

from imas_codex.discovery.signals.archive_grouping import (
    ArchiveGrouping,
    group_private_catalogue,
)
from imas_codex.remote.archive_catalogue import ArchiveCatalogue
from imas_codex.remote.catalogue_grouping import run_grouping


@pytest.fixture
def grouping(tmp_path):
    with closing(ArchiveGrouping(tmp_path / "groups.sqlite")) as store:
        yield store


def records(*names):
    return [{"name": name} for name in names]


def test_components_share_only_identical_complete_field_sets(grouping):
    names = [
        "DEMO-COOL-PUMP:FCV9001-MEAS",
        "DEMO-COOL-PUMP:FCV9001-TARGET",
        "DEMO-COOL-VALVE:FCV9001-TARGET",
        "DEMO-COOL-VALVE:FCV9001-MEAS",
        "DEMO-HEAT-PUMP:FCV9001-MEAS",
        "DEMO-HEAT-VALVE:FCV9002-MEAS",
    ]
    receipt = grouping.rebuild("demo", records(*reversed(names)))
    assert (receipt["variables"], receipt["grouped"], receipt["ungrouped"]) == (6, 6, 0)
    assert receipt["component_groups"] == 4
    assert receipt["field_patterns"] == 3
    assert receipt["group_size_distribution"] == {"1": 2, "2": 2}
    assert receipt["singleton_signal_share"] == 2 / 6
    assert receipt["group_descriptions_needed"] == 4
    assert receipt["pattern_descriptions_needed"] == 3
    components = [
        grouping.component(grouping.member("demo", name)["component"]) for name in names
    ]
    assert (
        components[0]["archive_field_pattern"] == components[2]["archive_field_pattern"]
    )
    assert (
        components[0]["archive_field_pattern"] != components[4]["archive_field_pattern"]
    )
    assert (
        components[4]["archive_field_pattern"] != components[5]["archive_field_pattern"]
    )
    largest = receipt["largest_patterns"][0]
    assert largest["fields"] == ["FCV9001-MEAS", "FCV9001-TARGET"]
    assert (
        largest["field_count"],
        largest["component_count"],
        largest["member_count"],
    ) == (2, 2, 4)
    repeat = grouping.rebuild("demo", records(*names))
    assert repeat["groups_added"] == repeat["patterns_added"] == 0
    assert repeat["groups_removed"] == repeat["patterns_removed"] == 0
    assert repeat["largest_patterns"] == receipt["largest_patterns"]


@pytest.mark.parametrize(
    "name", ["unparseable fixture", "DEMO-COOL:FCV9001", "DEMO-COOL:UNKNOWN-MEAS"]
)
def test_unparseable_name_is_counted_without_component_membership(grouping, name):
    receipt = grouping.rebuild("demo", records("DEMO-COOL:FCV9001-MEAS", name))
    assert receipt["variables"] == 2
    assert receipt["grouped"] == receipt["ungrouped"] == 1
    assert receipt["component_groups"] == receipt["field_patterns"] == 1
    member = grouping.member("demo", name)
    assert member["state"] == "ungrouped"
    assert member["reason"]
    assert member["component"] is None
    assert member["field"] is None


def test_growth_changes_pattern_without_duplicating_component(grouping):
    initial = records("DEMO-COOL:FCV9001-MEAS")
    first = grouping.rebuild("demo", initial)
    component = grouping.member("demo", initial[0]["name"])["component"]
    enlarged = initial + records("DEMO-COOL:FCV9001-TARGET")
    grown = grouping.rebuild("demo", enlarged)
    assert grouping.member("demo", initial[0]["name"])["component"] == component
    assert grown["groups_added"] == grown["groups_removed"] == 0
    assert grown["patterns_added"] == grown["patterns_removed"] == 1
    assert grown["largest_patterns"][0]["id"] != first["largest_patterns"][0]["id"]
    assert grouping.component(component)["archive_member_count"] == 2
    assert grouping.rebuild("demo", enlarged)["patterns_added"] == 0


def test_failed_or_duplicate_snapshot_preserves_committed_groups(grouping):
    initial = records("DEMO-COOL:FCV9001-MEAS")
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


def test_facilities_have_independent_membership_and_pattern_identity(grouping):
    names = records("DEMO-COOL:FCV9001-MEAS")
    alpha = grouping.rebuild("alpha", names)
    beta = grouping.rebuild("beta", names)
    assert alpha["largest_patterns"][0]["id"] != beta["largest_patterns"][0]["id"]
    grouping.rebuild("alpha", [])
    assert grouping.summary("beta")["variables"] == 1
    assert grouping.member("beta", names[0]["name"])["state"] == "grouped"


def test_largest_patterns_are_ranked_by_signal_count_and_bounded(grouping):
    # Every component exposes a different exact field set.
    items = records(*(f"DEMO-UNIT{index}:FCV{index}-MEAS" for index in range(25)))
    items += records("DEMO-EXTRA:FCV0-MEAS")
    receipt = grouping.rebuild("demo", items)
    assert receipt["field_patterns"] == 25
    assert len(receipt["largest_patterns"]) == 20
    assert receipt["largest_patterns"][0]["member_count"] == 2
    assert all(pattern["field_count"] == 1 for pattern in receipt["largest_patterns"])


def test_private_catalogue_union_keeps_pending_refused_and_graph_only_variables(
    tmp_path, grouping
):
    source = tmp_path / "catalogue.sqlite"
    measured = "DEMO-COOL:FCV9001-MEAS"
    pending = "DEMO-COOL:FCV9001-TARGET"
    graph_only = "DEMO-HEAT:FCV9001-MEAS"
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
    assert (receipt["variables"], receipt["grouped"], receipt["ungrouped"]) == (4, 3, 1)
    member = grouping.member("demo", measured)
    assert member["archives"]["slow"]["unit"] == "K"
    assert member["archives"]["slow"]["data_type"] == "float64"
    assert member["archives"]["fast"]["metadata_state"] == "unavailable"
    assert grouping.member("demo", pending)["state"] == "grouped"


def test_invalid_source_census_cannot_replace_grouping(tmp_path, grouping):
    grouping.rebuild("demo", records("DEMO-COOL:FCV9001-MEAS"))
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
    name = "DEMO-COOL:FCV9001-MEAS"
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


def test_runner_publishes_repeated_snapshot_and_a_failure_receipt(tmp_path):
    source, store, receipt = (
        tmp_path / name for name in ("source.sqlite", "groups.sqlite", "receipt.json")
    )
    with closing(ArchiveCatalogue(source)) as catalogue:
        catalogue.seed(
            {"slow": ["DEMO-COOL:FCV9001-MEAS", "unparseable fixture"]},
            timestamp="invented",
        )
    result = run_grouping(
        catalogue=source, store=store, facility="demo", receipt_path=receipt
    )
    assert json.loads(receipt.read_text()) == result
    assert result["status"] == "complete"
    assert result["first"]["ungrouped"] == 1
    assert result["repeat"]["groups_added"] == result["repeat"]["patterns_added"] == 0
    with pytest.raises(ValueError, match="separate"):
        run_grouping(
            catalogue=source, store=source, facility="demo", receipt_path=receipt
        )
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        run_grouping(
            catalogue=missing, store=store, facility="demo", receipt_path=receipt
        )
    assert json.loads(receipt.read_text())["status"] == "failed"
