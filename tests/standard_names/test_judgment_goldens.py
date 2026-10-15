"""Independent-label provenance and composition checks for judgment fixtures."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from copy import deepcopy

import pytest

from imas_codex.discovery.base.evaluation import load_labelled_fixture
from imas_codex.standard_names.judgment_goldens import (
    OUTPUT,
    _finish,
    _row,
    _west_verdicts,
    assign_group_splits,
    build_documentation_sheet,
    build_near_duplicate,
    write_fixture,
    write_sheet,
)

FIXTURES = (
    "golden-name-acceptable.json",
    "golden-description-object.json",
    "golden-near-duplicate.json",
)
LEAKED_SOURCE = re.compile(
    r"standard_?name_?review|reviewer[_ -]?(score|tier|verdict|rating)|"
    r"review[_ -]?tier|name_stage|docs_stage|accepted[_ -]?status",
    re.IGNORECASE,
)


def assert_independent_sources(rows: list[dict]) -> None:
    """Reject label provenance copied from the incumbent's review state."""
    for row in rows:
        assert not LEAKED_SOURCE.search(row["label_source"]), row


@pytest.mark.parametrize("filename", FIXTURES)
def test_fixtures_load_with_independent_labels(filename: str) -> None:
    rows = load_labelled_fixture(OUTPUT / filename)
    assert rows
    assert_independent_sources(rows)
    assert {row["split"] for row in rows} == {"calibration", "validation"}
    assert all(row["label"] in (0, 1) for row in rows)
    assert all(row["source_evidence"] for row in rows)
    assert len({row["domain"] for row in rows}) >= 3
    assert len({row["pattern"] for row in rows}) >= 3


def test_name_source_caps_and_balance() -> None:
    rows = load_labelled_fixture(OUTPUT / "golden-name-acceptable.json")
    assert 700 <= len(rows) <= 800
    sources = Counter(row["label_source"] for row in rows)
    assert max(sources.values()) <= 200
    assert {row["label"] for row in rows} == {0, 1}
    edits: dict[str, set[int]] = defaultdict(set)
    copies: dict[str, set[str]] = defaultdict(set)
    labels_by_unit: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        copies[row["canonical_unit"]].add(row["split"])
        labels_by_unit[row["canonical_unit"]].add(row["label"])
        if row["label_source"].startswith("human_edit_"):
            edits[row["copy_group"]].add(row["label"])
    assert edits and all(labels == {0, 1} for labels in edits.values())
    assert all(len(splits) == 1 for splits in copies.values())
    assert all(len(labels) == 1 for labels in labels_by_unit.values())


def test_rows_naming_one_item_share_canonical_unit() -> None:
    left = _row("dd/one/x::shared_name", 0, "src", "ev", canonical="shared_name")
    right = _row("dd/two/y::shared_name", 0, "src", "ev", canonical="shared_name")
    assert left["unit"] != right["unit"]
    assert left["canonical_unit"] == right["canonical_unit"] == "shared_name"

    rows = load_labelled_fixture(OUTPUT / "golden-name-acceptable.json")
    units_by_name: dict[str, set[str]] = defaultdict(set)
    canonical_by_name: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        name = row["unit"].rsplit("::", 1)[-1]
        units_by_name[name].add(row["unit"])
        canonical_by_name[name].add(row["canonical_unit"])
    assert any(len(units) > 1 for units in units_by_name.values())
    assert all(len(canonical) == 1 for canonical in canonical_by_name.values())
    assert len({row["canonical_unit"] for row in rows}) < len(rows)


def test_finished_rows_use_shared_group_split_rule() -> None:
    groups = ("alpha", "beta", "gamma", "delta", "alpha")
    rows = [
        _row(f"fixture/{index}", 0, "fixture", "fixture", group=group)
        for index, group in enumerate(groups)
    ]
    expected = {
        "alpha": "calibration",
        "beta": "calibration",
        "gamma": "validation",
        "delta": "validation",
    }
    assert assign_group_splits(rows, "copy_group") == expected
    finished = _finish(deepcopy(rows), _StubGraph([]))
    assert {row["unit"]: row["split"] for row in finished} == {
        row["unit"]: expected[row["copy_group"]] for row in rows
    }


def test_rebuild_keeps_every_existing_groups_split(tmp_path) -> None:
    golden = tmp_path / "golden.json"
    base = [
        _row(f"fixture/{group}", 0, "fixture", "fixture", group=group)
        for group in ("alpha", "beta", "gamma", "delta")
    ]
    first = _finish(deepcopy(base), _StubGraph([]), golden_path=None)
    write_fixture(golden, first)
    before = {row["copy_group"]: row["split"] for row in first}
    committed = {
        row["copy_group"]: row["split"] for row in json.loads(golden.read_text())
    }
    assert committed == before

    new_groups = ("epsilon", "zeta", "eta", "theta", "iota")
    grown = base + [
        _row(f"fixture/{group}", 0, "fixture", "fixture", group=group)
        for group in new_groups
    ]
    rebuilt = _finish(deepcopy(grown), _StubGraph([]), golden_path=golden)
    after = {row["copy_group"]: row["split"] for row in rebuilt}
    assert set(after) == set(before) | set(new_groups)
    assert {group: after[group] for group in before} == before

    fresh = Counter(after[group] for group in new_groups)
    assert abs(fresh["calibration"] - fresh["validation"]) <= 1


def test_golden_without_a_committed_file_splits_as_new(tmp_path) -> None:
    rows = [
        _row(f"fixture/{group}", 0, "fixture", "fixture", group=group)
        for group in ("alpha", "beta", "gamma", "delta")
    ]
    absent = tmp_path / "absent.json"
    finished = _finish(deepcopy(rows), _StubGraph([]), golden_path=absent)
    expected = assign_group_splits(rows, "copy_group")
    assert {row["unit"]: row["split"] for row in finished} == {
        row["unit"]: expected[row["copy_group"]] for row in rows
    }


def test_name_golden_admits_no_binding_positive() -> None:
    rows = load_labelled_fixture(OUTPUT / "golden-name-acceptable.json")
    assert any(
        row["label_source"] == "west_audit" and row["label"] == 0 for row in rows
    )
    assert not any(
        row["label_source"] == "west_audit" and row["label"] == 1 for row in rows
    )


def test_independent_source_guard_rejects_review_derived_label() -> None:
    planted = [
        {
            "unit": "planted::reviewer_tier",
            "label": 1,
            "canonical_unit": "planted",
            "copy_group": "planted",
            "label_source": "reviewer_tier",
            "source_evidence": "planted",
        }
    ]
    with pytest.raises(AssertionError):
        assert_independent_sources(planted)


def test_object_source_caps() -> None:
    rows = load_labelled_fixture(OUTPUT / "golden-description-object.json")
    assert 200 <= len(rows) <= 330
    west_count = sum(row["label_source"] == "west_audit" for row in rows)
    assert west_count <= 150
    assert west_count * 2 <= len(rows)
    assert {row["label"] for row in rows} == {0, 1}
    recorded_description_issues = {
        f"{item['path']}::{item['name']}"
        for item in _west_verdicts()
        if item["description_issue"]
    }
    assert recorded_description_issues
    assert not any(
        row["label"] == 1 and row["unit"] in recorded_description_issues for row in rows
    )


def test_recorded_name_verdicts() -> None:
    rows = load_labelled_fixture(OUTPUT / "golden-name-acceptable.json")
    by_unit = {row["unit"]: row for row in rows}
    assert (
        by_unit[
            "camera_ir/channel/camera/frame/apparent_temperature::surface_temperature"
        ]["label"]
        == 0
    )
    assert by_unit["equilibrium/time_slice/profiles_1d/q::safety_factor"]["label"] == 1
    assert (
        by_unit["expert-review-reply:comment=10::thermal_electron_energy"]["label"] == 1
    )


class _EmptyGraph:
    """Answers every read with no rows, so only recorded verdicts build rows."""

    def query(self, cypher: str, **params):
        return []


RULED_SOURCE = "coordinator_synonym_ruling"
CHARGE_NUMBER_NAMES = {
    "average_charge_number",
    "ion_charge",
    "ion_charge_state_average_charge_number",
}


def _ruled(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["label_source"] == RULED_SOURCE]


def _rulings() -> list[dict]:
    path = OUTPUT / "golden-synonym-rulings.json"
    return json.loads(path.read_text())["rows"]


def test_path_pair_rows_stay_on_the_distinct_side() -> None:
    rows = load_labelled_fixture(OUTPUT / "golden-near-duplicate.json")
    others = [row for row in rows if row["label_source"] != RULED_SOURCE]
    assert others
    assert all(row["label"] == 0 for row in others)
    assert {row["label_source"] for row in rows} == {
        "west_shared_identity",
        "isn_duplicate_path_review",
        RULED_SOURCE,
    }


@pytest.mark.parametrize("origin", ("built", "committed"))
def test_near_duplicate_carries_every_ruled_pair(origin: str) -> None:
    if origin == "built":
        rows = build_near_duplicate(_EmptyGraph())
    else:
        rows = load_labelled_fixture(OUTPUT / "golden-near-duplicate.json")
    ruled = _ruled(rows)
    assert len(ruled) == 47
    assert sum(row["label"] == 1 for row in ruled) == 7
    by_unit = {row["unit"]: row for row in ruled}
    for ruling in _rulings():
        left, right = sorted((ruling["left"], ruling["right"]))
        row = by_unit[f"{left}::{right}"]
        assert row["label"] == ruling["label"]
        assert row["canonical_unit"] == row["unit"]
        assert row["source_evidence"] == (
            f"golden-synonym-rulings.json:pair={ruling['pair']}"
        )


@pytest.mark.parametrize("origin", ("built", "committed"))
def test_ruled_names_sit_in_exactly_one_copy_group(origin: str) -> None:
    if origin == "built":
        rows = build_near_duplicate(_EmptyGraph())
    else:
        rows = load_labelled_fixture(OUTPUT / "golden-near-duplicate.json")
    groups: dict[str, set[str]] = defaultdict(set)
    splits: dict[str, set[str]] = defaultdict(set)
    for row in _ruled(rows):
        for name in row["unit"].split("::"):
            groups[name].add(row["copy_group"])
        splits[row["copy_group"]].add(row["split"])
    assert groups
    assert all(len(found) == 1 for found in groups.values())
    assert all(len(found) == 1 for found in splits.values())
    assert len(set.union(*groups.values())) < len(_ruled(rows))
    charge = {next(iter(groups[name])) for name in CHARGE_NUMBER_NAMES}
    assert charge == {"average_charge_number"}


def test_built_ruled_rows_match_the_committed_fixture() -> None:
    built = {row["unit"]: row for row in _ruled(build_near_duplicate(_EmptyGraph()))}
    stored = {
        row["unit"]: row
        for row in _ruled(load_labelled_fixture(OUTPUT / "golden-near-duplicate.json"))
    }
    assert built.keys() == stored.keys()
    for unit, row in built.items():
        for key in ("label", "copy_group", "label_source", "source_evidence"):
            assert row[key] == stored[unit][key], (unit, key)


# --- documentation-acceptability sheet -------------------------------------

DOCUMENTATION_DOMAINS = ("core_profiles", "mhd", "equilibrium")
DOCUMENTATION_FAMILIES = {
    "temperature": ("electron_temperature", "ion_temperature", "total_temperature"),
    "density": ("electron_density", "ion_density", "neutral_density"),
    "current": ("plasma_current", "bootstrap_current", "ohmic_current"),
    "power": ("heating_power", "radiated_power", "auxiliary_power"),
}
DD_SOURCE = "equilibrium/time_slice/profiles_1d/psi"
INCIDENT_GUARD = re.compile(r"incumbent|review|score|verdict|tier", re.IGNORECASE)


class _StubGraph:
    """Serves the sheet builder the sheet-local reads and nothing else."""

    def __init__(self, names: list[dict]) -> None:
        self._names = names

    def query(self, cypher: str, **params):
        if "UNWIND $names" in cypher:
            return []
        if "sn.name_stage IN" in cypher:
            return list(self._names)
        if "MATCH (unit:Unit" in cypher:
            return [{"id": unit, "symbol": unit} for unit in params.get("units", [])]
        if "MATCH (dd:IMASNode" in cypher:
            return [
                {
                    "path": path,
                    "documentation": f"DD documentation for {path}. " + "d" * 700,
                }
                for path in params.get("paths", [])
            ]
        return []


def _stub_names() -> list[dict]:
    names = []
    index = 0
    for members in DOCUMENTATION_FAMILIES.values():
        for member in members:
            names.append(
                {
                    "name": member,
                    "description": f"Description of {member}.",
                    "documentation": f"Documentation of {member}. " + member * 5,
                    "unit": "eV",
                    "physics_domain": DOCUMENTATION_DOMAINS[index % 3],
                    "source_paths": [f"dd:{DD_SOURCE}"] if index % 2 == 0 else None,
                }
            )
            index += 1
    names.append(
        {
            "name": "electron_temperature_of_plasma",
            "description": "An id the strict parse refuses.",
            "documentation": "Documentation of a refused id.",
            "unit": "eV",
            "physics_domain": "core_profiles",
            "source_paths": None,
        }
    )
    names.sort(key=lambda row: row["name"])
    return names


def _sheet_rows(count: int = 8, reserve: int = 5, lead_overlap: int = 3):
    return build_documentation_sheet(
        _StubGraph(_stub_names()),
        count=count,
        reserve=reserve,
        lead_overlap=lead_overlap,
    )


def test_documentation_sheet_draws_deterministic_rank_order(tmp_path) -> None:
    count, reserve = 8, 5
    rows = _sheet_rows(count=count, reserve=reserve)
    assert [row["draw_rank"] for row in rows] == list(range(1, count + reserve + 1))
    assert len({row["unit"] for row in rows}) == count + reserve
    assert [row["draw_rank"] for row in rows] == sorted(
        row["draw_rank"] for row in rows
    )

    def reload() -> list[dict]:
        return build_documentation_sheet(
            _StubGraph(_stub_names()), count=count, reserve=reserve, lead_overlap=3
        )

    assert [row["unit"] for row in reload()] == [row["unit"] for row in rows]
    assert reload() == rows


def test_documentation_sheet_marks_exact_lead_overlap() -> None:
    count = 8
    rows = _sheet_rows(count=count, reserve=5, lead_overlap=3)
    marked = [row for row in rows if row["lead_overlap"]]
    assert len(marked) == 3
    assert all(row["draw_rank"] <= count for row in marked)
    assert not any(row["lead_overlap"] for row in rows if row["draw_rank"] > count)


def test_documentation_sheet_carries_no_incumbent_field() -> None:
    rows = _sheet_rows()
    for row in rows:
        leaking = [key for key in row if INCIDENT_GUARD.search(key)]
        assert not leaking, (row["unit"], leaking)
    assert all(row["label"] is None for row in rows)
    assert {row["label_source"] for row in rows} == {"coordinator"}


def test_documentation_sheet_keeps_family_on_one_split() -> None:
    rows = _sheet_rows(count=8, reserve=5, lead_overlap=3)
    by_family: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_family[row["pattern_family"]].add(row["split"])
    multi = [family for family, members in _family_counts(rows).items() if members > 1]
    assert multi
    assert all(len(splits) == 1 for splits in by_family.values())


def _family_counts(rows: list[dict]) -> dict[str, int]:
    return Counter(row["pattern_family"] for row in rows)


def test_documentation_sheet_loads_as_a_labelled_fixture(tmp_path) -> None:
    rows = _sheet_rows()
    path = tmp_path / "sheet.json"
    composition = write_sheet(path, rows, 8)
    loaded = load_labelled_fixture(path)
    assert len(loaded) == composition["rows"] == len(rows)
    assert composition["sheet"] == 8 and composition["reserve"] == 5
    assert composition["lead_overlap"] == 3
    assert all(row["source_evidence"] for row in loaded)
    assert all(len(row["source_evidence"]) == 64 for row in loaded)
    assert any(row["conventions"]["parse_refused"] for row in rows)


def test_documentation_sheet_truncates_dd_documentation() -> None:
    rows = _sheet_rows()
    sourced = [row for row in rows if row["dd_sources"]]
    assert sourced
    assert all(
        len(source["documentation"]) <= 600
        for row in sourced
        for source in row["dd_sources"]
    )
    assert all(len(row["dd_sources"]) <= 2 for row in rows)
