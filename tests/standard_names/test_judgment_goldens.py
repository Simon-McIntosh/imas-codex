"""Independent-label provenance and composition checks for judgment fixtures."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import pytest

from imas_codex.discovery.base.evaluation import load_labelled_fixture
from imas_codex.standard_names.judgment_goldens import OUTPUT, _row, _west_verdicts

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
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[row["copy_group"]].add(row["split"])
    assert all(len(splits) == 1 for splits in grouped.values())
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


def test_distinct_side_only() -> None:
    rows = load_labelled_fixture(OUTPUT / "golden-near-duplicate.json")
    assert all(row["label"] == 0 for row in rows)
    assert {row["label_source"] for row in rows} == {
        "west_shared_identity",
        "isn_duplicate_path_review",
    }
