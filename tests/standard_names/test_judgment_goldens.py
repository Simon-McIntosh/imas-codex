"""Independent-label provenance and composition checks for judgment fixtures."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import pytest

from imas_codex.discovery.base.evaluation import load_labelled_fixture
from imas_codex.standard_names.judgment_goldens import OUTPUT, _west_verdicts

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
    assert sources["human_edit_source"] + sources["human_edit_target"] <= 200
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
