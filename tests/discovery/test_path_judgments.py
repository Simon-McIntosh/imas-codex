"""The path decision questions track the FacilityPath schema."""

from __future__ import annotations

from imas_codex.discovery.paths.scorer import (
    build_path_judgment_questions,
    build_path_judgment_state,
)
from imas_codex.graph.schema import get_schema


def test_path_questions_cover_schema_purposes_and_scores():
    schema = get_schema()
    questions = build_path_judgment_questions()
    purposes = schema.get_enum_with_descriptions("PathPurpose") or []
    expected_purposes = {item["value"] for item in purposes} - {"empty"}
    expected_purposes.add("other")
    assert set(questions["path_purpose"]["criteria"]) == expected_purposes
    assert questions["path_purpose"]["criteria"]["empty_directory"]
    for item in purposes:
        if item["value"] != "empty":
            assert (
                questions["path_purpose"]["criteria"][item["value"]]
                == item["description"]
            )

    score_fields = {
        name
        for name, slot in schema.get_all_slots("FacilityPath").items()
        if name.startswith("score_")
        and name not in {"score_composite", "score_percentile", "score_cost"}
        and slot["type"] == "float"
    }
    assert set(questions) == score_fields | {
        "path_purpose",
        "children_worth_listing",
    }
    for field in score_fields:
        assert questions[field]["type"] == "score"
        assert (
            schema.get_all_slots("FacilityPath")[field]["description"].lower()
            in questions[field]["instructions"]
        )
        assert len(questions[field]["criteria"]) >= 3
    assert questions["children_worth_listing"]["type"] == "noul"


def test_path_state_carries_scanner_evidence_and_facility_patterns():
    row = {
        "path": "/analysis/src/getseldata_v4.2",
        "depth": 2,
        "total_files": 12,
        "total_dirs": 3,
        "child_names": '["src/", "README"]',
        "file_type_counts": '{".f": 8, ".h": 4}',
        "tree_context": "src/\nREADME",
        "has_readme": True,
        "has_makefile": True,
        "vcs_type": "git",
        "patterns_detected": ["eddbreadTime"],
        "description": "Reads EDDB channels into SELENE arrays",
    }
    config = {
        "data_access_patterns": {
            "primary_method": "edas",
            "key_tools": ["getseldata"],
            "code_import_patterns": ["eddbreadTime"],
        }
    }
    state = build_path_judgment_state(row, "jt-60sa", config)
    directory = state["directory"]
    assert directory["path"] == row["path"]
    assert directory["file_type_counts"] == {".f": 8, ".h": 4}
    assert directory["child_names"] == ["src/", "README"]
    assert directory["tree_context"] == row["tree_context"]
    assert directory["patterns_detected"] == ["eddbreadTime"]
    assert directory["has_readme"] and directory["has_makefile"]
    assert state["facility"]["data_access_tools"] == ["getseldata"]
    assert state["facility"]["data_access_code_patterns"] == ["eddbreadTime"]
