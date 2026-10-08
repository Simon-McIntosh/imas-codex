"""The path decision questions track the FacilityPath schema."""

from __future__ import annotations

from imas_codex.discovery.paths.scorer import (
    build_path_judgment_questions,
    build_path_judgment_state,
    path_judgment_fields,
    path_scan_relevance,
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
        and name
        not in {"score_composite", "score_percentile", "score_reason", "score_cost"}
        and not name.endswith(("_probs", "_confidence"))
        and slot["type"] == "float"
    }
    assert set(questions) == score_fields | {
        "path_purpose",
        "children_worth_listing",
    }
    for field in score_fields:
        assert questions[field]["type"] == "score"
        slots = schema.get_all_slots("FacilityPath")
        assert slots[f"{field}_probs"]["multivalued"] is True
        assert slots[f"{field}_confidence"]["type"] == "float"
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


def test_typed_path_judgment_stores_distributions_and_computes_gates():
    questions = build_path_judgment_questions()
    options = list(questions["path_purpose"]["criteria"])
    answers = {
        "path_purpose": {
            "choice": "analysis_code",
            "probabilities": {name: float(name == "analysis_code") for name in options},
            "confidence": 0.91,
        },
        "children_worth_listing": {"noul": 0.81},
    }
    for name in questions:
        if name.startswith("score_"):
            probabilities = {"0": 0.4, "1": 0.2, "2": 0.3, "3": 0.1}
            answers[name] = {
                "score": 1.1 if name == "score_data_access" else 0,
                "probabilities": probabilities,
                "confidence": 0.6,
            }
    fields = path_judgment_fields(answers, "test-judge", prefix="triage")
    assert fields["path_purpose_probs"] == [
        float(x == "analysis_code") for x in options
    ]
    assert fields["path_purpose_confidence"] == 0.91
    assert fields["triage_data_access"] == 1.1 / 3
    assert fields["triage_data_access_probs"] == [0.4, 0.2, 0.3, 0.1]
    assert fields["triage_data_access_confidence"] == 0.6
    assert fields["scan_relevance"] == 1.1 / 3
    assert fields["should_enrich"] is True
    assert fields["should_expand"] is True
    assert fields["judgment_model"] == "test-judge"
    assert (
        path_scan_relevance({"score_documentation": 1, "score_data_access": 0.1}) == 0.1
    )


def test_code_scan_claim_uses_judged_facets(monkeypatch):
    from imas_codex.discovery.code import graph_ops

    queries = []

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def query(self, statement, **_params):
            queries.append(statement)
            return []

    monkeypatch.setattr(graph_ops, "GraphClient", FakeGraph)
    graph_ops.claim_paths_for_file_scan("jt-60sa", limit=1)
    claim = next(query for query in queries if "SET p.files_claimed_at" in query)
    assert "p.scan_relevance >= $min_score" in claim
    assert "p.score_composite >= $min_score" not in claim
