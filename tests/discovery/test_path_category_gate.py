"""Path actions use stored category judgments and child-listing evidence."""

import json
from pathlib import Path

import pytest
from sklearn.metrics import roc_auc_score

from imas_codex.discovery.paths.scorer import (
    CODE_BEARING_PURPOSES,
    build_path_judgment_questions,
    path_category_gate,
    path_judgment_fields,
)


def _answers(purpose: str, children: float) -> dict:
    questions = build_path_judgment_questions()
    options = questions["path_purpose"]["criteria"]
    answers = {
        "path_purpose": {
            "choice": purpose,
            "probabilities": {name: float(name == purpose) for name in options},
            "confidence": 1.0,
        },
        "children_worth_listing": {"noul": children},
    }
    for name in questions:
        if name.startswith("score_"):
            answers[name] = {
                "score": 3,
                "probabilities": {"0": 0, "1": 0, "2": 0, "3": 1},
                "confidence": 1,
            }
    return answers


@pytest.mark.parametrize(
    ("purpose", "children", "scan", "expand"),
    [
        ("analysis_code", 0.1, True, False),
        ("container", 0.8, False, True),
        ("archive", 0.9, False, False),
        ("modeling_data", 0.9, False, False),
    ],
)
def test_category_actions_ignore_high_facet_scores(purpose, children, scan, expand):
    fields = path_judgment_fields(_answers(purpose, children), "test-judge")
    assert (fields["scan_relevance"] >= 0.3) is scan
    assert fields["should_expand"] is expand
    assert fields["should_enrich"] is scan


def test_scored_container_claim_excludes_archive(monkeypatch):
    from imas_codex.discovery.paths import parallel

    queries = []

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def query(self, statement, **params):
            queries.append((statement, params))
            if "RETURN p.id AS id" in statement:
                return [
                    {
                        "id": "jt-60sa:/home/example",
                        "path": "/home/example",
                        "depth": 2,
                        "is_expanding": True,
                    }
                ]
            return []

    monkeypatch.setattr("imas_codex.graph.GraphClient", FakeGraph)
    claimed = parallel.claim_paths_for_expanding(
        "jt-60sa", limit=1, root_filter=["/home/example"]
    )
    assert [item["path"] for item in claimed] == ["/home/example"]
    claim, params = next(
        (query, params) for query, params in queries if "SET p.claimed_at" in query
    )
    assert "scored" in params["expand_statuses"]
    assert "archive" in params["excluded_purposes"]
    assert "p.children_worth_listing >= $expand_threshold" in claim
    assert "p.scan_relevance" not in claim
    assert "p.should_expand" not in claim


def test_stored_judgments_refresh_scan_gate_without_rejudging(monkeypatch):
    from imas_codex.discovery.paths import parallel

    calls = []

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def query(self, statement, **params):
            calls.append((statement, params))
            return [{"updated": 1 if len(calls) == 1 else 0}]

    monkeypatch.setattr("imas_codex.graph.GraphClient", FakeGraph)
    assert parallel.refresh_stored_path_gates("jt-60sa", batch_size=1) == 1
    query, params = calls[0]
    assert "p.path_purpose_probs[index]" in query
    assert "SET p.scan_relevance = score" in query
    assert "archive" in params["excluded_purposes"]
    options = list(build_path_judgment_questions()["path_purpose"]["criteria"])
    assert {options[index] for index in params["scan_indexes"]} == CODE_BEARING_PURPOSES


def test_category_gate_reproduces_source_label_auc():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/evidence/fragments/discovery-judgments-through-jev/labelled-path-categories.json"
    )
    rows = json.loads(path.read_text())
    options = list(build_path_judgment_questions()["path_purpose"]["criteria"])
    units = {}
    for row in rows:
        if "judgment" in row:
            units.setdefault(row["canonical_copy"], row)
    labels, scores = [], []
    for row in units.values():
        judgment = row["judgment"]
        distribution = dict(zip(options, judgment["path_purpose_probs"], strict=True))
        scan, expand = path_category_gate(
            distribution, judgment["children_worth_listing"]
        )
        labels.append(row["category"] in CODE_BEARING_PURPOSES | {"container"})
        scores.append(max(scan, expand))
    auc = roc_auc_score(labels, scores)
    assert len(units) == 222
    assert sum(labels) == 134
    assert 0.8858 <= auc <= 0.9608
    assert auc == pytest.approx(0.9244, abs=0.0001)
