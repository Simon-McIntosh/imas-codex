"""Documentation-review method resolution at the export boundary."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import yaml

from imas_codex.standard_names.export import run_export


def _candidate(name: str, **overrides) -> dict:
    candidate = {
        "id": name,
        "name_stage": "accepted",
        "validation_status": "valid",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "docs_review_quorum_shortfall": None,
        "reviewer_score_name": 0.95,
        "reviewer_score_docs": 0.95,
        "description": f"Description for {name}.",
        "documentation": f"Documentation for {name}.",
        "kind": "scalar",
        "unit": "1",
        "physics_domain": "general",
        "links": [],
        "_has_dd_source_binding": True,
        "_has_derived_producer": False,
        "_has_non_derived_producer": True,
        "_is_parent": False,
    }
    candidate.update(overrides)
    return candidate


class _PopulationGraph:
    def __init__(self, population: list[dict]) -> None:
        self.population = population
        self.population_query = ""
        self.population_params: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        if "accepting_docs_reviews" in cypher:
            self.population_query = cypher
            self.population_params = params
            return [{"record": candidate} for candidate in self.population]
        return []


def test_export_resolves_accepting_docs_review_method_and_keeps_absence_closed(
    tmp_path: Path,
) -> None:
    graph = _PopulationGraph(
        [
            _candidate(
                "accepted_documentation",
                _has_docs_review=True,
                _has_winning_docs_review=True,
                docs_review_resolution_method="quorum_consensus",
                _docs_review_group_id="accepted-group",
                _docs_review_id="accepted-review",
            ),
            _candidate(
                "documentation_without_review",
                _has_docs_review=False,
                _has_winning_docs_review=False,
                docs_review_resolution_method=None,
                _docs_review_group_id=None,
                _docs_review_id=None,
            ),
        ]
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch("imas_codex.graph.client.GraphClient", return_value=graph)
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.graph_ops.docs_review_property_coverage",
                return_value={},
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._validate_entry",
                side_effect=lambda entry: entry,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_deprecation_stubs",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.export._fetch_ordering_edges_for_domain",
                return_value=([], set()),
            )
        )
        report = run_export(
            tmp_path,
            skip_gate=True,
            force=True,
            include_sources=False,
        )

    assert "accepting_review.resolution_method" in graph.population_query
    assert (
        "non_winner.review_group_id =\n                         "
        "accepting_review.review_group_id"
    ) in graph.population_query
    assert graph.population_params["docs_review_winning_methods"] == [
        "authoritative_escalation",
        "quorum_consensus",
        "single_review",
    ]
    assert report.exported_names == ["accepted_documentation"]
    assert [
        (record.standard_name_id, record.reason) for record in report.exclusion_records
    ] == [("documentation_without_review", "never_reviewed")]

    entries = yaml.safe_load(
        (tmp_path / "standard_names" / "general.yml").read_text(encoding="utf-8")
    )
    assert [entry["name"] for entry in entries] == ["accepted_documentation"]
