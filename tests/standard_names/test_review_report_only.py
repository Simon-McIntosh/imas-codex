"""The full review can report its findings without mutating the graph."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from click.testing import CliRunner

_CATALOG = [
    {
        "id": "electron_temperature",
        "description": "Electron temperature.",
        "documentation": "Electron temperature in the plasma.",
        "name_stage": "drafted",
        "physics_domain": "transport",
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
        "embedding": [0.1, 0.2],
    }
]


class _ObservedGraph:
    mutation_queries: list[str] = []
    review_nodes = 0

    def __enter__(self) -> _ObservedGraph:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    @contextmanager
    def session(self) -> Iterator[_ObservedGraph]:
        """Every read and write on the real client opens a session first."""
        yield self

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self.session() as session:
            return session.run(cypher, **params)

    def run(self, cypher: str, *_args: Any, **params: Any) -> list[dict[str, Any]]:
        from imas_codex.cli.sn import _cypher_mutation_clause

        if _cypher_mutation_clause(cypher) is not None:
            type(self).mutation_queries.append(cypher)
            if "StandardNameReview" in cypher:
                type(self).review_nodes += len(params.get("batch") or [None])
            return []
        if "MATCH (sn:StandardName)" in cypher:
            return deepcopy(_CATALOG)
        return []


def _summary() -> SimpleNamespace:
    return SimpleNamespace(
        total_scored=1,
        total_catalog_size=1,
        coverage_pct=100.0,
        total_unscored=0,
        total_cost=0.125,
        tier_distribution={"good": 1},
        duplicate_candidates=[],
        drift_warnings=[],
        outliers=[],
        lowest_scorers=[],
    )


def _install_review_layers(monkeypatch: Any, observed_layers: list[str]) -> None:
    from imas_codex.graph import client as graph_client
    from imas_codex.standard_names.review import audits, consolidation, pipeline

    monkeypatch.setattr(graph_client, "GraphClient", _ObservedGraph)

    def _run_audits(names: list[dict[str, Any]]) -> SimpleNamespace:
        observed_layers.append("Layer 1")
        assert [name["id"] for name in names] == ["electron_temperature"]
        with graph_client.GraphClient() as gc:
            gc.query(
                "MATCH (sn:StandardName {id: $id}) SET sn.embedding = $embedding",
                id="electron_temperature",
                embedding=[0.1, 0.2],
            )
        return SimpleNamespace(
            embedding=SimpleNamespace(
                missing_count=0, stale_count=0, refreshed_count=0
            ),
            lint_findings=[],
            link_findings=[],
            duplicate_components=[],
        )

    async def _run_engine(state: Any, *, stop_event: Any) -> None:
        observed_layers.append("Layer 2")
        assert stop_event is not None
        state.review_results = [
            {
                "id": "electron_temperature",
                "reviewer_score": 0.75,
                "review_tier": "good",
            }
        ]
        state.review_records = [
            {
                "id": "electron_temperature:name:group:0",
                "standard_name_id": "electron_temperature",
            }
        ]
        state.stats["review_cost"] = 0.125
        state.review_stats.cost = 0.125
        with graph_client.GraphClient() as gc:
            gc.query(
                "UNWIND $batch AS b MERGE (r:StandardNameReview {id: b.id})",
                batch=state.review_records,
            )
            gc.query(
                "UNWIND $batch AS b MATCH (sn:StandardName {id: b.id}) "
                "SET sn.reviewer_score_name = b.score",
                batch=[{"id": "electron_temperature", "score": 0.75}],
            )
            gc.query(
                "MATCH (sn:StandardName {id: $id}) SET sn.review_count = 1",
                id="electron_temperature",
            )

    def _consolidate(state: Any) -> SimpleNamespace:
        observed_layers.append("Layer 3")
        assert state.review_results[0]["reviewer_score"] == 0.75
        return _summary()

    monkeypatch.setattr(audits, "run_all_audits", _run_audits)
    monkeypatch.setattr(pipeline, "run_sn_review_engine", _run_engine)
    monkeypatch.setattr(consolidation, "run_consolidation", _consolidate)


def _review_summary_text(output: str) -> str:
    return output.split("Review Summary", maxsplit=1)[1]


def test_report_only_runs_every_layer_without_creating_review_nodes(
    monkeypatch: Any,
) -> None:
    """Report-only changes persistence, not the computed review or exit status."""
    from imas_codex.cli.sn import sn

    observed_layers: list[str] = []
    _install_review_layers(monkeypatch, observed_layers)

    runner = CliRunner()
    _ObservedGraph.mutation_queries = []
    _ObservedGraph.review_nodes = 0
    report_only = runner.invoke(
        sn,
        ["review", "--report-only", "--models", "test/reviewer"],
        catch_exceptions=False,
    )

    assert report_only.exit_code == 0
    assert observed_layers == ["Layer 1", "Layer 2", "Layer 3"]
    assert _ObservedGraph.mutation_queries == []
    assert _ObservedGraph.review_nodes == 0
    receipt = re.search(
        r"Report mode: report-only \((\d+) graph mutations suppressed\)",
        report_only.output,
    )
    assert receipt is not None, report_only.output
    # The guard sits on the session boundary, so it cannot receipt fewer
    # mutations than the narrower query-only aperture saw on this same run.
    assert int(receipt.group(1)) >= 4

    observed_layers.clear()
    _ObservedGraph.mutation_queries = []
    _ObservedGraph.review_nodes = 0
    persisting = runner.invoke(
        sn,
        ["review", "--models", "test/reviewer"],
        catch_exceptions=False,
    )

    assert persisting.exit_code == report_only.exit_code == 0
    assert observed_layers == ["Layer 1", "Layer 2", "Layer 3"]
    assert len(_ObservedGraph.mutation_queries) == 4
    assert _ObservedGraph.review_nodes == 1
    assert "Report mode: persist review results" in persisting.output
    assert _review_summary_text(report_only.output) == _review_summary_text(
        persisting.output
    )


def test_report_only_refuses_dry_run_combination() -> None:
    """A Layer-1 preview cannot masquerade as a full report-only review."""
    from imas_codex.cli.sn import sn

    result = CliRunner().invoke(sn, ["review", "--dry-run", "--report-only"])

    assert result.exit_code == 2
    assert "--dry-run and --report-only are mutually exclusive" in result.output


def test_report_only_refuses_skipping_the_audit_layer() -> None:
    """The report-only contract always includes deterministic review."""
    from imas_codex.cli.sn import sn

    result = CliRunner().invoke(sn, ["review", "--report-only", "--skip-audit"])

    assert result.exit_code == 2
    assert "runs all three layers" in result.output
